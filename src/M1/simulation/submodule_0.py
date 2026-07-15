import sys
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RotationMatrix

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

KINEMATICS_DIR = SRC_DIR.parent / "materials" / "practical_3_planning"
if str(KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(KINEMATICS_DIR))

from kinematics import RobotKinematics
from scene import (
    CAMERA_TOOL0_OFFSET,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ROBOT_OFFSET_X,
    ROBOT_OFFSET_Y,
    TABLE_LENGTH,
    TABLE_WIDTH,
    TCP_OFFSET,
    build_arm_gripper_scene,
    resting_z_offset,
    set_gripper_opening,
)

LEGO_URDF_DIR = SRC_DIR.parent / "lego_3d" / "urdf"

BRICK_URDFS = {
    "69729": LEGO_URDF_DIR / "69729__light_bluish_gray.urdf",
    "3008": LEGO_URDF_DIR / "3008__tan.urdf",
    "3021": LEGO_URDF_DIR / "3021__tan.urdf",
}

STEP_DELAY = 0.05
REACH_STEPS = 40
DESCEND_STEPS = 25
GRIPPER_STEPS = 30
LIFT_STEPS = 40
INTER_BRICK_PAUSE = 2.0
LIFT_HEIGHT = 0.15
PREGRASP_CLEARANCE = 0.12

MIN_REACH = 0.25
MAX_REACH = 0.45
MAX_SAMPLE_ATTEMPTS = 20

MIN_HEIGHT_ABOVE_TABLE = 0.002
MAX_HEIGHT_ABOVE_TABLE = 0.05
BASE_EXCLUSION_RADIUS = 0.15

GRASP_ROTATION_BASE = RotationMatrix.MakeXRotation(np.pi)

OBSERVE_EYE_RADIUS = 0.15
OBSERVE_EYE_HEIGHT = 0.4
OBSERVE_EYE_ANGLE = np.deg2rad(35)
OBSERVE_TARGET = np.array([0.2, 0.15, 0.0])
OBSERVE_INITIAL_GUESS = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _look_at_rotation(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.97:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.column_stack([right, -true_up, forward])


def _observe_tcp_pose() -> RigidTransform:
    eye = np.array(
        [
            OBSERVE_EYE_RADIUS * np.cos(OBSERVE_EYE_ANGLE),
            OBSERVE_EYE_RADIUS * np.sin(OBSERVE_EYE_ANGLE),
            OBSERVE_EYE_HEIGHT,
        ]
    )
    R_camera = _look_at_rotation(eye, OBSERVE_TARGET)
    X_W_Camera = RigidTransform(RotationMatrix(R_camera), eye)
    X_Tool0_Tcp = RigidTransform(RotationMatrix.Identity(), [0, 0, TCP_OFFSET])
    X_Tcp_Camera = X_Tool0_Tcp.inverse() @ CAMERA_TOOL0_OFFSET
    return X_W_Camera @ X_Tcp_Camera.inverse()


def _interpolate(start, end, num_steps: int):
    if num_steps <= 1:
        return [end]
    return np.linspace(start, end, num_steps)


def _brick_body_name(plant, brick_index) -> str:
    body_indices = plant.GetBodyIndices(brick_index)
    return plant.get_body(body_indices[0]).name()


def _sample_ground_truth_pose(rng: np.random.Generator, z: float):
    x_min, x_max = -ROBOT_OFFSET_X, TABLE_LENGTH - ROBOT_OFFSET_X
    y_min, y_max = -ROBOT_OFFSET_Y, TABLE_WIDTH - ROBOT_OFFSET_Y

    x, y = MIN_REACH, 0.0
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if MIN_REACH <= np.hypot(x, y) <= MAX_REACH:
            break
    else:
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(MIN_REACH, MAX_REACH)
        x, y = radius * np.cos(angle), radius * np.sin(angle)

    yaw = rng.uniform(0, 2 * np.pi)
    return RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, z])


def capture_world_points(plant, context, sensor, camera_pose: RigidTransform):
    sensor_context = sensor.GetMyContextFromRoot(context)
    depth_image = sensor.depth_image_32F_output_port().Eval(sensor_context)
    depth = np.array(depth_image.data).squeeze(-1)
    intrinsics = sensor.depth_camera_info()
    fx, fy = intrinsics.focal_x(), intrinsics.focal_y()
    cx, cy = intrinsics.center_x(), intrinsics.center_y()

    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    valid = np.isfinite(depth) & (depth > 0)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    points_cam = np.stack([x[valid], y[valid], depth[valid]], axis=-1)

    R = camera_pose.rotation().matrix()
    t = camera_pose.translation()
    return points_cam @ R.T + t


def estimate_brick_pose(points_world):
    x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    mask = (
        (z > MIN_HEIGHT_ABOVE_TABLE)
        & (z < MAX_HEIGHT_ABOVE_TABLE)
        & (x > -ROBOT_OFFSET_X)
        & (x < TABLE_LENGTH - ROBOT_OFFSET_X)
        & (y > -ROBOT_OFFSET_Y)
        & (y < TABLE_WIDTH - ROBOT_OFFSET_Y)
        & (np.hypot(x, y) > BASE_EXCLUSION_RADIUS)
    )
    candidate = points_world[mask]
    if candidate.shape[0] < 10:
        return None

    centroid = candidate.mean(axis=0)
    xy = candidate[:, :2] - centroid[:2]
    cov = xy.T @ xy
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(principal[1], principal[0])
    return centroid[0], centroid[1], yaw


def _solve_ik(kinematics: RobotKinematics, plant, arm_index, X_W_Base: RigidTransform, X_W_Target: RigidTransform, q_init):
    X_Base_Target = X_W_Base.inverse() @ X_W_Target
    q_target = kinematics.inverse_kinematics_from_q0(q_init, X_Base_Target, ignore_collisions=True)
    if q_target is None:
        raise RuntimeError(f"IK failed to reach target pose:\n{X_W_Target}")
    return q_target


def run_pick_demo(meshcat: Meshcat, brick_urdf_path: Path, rng_seed=None) -> None:
    scene = build_arm_gripper_scene(
        meshcat,
        brick_urdf_path=brick_urdf_path,
        add_table=True,
        add_camera=True,
        rng_seed=rng_seed,
    )
    plant, context = scene.plant, scene.context
    plant_context = plant.GetMyContextFromRoot(context)
    kinematics = RobotKinematics(scene.robot_diagram, scene.arm_index, meshcat=meshcat)

    X_W_Base = plant.GetFrameByName("base", scene.arm_index).CalcPoseInWorld(plant_context)
    observe_configuration = _solve_ik(
        kinematics, plant, scene.arm_index, X_W_Base, _observe_tcp_pose(), OBSERVE_INITIAL_GUESS
    )

    plant.SetPositions(plant_context, scene.arm_index, observe_configuration)
    set_gripper_opening(plant, plant_context, scene.gripper_index, GRIPPER_OPEN)
    scene.robot_diagram.ForcedPublish(context)
    time.sleep(STEP_DELAY * 4)

    rng = np.random.default_rng(rng_seed)
    brick_body = plant.GetBodyByName(_brick_body_name(plant, scene.brick_index), scene.brick_index)
    ground_truth_z = resting_z_offset(brick_urdf_path)

    for attempt in range(MAX_SAMPLE_ATTEMPTS):
        X_W_GroundTruth = _sample_ground_truth_pose(rng, ground_truth_z)
        plant.SetFreeBodyPose(plant_context, brick_body, X_W_GroundTruth)
        scene.robot_diagram.ForcedPublish(context)

        camera_pose = scene.arm_camera_frame.CalcPoseInWorld(plant_context)
        points = capture_world_points(plant, context, scene.camera_sensor, camera_pose)
        estimate = estimate_brick_pose(points)
        if estimate is None:
            continue
        x, y, yaw = estimate
        grasp_rotation = RotationMatrix.MakeZRotation(yaw) @ GRASP_ROTATION_BASE
        X_W_Pregrasp = RigidTransform(grasp_rotation, [x, y, PREGRASP_CLEARANCE])
        X_W_Grasp = RigidTransform(grasp_rotation, [x, y, 0.0])
        X_W_Lift = RigidTransform(grasp_rotation, [x, y, LIFT_HEIGHT])
        try:
            q_pregrasp = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Pregrasp, observe_configuration)
            q_grasp = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Grasp, q_pregrasp)
            q_lift = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Lift, q_grasp)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"Could not perceive and reach a brick after {MAX_SAMPLE_ATTEMPTS} attempts.")

    print(f"  Perceived brick at (x={x:.3f}, y={y:.3f}, yaw={yaw:.2f}) from RGBD camera")

    print("  Reaching towards the brick...")
    for q in _interpolate(observe_configuration, q_pregrasp, REACH_STEPS):
        plant.SetPositions(plant_context, scene.arm_index, q)
        scene.robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    print("  Descending onto the brick...")
    for q in _interpolate(q_pregrasp, q_grasp, DESCEND_STEPS):
        plant.SetPositions(plant_context, scene.arm_index, q)
        scene.robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    print("  Closing the gripper...")
    for angle in _interpolate(GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_STEPS):
        set_gripper_opening(plant, plant_context, scene.gripper_index, angle)
        scene.robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    X_W_Tcp_at_grasp = scene.arm_tcp_frame.CalcPoseInWorld(plant_context)
    X_Tcp_Brick = X_W_Tcp_at_grasp.inverse() @ plant.EvalBodyPoseInWorld(plant_context, brick_body)

    print(f"  Lifting {LIFT_HEIGHT * 100:.0f}cm straight up...")
    for q in _interpolate(q_grasp, q_lift, LIFT_STEPS):
        plant.SetPositions(plant_context, scene.arm_index, q)
        X_W_Tcp = scene.arm_tcp_frame.CalcPoseInWorld(plant_context)
        plant.SetFreeBodyPose(plant_context, brick_body, X_W_Tcp @ X_Tcp_Brick)
        scene.robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)


def run_all(meshcat: Meshcat) -> None:
    for name, urdf_path in BRICK_URDFS.items():
        if not urdf_path.exists():
            raise FileNotFoundError(f"Missing URDF for brick {name}: {urdf_path}")
        print(f"Picking up brick {name} ({urdf_path.name})...")
        run_pick_demo(meshcat, urdf_path)
        time.sleep(INTER_BRICK_PAUSE)


if __name__ == "__main__":
    meshcat = Meshcat()
    print(f"Meshcat running at {meshcat.web_url()}")
    run_all(meshcat)
    print("Done")