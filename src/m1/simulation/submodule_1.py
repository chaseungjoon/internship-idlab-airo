"""M1 submodule 1: grasp a specific (color-identified) brick from a pile.

A pile of distractor bricks plus one target brick is dropped near a known pile
location. The wrist RGBD camera segments the target by color, estimates its
pose from the segmented point cloud, and the arm picks it out of the pile.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.analysis import Simulator

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

KINEMATICS_DIR = SRC_DIR.parent / "materials" / "practical_3_planning"
if str(KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(KINEMATICS_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
)
from submodule_0 import (
    BASE_EXCLUSION_RADIUS,
    DESCEND_DURATION,
    GRASP_ROTATION_BASE,
    GRIPPER_DURATION,
    LIFT_DURATION,
    LIFT_HEIGHT,
    LIFT_SUCCESS_TOLERANCE,
    MAX_SAMPLE_ATTEMPTS,
    MIN_HEIGHT_ABOVE_TABLE,
    OBSERVE_INITIAL_GUESS,
    PREGRASP_CLEARANCE,
    REACH_DURATION,
    SETTLE_DURATION,
    _arm_setpoint,
    _brick_body_name,
    _gripper_setpoint,
    _look_at_rotation,
    _set_setpoint,
    _solve_ik,
)

LEGO_URDF_DIR = SRC_DIR.parent / "lego_3d" / "urdf"

TARGET_BRICK_URDF = LEGO_URDF_DIR / "3005__dark_green.urdf"
PILE_BRICK_URDFS = [
    LEGO_URDF_DIR / "3005__light_bluish_gray.urdf",
    LEGO_URDF_DIR / "3021__tan.urdf",
    LEGO_URDF_DIR / "3008__tan.urdf",
    LEGO_URDF_DIR / "2431__tan.urdf",
    LEGO_URDF_DIR / "3022__light_bluish_gray.urdf",
]

PILE_CENTER = np.array([0.30, 0.10])
PILE_RADIUS = 0.05
PILE_LAYER_HEIGHT = 0.03
PILE_SETTLE_DURATION = 2.0
RESAMPLE_DURATION = 1.0

OBSERVE_EYE_HEIGHT = 0.4
OBSERVE_EYE_BACKOFF = 0.4

HUE_TOLERANCE = 0.08
MIN_SATURATION = 0.2
MIN_VALUE = 0.05
MIN_TARGET_POINTS = 25
MAX_PILE_HEIGHT = 0.15
GRASP_DEPTH = 0.02


def _urdf_color(urdf_path: Path) -> np.ndarray:
    color = ET.parse(urdf_path).getroot().find(".//visual/material/color")
    return np.array([float(v) for v in color.get("rgba").split()[:3]])


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    safe_delta = np.where(delta > 0, delta, 1.0)

    hue = np.zeros_like(maxc)
    hue = np.where(maxc == r, ((g - b) / safe_delta) % 6.0, hue)
    hue = np.where(maxc == g, (b - r) / safe_delta + 2.0, hue)
    hue = np.where(maxc == b, (r - g) / safe_delta + 4.0, hue)
    hue = np.where(delta > 0, hue / 6.0, 0.0)

    saturation = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1.0), 0.0)
    return np.stack([hue, saturation, maxc], axis=-1)


def _observe_tcp_pose(pile_center: np.ndarray) -> RigidTransform:
    eye = np.array([pile_center[0] * OBSERVE_EYE_BACKOFF, pile_center[1] * OBSERVE_EYE_BACKOFF, OBSERVE_EYE_HEIGHT])
    R_camera = _look_at_rotation(eye, np.array([pile_center[0], pile_center[1], 0.0]))
    X_W_Camera = RigidTransform(RotationMatrix(R_camera), eye)
    X_Tcp_Camera = TCP_OFFSET.inverse() @ CAMERA_TOOL0_OFFSET
    return X_W_Camera @ X_Tcp_Camera.inverse()


def capture_rgbd(sensor, root_context):
    sensor_context = sensor.GetMyContextFromRoot(root_context)
    color = np.array(sensor.color_image_output_port().Eval(sensor_context).data, copy=True)
    depth = np.array(sensor.depth_image_32F_output_port().Eval(sensor_context).data).squeeze(-1)
    return color, depth


def segment_target_points(color, depth, intrinsics, camera_pose: RigidTransform, target_rgb: np.ndarray):
    """Color-segments the target brick and returns its points in world frame."""
    hsv = _rgb_to_hsv(color[..., :3].astype(np.float64) / 255.0)
    target_hue = _rgb_to_hsv(target_rgb[None, None, :])[0, 0, 0]

    hue_diff = np.abs(hsv[..., 0] - target_hue)
    hue_diff = np.minimum(hue_diff, 1.0 - hue_diff)
    mask = (
        (hue_diff < HUE_TOLERANCE)
        & (hsv[..., 1] > MIN_SATURATION)
        & (hsv[..., 2] > MIN_VALUE)
        & np.isfinite(depth)
        & (depth > 0)
    )
    if not mask.any():
        return np.empty((0, 3))

    fx, fy = intrinsics.focal_x(), intrinsics.focal_y()
    cx, cy = intrinsics.center_x(), intrinsics.center_y()
    v, u = np.where(mask)
    d = depth[v, u]
    points_cam = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=-1)

    R = camera_pose.rotation().matrix()
    t = camera_pose.translation()
    return points_cam @ R.T + t


def estimate_target_grasp(points_world: np.ndarray):
    """Returns (x, y, z_top, yaw) of the target from its segmented points, or None."""
    x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    in_workspace = (
        (z > MIN_HEIGHT_ABOVE_TABLE)
        & (z < MAX_PILE_HEIGHT)
        & (x > -ROBOT_OFFSET_X)
        & (x < TABLE_LENGTH - ROBOT_OFFSET_X)
        & (y > -ROBOT_OFFSET_Y)
        & (y < TABLE_WIDTH - ROBOT_OFFSET_Y)
        & (np.hypot(x, y) > BASE_EXCLUSION_RADIUS)
    )
    candidate = points_world[in_workspace]
    if candidate.shape[0] < MIN_TARGET_POINTS:
        return None

    centroid = candidate.mean(axis=0)
    xy = candidate[:, :2] - centroid[:2]
    cov = xy.T @ xy
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(principal[1], principal[0])
    z_top = np.percentile(candidate[:, 2], 95)
    return centroid[0], centroid[1], z_top, yaw


def _scatter_pile(plant, plant_context, brick_bodies, brick_urdfs, rng: np.random.Generator) -> None:
    """Drops the bricks in a staggered stack around PILE_CENTER so they settle into a pile."""
    order = rng.permutation(len(brick_bodies))
    for layer, index in enumerate(order):
        body, urdf = brick_bodies[index], brick_urdfs[index]
        radius = rng.uniform(0, PILE_RADIUS)
        angle = rng.uniform(0, 2 * np.pi)
        position = [
            PILE_CENTER[0] + radius * np.cos(angle),
            PILE_CENTER[1] + radius * np.sin(angle),
            resting_z_offset(urdf) + 0.005 + layer * PILE_LAYER_HEIGHT,
        ]
        yaw = rng.uniform(0, 2 * np.pi)
        plant.SetFreeBodyPose(plant_context, body, RigidTransform(RotationMatrix.MakeZRotation(yaw), position))


def run_pile_pick_demo(meshcat: Meshcat, rng_seed=None, realtime_rate: float = 1.0) -> bool:
    scene = build_arm_gripper_scene(
        meshcat,
        brick_urdf_path=TARGET_BRICK_URDF,
        add_table=True,
        scattered_brick_urdf_paths=PILE_BRICK_URDFS,
        add_camera=True,
        add_controllers=True,
        rng_seed=rng_seed,
    )
    plant = scene.plant
    root_context = scene.context
    plant_context = plant.GetMyContextFromRoot(root_context)
    diagram = scene.robot_diagram
    kinematics = RobotKinematics(diagram, scene.arm_index, meshcat=meshcat)

    X_W_Base = plant.GetFrameByName("base", scene.arm_index).CalcPoseInWorld(plant_context)
    observe_configuration = _solve_ik(
        kinematics, plant, scene.arm_index, X_W_Base, _observe_tcp_pose(PILE_CENTER), OBSERVE_INITIAL_GUESS
    )
    plant.SetPositions(plant_context, scene.arm_index, observe_configuration)

    target_body = plant.GetBodyByName(_brick_body_name(plant, scene.brick_index), scene.brick_index)
    pile_indices = [scene.brick_index] + list(scene.scattered_brick_indices)
    pile_bodies = [target_body] + [
        plant.GetBodyByName(_brick_body_name(plant, index), index) for index in scene.scattered_brick_indices
    ]
    pile_urdfs = [TARGET_BRICK_URDF] + PILE_BRICK_URDFS
    target_rgb = _urdf_color(TARGET_BRICK_URDF)

    rng = np.random.default_rng(rng_seed)
    _scatter_pile(plant, plant_context, pile_bodies, pile_urdfs, rng)

    _set_setpoint(diagram, root_context, scene.arm_setpoint_source, _arm_setpoint(observe_configuration))
    _set_setpoint(diagram, root_context, scene.gripper_setpoint_source, _gripper_setpoint(GRIPPER_OPEN))

    simulator = Simulator(diagram, root_context)
    simulator.set_target_realtime_rate(realtime_rate)
    simulator.Initialize()

    elapsed = 0.0

    def advance(duration: float):
        nonlocal elapsed
        elapsed += duration
        simulator.AdvanceTo(elapsed)

    advance(PILE_SETTLE_DURATION)

    for attempt in range(MAX_SAMPLE_ATTEMPTS):
        sim_context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(sim_context)
        camera_pose = scene.arm_camera_frame.CalcPoseInWorld(plant_context)
        color, depth = capture_rgbd(scene.camera_sensor, sim_context)
        points = segment_target_points(color, depth, scene.camera_sensor.depth_camera_info(), camera_pose, target_rgb)
        estimate = estimate_target_grasp(points)
        if estimate is not None:
            x, y, z_top, yaw = estimate
            grasp_z = max(0.0, z_top - GRASP_DEPTH)
            grasp_rotation = RotationMatrix.MakeZRotation(yaw) @ GRASP_ROTATION_BASE
            X_W_Pregrasp = RigidTransform(grasp_rotation, [x, y, grasp_z + PREGRASP_CLEARANCE])
            X_W_Grasp = RigidTransform(grasp_rotation, [x, y, grasp_z])
            X_W_Lift = RigidTransform(grasp_rotation, [x, y, grasp_z + LIFT_HEIGHT])
            try:
                q_pregrasp = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Pregrasp, observe_configuration)
                q_grasp = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Grasp, q_pregrasp)
                q_lift = _solve_ik(kinematics, plant, scene.arm_index, X_W_Base, X_W_Lift, q_grasp)
                break
            except RuntimeError:
                pass
        print(f"  Target not visible/reachable (attempt {attempt + 1}), re-scattering the pile...")
        _scatter_pile(plant, plant_context, pile_bodies, pile_urdfs, rng)
        for index in pile_indices:
            plant.SetVelocities(plant_context, index, np.zeros(6))
        advance(RESAMPLE_DURATION)
    else:
        raise RuntimeError(f"Could not perceive and reach the target brick after {MAX_SAMPLE_ATTEMPTS} attempts.")

    print(f"  Segmented target brick at (x={x:.3f}, y={y:.3f}, z_top={z_top:.3f}, yaw={yaw:.2f}) by color")
    start_z = plant.EvalBodyPoseInWorld(plant_context, target_body).translation()[2]

    print("  Reaching above the target...")
    _set_setpoint(diagram, root_context, scene.arm_setpoint_source, _arm_setpoint(q_pregrasp))
    advance(REACH_DURATION)

    print("  Descending into the pile...")
    _set_setpoint(diagram, root_context, scene.arm_setpoint_source, _arm_setpoint(q_grasp))
    advance(DESCEND_DURATION)

    print("  Closing the gripper...")
    _set_setpoint(diagram, root_context, scene.gripper_setpoint_source, _gripper_setpoint(GRIPPER_CLOSED))
    advance(GRIPPER_DURATION)

    print(f"  Lifting {LIFT_HEIGHT * 100:.0f}cm straight up...")
    _set_setpoint(diagram, root_context, scene.arm_setpoint_source, _arm_setpoint(q_lift))
    advance(LIFT_DURATION)

    plant_context = plant.GetMyContextFromRoot(simulator.get_context())
    final_z = plant.EvalBodyPoseInWorld(plant_context, target_body).translation()[2]
    lifted = final_z > start_z + LIFT_HEIGHT - LIFT_SUCCESS_TOLERANCE
    print(f"  Target brick height: {start_z:.3f}m -> {final_z:.3f}m ({'grasped' if lifted else 'dropped'})")
    return lifted


if __name__ == "__main__":
    meshcat = Meshcat()
    print(f"Meshcat running at {meshcat.web_url()}")
    print(f"Picking target brick {TARGET_BRICK_URDF.stem} from a pile of {len(PILE_BRICK_URDFS)} bricks...")
    run_pile_pick_demo(meshcat, rng_seed=0)
    print("Done")
