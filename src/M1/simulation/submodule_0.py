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
from scene import(
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ROBOT_OFFSET_X,
    ROBOT_OFFSET_Y,
    TABLE_LENGTH,
    TABLE_WIDTH,
    build_arm_gripper_scene,
    set_gripper_opening,
)

LEGO_URDF_DIR = SRC_DIR.parent / "lego_3d" / "urdf"

BRICK_URDFS = {
    "3008": LEGO_URDF_DIR / "3008__tan.urdf",
    "3021": LEGO_URDF_DIR / "3021__tan.urdf",
    "69729": LEGO_URDF_DIR / "69729__light_bluish_gray.urdf"
}

SCATTERED_BRICK_URDFS = list(BRICK_URDFS.values()) * 3

STEP_DELAY = 0.05 
REACH_STEPS = 40
DESCEND_STEPS = 25
GRIPPER_STEPS = 30
LIFT_STEPS = 40
INTER_BRICK_PAUSE = 2.0
LIFT_HEIGHT = 0.15
PREGRASP_CLEARANCE = 0.12
GRASP_HEIGHT = 0.01

MIN_REACH = 0.20 
MAX_REACH = 0.40

GRASP_ROTATION = RotationMatrix.MakeXRotation(np.pi)
HOME_CONFIGURATION = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
MAX_SAMPLE_ATTEMPTS = 20


def _interpolate(start, end, num_steps: int):
    if num_steps <= 1:
        return [end]
    return np.linspace(start, end, num_steps)


def _brick_body_name(plant, brick_index) -> str:
    """The lego_3d URDFs are single-link models named after the part; look up that name."""
    body_indices = plant.GetBodyIndices(brick_index)
    return plant.get_body(body_indices[0]).name()


def _sample_target_pose(rng: np.random.Generator):
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
    return x, y, yaw


def _solve_ik(kinematics: RobotKinematics, plant, arm_index, X_W_Base: RigidTransform, X_W_Target: RigidTransform, q_init):
    X_Base_Target = X_W_Base.inverse() @ X_W_Target
    q_target = kinematics.inverse_kinematics_from_q0(q_init, X_Base_Target, ignore_collisions=True)
    if q_target is None:
        raise RuntimeError(f"IK failed to reach target pose:\n{X_W_Target}")
    return q_target


def run_pick_demo(meshcat: Meshcat, brick_urdf_path: Path, rng_seed=None) -> None:
    (
        robot_diagram,
        context,
        plant,
        arm_index,
        gripper_index,
        brick_index,
        arm_tcp_frame,
        _scattered_brick_indices,
    ) = build_arm_gripper_scene(
        meshcat,
        brick_urdf_path=brick_urdf_path,
        add_table=True,
        scattered_brick_urdf_paths=SCATTERED_BRICK_URDFS,
        rng_seed=rng_seed,
    )
    plant_context = plant.GetMyContextFromRoot(context)
    kinematics = RobotKinematics(robot_diagram, arm_index, meshcat=meshcat)

    plant.SetPositions(plant_context, arm_index, HOME_CONFIGURATION)
    set_gripper_opening(plant, plant_context, gripper_index, GRIPPER_OPEN)
    X_W_Base = plant.GetFrameByName("base", arm_index).CalcPoseInWorld(plant_context)
    robot_diagram.ForcedPublish(context)
    time.sleep(STEP_DELAY * 4)

    rng = np.random.default_rng(rng_seed)
    brick_body = plant.GetBodyByName(_brick_body_name(plant, brick_index), brick_index)

    for attempt in range(MAX_SAMPLE_ATTEMPTS):
        x, y, yaw = _sample_target_pose(rng)
        X_W_Pregrasp = RigidTransform(GRASP_ROTATION, [x, y, GRASP_HEIGHT + PREGRASP_CLEARANCE])
        X_W_Grasp = RigidTransform(GRASP_ROTATION, [x, y, GRASP_HEIGHT])
        X_W_Lift = RigidTransform(GRASP_ROTATION, [x, y, GRASP_HEIGHT + LIFT_HEIGHT])
        try:
            q_pregrasp = _solve_ik(kinematics, plant, arm_index, X_W_Base, X_W_Pregrasp, HOME_CONFIGURATION)
            q_grasp = _solve_ik(kinematics, plant, arm_index, X_W_Base, X_W_Grasp, q_pregrasp)
            q_lift = _solve_ik(kinematics, plant, arm_index, X_W_Base, X_W_Lift, q_grasp)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"Could not find a reachable brick pose after {MAX_SAMPLE_ATTEMPTS} attempts.")

    X_W_Brick = RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, 0.0])
    plant.SetFreeBodyPose(plant_context, brick_body, X_W_Brick)
    robot_diagram.ForcedPublish(context)
    time.sleep(STEP_DELAY * 4)

    print(f"  Reaching towards brick at (x={x:.3f}, y={y:.3f})...")
    for q in _interpolate(HOME_CONFIGURATION, q_pregrasp, REACH_STEPS):
        plant.SetPositions(plant_context, arm_index, q)
        robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    print("  Descending onto the brick...")
    for q in _interpolate(q_pregrasp, q_grasp, DESCEND_STEPS):
        plant.SetPositions(plant_context, arm_index, q)
        robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    print("  Closing the gripper...")
    for angle in _interpolate(GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_STEPS):
        set_gripper_opening(plant, plant_context, gripper_index, angle)
        robot_diagram.ForcedPublish(context)
        time.sleep(STEP_DELAY)

    X_W_Tcp_at_grasp = arm_tcp_frame.CalcPoseInWorld(plant_context)
    X_Tcp_Brick = X_W_Tcp_at_grasp.inverse() @ plant.EvalBodyPoseInWorld(plant_context, brick_body)

    print(f"  Lifting {LIFT_HEIGHT * 100:.0f}cm straight up...")
    for q in _interpolate(q_grasp, q_lift, LIFT_STEPS):
        plant.SetPositions(plant_context, arm_index, q)
        X_W_Tcp = arm_tcp_frame.CalcPoseInWorld(plant_context)
        plant.SetFreeBodyPose(plant_context, brick_body, X_W_Tcp @ X_Tcp_Brick)
        robot_diagram.ForcedPublish(context)
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
