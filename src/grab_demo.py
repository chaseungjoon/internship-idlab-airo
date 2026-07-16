import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RotationMatrix

from scene import DEFAULT_BRICK_URDF, GRIPPER_CLOSED, GRIPPER_OPEN, build_arm_gripper_scene, set_gripper_opening

# An arbitrary, non-degenerate RM65 joint configuration (radians) to pose the
# arm for the demo. Not calibrated to any physical mounting -- MVP item 1 only
# needs a working sim pipeline, not a real workspace layout.
DEMO_ARM_CONFIGURATION = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


def _interpolate(start, end, num_steps: int):
    """Returns num_steps points from start to end inclusive (num_steps<=1 -> just [end])."""
    if num_steps <= 1:
        return [end]
    return np.linspace(start, end, num_steps)


def run_grab_demo(
    meshcat: Meshcat,
    publish_delay: float = 0.3,
    brick_urdf_path: Path = DEFAULT_BRICK_URDF,
    add_table: bool = False,
    scattered_brick_urdf_paths: Optional[List[Path]] = None,
    rng_seed: Optional[int] = None,
    arm_animation_steps: int = 1,
    brick_animation_steps: int = 1,
    gripper_animation_steps: int = 8,
):
    """Runs the grab demo against the given Meshcat instance. Returns the final plant_context.

    Each phase (arm into position / brick to the gripper / gripper closing) is
    linearly interpolated over its own *_animation_steps frames, publishing
    and pausing publish_delay seconds after every single frame - so a phase
    with N steps takes roughly N * publish_delay seconds and is visibly
    smooth, not a single jump. The defaults (1, 1, 8) are a quick sanity-check
    demo; pass larger step counts (e.g. from submodule_0) to watch every
    movement clearly.
    """
    scene = build_arm_gripper_scene(
        meshcat,
        brick_urdf_path=brick_urdf_path,
        add_table=add_table,
        scattered_brick_urdf_paths=scattered_brick_urdf_paths,
        rng_seed=rng_seed,
    )
    robot_diagram = scene.robot_diagram
    context = scene.context
    plant = scene.plant
    arm_index = scene.arm_index
    gripper_index = scene.gripper_index
    brick_index = scene.brick_index
    arm_tcp_frame = scene.arm_tcp_frame
    plant_context = plant.GetMyContextFromRoot(context)
    set_gripper_opening(plant, plant_context, gripper_index, GRIPPER_OPEN)

    # 1. Move the arm to the demo configuration.
    q_start = plant.GetPositions(plant_context, arm_index)
    for q in _interpolate(q_start, DEMO_ARM_CONFIGURATION, arm_animation_steps):
        plant.SetPositions(plant_context, arm_index, q)
        robot_diagram.ForcedPublish(context)
        time.sleep(publish_delay)

    # 2. Bring the brick to the gripper's TCP (i.e. "presented" to the gripper).
    X_W_Tcp = arm_tcp_frame.CalcPoseInWorld(plant_context)
    brick_body = plant.GetBodyByName(_brick_body_name(plant, brick_index), brick_index)
    p_start = plant.EvalBodyPoseInWorld(plant_context, brick_body).translation()
    p_end = X_W_Tcp.translation()
    for p in _interpolate(p_start, p_end, brick_animation_steps):
        X_W_Brick = RigidTransform(RotationMatrix.Identity(), p)
        plant.SetFreeBodyPose(plant_context, brick_body, X_W_Brick)
        robot_diagram.ForcedPublish(context)
        time.sleep(publish_delay)

    # 3. Close the gripper around the brick.
    for angle in _interpolate(GRIPPER_OPEN, GRIPPER_CLOSED, gripper_animation_steps):
        set_gripper_opening(plant, plant_context, gripper_index, angle)
        robot_diagram.ForcedPublish(context)
        time.sleep(publish_delay)

    return robot_diagram, context, plant, arm_index, gripper_index, brick_index


def _brick_body_name(plant, brick_index) -> str:
    """The lego_3d URDFs are single-link models named after the part; look up that name."""
    body_indices = plant.GetBodyIndices(brick_index)
    return plant.get_body(body_indices[0]).name()


if __name__ == "__main__":
    meshcat = Meshcat()
    print(f"Meshcat running at {meshcat.web_url()}")
    run_grab_demo(meshcat)
    input("Press Enter to exit...")
