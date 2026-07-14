"""MVP item 1: load the UR3e + Robotiq 2F-85 URDFs into Drake, place a lego
brick at the gripper's TCP, and kinematically animate the gripper closing
around it, visualized live in Meshcat.

This is a kinematic demo (SetPositions + ForcedPublish), not a contact-rich
dynamics simulation -- the same style used in materials/practical_2.ipynb and
materials/homework_3_modeling_environments for scripted visualization.
"""

import time

import numpy as np
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RotationMatrix

from scene import GRIPPER_CLOSED, GRIPPER_OPEN, build_arm_gripper_scene, set_gripper_opening

# An arbitrary, non-degenerate UR3e joint configuration (radians) to pose the
# arm for the demo. Not calibrated to any physical mounting -- MVP item 1 only
# needs a working sim pipeline, not a real workspace layout.
DEMO_ARM_CONFIGURATION = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


def run_grab_demo(meshcat: Meshcat, publish_delay: float = 0.3):
    """Runs the grab demo against the given Meshcat instance. Returns the final plant_context."""
    robot_diagram, context, plant, arm_index, gripper_index, brick_index, arm_tcp_frame = build_arm_gripper_scene(
        meshcat
    )
    plant_context = plant.GetMyContextFromRoot(context)

    # 1. Move the arm to the demo configuration.
    plant.SetPositions(plant_context, arm_index, DEMO_ARM_CONFIGURATION)
    set_gripper_opening(plant, plant_context, gripper_index, GRIPPER_OPEN)
    robot_diagram.ForcedPublish(context)
    time.sleep(publish_delay)

    # 2. Place the brick at the gripper's TCP (i.e. "presented" to the gripper).
    X_W_Tcp = arm_tcp_frame.CalcPoseInWorld(plant_context)
    brick_body = plant.GetBodyByName(_brick_body_name(plant, brick_index), brick_index)
    X_W_Brick = RigidTransform(RotationMatrix.Identity(), X_W_Tcp.translation())
    plant.SetFreeBodyPose(plant_context, brick_body, X_W_Brick)
    robot_diagram.ForcedPublish(context)
    time.sleep(publish_delay)

    # 3. Animate the gripper closing around the brick.
    for angle in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 8):
        set_gripper_opening(plant, plant_context, gripper_index, angle)
        robot_diagram.ForcedPublish(context)
        time.sleep(publish_delay / 4)

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
