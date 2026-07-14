"""Shared Drake + Meshcat scene-building helpers, following the pattern used
in materials/homework_3_modeling_environments and materials/practical_5.

Builds a UR3e + Robotiq 2F-85 scene (world -> arm base_link, arm tool0 ->
gripper base_link), with a `tcp` frame offset from tool0, and optionally a
lego brick (a free body) from the repo's lego_3d/urdf assets.
"""

from pathlib import Path

import airo_models
from airo_drake import finish_build
from pydrake.geometry import Meshcat, MeshcatVisualizer, MeshcatVisualizerParams, Role
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.planning import RobotDiagramBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRICK_URDF = REPO_ROOT / "lego_3d" / "urdf" / "3005__light_bluish_gray.urdf"

TCP_OFFSET = 0.174  # meters from tool0 to the point between the gripper fingers

# Robotiq 2F-85 finger joints are declared with <mimic joint="finger_joint" .../>
# in the URDF. Drake ignores mimic constraints unless the plant uses a discrete
# time step + SAP solver, so for kinematic-only animation (no dynamics/contact)
# we replicate the mimic relationship by hand: {joint_name: multiplier}.
GRIPPER_MIMIC_SIGNS = {
    "finger_joint": 1,
    "left_inner_knuckle_joint": 1,
    "left_inner_finger_joint": -1,
    "right_outer_knuckle_joint": 1,
    "right_inner_knuckle_joint": 1,
    "right_inner_finger_joint": -1,
}
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.68  # radians; joint limit is 0.8, brick-sized objects grasp around here


def add_meshcat_visualizer(robot_diagram_builder: RobotDiagramBuilder, meshcat: Meshcat = None) -> Meshcat:
    """Attach a Meshcat visualizer (+ collision-geometry layer) to the builder. Must run before Finalize."""
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()

    meshcat = Meshcat() if meshcat is None else meshcat
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    collision_params = MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, collision_params)

    return meshcat


def build_arm_gripper_scene(meshcat: Meshcat, brick_urdf_path: Path = DEFAULT_BRICK_URDF):
    """Build a UR3e + Robotiq 2F-85 + lego-brick scene, welded arm base to world.

    Returns (robot_diagram, context, plant, arm_index, gripper_index, brick_index, arm_tcp_frame).
    """
    robot_diagram_builder = RobotDiagramBuilder()
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)

    meshcat.Delete()
    meshcat.DeleteAddedControls()
    add_meshcat_visualizer(robot_diagram_builder, meshcat)

    arm_urdf_path = airo_models.get_urdf_path("ur3e")
    gripper_urdf_path = airo_models.get_urdf_path("robotiq_2f_85")

    arm_index = parser.AddModels(arm_urdf_path)[0]
    gripper_index = parser.AddModels(gripper_urdf_path)[0]
    brick_index = parser.AddModels(str(Path(brick_urdf_path).resolve()))[0]

    world_frame = plant.world_frame()
    arm_frame = plant.GetFrameByName("base_link", arm_index)
    arm_tool_frame = plant.GetFrameByName("tool0", arm_index)
    gripper_frame = plant.GetFrameByName("base_link", gripper_index)

    plant.WeldFrames(world_frame, arm_frame)
    plant.WeldFrames(arm_tool_frame, gripper_frame)

    X_Tool0Tcp = RigidTransform(RotationMatrix.Identity(), [0, 0, TCP_OFFSET])
    arm_tcp_frame = plant.AddFrame(FixedOffsetFrame("tcp", arm_tool_frame, X_Tool0Tcp))

    robot_diagram, context = finish_build(robot_diagram_builder, meshcat)

    return robot_diagram, context, plant, arm_index, gripper_index, brick_index, arm_tcp_frame


def set_gripper_opening(plant, plant_context, gripper_index, angle: float) -> None:
    """Kinematically drive all 6 Robotiq 2F-85 finger joints per the mimic relationship."""
    for joint_name, sign in GRIPPER_MIMIC_SIGNS.items():
        joint = plant.GetJointByName(joint_name, gripper_index)
        joint.set_angle(plant_context, sign * angle)
