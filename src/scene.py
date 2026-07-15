"""Shared Drake + Meshcat scene-building helpers, following the pattern used
in materials/homework_3_modeling_environments and materials/practical_5.

Builds a UR3e + Robotiq 2F-85 scene (world -> arm base_link, arm tool0 ->
gripper base_link), with a `tcp` frame offset from tool0, and optionally a
lego brick (a free body) from the repo's lego_3d/urdf assets. Can also add
the real lab table (with the arm mounted on it) plus decorative bricks
scattered at random positions on the tabletop.
"""

from pathlib import Path
from typing import List, Optional

import airo_models
import numpy as np
from airo_drake import finish_build
from pydrake.geometry import Box, Meshcat, MeshcatVisualizer, MeshcatVisualizerParams, Role
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

# --- Real lab table -------------------------------------------------------
# 80cm x 60cm. The UR3e's base is mounted on the table, inset 9cm from the
# table's short (0.6m) edge and 7cm from the long (0.8m) edge, near one
# corner - i.e. the robot base sits at local table coordinates (0.09, 0.07)
# measured from that corner. The world origin is the robot base (as
# elsewhere in this file), so the table's near corner is at
# (-ROBOT_OFFSET_X, -ROBOT_OFFSET_Y) in world coordinates. If the real
# mounting turns out to be the other way round, just swap the two offsets.
TABLE_LENGTH = 0.80  # meters, along the table's local x-axis (the 0.8m edge)
TABLE_WIDTH = 0.60  # meters, along the table's local y-axis (the 0.6m edge)
TABLE_THICKNESS = 0.03  # meters; a tabletop-only box, no legs modeled
ROBOT_OFFSET_X = 0.09  # meters, robot base inset from the table's short edge
ROBOT_OFFSET_Y = 0.07  # meters, robot base inset from the table's long edge
TABLE_COLOR = np.array([0.55, 0.38, 0.20, 1.0])  # plain wood brown

# Table center, expressed relative to the robot base (world origin).
TABLE_CENTER_X = TABLE_LENGTH / 2 - ROBOT_OFFSET_X
TABLE_CENTER_Y = TABLE_WIDTH / 2 - ROBOT_OFFSET_Y
# Yaw (about world z) that points the robot's local +x axis at the table's
# center - an approximation of "pointing towards the center of the table",
# since the UR3e's true heading convention on the real mount isn't specified.
ROBOT_BASE_YAW = np.arctan2(TABLE_CENTER_Y, TABLE_CENTER_X)


def add_meshcat_visualizer(robot_diagram_builder: RobotDiagramBuilder, meshcat: Meshcat = None) -> Meshcat:
    """Attach a Meshcat visualizer (+ collision-geometry layer) to the builder. Must run before Finalize."""
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()

    meshcat = Meshcat() if meshcat is None else meshcat
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    collision_params = MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, collision_params)

    return meshcat


def _add_table(plant) -> None:
    """Registers a static, visual-only box for the tabletop (see the TABLE_* constants above)."""
    X_W_Table = RigidTransform(RotationMatrix.Identity(), [TABLE_CENTER_X, TABLE_CENTER_Y, -TABLE_THICKNESS / 2])
    plant.RegisterVisualGeometry(
        plant.world_body(), X_W_Table, Box(TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS), "table", TABLE_COLOR
    )


def _sample_table_pose(rng: np.random.Generator, margin: float = 0.03, base_clearance: float = 0.12) -> RigidTransform:
    """Samples a random pose resting on the tabletop, clear of the table edges and the robot base."""
    x_min, x_max = -ROBOT_OFFSET_X + margin, TABLE_LENGTH - ROBOT_OFFSET_X - margin
    y_min, y_max = -ROBOT_OFFSET_Y + margin, TABLE_WIDTH - ROBOT_OFFSET_Y - margin

    x, y = 0.0, 0.0
    for _ in range(20):  # rejection sampling, to keep bricks clear of the robot base
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if np.hypot(x, y) >= base_clearance:
            break

    yaw = rng.uniform(0, 2 * np.pi)
    return RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, 0.0])


def build_arm_gripper_scene(
    meshcat: Meshcat,
    brick_urdf_path: Path = DEFAULT_BRICK_URDF,
    add_table: bool = False,
    scattered_brick_urdf_paths: Optional[List[Path]] = None,
    rng_seed: Optional[int] = None,
):
    """Build a UR3e + Robotiq 2F-85 + lego-brick scene, welded arm base to world.

    If add_table is True, also adds the real lab table (see the TABLE_*
    constants above) with the arm's base yawed to face its center, plus one
    randomly-posed, resting-on-the-table decorative brick per path in
    scattered_brick_urdf_paths. These are purely visual context - they are
    not grasped by anything here (that's Module 1 submodule 1's job).

    Returns (robot_diagram, context, plant, arm_index, gripper_index, brick_index,
    arm_tcp_frame, scattered_brick_indices).
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

    scattered_brick_indices = [
        parser.AddModels(str(Path(p).resolve()))[0] for p in (scattered_brick_urdf_paths or [])
    ]

    world_frame = plant.world_frame()
    arm_frame = plant.GetFrameByName("base_link", arm_index)
    arm_tool_frame = plant.GetFrameByName("tool0", arm_index)
    gripper_frame = plant.GetFrameByName("base_link", gripper_index)

    X_W_ArmBase = RigidTransform(RotationMatrix.MakeZRotation(ROBOT_BASE_YAW)) if add_table else RigidTransform()
    plant.WeldFrames(world_frame, arm_frame, X_W_ArmBase)
    plant.WeldFrames(arm_tool_frame, gripper_frame)

    X_Tool0Tcp = RigidTransform(RotationMatrix.Identity(), [0, 0, TCP_OFFSET])
    arm_tcp_frame = plant.AddFrame(FixedOffsetFrame("tcp", arm_tool_frame, X_Tool0Tcp))

    if add_table:
        _add_table(plant)

    robot_diagram, context = finish_build(robot_diagram_builder, meshcat)

    if scattered_brick_indices:
        plant_context = plant.GetMyContextFromRoot(context)
        rng = np.random.default_rng(rng_seed)
        for scattered_index in scattered_brick_indices:
            body = plant.get_body(plant.GetBodyIndices(scattered_index)[0])
            plant.SetFreeBodyPose(plant_context, body, _sample_table_pose(rng))
        robot_diagram.ForcedPublish(context)

    return robot_diagram, context, plant, arm_index, gripper_index, brick_index, arm_tcp_frame, scattered_brick_indices


def set_gripper_opening(plant, plant_context, gripper_index, angle: float) -> None:
    """Kinematically drive all 6 Robotiq 2F-85 finger joints per the mimic relationship."""
    for joint_name, sign in GRIPPER_MIMIC_SIGNS.items():
        joint = plant.GetJointByName(joint_name, gripper_index)
        joint.set_angle(plant_context, sign * angle)
