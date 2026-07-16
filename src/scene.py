import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from airo_drake import finish_build
from pydrake.geometry import (
    Box,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RenderCameraCore,
    RenderEngineVtkParams,
    Role,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import CoulombFriction, DiscreteContactApproximation
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.planning import RobotDiagramBuilder
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRICK_URDF = REPO_ROOT / "lego_3d" / "urdf" / "3005__light_bluish_gray.urdf"
ARM_URDF = REPO_ROOT / "src" / "assets" / "rm65" / "urdf" / "rm65.urdf"
GRIPPER_URDF = REPO_ROOT / "src" / "assets" / "revo2" / "urdf" / "revo2_right_hand.urdf"
NORMALS_CACHE_DIR = REPO_ROOT / "src" / "assets" / "normals_cache"


def _has_normals(obj_path: Path) -> bool:
    with open(obj_path) as f:
        for line in f:
            if line.startswith("vn "):
                return True
    return False


def _add_normals_to_obj(src_path: Path, dst_path: Path) -> None:
    vertices = []
    faces = []
    with open(src_path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])

    vertices = np.array(vertices)
    faces = np.array(faces)
    normals = np.zeros_like(vertices)

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    with open(dst_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            i, j, k = face + 1
            f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")


def ensure_renderable_urdf(urdf_path: Path) -> Path:
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False

    for mesh in root.iter("mesh"):
        mesh_path = (urdf_path.parent / mesh.get("filename")).resolve()
        mesh.set("filename", str(mesh_path))

    for visual in root.iter("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        mesh_path = Path(mesh.get("filename"))
        if _has_normals(mesh_path):
            continue
        NORMALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_mesh_path = NORMALS_CACHE_DIR / mesh_path.name
        if not cached_mesh_path.exists():
            _add_normals_to_obj(mesh_path, cached_mesh_path)
        mesh.set("filename", str(cached_mesh_path))
        changed = True

    if not changed:
        return urdf_path

    NORMALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_urdf_path = NORMALS_CACHE_DIR / urdf_path.name
    tree.write(cached_urdf_path)
    return cached_urdf_path


def _mesh_z_min(obj_path: Path) -> float:
    z_min = float("inf")
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                z = float(line.split()[3])
                z_min = min(z_min, z)
    return z_min


def resting_z_offset(urdf_path: Path) -> float:
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    visual = tree.getroot().find(".//visual")
    mesh = visual.find("geometry/mesh")
    mesh_path = (urdf_path.parent / mesh.get("filename")).resolve()
    scale = mesh.get("scale")
    scale_z = float(scale.split()[2]) if scale else 1.0
    return -_mesh_z_min(mesh_path) * scale_z


TCP_OFFSET = RigidTransform(RotationMatrix.Identity(), [0.033, 0.019, 0.074])

REVO2_FINGER_JOINTS = {
    "right_thumb_metacarpal_joint": 0.9,
    "right_thumb_proximal_joint": 1.0,
    "right_index_proximal_joint": 1.2,
    "right_middle_proximal_joint": 1.2,
    "right_ring_proximal_joint": 1.2,
    "right_pinky_proximal_joint": 1.2,
}
REVO2_MIMIC_JOINTS = {
    "right_thumb_distal_joint": ("right_thumb_proximal_joint", 1.0),
    "right_index_distal_joint": ("right_index_proximal_joint", 1.155),
    "right_middle_distal_joint": ("right_middle_proximal_joint", 1.155),
    "right_ring_distal_joint": ("right_ring_proximal_joint", 1.155),
    "right_pinky_distal_joint": ("right_pinky_proximal_joint", 1.155),
}
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

ARM_JOINT_EFFORTS = {
    "joint_1": 60.0,
    "joint_2": 60.0,
    "joint_3": 30.0,
    "joint_4": 10.0,
    "joint_5": 10.0,
    "joint_6": 10.0,
}
ARM_PD_GAINS = {
    "joint_1": (400.0, 40.0),
    "joint_2": (400.0, 40.0),
    "joint_3": (300.0, 30.0),
    "joint_4": (100.0, 10.0),
    "joint_5": (100.0, 10.0),
    "joint_6": (50.0, 5.0),
}
REVO2_JOINT_EFFORTS = {
    "right_thumb_metacarpal_joint": 0.5,
    "right_thumb_proximal_joint": 1.1,
    "right_index_proximal_joint": 2.0,
    "right_middle_proximal_joint": 2.0,
    "right_ring_proximal_joint": 2.0,
    "right_pinky_proximal_joint": 2.0,
}
REVO2_PD_GAINS = {
    "right_thumb_metacarpal_joint": (5.0, 0.3),
    "right_thumb_proximal_joint": (5.0, 0.3),
    "right_index_proximal_joint": (5.0, 0.3),
    "right_middle_proximal_joint": (5.0, 0.3),
    "right_ring_proximal_joint": (5.0, 0.3),
    "right_pinky_proximal_joint": (5.0, 0.3),
}

TABLE_LENGTH = 0.80
TABLE_WIDTH = 0.60
TABLE_THICKNESS = 0.03
ROBOT_OFFSET_X = 0.09
ROBOT_OFFSET_Y = 0.07
TABLE_COLOR = np.array([0.55, 0.38, 0.20, 1.0])
TABLE_FRICTION = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)

TABLE_CENTER_X = TABLE_LENGTH / 2 - ROBOT_OFFSET_X
TABLE_CENTER_Y = TABLE_WIDTH / 2 - ROBOT_OFFSET_Y
ROBOT_BASE_YAW = np.arctan2(TABLE_CENTER_Y, TABLE_CENTER_X)

CAMERA_RENDERER_NAME = "vtk_renderer"
CAMERA_WIDTH_PX = 640
CAMERA_HEIGHT_PX = 480
CAMERA_FOV_Y = np.deg2rad(70)
CAMERA_TOOL0_OFFSET = RigidTransform(RotationMatrix.MakeYRotation(np.deg2rad(30)), [0.20, 0.0, 0.02])


@dataclass
class ArmGripperScene:
    robot_diagram: object
    context: object
    plant: object
    arm_index: object
    gripper_index: object
    brick_index: object
    arm_tcp_frame: object
    scattered_brick_indices: list = field(default_factory=list)
    camera_sensor: Optional[RgbdSensor] = None
    arm_camera_frame: object = None
    arm_setpoint_source: object = None
    gripper_setpoint_source: object = None


def add_meshcat_visualizer(robot_diagram_builder: RobotDiagramBuilder, meshcat: Meshcat = None) -> Meshcat:
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()

    meshcat = Meshcat() if meshcat is None else meshcat
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    collision_params = MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, collision_params)

    return meshcat


def _add_table(plant) -> None:
    X_W_Table = RigidTransform(RotationMatrix.Identity(), [TABLE_CENTER_X, TABLE_CENTER_Y, -TABLE_THICKNESS / 2])
    table_box = Box(TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS)
    plant.RegisterVisualGeometry(plant.world_body(), X_W_Table, table_box, "table", TABLE_COLOR)
    plant.RegisterCollisionGeometry(plant.world_body(), X_W_Table, table_box, "table_collision", TABLE_FRICTION)


def _add_actuators(plant, arm_index, gripper_index) -> None:
    for joint_name, effort in ARM_JOINT_EFFORTS.items():
        joint = plant.GetJointByName(joint_name, arm_index)
        plant.AddJointActuator(f"{joint_name}_actuator", joint, effort)
    for joint_name, effort in REVO2_JOINT_EFFORTS.items():
        joint = plant.GetJointByName(joint_name, gripper_index)
        plant.AddJointActuator(f"{joint_name}_actuator", joint, effort)


class _PdGravityCompensationController(LeafSystem):
    def __init__(self, plant, joint_names, model_instance, pd_gains):
        super().__init__()
        self._plant = plant
        self._scratch_context = plant.CreateDefaultContext()
        self._joints = [plant.GetJointByName(name, model_instance) for name in joint_names]
        self._gains = [pd_gains[name] for name in joint_names]

        num_q = plant.num_positions()
        num_v = plant.num_velocities()
        self.DeclareVectorInputPort("plant_state", num_q + num_v)
        self.DeclareVectorInputPort("desired_position", len(self._joints))
        self.DeclareVectorOutputPort("actuation", len(self._joints), self._calc_actuation)

    def _calc_actuation(self, context, output):
        state = self.get_input_port(0).Eval(context)
        q_desired = self.get_input_port(1).Eval(context)
        num_q = self._plant.num_positions()
        q, v = state[:num_q], state[num_q:]

        self._plant.SetPositions(self._scratch_context, q)
        self._plant.SetVelocities(self._scratch_context, v)
        tau_gravity = self._plant.CalcGravityGeneralizedForces(self._scratch_context)

        tau = np.empty(len(self._joints))
        for i, (joint, (kp, kd)) in enumerate(zip(self._joints, self._gains)):
            q_i = q[joint.position_start()]
            v_i = v[joint.velocity_start()]
            tau[i] = kp * (q_desired[i] - q_i) - kd * v_i - tau_gravity[joint.velocity_start()]
        output.SetFromVector(tau)


def _add_pd_controller(builder, plant, joint_names, model_instance, pd_gains):
    controller = builder.AddSystem(_PdGravityCompensationController(plant, joint_names, model_instance, pd_gains))
    setpoint_source = builder.AddSystem(ConstantVectorSource(np.zeros(len(joint_names))))
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    builder.Connect(setpoint_source.get_output_port(), controller.get_input_port(1))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port(model_instance))
    return setpoint_source


def _sample_table_pose(
    rng: np.random.Generator, z: float = 0.0, margin: float = 0.03, base_clearance: float = 0.12
) -> RigidTransform:
    x_min, x_max = -ROBOT_OFFSET_X + margin, TABLE_LENGTH - ROBOT_OFFSET_X - margin
    y_min, y_max = -ROBOT_OFFSET_Y + margin, TABLE_WIDTH - ROBOT_OFFSET_Y - margin

    x, y = 0.0, 0.0
    for _ in range(20):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if np.hypot(x, y) >= base_clearance:
            break

    yaw = rng.uniform(0, 2 * np.pi)
    return RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, z])


def _add_wrist_camera(robot_diagram_builder: RobotDiagramBuilder, plant, arm_index) -> RgbdSensor:
    scene_graph = robot_diagram_builder.scene_graph()
    scene_graph.AddRenderer(CAMERA_RENDERER_NAME, MakeRenderEngineVtk(RenderEngineVtkParams()))

    intrinsics = CameraInfo(width=CAMERA_WIDTH_PX, height=CAMERA_HEIGHT_PX, fov_y=CAMERA_FOV_Y)
    clipping = ClippingRange(0.05, 3.0)
    color_camera = ColorRenderCamera(RenderCameraCore(CAMERA_RENDERER_NAME, intrinsics, clipping, RigidTransform()))
    depth_camera = DepthRenderCamera(
        RenderCameraCore(CAMERA_RENDERER_NAME, intrinsics, clipping, RigidTransform()), DepthRange(0.05, 3.0)
    )

    flange_body = plant.GetBodyByName("link_6", arm_index)
    flange_frame_id = plant.GetBodyFrameIdOrThrow(flange_body.index())

    builder = robot_diagram_builder.builder()
    sensor = builder.AddSystem(RgbdSensor(flange_frame_id, CAMERA_TOOL0_OFFSET, color_camera, depth_camera))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    return sensor


def build_arm_gripper_scene(
    meshcat: Meshcat,
    brick_urdf_path: Path = DEFAULT_BRICK_URDF,
    add_table: bool = False,
    scattered_brick_urdf_paths: Optional[List[Path]] = None,
    rng_seed: Optional[int] = None,
    add_camera: bool = False,
    add_controllers: bool = False,
) -> ArmGripperScene:
    robot_diagram_builder = RobotDiagramBuilder()
    plant = robot_diagram_builder.plant()
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)

    meshcat.Delete()
    meshcat.DeleteAddedControls()
    add_meshcat_visualizer(robot_diagram_builder, meshcat)

    resolved_brick_urdf = ensure_renderable_urdf(brick_urdf_path) if add_camera else Path(brick_urdf_path).resolve()
    resolved_scattered_urdfs = [
        ensure_renderable_urdf(p) if add_camera else Path(p).resolve() for p in (scattered_brick_urdf_paths or [])
    ]

    arm_index = parser.AddModels(str(ARM_URDF))[0]
    gripper_index = parser.AddModels(str(GRIPPER_URDF))[0]
    brick_index = parser.AddModels(str(resolved_brick_urdf))[0]

    scattered_brick_indices = [parser.AddModels(str(p))[0] for p in resolved_scattered_urdfs]

    camera_sensor = _add_wrist_camera(robot_diagram_builder, plant, arm_index) if add_camera else None

    world_frame = plant.world_frame()
    arm_frame = plant.GetFrameByName("base_link", arm_index)
    arm_tool_frame = plant.GetFrameByName("link_6", arm_index)
    gripper_frame = plant.GetFrameByName("right_base_link", gripper_index)

    X_W_ArmBase = RigidTransform(RotationMatrix.MakeZRotation(ROBOT_BASE_YAW)) if add_table else RigidTransform()
    plant.WeldFrames(world_frame, arm_frame, X_W_ArmBase)
    plant.WeldFrames(arm_tool_frame, gripper_frame)

    plant.AddFrame(FixedOffsetFrame("base", arm_frame, RigidTransform()))
    arm_tcp_frame = plant.AddFrame(FixedOffsetFrame("tcp", arm_tool_frame, TCP_OFFSET))
    arm_camera_frame = plant.AddFrame(FixedOffsetFrame("camera", arm_tool_frame, CAMERA_TOOL0_OFFSET))

    if add_table:
        _add_table(plant)

    _add_actuators(plant, arm_index, gripper_index)

    arm_setpoint_source = None
    gripper_setpoint_source = None
    if add_controllers:
        builder = robot_diagram_builder.builder()
        plant.Finalize()
        config = VisualizationConfig(publish_contacts=True, enable_alpha_sliders=True)
        ApplyVisualizationConfig(config, builder=builder, plant=plant, meshcat=meshcat)
        arm_setpoint_source = _add_pd_controller(
            builder, plant, list(ARM_PD_GAINS.keys()), arm_index, ARM_PD_GAINS
        )
        gripper_setpoint_source = _add_pd_controller(
            builder, plant, list(REVO2_PD_GAINS.keys()), gripper_index, REVO2_PD_GAINS
        )
        robot_diagram = robot_diagram_builder.Build()
        context = robot_diagram.CreateDefaultContext()
        robot_diagram.ForcedPublish(context)
    else:
        robot_diagram, context = finish_build(robot_diagram_builder, meshcat)

    if scattered_brick_indices:
        plant_context = plant.GetMyContextFromRoot(context)
        rng = np.random.default_rng(rng_seed)
        for scattered_index, urdf_path in zip(scattered_brick_indices, scattered_brick_urdf_paths or []):
            body = plant.get_body(plant.GetBodyIndices(scattered_index)[0])
            z_offset = resting_z_offset(urdf_path)
            plant.SetFreeBodyPose(plant_context, body, _sample_table_pose(rng, z=z_offset))
        robot_diagram.ForcedPublish(context)

    return ArmGripperScene(
        robot_diagram=robot_diagram,
        context=context,
        plant=plant,
        arm_index=arm_index,
        gripper_index=gripper_index,
        brick_index=brick_index,
        arm_tcp_frame=arm_tcp_frame,
        scattered_brick_indices=scattered_brick_indices,
        camera_sensor=camera_sensor,
        arm_camera_frame=arm_camera_frame if add_camera else None,
        arm_setpoint_source=arm_setpoint_source,
        gripper_setpoint_source=gripper_setpoint_source,
    )


def set_gripper_opening(plant, plant_context, gripper_index, closure: float) -> None:
    for joint_name, closed_angle in REVO2_FINGER_JOINTS.items():
        joint = plant.GetJointByName(joint_name, gripper_index)
        joint.set_angle(plant_context, closure * closed_angle)
    for joint_name, (source_name, multiplier) in REVO2_MIMIC_JOINTS.items():
        joint = plant.GetJointByName(joint_name, gripper_index)
        joint.set_angle(plant_context, closure * REVO2_FINGER_JOINTS[source_name] * multiplier)
