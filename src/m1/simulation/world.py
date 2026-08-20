"""The simulated cell: a UR3e with a Robotiq 2F-85, a wrist RealSense, and a flat pile of lego.

Everything ``main.ipynb`` needs to stand up a world, and everything submodule_1 and submodule_2 need
to drive it. Nothing here decides anything -- it is the hardware, and the two submodules are the
software, exactly as on the bench.

The point of the simulation is that the *perception and the decision* are not simulated: submodule_1
hands the rendered colour and depth to ``m1.perception_rgbd.analyse_pile``, the same code that
runs against the real RealSense. What this module has to get right, therefore, is the interface that
code sees -- a colour frame, a metric depth map, an intrinsics matrix, and the camera's pose in the
robot's base frame -- and it gets it right by construction, because Drake knows all four exactly.

Three details of the cell are worth stating outright, because every number downstream rests on them:

* **World frame == robot base frame.** The UR3e's ``base_link`` is welded at the world origin with no
  rotation, so a position in the base frame is a position in Drake's world, and the base-frame
  geometry ``analyse_pile`` returns can be compared to ground truth without a single transform.
* **The tabletop is exactly ``z = 0``**, so the "touched-off table plane" that the physical pipeline
  spends a whole calibration script measuring is ``(a, b, c) = (0, 0, 0)`` here, known perfectly.
* **The TCP is the fingertip plane, and it moves with the opening.** A 2F-85's pads are 37.5 mm tall
  and swing forward as they close, so there is no one point on the gripper that is "the TCP". What
  matters for grasping a 9.6 mm brick is where the *tips* of the pads are, and that is what
  :class:`GripperCalibration` reports, as a function of the commanded opening.

The renderer's noise is deliberate too. A rendered tabletop is one flat colour, and several of the
perception's statistics are measured in units of the table's own scatter -- which, on a table with no
scatter, is a division by zero waiting to happen. Real sensors have noise; adding it back is what
makes the simulated frame a fair test of code tuned on real ones.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from pydrake.geometry import (
    Box,
    ClippingRange,
    CollisionFilterDeclaration,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    LightParameter,
    MakeRenderEngineVtk,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RenderCameraCore,
    RenderEngineVtkParams,
    Role,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import CoulombFriction, DiscreteContactApproximation, MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, PdControllerGains
from pydrake.planning import RobotDiagramBuilder
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.primitives import ConstantVectorSource, MatrixGain
from pydrake.systems.sensors import CameraInfo, RgbdSensor

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from m1.perception_rgbd import PileView  # noqa: E402

REPO_ROOT = Path(_SRC_DIR).parent
ARM_URDF = REPO_ROOT / "src" / "assets" / "ur3e" / "ur3e.urdf"
GRIPPER_URDF = REPO_ROOT / "src" / "assets" / "robotiq_2f_85" / "urdf" / "robotiq_2f_85.urdf"
LEGO_URDF_DIR = REPO_ROOT / "lego_3d" / "urdf"
NORMALS_CACHE_DIR = REPO_ROOT / "src" / "assets" / "normals_cache"

ARM_BASE_LINK = "base_link"
ARM_TOOL_FRAME = "tool0"
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# --- the table --------------------------------------------------------------------------------------
# Sized and placed so the whole pile is inside a UR3e's 0.5 m reach with room to look down at it from
# two angles; the top face is the world's z = 0 plane.
TABLE_LENGTH = 0.70
TABLE_WIDTH = 0.60
TABLE_THICKNESS = 0.03
TABLE_CENTER = (0.28, 0.02)
TABLE_COLOR = np.array([0.55, 0.38, 0.20, 1.0])  # plywood, like the real bench
TABLE_FRICTION = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)

# --- the pile ---------------------------------------------------------------------------------------
# A *flat* pile: every brick lying on its own largest face, none stacked. That is what the real pile
# looks like once it has been spread out, it is the case the grasp planner is built for, and it keeps
# the physics honest -- a heap of mesh-on-mesh contacts in a rigid-body simulator is mostly a study of
# the contact solver, which is not what this simulation is for.
PILE_CENTER = (0.30, 0.02)
PILE_RADIUS = 0.085
PILE_MIN_GAP = 0.004  # metres of bare table between two bricks' footprint circles at spawn
PILE_PLACEMENT_ATTEMPTS = 250
PILE_SETTLE_DURATION = 0.6

#: A representative handful of the real set: bricks and plates, 1x1 to 2x4, in the colours the pile
#: actually contains. Each is a real lego mesh with a real convex collision hull.
DEFAULT_PILE_PARTS: Tuple[str, ...] = (
    "3001__dark_green.urdf" if (LEGO_URDF_DIR / "3001__dark_green.urdf").exists() else "3004__dark_green.urdf",
    "3010__dark_green.urdf",
    "3022__light_bluish_gray.urdf",
    "3021__tan.urdf",
    "3023__medium_nougat.urdf",
    "3069__white.urdf",
    "3070__reddish_brown.urdf",
    "3004__tan.urdf",
    "2431__light_bluish_gray.urdf",
    "3005__dark_bluish_gray.urdf",
    "3024__tan.urdf",
    "3068__light_bluish_gray.urdf",
)

# --- the gripper ------------------------------------------------------------------------------------
# The 2F-85's five other finger joints all mimic ``finger_joint``; Drake honours the URDF's <mimic>
# elements as constraints, but only on a discrete plant using the SAP contact solver, which is why the
# approximation is set before anything is parsed.
GRIPPER_DRIVER_JOINT = "finger_joint"
GRIPPER_MIMIC_JOINTS = {
    "left_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
}
GRIPPER_PAD_FRAMES = ("left_inner_finger_pad", "right_inner_finger_pad")
GRIPPER_PAD_HALF_THICKNESS = 0.00635 / 2  # the pad box's thin axis, from the URDF
GRIPPER_PAD_HALF_HEIGHT = 0.0375 / 2  # ...and its long axis, which is what "fingertip plane" means
GRIPPER_MAX_ANGLE = 0.8  # rad, the driver joint's travel from fully open to fully closed
# The driver joint's stiffness and its torque ceiling, chosen together to land the pinch force where
# a real 2F-85's does. The jaws are commanded 3 mm inside the brick, which is about 0.03 rad of joint
# error, so 27 N*m/rad is a shade under a newton-metre of squeeze -- some tens of newtons at the pads,
# against the 20 N the real gripper's gentlest setting delivers. The ceiling stops a mistimed close
# from flicking a gram of plastic across the table.
GRIPPER_PD_GAINS = (27.0, 1.5)
GRIPPER_EFFORT = 3.0  # N*m
# The fingers close along the gripper's own x. Mounting it a quarter turn round puts that along the
# flange's y, which is the axis the physical submodule_2 defaults to (--closing-axis y).
GRIPPER_MOUNT_YAW = np.pi / 2
GRIPPER_PAD_FRICTION = CoulombFriction(static_friction=1.2, dynamic_friction=1.0)
BRICK_FRICTION = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)

ARM_JOINT_EFFORTS = {
    "shoulder_pan_joint": 56.0,
    "shoulder_lift_joint": 56.0,
    "elbow_joint": 28.0,
    "wrist_1_joint": 12.0,
    "wrist_2_joint": 12.0,
    "wrist_3_joint": 12.0,
}
# Stiff enough that the arm actually arrives. The camera is bolted to the flange, so a steady-state
# joint error is a steady-state error in the pose every measurement is expressed in -- and later, in
# where the fingers come down. Well inside what a one-millisecond step integrates for link inertias
# of this size.
ARM_PD_GAINS = {name: (2000.0, 120.0) for name in ARM_JOINT_NAMES}
#: Held at the final setpoint after each move, so the controller has settled before anything is
#: measured from where the arm ended up.
ARM_SETTLE_DURATION = 0.35

# --- the wrist camera ---------------------------------------------------------------------------------
# A RealSense D435's colour stream: 1280x720 at roughly 42 degrees vertical field of view, which puts
# the focal length near 930 px. That number matters -- the perception derives every pixel threshold it
# uses from ``focal / distance``, so a camera with the wrong focal length would be a different
# instrument, and the thresholds tuned against the real one would not transfer.
CAMERA_WIDTH_PX = 1280
CAMERA_HEIGHT_PX = 720
CAMERA_FOV_Y = 2 * np.arctan(0.5 * CAMERA_HEIGHT_PX / 931.0)
CAMERA_NEAR, CAMERA_FAR = 0.05, 3.0
CAMERA_RENDERER = "vtk"
#: Eye-in-hand mount: on a bracket clear of the fingers, tilted back towards the tool axis so the
#: pile is centred in frame at working distance. The stand-off is set by what has to stay *out* of
#: frame -- a 2F-85 is 160 mm long and 85 mm across, and a gripper filling a third of the picture is
#: a third of the pile the perception never sees.
CAMERA_TOOL_OFFSET = RigidTransform(RotationMatrix.MakeYRotation(np.deg2rad(-16.0)), [0.105, 0.0, 0.02])

# Sensor noise. See the module docstring: without it the table has no scatter, and several of the
# perception's cues are measured in units of exactly that.
RGB_NOISE_COUNTS = 2.0
DEPTH_NOISE_M = 0.0004
DEPTH_DROPOUT_FRACTION = 0.01

# --- contact ------------------------------------------------------------------------------------------
# A lego brick weighs a gram and a half and is a centimetre tall, so how far it sinks into the table
# is not a cosmetic detail: the perception measures every brick's height above that table, and a brick
# resting 4 mm inside it reads as a plate. Drake's default SAP approximation at a millisecond step
# gives exactly that; kLagged with a tight penetration allowance gives 0.14 mm, which is under the
# depth noise. Both are SAP-solver approximations, so the gripper's <mimic> constraints still hold.
CONTACT_APPROXIMATION = DiscreteContactApproximation.kLagged
PENETRATION_ALLOWANCE = 1e-4

# --- control ------------------------------------------------------------------------------------------
#: A comfortable, reachable starting configuration: elbow up, wrist over the table, nothing in frame.
HOME_CONFIGURATION = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

SIM_TIME_STEP = 0.001
IK_POSITION_TOLERANCE = 0.0015
IK_ANGLE_TOLERANCE = np.deg2rad(1.5)
IK_RESTARTS = 12
MOVE_SETPOINT_STEPS = 40  # a move is fed to the controller as this many interpolated setpoints


# =================================================================================================
# meshes
# =================================================================================================


#: Drake's URDF extensions live in this namespace. The stock Robotiq URDF uses ``drake:`` tags without
#: ever declaring the prefix -- Drake's own parser is relaxed about that, ElementTree is not -- so the
#: declaration is added on the way in and registered so it comes back out spelled the same.
DRAKE_NS = "http://drake.mit.edu"
ET.register_namespace("drake", DRAKE_NS)


def _load_urdf(urdf_path: Path) -> ET.ElementTree:
    text = Path(urdf_path).read_text()
    if "xmlns:drake" not in text:
        text = text.replace("<robot ", f'<robot xmlns:drake="{DRAKE_NS}" ', 1)
    return ET.ElementTree(ET.fromstring(text))


def _drake_tag(name: str) -> str:
    return f"{{{DRAKE_NS}}}{name}"


def _mesh_bounds(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                v = np.array([float(x) for x in line.split()[1:4]])
                low = np.minimum(low, v)
                high = np.maximum(high, v)
    return low, high


def _visual_mesh(urdf_path: Path) -> Tuple[Path, float]:
    tree = _load_urdf(urdf_path)
    mesh = tree.getroot().find(".//visual/geometry/mesh")
    scale = mesh.get("scale")
    return (urdf_path.parent / mesh.get("filename")).resolve(), (float(scale.split()[2]) if scale else 1.0)


def brick_footprint(urdf_path: Path) -> Tuple[float, float, float]:
    """``(x_extent, y_extent, resting_z_offset)`` of a part's visual mesh, in metres.

    The resting offset is what lifts the mesh so its lowest vertex sits exactly on the table, which is
    how a flat pile is spawned without waiting for anything to fall.
    """
    mesh_path, scale = _visual_mesh(Path(urdf_path).resolve())
    low, high = _mesh_bounds(mesh_path)
    extent = (high - low) * scale
    return float(extent[0]), float(extent[1]), float(-low[2] * scale)


def _has_normals(obj_path: Path) -> bool:
    with open(obj_path) as f:
        return any(line.startswith("vn ") for line in f)


def _add_normals(src_path: Path, dst_path: Path) -> None:
    """Give a mesh vertex normals, without which the VTK renderer shades it flat black.

    The lego meshes are exported as bare triangle soups. Area-weighted vertex normals -- each face's
    normal, unnormalised so its magnitude carries the face's area, accumulated onto the face's three
    vertices -- are the standard reconstruction and are what the renderer would have wanted.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(src_path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])

    v = np.array(vertices)
    faces_array = np.array(faces)
    normals = np.zeros_like(v)
    v0, v1, v2 = v[faces_array[:, 0]], v[faces_array[:, 1]], v[faces_array[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces_array[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths == 0, 1.0, lengths)

    with open(dst_path, "w") as f:
        for vertex in v:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for normal in normals:
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        for face in faces_array:
            i, j, k = face + 1
            f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")


def prepared_urdf(
    urdf_path: Path,
    friction: Optional[CoulombFriction] = None,
    friction_links: Optional[Sequence[str]] = None,
    tag: str = "sim",
) -> Path:
    """A cached copy of ``urdf_path`` fit to be simulated and rendered.

    Three fixes, none of which belong in the source asset:

    * mesh paths made absolute, so the copy can live in the cache directory;
    * vertex normals added to any visual mesh missing them, or VTK renders the part matte black and
      the colour cue the perception runs on is destroyed before it starts;
    * a friction pair written onto the named links' collisions. Drake's default contact material is
      slippery enough that a 0.7 g brick pinched between two 3 mm strips of finger pad slides straight
      back out, which looks exactly like a grasp planning failure and is not one.

    ``<transmission>`` elements are dropped as well. They are ros_control's way of saying which joints
    a hardware interface drives, and Drake reads them as joint actuators -- which for the 2F-85 means
    inheriting an actuator on ``right_outer_knuckle_joint``, a joint whose whole job is to mimic
    ``finger_joint``. One driver joint, one actuator; the linkage does the rest.
    """
    urdf_path = Path(urdf_path).resolve()
    tree = _load_urdf(urdf_path)
    root = tree.getroot()
    for transmission in root.findall("transmission"):
        root.remove(transmission)

    for mesh in root.iter("mesh"):
        mesh.set("filename", str((urdf_path.parent / mesh.get("filename")).resolve()))
    NORMALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for visual in root.iter("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None or _has_normals(Path(mesh.get("filename"))):
            continue
        mesh_path = Path(mesh.get("filename"))
        cached = NORMALS_CACHE_DIR / mesh_path.name
        if not cached.exists():
            _add_normals(mesh_path, cached)
        mesh.set("filename", str(cached))

    if friction is not None:
        for link in root.iter("link"):
            if friction_links is not None and link.get("name") not in friction_links:
                continue
            for collision in link.findall("collision"):
                properties = ET.SubElement(collision, _drake_tag("proximity_properties"))
                ET.SubElement(properties, _drake_tag("mu_static")).set("value", str(friction.static_friction()))
                ET.SubElement(properties, _drake_tag("mu_dynamic")).set("value", str(friction.dynamic_friction()))

    cached_urdf = NORMALS_CACHE_DIR / f"{tag}_{urdf_path.name}"
    tree.write(cached_urdf)
    return cached_urdf


# =================================================================================================
# the gripper's own geometry, measured rather than assumed
# =================================================================================================


@dataclass(frozen=True)
class GripperCalibration:
    """Opening and fingertip position as functions of the driver joint, swept off the real URDF.

    A 2F-85's fingers travel on a four-bar linkage, so neither the opening nor the height of the
    fingertips is a linear function of the joint angle, and neither is worth deriving by hand when the
    kinematics are sitting right there in the model. This sweeps the linkage once at build time and
    interpolates, which makes every width this simulation commands a real distance between the pads
    rather than a guess dressed up as one.
    """

    angles: np.ndarray  # driver joint, rad
    widths: np.ndarray  # clear distance between the pad faces, metres
    tip_z: np.ndarray  # fingertip plane, in the gripper base frame, metres

    @property
    def max_width(self) -> float:
        return float(self.widths[0])

    @property
    def min_width(self) -> float:
        return float(self.widths[-1])

    def angle_for_width(self, width: float) -> float:
        """The driver angle that opens the pads to ``width``, clamped to what the gripper can do."""
        clamped = float(np.clip(width, self.min_width, self.max_width))
        # ``widths`` decreases with angle, so it has to be reversed for np.interp's ascending-x rule.
        return float(np.interp(clamped, self.widths[::-1], self.angles[::-1]))

    def width_for_angle(self, angle: float) -> float:
        return float(np.interp(float(np.clip(angle, self.angles[0], self.angles[-1])), self.angles, self.widths))

    def tip_offset(self, width: float) -> float:
        """Fingertip plane above the gripper's mounting face, at the given opening."""
        return float(np.interp(self.angle_for_width(width), self.angles, self.tip_z))


def _calibrate_gripper() -> GripperCalibration:
    plant = MultibodyPlant(time_step=SIM_TIME_STEP)
    plant.set_discrete_contact_approximation(CONTACT_APPROXIMATION)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    model = parser.AddModels(str(GRIPPER_URDF))[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", model))
    plant.Finalize()
    context = plant.CreateDefaultContext()

    angles = np.linspace(0.0, GRIPPER_MAX_ANGLE, 81)
    widths, tips = [], []
    for angle in angles:
        plant.GetJointByName(GRIPPER_DRIVER_JOINT, model).set_angle(context, angle)
        for name, multiplier in GRIPPER_MIMIC_JOINTS.items():
            plant.GetJointByName(name, model).set_angle(context, angle * multiplier)
        left, right = (plant.GetFrameByName(n, model).CalcPoseInWorld(context) for n in GRIPPER_PAD_FRAMES)
        separation = float(np.linalg.norm(left.translation() - right.translation()))
        widths.append(separation - 2 * GRIPPER_PAD_HALF_THICKNESS)
        # The pads' own +z is the gripper's +z, so the tips are half a pad above the pad centres.
        tips.append(0.5 * (left.translation()[2] + right.translation()[2]) + GRIPPER_PAD_HALF_HEIGHT)
    return GripperCalibration(angles=angles, widths=np.array(widths), tip_z=np.array(tips))


# =================================================================================================
# the world
# =================================================================================================


@dataclass
class PlacedBrick:
    """One part in the pile, and where the simulator put it. Ground truth, for scoring the perception."""

    index: int
    part: str
    urdf: Path
    model_instance: object
    body_name: str
    footprint_m: Tuple[float, float]

    def pose(self, world: "SimWorld") -> RigidTransform:
        return world.plant.EvalBodyPoseInWorld(world.plant_context, world.plant.GetBodyByName(self.body_name))


@dataclass
class SimWorld:
    """A built, initialized cell. Hold on to it; every other function here takes one."""

    meshcat: Meshcat
    diagram: object
    simulator: Simulator
    plant: MultibodyPlant
    arm_index: object
    gripper_index: object
    tool_frame: object
    camera_frame: object
    camera: RgbdSensor
    arm_setpoint: ConstantVectorSource
    gripper_setpoint: ConstantVectorSource
    gripper_calibration: GripperCalibration
    bricks: List[PlacedBrick]
    ik_plant: MultibodyPlant
    ik_context: object
    rng: np.random.Generator
    table_plane: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _elapsed: float = 0.0
    _commanded_width: float = 0.085

    # --- context ----------------------------------------------------------------------------------
    @property
    def context(self):
        return self.simulator.get_mutable_context()

    @property
    def plant_context(self):
        return self.plant.GetMyContextFromRoot(self.context)

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def advance(self, duration: float) -> None:
        self._elapsed += float(duration)
        self.simulator.AdvanceTo(self._elapsed)

    # --- arm --------------------------------------------------------------------------------------
    def arm_positions(self) -> np.ndarray:
        return self.plant.GetPositions(self.plant_context, self.arm_index).copy()

    def tool_pose(self) -> RigidTransform:
        return self.tool_frame.CalcPoseInWorld(self.plant_context)

    def tcp_pose(self, width: Optional[float] = None) -> RigidTransform:
        """Where the fingertip plane is right now, on the tool's axis."""
        if width is None:
            width = self.finger_width()
        return self.tool_pose() @ self.X_tool_tcp(width)

    def X_tool_tcp(self, width: float) -> RigidTransform:
        """Tool flange -> fingertip plane, at the given opening (see the module docstring)."""
        return RigidTransform([0.0, 0.0, self.gripper_calibration.tip_offset(width)])

    # --- gripper ----------------------------------------------------------------------------------
    def finger_width(self) -> float:
        """The clear distance between the pad faces, measured off the two pad frames.

        Measured rather than commanded, and so it means the same thing the physical
        ``gripper.get_current_width()`` does: with a brick in the jaws this stops where the brick is,
        which is how a grasp is told apart from the fingers meeting on nothing.
        """
        poses = [self.plant.GetFrameByName(n, self.gripper_index).CalcPoseInWorld(self.plant_context) for n in GRIPPER_PAD_FRAMES]
        separation = float(np.linalg.norm(poses[0].translation() - poses[1].translation()))
        return separation - 2 * GRIPPER_PAD_HALF_THICKNESS

    def set_gripper_width(self, width: float) -> None:
        """Command an opening. What the pads actually reach is :meth:`finger_width`, which is the point.

        The command is a *position*, exactly as the real driver takes one: told to close past a brick,
        the controller keeps pushing and the fingers stop where the brick is. That gap between what was
        asked for and what was reached is the whole grasp check, here and on the bench.
        """
        self._commanded_width = float(np.clip(width, self.gripper_calibration.min_width, self.gripper_calibration.max_width))
        angle = self.gripper_calibration.angle_for_width(self._commanded_width)
        _set_source(self.diagram, self.context, self.gripper_setpoint, np.array([angle, 0.0]))

    @property
    def commanded_gripper_width(self) -> float:
        return self._commanded_width

    # --- moving -----------------------------------------------------------------------------------
    def set_arm_target(self, q: np.ndarray) -> None:
        _set_source(self.diagram, self.context, self.arm_setpoint, np.concatenate([np.asarray(q, float), np.zeros(len(q))]))

    def move_arm_to(self, q_goal: np.ndarray, duration: float, steps: int = MOVE_SETPOINT_STEPS) -> None:
        """Drive the arm to ``q_goal`` over ``duration`` seconds.

        The setpoint is walked there in a straight line in joint space rather than stepped to the goal
        in one go. A stiff inverse-dynamics controller handed a distant setpoint accelerates as hard as
        its effort limits allow, which on a table covered in loose lego is the difference between
        picking a brick up and sweeping the pile off the bench.
        """
        q_start = self.arm_positions()
        q_goal = np.asarray(q_goal, float)
        for step in range(1, steps + 1):
            self.set_arm_target(q_start + (q_goal - q_start) * step / steps)
            self.advance(duration / steps)
        self.advance(ARM_SETTLE_DURATION)

    def move_gripper_to_width(self, width: float, duration: float) -> None:
        self.set_gripper_width(width)
        self.advance(duration)

    # --- perception -------------------------------------------------------------------------------
    def capture(self, name: str = "sim view") -> PileView:
        """Render colour and depth, and pair them with the camera pose the arm is actually holding.

        The returned :class:`PileView` is exactly what ``m1.perception_rgbd`` gets from a real
        RealSense, so the perception cannot tell where it is running -- which is the point.
        """
        camera_context = self.camera.GetMyContextFromRoot(self.context)
        colour = np.array(self.camera.color_image_output_port().Eval(camera_context).data)[:, :, :3]
        depth = np.array(self.camera.depth_image_32F_output_port().Eval(camera_context).data).squeeze(-1)

        if RGB_NOISE_COUNTS > 0:
            colour = np.clip(colour.astype(np.float32) + self.rng.normal(0, RGB_NOISE_COUNTS, colour.shape), 0, 255)
        colour = colour.astype(np.uint8)

        depth = depth.astype(np.float32).copy()
        finite = np.isfinite(depth) & (depth > 0)
        if DEPTH_NOISE_M > 0:
            depth[finite] += self.rng.normal(0, DEPTH_NOISE_M, int(finite.sum())).astype(np.float32)
        if DEPTH_DROPOUT_FRACTION > 0:
            depth[self.rng.random(depth.shape) < DEPTH_DROPOUT_FRACTION] = 0.0

        info = self.camera.color_camera_info()
        intrinsics = np.array(
            [[info.focal_x(), 0.0, info.center_x()], [0.0, info.focal_y(), info.center_y()], [0.0, 0.0, 1.0]]
        )
        return PileView(
            image_rgb=colour,
            depth_map=depth,
            intrinsics_matrix=intrinsics,
            X_base_camera=self.camera_frame.CalcPoseInWorld(self.plant_context).GetAsMatrix4(),
            joint_configuration=self.arm_positions(),
            name=name,
        )

    # --- ground truth -----------------------------------------------------------------------------
    def brick_poses(self) -> List[Tuple[PlacedBrick, RigidTransform]]:
        return [(brick, brick.pose(self)) for brick in self.bricks]

    def nearest_brick(self, position_xy: Sequence[float]) -> Tuple[Optional[PlacedBrick], float]:
        """The real brick closest to a point, and how far away it is. For scoring, never for planning."""
        best, best_distance = None, float("inf")
        for brick, pose in self.brick_poses():
            distance = float(np.linalg.norm(pose.translation()[:2] - np.asarray(position_xy, float)[:2]))
            if distance < best_distance:
                best, best_distance = brick, distance
        return best, best_distance


def _set_source(diagram, root_context, source: ConstantVectorSource, value: np.ndarray) -> None:
    source.get_mutable_source_value(diagram.GetMutableSubsystemContext(source, root_context)).set_value(value)


# =================================================================================================
# building it
# =================================================================================================


def _add_table(plant: MultibodyPlant) -> None:
    pose = RigidTransform([TABLE_CENTER[0], TABLE_CENTER[1], -TABLE_THICKNESS / 2])
    box = Box(TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS)
    plant.RegisterVisualGeometry(plant.world_body(), pose, box, "table", TABLE_COLOR)
    plant.RegisterCollisionGeometry(plant.world_body(), pose, box, "table_collision", TABLE_FRICTION)


def _strip_geometry(root: ET.Element) -> None:
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)


def _rigid_model(urdf_path: Path, cache_name: str, freeze_joints: bool) -> Path:
    """A geometry-free copy of a model, optionally with every moving joint welded shut."""
    tree = _load_urdf(urdf_path)
    root = tree.getroot()
    _strip_geometry(root)
    for transmission in root.findall("transmission"):
        root.remove(transmission)
    if freeze_joints:
        for joint in root.findall("joint"):
            if joint.get("type") in ("fixed", None):
                continue
            joint.set("type", "fixed")
            for tag in ("mimic", "limit", "axis", "dynamics"):
                element = joint.find(tag)
                if element is not None:
                    joint.remove(element)
    NORMALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NORMALS_CACHE_DIR / cache_name
    tree.write(out_path)
    return out_path


def _arm_control_plant() -> MultibodyPlant:
    """A six-joint UR3e *carrying the gripper*, for the inverse-dynamics controller to think with.

    Carrying the gripper is the whole point. An inverse-dynamics controller cancels the gravity its
    model knows about, so a model of the bare arm leaves the 2F-85's kilogram uncompensated and the
    proportional term has to take up the slack -- the arm settles wherever the error times the gain
    balances the load, which here was five centimetres from where it was sent. That is invisible when
    the arm is only posing, and fatal when the thing bolted to the flange is the camera whose pose
    every measurement is expressed in.

    The gripper's own joints are welded shut in this copy, so the model is exactly the six degrees of
    freedom the controller commands, with the gripper's mass and inertia rigidly attached.
    """
    arm_urdf = _rigid_model(ARM_URDF, "control_ur3e.urdf", freeze_joints=False)
    gripper_urdf = _rigid_model(GRIPPER_URDF, "control_robotiq.urdf", freeze_joints=True)

    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    arm = parser.AddModels(str(arm_urdf))[0]
    gripper = parser.AddModels(str(gripper_urdf))[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(ARM_BASE_LINK, arm))
    plant.WeldFrames(
        plant.GetFrameByName(ARM_TOOL_FRAME, arm),
        plant.GetFrameByName("base_link", gripper),
        RigidTransform(RotationMatrix.MakeZRotation(GRIPPER_MOUNT_YAW)),
    )
    for name, effort in ARM_JOINT_EFFORTS.items():
        plant.AddJointActuator(f"{name}_actuator", plant.GetJointByName(name, arm), effort)
    plant.Finalize()
    return plant


def _ik_plant() -> Tuple[MultibodyPlant, object]:
    """A UR3e on its own, for inverse kinematics.

    Solving IK on the full scene would put the ten loose bricks' floating positions into the decision
    variables -- seventy variables that no constraint touches, and a solver free to slide the pile
    around inside its own copy of the world while it looks for a wrist angle.
    """
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    model = parser.AddModels(str(ARM_URDF))[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(ARM_BASE_LINK, model))
    plant.Finalize()
    return plant, plant.CreateDefaultContext()


def _add_arm_controller(builder, plant, arm_index) -> ConstantVectorSource:
    control_plant = _arm_control_plant()
    kp = np.array([ARM_PD_GAINS[name][0] for name in ARM_JOINT_NAMES])
    kd = np.array([ARM_PD_GAINS[name][1] for name in ARM_JOINT_NAMES])
    controller = builder.AddSystem(InverseDynamicsController(control_plant, kp, np.zeros(6), kd, False))

    selector = builder.AddSystem(
        MatrixGain(plant.MakeStateSelectorMatrix([plant.GetJointByName(n, arm_index).index() for n in ARM_JOINT_NAMES]))
    )
    builder.Connect(plant.get_state_output_port(), selector.get_input_port())
    builder.Connect(selector.get_output_port(), controller.get_input_port_estimated_state())

    setpoint = builder.AddSystem(ConstantVectorSource(np.zeros(12)))
    builder.Connect(setpoint.get_output_port(), controller.get_input_port_desired_state())
    builder.Connect(controller.get_output_port_control(), plant.get_actuation_input_port(arm_index))
    return setpoint


def _add_gripper_actuator(plant, gripper_index) -> None:
    """Actuate the one driver joint, with its PD gains carried by the actuator itself (pre-Finalize)."""
    joint = plant.GetJointByName(GRIPPER_DRIVER_JOINT, gripper_index)
    actuator = plant.AddJointActuator("finger_actuator", joint, GRIPPER_EFFORT)
    actuator.set_controller_gains(PdControllerGains(p=GRIPPER_PD_GAINS[0], d=GRIPPER_PD_GAINS[1]))


def _add_gripper_controller(builder, plant, gripper_index) -> ConstantVectorSource:
    """Feed the driver joint's desired state; the other five follow through the mimic constraints.

    The gains live *inside* the actuator (``set_controller_gains``), not in a PidController system
    outside the plant, and that distinction is the difference between a gripper and a bomb. A finger
    linkage has a rotational inertia of about 1e-5 kg m^2, so an external PD stiff enough to pinch a
    brick -- tens of newton-metres per radian -- oscillates at a couple of hundred hertz, which a
    one-millisecond discrete step cannot integrate: the loop diverges within a few steps and the
    joint's velocity runs away to hundreds of radians a second while the constraint solver pins the
    position, so the fingers simply never move. Gains handed to the actuator are folded into SAP's
    implicit solve instead, which is unconditionally stable at any stiffness.
    """
    # A PD-controlled actuator still expects a feed-forward torque, and an unconnected input port is
    # an error rather than a zero.
    feedforward = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
    builder.Connect(feedforward.get_output_port(), plant.get_actuation_input_port(gripper_index))

    setpoint = builder.AddSystem(ConstantVectorSource(np.zeros(2)))
    builder.Connect(setpoint.get_output_port(), plant.get_desired_state_input_port(gripper_index))
    return setpoint


def _render_params() -> RenderEngineVtkParams:
    """Light the cell the way a bench is lit: overhead, plus enough fill to kill the black shadows.

    VTK's default is a single headlamp at the camera, which renders every part several shades darker
    than its own colour and drags the whole scene towards black. That is not a cosmetic complaint --
    the perception's colour cue is a distance from the table's colour, and the palette it names the
    result against is a list of real lego colours. Under a headlamp, tan reads as dark tan and light
    grey as sand blue.
    """
    #: ``direction`` is the direction the light *travels*, so a lamp over the table points down.
    return RenderEngineVtkParams(
        lights=[
            LightParameter(type="directional", direction=[0.15, 0.25, -1.0], intensity=0.8, frame="world"),
            LightParameter(type="directional", direction=[-0.6, -0.4, -1.0], intensity=0.4, frame="world"),
            LightParameter(type="directional", direction=[0.0, 0.0, 1.0], intensity=0.35, frame="camera"),
        ]
    )


def _add_wrist_camera(robot_diagram_builder, plant, arm_index) -> RgbdSensor:
    scene_graph = robot_diagram_builder.scene_graph()
    scene_graph.AddRenderer(CAMERA_RENDERER, MakeRenderEngineVtk(_render_params()))

    intrinsics = CameraInfo(width=CAMERA_WIDTH_PX, height=CAMERA_HEIGHT_PX, fov_y=CAMERA_FOV_Y)
    clipping = ClippingRange(CAMERA_NEAR, CAMERA_FAR)
    core = RenderCameraCore(CAMERA_RENDERER, intrinsics, clipping, RigidTransform())
    colour_camera = ColorRenderCamera(core)
    depth_camera = DepthRenderCamera(core, DepthRange(CAMERA_NEAR, CAMERA_FAR))

    flange = plant.GetBodyByName(ARM_TOOL_FRAME, arm_index)
    sensor = robot_diagram_builder.builder().AddSystem(
        RgbdSensor(plant.GetBodyFrameIdOrThrow(flange.index()), CAMERA_TOOL_OFFSET, colour_camera, depth_camera)
    )
    robot_diagram_builder.builder().Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    return sensor


def _filter_robot_self_collisions(scene_graph, plant, arm_index, gripper_index) -> None:
    """Stop the arm and the gripper colliding with themselves.

    The 2F-85's fingers are two branches of a four-bar linkage that the URDF had to cut open to be a
    tree, so the two halves overlap geometrically at every opening; left to itself the contact solver
    spends the whole simulation pushing the gripper apart. The arm's own consecutive links likewise
    touch by construction. Neither is a collision anybody wants reported, and filtering them leaves the
    contacts that matter -- fingers against bricks, bricks against the table -- untouched.
    """
    bodies = [
        plant.get_body(index)
        for model in (arm_index, gripper_index)
        for index in plant.GetBodyIndices(model)
    ]
    geometries = plant.CollectRegisteredGeometries(bodies)
    scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(geometries))


def _sample_flat_pile(
    rng: np.random.Generator, footprints: Sequence[Tuple[float, float]]
) -> List[Optional[Tuple[float, float, float]]]:
    """Lay the parts out in one flat layer inside the pile disc, none of them overlapping.

    Rejection sampling against the circle circumscribing each part's footprint. A circle is generous
    for a 2x4 brick lying at 40 degrees, which is the point: the bricks end up with a little bare table
    between them, and the grasp planner gets a pile it can actually reach into rather than a solid raft
    of plastic.
    """
    placed: List[Optional[Tuple[float, float, float]]] = []
    accepted: List[Tuple[float, float, float]] = []
    for x_extent, y_extent in footprints:
        radius = 0.5 * float(np.hypot(x_extent, y_extent))
        for _ in range(PILE_PLACEMENT_ATTEMPTS):
            angle = rng.uniform(0, 2 * np.pi)
            distance = np.sqrt(rng.uniform(0, 1)) * max(PILE_RADIUS - radius, 0.0)
            x = PILE_CENTER[0] + distance * np.cos(angle)
            y = PILE_CENTER[1] + distance * np.sin(angle)
            if all(np.hypot(x - px, y - py) >= radius + pr + PILE_MIN_GAP for px, py, pr in accepted):
                accepted.append((x, y, radius))
                placed.append((x, y, float(rng.uniform(0, 2 * np.pi))))
                break
        else:
            placed.append(None)
    return placed


def build_world(
    meshcat: Meshcat,
    parts: Sequence[str] = DEFAULT_PILE_PARTS,
    seed: int = 0,
    home_configuration: Optional[Sequence[float]] = None,
) -> SimWorld:
    """Stand up the cell: arm, gripper, wrist camera, table and a flat pile of real lego parts."""
    robot_diagram_builder = RobotDiagramBuilder(time_step=SIM_TIME_STEP)
    plant = robot_diagram_builder.plant()
    # Before anything is parsed: the SAP solver is the only one Drake honours <mimic> under, and the
    # 2F-85's five mimic joints are the whole reason its fingers stay a linkage rather than five
    # independent flippers.
    plant.set_discrete_contact_approximation(CONTACT_APPROXIMATION)
    plant.set_penetration_allowance(PENETRATION_ALLOWANCE)
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)

    meshcat.Delete()
    meshcat.DeleteAddedControls()
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph.get_query_output_port(),
        meshcat,
        MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False),
    )

    arm_index = parser.AddModels(str(ARM_URDF))[0]
    gripper_index = parser.AddModels(
        str(prepared_urdf(GRIPPER_URDF, GRIPPER_PAD_FRICTION, GRIPPER_PAD_FRAMES, tag="grip"))
    )[0]

    urdfs = [LEGO_URDF_DIR / part for part in parts]
    missing = [str(p) for p in urdfs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No such lego URDF(s): {', '.join(missing)}")
    brick_indices = [parser.AddModels(str(prepared_urdf(path, BRICK_FRICTION, tag="lego")))[0] for path in urdfs]

    camera = _add_wrist_camera(robot_diagram_builder, plant, arm_index)

    # World frame == robot base frame: no rotation, no offset, nothing to undo downstream.
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(ARM_BASE_LINK, arm_index))
    tool_frame = plant.GetFrameByName(ARM_TOOL_FRAME, arm_index)
    plant.WeldFrames(
        tool_frame,
        plant.GetFrameByName("base_link", gripper_index),
        RigidTransform(RotationMatrix.MakeZRotation(GRIPPER_MOUNT_YAW)),
    )
    camera_frame = plant.AddFrame(FixedOffsetFrame("wrist_camera", tool_frame, CAMERA_TOOL_OFFSET))
    _add_table(plant)

    for name, effort in ARM_JOINT_EFFORTS.items():
        plant.AddJointActuator(f"{name}_actuator", plant.GetJointByName(name, arm_index), effort)
    _add_gripper_actuator(plant, gripper_index)
    plant.Finalize()
    _filter_robot_self_collisions(scene_graph, plant, arm_index, gripper_index)

    arm_setpoint = _add_arm_controller(builder, plant, arm_index)
    gripper_setpoint = _add_gripper_controller(builder, plant, gripper_index)

    diagram = robot_diagram_builder.Build()
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    calibration = _calibrate_gripper()
    rng = np.random.default_rng(seed)

    home = np.asarray(home_configuration if home_configuration is not None else HOME_CONFIGURATION, float)
    plant.SetPositions(plant_context, arm_index, home)

    bricks: List[PlacedBrick] = []
    footprints = [brick_footprint(path) for path in urdfs]
    layout = _sample_flat_pile(rng, [(fx, fy) for fx, fy, _ in footprints])
    for i, (part, path, model, (x_extent, y_extent, z_offset), place) in enumerate(
        zip(parts, urdfs, brick_indices, footprints, layout)
    ):
        body = plant.get_body(plant.GetBodyIndices(model)[0])
        if place is None:
            logger.warning(f"No room left in the pile for {part}; parking it off the table.")
            plant.SetFreeBodyPose(plant_context, body, RigidTransform([0.0, -0.6, 0.02 * i]))
            continue
        x, y, yaw = place
        plant.SetFreeBodyPose(
            plant_context, body, RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, z_offset])
        )
        bricks.append(
            PlacedBrick(
                index=len(bricks),
                part=part,
                urdf=path,
                model_instance=model,
                body_name=body.name(),
                footprint_m=(x_extent, y_extent),
            )
        )

    ik_plant, ik_context = _ik_plant()
    world = SimWorld(
        meshcat=meshcat,
        diagram=diagram,
        simulator=simulator,
        plant=plant,
        arm_index=arm_index,
        gripper_index=gripper_index,
        tool_frame=tool_frame,
        camera_frame=camera_frame,
        camera=camera,
        arm_setpoint=arm_setpoint,
        gripper_setpoint=gripper_setpoint,
        gripper_calibration=calibration,
        bricks=bricks,
        ik_plant=ik_plant,
        ik_context=ik_context,
        rng=rng,
    )
    world.set_arm_target(home)
    world.set_gripper_width(calibration.max_width)
    simulator.Initialize()
    world.advance(PILE_SETTLE_DURATION)

    logger.info(
        f"Cell ready: {len(bricks)} brick(s) flat on the table around "
        f"({PILE_CENTER[0]:.2f}, {PILE_CENTER[1]:.2f}) m, gripper opening "
        f"{calibration.min_width * 1000:.0f}..{calibration.max_width * 1000:.0f} mm, "
        f"camera {CAMERA_WIDTH_PX}x{CAMERA_HEIGHT_PX} at f={0.5 * CAMERA_HEIGHT_PX / np.tan(CAMERA_FOV_Y / 2):.0f} px."
    )
    logger.info(f"Meshcat: {meshcat.web_url()}")
    return world


# =================================================================================================
# inverse kinematics
# =================================================================================================


def solve_tool_ik(
    world: SimWorld,
    X_W_tool: RigidTransform,
    q_seed: Optional[np.ndarray] = None,
    position_tolerance: float = IK_POSITION_TOLERANCE,
    angle_tolerance: float = IK_ANGLE_TOLERANCE,
    restarts: int = IK_RESTARTS,
) -> Optional[np.ndarray]:
    """Arm joints that put the flange at ``X_W_tool``, or ``None`` if there are none.

    A UR3e reaches most poses in up to eight configurations, and a nonlinear solver only ever finds
    the one nearest its seed -- so this seeds from where the arm is standing *and* from a spread of
    random configurations, and then returns the solution that moves the arm least.

    That last step is not a nicety. Moves here are straight lines in joint space, so a solution on a
    different branch -- elbow flipped, shoulder round the other way -- is a valid answer to "where
    should the joints end up" and a catastrophic answer to "how do I get there": the arm sweeps a
    completely different path through the world on the way, and the path it picks goes through the
    table about as often as not. Picking the nearest branch keeps the interpolation local, and local
    is the only thing that makes it safe.

    Returning ``None`` rather than raising is deliberate: "can the arm get there" is a question the
    grasp planner asks of a dozen candidate bricks and expects a plain answer to.
    """
    plant, context = world.ik_plant, world.ik_context
    tool = plant.GetFrameByName(ARM_TOOL_FRAME)
    lower = np.array([plant.GetJointByName(n).position_lower_limits()[0] for n in ARM_JOINT_NAMES])
    upper = np.array([plant.GetJointByName(n).position_upper_limits()[0] for n in ARM_JOINT_NAMES])

    reference = np.asarray(q_seed, float) if q_seed is not None else world.arm_positions()
    rng = np.random.default_rng(0)  # fixed, so a failed reachability check is reproducible
    seeds = [reference] + [rng.uniform(lower, upper) for _ in range(restarts)]

    solutions: List[np.ndarray] = []
    for seed in seeds:
        ik = InverseKinematics(plant, context)
        ik.AddPositionConstraint(
            tool,
            np.zeros(3),
            plant.world_frame(),
            X_W_tool.translation() - position_tolerance,
            X_W_tool.translation() + position_tolerance,
        )
        ik.AddOrientationConstraint(
            tool, RotationMatrix(), plant.world_frame(), X_W_tool.rotation(), angle_tolerance
        )
        program = ik.prog()
        program.SetInitialGuess(ik.q(), np.asarray(seed, float))
        result = Solve(program)
        if result.is_success():
            solutions.append(np.asarray(result.GetSolution(ik.q()), float))

    if not solutions:
        return None
    return min(solutions, key=lambda q: float(np.linalg.norm(q - reference)))


def look_at_tool_pose(world: SimWorld, eye: Sequence[float], target: Sequence[float]) -> RigidTransform:
    """The flange pose that puts the wrist camera at ``eye`` looking at ``target``.

    Built camera-first, because the camera is the thing with a job: pick where it should be and what it
    should see, then let the mount offset say where the flange has to be for that. The camera frame
    follows the optical convention the depth back-projection assumes -- +z out of the lens, +y down --
    so "down" in the image is world-down projected into the sensor plane.
    """
    eye = np.asarray(eye, float)
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    world_down = np.array([0.0, 0.0, -1.0])
    if abs(float(forward @ world_down)) > 0.98:  # looking straight down: pick any consistent right
        world_down = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_down)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    X_W_camera = RigidTransform(RotationMatrix(np.column_stack([right, down, forward])), eye)
    return X_W_camera @ CAMERA_TOOL_OFFSET.inverse()


def top_down_tool_pose(world: SimWorld, position: Sequence[float], closing_heading: float, width: float) -> RigidTransform:
    """Flange pose that puts the fingertip plane at ``position``, jaws closing along ``closing_heading``.

    The tool points straight down, and the yaw is *solved* rather than guessed: with yaw = 0 the tool
    frame is a half turn about x, which sends the gripper's closing axis (the flange's +y, after the
    quarter-turn mount) to a known heading; adding the difference turns the whole thing about the
    vertical until the jaws line up square to the brick's long axis.
    """
    reference = RotationMatrix.MakeXRotation(np.pi).matrix() @ np.array([0.0, 1.0, 0.0])
    yaw = float(closing_heading) - float(np.arctan2(reference[1], reference[0]))
    rotation = RotationMatrix.MakeZRotation(yaw) @ RotationMatrix.MakeXRotation(np.pi)
    X_W_tcp = RigidTransform(rotation, np.asarray(position, float))
    return X_W_tcp @ world.X_tool_tcp(width).inverse()
