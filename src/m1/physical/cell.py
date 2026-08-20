"""m1 physical: the cell -- arm, gripper, camera, table. Hardware, no decisions.

Bench counterpart of :mod:`m1.simulation.world`, with the same verbs and units. Differences from the
simulator: the TCP is the fingertips (the UR controller carries the tool offset), moves take a speed
rather than a duration, there is no ground truth, and ``X_base_camera`` carries hand-eye error.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_robots.awaitable_action import ACTION_STATUS_ENUM
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.config import (  # noqa: E402
    APPROX_ARM_REACH,
    CAMERA_RESOLUTIONS,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    TABLE_Z,
    connect_arm,
    ensure_control_ready,
    load_camera_pose_in_tcp,
    load_table_plane,
    open_camera,
)
from m1.perception_rgbd import PileView, capture_pile_view  # noqa: E402

# --- the table and the pile -----------------------------------------------------------------------

#: Where the pile is tipped out, base frame. Measure on your own bench; a few cm is close enough.
#: Taken from the centre of the frame the two viewpoints actually see, as reported by
#: ``src/tools/diagnose_table.py``. Confirm it with ``src/tools/teach_pose.py`` if the pile moves.
PILE_CENTER: Tuple[float, float] = (-0.20, -0.32)

#: Parking configuration: elbow up, central, clear of the table. Every cross-table leg goes via here.
HOME_CONFIGURATION = np.array([-0.0834, -1.3199, 0.2621, -0.4055, -1.2062, -1.6360])

#: Measured on the bench, preferred over IK: known reachable and known to see the whole pile. Set an
#: entry to ``None`` to have :func:`m1.physical.submodule_1.observe` solve for the eye position.
VIEWPOINT_JOINT_CONFIGURATIONS = {
    "view 1": np.array([-0.17383129, -1.51381945, 1.21881563, -1.11991935, -1.04903251, -1.86176998]),
    "view 2": np.array([0.88960707, -1.10264479, 0.41630060, -1.06990261, -1.71752865, 0.86208582]),
}

# --- speeds ---------------------------------------------------------------------------------------

DEFAULT_SPEED_RATIO = 10  # percent of max joint speed
DEFAULT_LINEAR_SPEED = 0.03  # m/s, for the descent and the lift
#: Joints ring after a move; a frame grabbed during the ringing gets a stale pose.
ARM_SETTLE_DURATION = 0.35

GRIPPER_FORCE = 50.0  # newtons
GRIPPER_SPEED = 0.05  # m/s
GRIPPER_MOVE_TIMEOUT = 8.0
#: Travel below which a timed-out move means "never moved", not "stopped on something". Above the
#: ~0.4 mm register quantisation, far below a grasp.
GRIPPER_STALL_TOLERANCE_M = 0.002


# --- the gripper ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperCalibration:
    """The opening range the jaws actually have."""

    max_width: float
    min_width: float

    def tip_offset(self, width: float) -> float:
        """Zero: the controller's tool offset already makes ``get_tcp_pose()`` the fingertip plane."""
        return 0.0


def _gripper_calibration(gripper: Optional[ParallelPositionGripper]) -> GripperCalibration:
    if gripper is None:
        return GripperCalibration(max_width=0.085, min_width=0.0)
    specs = gripper.gripper_specs
    return GripperCalibration(max_width=float(specs.max_width), min_width=float(specs.min_width))


# --- the cell -------------------------------------------------------------------------------------


@dataclass
class Cell:
    """A connected bench: arm, camera, optional gripper, and the calibrations tying them together."""

    arm: PositionManipulator
    camera: RGBDCamera
    X_tcp_camera: HomogeneousMatrixType
    table_plane: Tuple[float, float, float]
    robot_type: str = "ur3e"
    gripper: Optional[ParallelPositionGripper] = None
    joint_speed: float = 0.1
    linear_speed: float = DEFAULT_LINEAR_SPEED
    gripper_calibration: GripperCalibration = field(default_factory=lambda: GripperCalibration(0.085, 0.0))
    _started_at: float = field(default_factory=time.monotonic)
    _commanded_width: float = 0.085

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since the cell was built."""
        return time.monotonic() - self._started_at

    def advance(self, duration: float) -> None:
        """Wait."""
        time.sleep(max(0.0, float(duration)))

    def arm_positions(self) -> JointConfigurationType:
        return np.asarray(self.arm.get_joint_configuration(), dtype=float)

    def tool_pose(self) -> HomogeneousMatrixType:
        return self.arm.get_tcp_pose()

    def tcp_pose(self, width: Optional[float] = None) -> HomogeneousMatrixType:
        """The fingertip plane. Identical to :meth:`tool_pose`."""
        return self.arm.get_tcp_pose()

    def X_tool_tcp(self, width: float) -> HomogeneousMatrixType:
        return np.eye(4)

    def move_arm_to(self, q_goal: Sequence[float], duration: Optional[float] = None) -> None:
        """Drive the arm to a joint configuration. ``duration`` is ignored (signature parity)."""
        ensure_control_ready(self.arm)
        self.arm.move_to_joint_configuration(np.asarray(q_goal, float), joint_speed=self.joint_speed).wait()
        self.advance(ARM_SETTLE_DURATION)

    def move_tcp_to(self, pose: HomogeneousMatrixType, linear: bool = False) -> None:
        """Move the TCP to a pose. Use ``linear`` among the bricks: a joint move between poses
        centimetres apart still swings the fingers sideways through the neighbours.
        """
        ensure_control_ready(self.arm)
        if linear:
            self.arm.move_linear_to_tcp_pose(pose, linear_speed=self.linear_speed).wait()
        else:
            self.arm.move_to_tcp_pose(pose, joint_speed=self.joint_speed).wait()
        self.advance(ARM_SETTLE_DURATION)

    def finger_width(self) -> float:
        """The measured gap between the pads -- told to close past a brick, the fingers stop on it."""
        if self.gripper is None:
            raise RuntimeError("No gripper is connected to this cell; build it with `with_gripper=True`.")
        return float(self.gripper.get_current_width())

    @property
    def commanded_gripper_width(self) -> float:
        return self._commanded_width

    def move_gripper_to_width(self, width: float, duration: Optional[float] = None) -> None:
        """Command an opening and confirm the fingers moved.

        A Robotiq accepts commands it cannot execute, so comparing width before and after is the only
        way to tell a stuck gripper from a failed grasp.
        """
        if self.gripper is None:
            raise RuntimeError("No gripper is connected to this cell; build it with `with_gripper=True`.")
        width = float(np.clip(width, self.gripper_calibration.min_width, self.gripper_calibration.max_width))
        before = self.finger_width()
        self._commanded_width = width
        # Re-asserted per move: airo-mono's `move` writes only POS, where ur-rtde's reference driver
        # sets GTO together with POS/SPE/FOR in one command. Anything that cleared GTO since the cell
        # was built -- a stopped program, the URCap's own thread -- would otherwise silently swallow
        # this move and every one after it.
        with contextlib.suppress(Exception):
            self.gripper._communicate("SET GTO 1")
        status = self.gripper.move(width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE).wait(timeout=GRIPPER_MOVE_TIMEOUT)
        after = self.finger_width()
        if status is ACTION_STATUS_ENUM.TIMEOUT and abs(after - before) < GRIPPER_STALL_TOLERANCE_M:
            raise RuntimeError(
                f"The gripper did not move at all in {GRIPPER_MOVE_TIMEOUT:.0f} s: told to go to "
                f"{width * 1000:.0f} mm, still at {after * 1000:.1f} mm. It is accepting commands but not "
                "executing them -- check the Robotiq URCap on the teach pendant (activated and fault-free?), "
                "that the robot is in remote control, and that the fingers are not physically blocked."
            )
        logger.info(f"Jaws {before * 1000:.1f} -> {after * 1000:.1f} mm (commanded {width * 1000:.0f} mm).")

    def is_an_object_grasped(self) -> bool:
        """The Robotiq's object-detection flag. Used with the pad separation, since either alone lies."""
        if self.gripper is None:
            return False
        try:
            return bool(self.gripper.is_an_object_grasped())
        except Exception as exception:  # noqa: BLE001 - a missing flag degrades to the width check
            logger.debug(f"Object-detection flag unavailable: {exception}")
            return True

    def capture(self, name: str = "pile view") -> PileView:
        """Grab colour, depth and the camera pose together."""
        return capture_pile_view(self.arm, self.camera, self.X_tcp_camera, name=name)

    def table_z_at(self, x: float, y: float) -> float:
        a, b, c = self.table_plane
        return float(c + a * x + b * y)


# --- building one ---------------------------------------------------------------------------------


@contextlib.contextmanager
def build_cell(
    robot_type: str = "ur3e",
    ip_address: Optional[str] = None,
    port: int = DEFAULT_REALMAN_PORT,
    calibration_path: str = DEFAULT_CALIBRATION_DIR,
    camera_resolution: str = DEFAULT_CAMERA_RESOLUTION,
    speed_ratio: int = DEFAULT_SPEED_RATIO,
    linear_speed: float = DEFAULT_LINEAR_SPEED,
    table_z: Optional[float] = None,
    with_gripper: bool = True,
) -> Iterator[Cell]:
    """Connect everything and yield a :class:`Cell`, closing it again on the way out.

    The only place here that opens a connection; the hand-eye calibration and table plane load here.
    """
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]
    X_tcp_camera = load_camera_pose_in_tcp(calibration_path)
    plane = resolve_table_plane(table_z)

    with connect_arm(robot_type, ip_address, port) as arm, open_camera(
        CAMERA_RESOLUTIONS[camera_resolution]
    ) as camera:
        ensure_control_ready(arm)
        joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)

        gripper_context = connect_gripper(ip_address) if with_gripper else contextlib.nullcontext(None)
        with gripper_context as gripper:
            if gripper is not None:
                arm.gripper = gripper
            cell = Cell(
                arm=arm,
                camera=camera,
                X_tcp_camera=X_tcp_camera,
                table_plane=plane,
                robot_type=robot_type,
                gripper=gripper,
                joint_speed=joint_speed,
                linear_speed=linear_speed,
                gripper_calibration=_gripper_calibration(gripper),
            )
            logger.success(
                f"Cell up: {robot_type} at {ip_address}, camera at "
                f"{CAMERA_RESOLUTIONS[camera_resolution][0]}x{CAMERA_RESOLUTIONS[camera_resolution][1]}, "
                f"table plane z = {plane[2]:+.4f} + {plane[0]:+.5f} x + {plane[1]:+.5f} y, jaws "
                f"{cell.gripper_calibration.min_width * 1000:.0f}-{cell.gripper_calibration.max_width * 1000:.0f} mm."
            )
            yield cell


def resolve_table_plane(table_z: Optional[float] = None) -> Tuple[float, float, float]:
    """The tabletop as ``z = a*x + b*y + c``: touched-off plane, else ``table_z`` level, else config.

    The touched-off plane wins because it was measured by touching and so carries no hand-eye error.
    """
    if table_z is not None:
        logger.info(f"table_z {table_z:+.4f} m given; using a level plane at that height.")
        return 0.0, 0.0, float(table_z)

    plane = load_table_plane()
    if plane is not None:
        logger.info(f"Using the {plane.describe()}.")
        return plane.a, plane.b, plane.c

    logger.warning(
        f"The table has never been touched off, so every brick height is measured from "
        f"config.TABLE_Z={TABLE_Z:+.4f} m, which is a guess. A 2 cm error there turns every brick into a "
        "clump and every clump into nothing. Run `python src/tools/calibrate_table.py` first."
    )
    return 0.0, 0.0, TABLE_Z


@contextlib.contextmanager
def connect_gripper(robot_ip: str) -> Iterator[ParallelPositionGripper]:
    """Yield a connected, activated Robotiq 2F-85 that is armed to move.

    Reached through the UR controller's URCap socket on the robot's IP. No "open on exit": a
    successful run ends with the brick still held.
    """
    from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

    logger.info(f"Connecting to the Robotiq 2F-85 through the UR controller at {robot_ip} ...")
    try:
        gripper = Robotiq2F85(robot_ip)
    except Exception as exception:
        raise RuntimeError(
            f"Could not talk to the Robotiq 2F-85 via the UR controller at {robot_ip}:63352. Check that the "
            "Robotiq URCap is installed and running on the teach pendant, that the gripper moves from "
            f"Polyscope, and that the robot is in remote control. Original error: {exception}"
        ) from exception
    arm_gripper_for_motion(gripper)
    logger.info(f"Gripper connected and armed; currently {gripper.get_current_width() * 1000:.0f} mm open.")
    yield gripper


#: How long to wait for a ``SET GTO 1`` to show up on a read-back, and how many times to re-send it.
#: Every register write in airo-mono's own driver -- SPE, FOR, POS, ACT -- is wrapped in
#: ``wait_for_condition_with_timeout`` for exactly this reason: the value is written over Modbus and
#: read back over a *separate* TCP connection, so an immediate read races the write and legitimately
#: returns the old value.
GRIPPER_GTO_TIMEOUT = 1.0
GRIPPER_GTO_ATTEMPTS = 3
GRIPPER_GTO_POLL = 0.05


def read_gripper_register(gripper: ParallelPositionGripper, name: str) -> Optional[int]:
    """One of the Robotiq's registers as an int, or ``None`` if it could not be read or parsed."""
    try:
        return int(gripper._communicate(f"GET {name}").split(" ")[-1])
    except Exception as exception:  # noqa: BLE001 - a diagnostic read is never worth aborting on
        logger.debug(f"Could not read the gripper's {name} register: {exception}")
        return None


def set_gripper_gto(gripper: ParallelPositionGripper) -> bool:
    """Set the go-to bit and wait for it to actually read back set. ``True`` if it did.

    Re-sent rather than written once: the URCap's own background thread also writes the request
    registers, so a single external write can be overwritten before it takes.
    """
    for _ in range(GRIPPER_GTO_ATTEMPTS):
        gripper._communicate("SET GTO 1")
        deadline = time.monotonic() + GRIPPER_GTO_TIMEOUT
        while time.monotonic() < deadline:
            if read_gripper_register(gripper, "GTO") == 1:
                return True
            time.sleep(GRIPPER_GTO_POLL)
    return False


def arm_gripper_for_motion(gripper: ParallelPositionGripper) -> None:
    """Check the gripper will actually move: activated (``STA``), faultless (``FLT``), ``GTO`` set.

    A fault is fatal -- the gripper accepts commands and refuses to move, and only a re-activation on
    the pendant clears it. A ``GTO`` that will not read back is *not* fatal, and used to be: the read
    races the write, some URCap versions report the status bit rather than the request bit, and the
    real test of whether the fingers move is whether the fingers move.
    :meth:`Cell.move_gripper_to_width` measures that directly and raises on it, so a wrong guess here
    costs one clear error message at the first move instead of refusing to start at all.
    """
    fault = read_gripper_register(gripper, "FLT")
    if fault:
        raise RuntimeError(
            f"The Robotiq reports fault status FLT {fault}. It will accept commands but not move. Clear the "
            "fault from the Robotiq URCap toolbar on the teach pendant (re-activate the gripper), then re-run."
        )
    if not gripper.gripper_is_active():
        logger.warning("The gripper is not activated; activating it now (the fingers will open and close once).")
        gripper._activate_gripper()
    if not set_gripper_gto(gripper):
        logger.warning(
            f"The gripper's GTO bit still reads "
            f"{read_gripper_register(gripper, 'GTO')} after {GRIPPER_GTO_ATTEMPTS} attempts (STA "
            f"{read_gripper_register(gripper, 'STA')}, FLT {read_gripper_register(gripper, 'FLT')}). Some URCap "
            "versions never report it set over the socket even while moving perfectly well, so this is a "
            "warning and not a refusal: GTO is re-sent with every move, and a gripper that truly does not "
            "move is caught by the width check on the first command."
        )


# --- poses ----------------------------------------------------------------------------------------


def look_at_tool_pose(cell: Cell, eye: Sequence[float], target: Sequence[float]) -> HomogeneousMatrixType:
    """The TCP pose putting the wrist camera at ``eye`` looking at ``target``.

    Built camera-first, in the optical convention the depth back-projection assumes: +z out of the
    lens, +y down.
    """
    eye = np.asarray(eye, float)
    forward = np.asarray(target, float) - eye
    forward = forward / np.linalg.norm(forward)
    world_down = np.array([0.0, 0.0, -1.0])
    if abs(float(forward @ world_down)) > 0.98:  # looking straight down: pick any consistent right
        world_down = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_down)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    X_base_camera = np.eye(4)
    X_base_camera[:3, :3] = np.column_stack([right, down, forward])
    X_base_camera[:3, 3] = eye
    return X_base_camera @ np.linalg.inv(np.asarray(cell.X_tcp_camera, float))


#: All the same grasp: a parallel jaw is unchanged by flipping end for end, and a UR wrist often
#: reaches an orientation a full turn away when it cannot reach it directly.
EQUIVALENT_YAW_OFFSETS = (0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi)
#: The tool axis the fingers close along, set by how the gripper is coupled to the flange.
CLOSING_AXIS = np.array([0.0, 1.0, 0.0])


def top_down_tool_pose(
    cell: Optional[Cell], position: Sequence[float], closing_heading: float, width: float = 0.0
) -> HomogeneousMatrixType:
    """TCP pose at ``position``, tool straight down, fingers closing along ``closing_heading``.

    At yaw = 0 the tool frame is ``Ry(pi)``, which sends the finger axis to a known heading; ``Rz(yaw)``
    turns it square to the brick. ``width`` is unused -- the controller carries the tool offset.
    """
    reference = SE3Container.from_euler_angles_and_translation(np.array([0.0, np.pi, 0.0])).rotation_matrix @ CLOSING_AXIS
    yaw = float(closing_heading) - math.atan2(reference[1], reference[0])
    return SE3Container.from_euler_angles_and_translation(
        np.array([0.0, np.pi, yaw]), np.asarray(position, float)
    ).homogeneous_matrix


def pose_is_reachable(cell: Cell, pose: HomogeneousMatrixType) -> bool:
    """Reachable through the safety limits *and* IK.

    ``is_tcp_pose_reachable`` only checks the safety planes, which passes poses no configuration can
    produce. IK is queried without a seed: ur-rtde seeds itself, and a numpy seed trips a truthiness
    bug in its wrapper.
    """
    try:
        if not cell.arm.is_tcp_pose_reachable(pose):
            return False
    except Exception as exception:  # noqa: BLE001 - a driver without the check must not block us
        logger.debug(f"Safety-limit check unavailable: {exception}")

    try:
        solution = np.asarray(cell.arm.inverse_kinematics(pose), dtype=float)
    except Exception as exception:  # noqa: BLE001 - ur-rtde raises on an unsolvable pose
        logger.debug(f"IK failed for a candidate pose: {exception}")
        return False
    if solution.size != cell.arm.manipulator_specs.dof or not np.all(np.isfinite(solution)):
        return False

    joint_check = getattr(cell.arm, "_is_joint_configuration_reachable", None)
    if joint_check is None:
        return True
    try:
        return bool(joint_check(solution))
    except Exception:  # noqa: BLE001 - an unavailable check is not a reason to reject the pose
        return True


def solve_tool_ik(
    cell: Cell, pose: HomogeneousMatrixType, q_seed: Optional[np.ndarray] = None
) -> Optional[JointConfigurationType]:
    """A joint configuration reaching ``pose``, or ``None``. ``q_seed`` is not forwarded -- see
    :func:`pose_is_reachable`.
    """
    if not pose_is_reachable(cell, pose):
        return None
    try:
        return np.asarray(cell.arm.inverse_kinematics(pose), dtype=float)
    except Exception as exception:  # noqa: BLE001
        logger.debug(f"IK failed for {np.round(pose[:3, 3], 3)} m: {exception}")
        return None


def solve_top_down_ik(
    cell: Cell, position: Sequence[float], closing_heading: float, width: float = 0.0
) -> Optional[Tuple[HomogeneousMatrixType, JointConfigurationType, float]]:
    """``(pose, joint_configuration, heading_used)`` for a reachable equivalent yaw, else ``None``."""
    for offset in EQUIVALENT_YAW_OFFSETS:
        heading = float(closing_heading) + offset
        pose = top_down_tool_pose(cell, position, heading, width)
        q = solve_tool_ik(cell, pose)
        if q is not None:
            return pose, q, heading
    return None


def stop_arm(cell: Cell) -> None:
    """Bring the arm to a controlled stop, for aborts and unexpected exceptions."""
    rtde_control = getattr(cell.arm, "rtde_control", None)
    if rtde_control is None:
        return
    with contextlib.suppress(Exception):
        rtde_control.servoStop()
    with contextlib.suppress(Exception):
        rtde_control.stopL(2.0)


def reach_warning(cell: Cell, position: Sequence[float]) -> Optional[str]:
    """A sentence about the target being near the arm's limit, or ``None``."""
    horizontal = float(np.hypot(position[0], position[1]))
    limit = APPROX_ARM_REACH.get(cell.robot_type, 0.5)
    if horizontal <= 0.9 * limit:
        return None
    return (
        f"{horizontal * 100:.0f} cm from the base horizontally is near the {cell.robot_type}'s "
        f"~{limit * 100:.0f} cm reach; the move may fail as unreachable."
    )
