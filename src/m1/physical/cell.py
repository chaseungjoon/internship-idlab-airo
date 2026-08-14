"""m1 physical: the cell -- arm, gripper, camera, table. Hardware, no decisions.

The bench counterpart of :mod:`m1.simulation.world`, and deliberately the same shape: the same verbs,
the same names, the same units, so that :mod:`m1.physical.submodule_1` and
:mod:`m1.physical.submodule_2` can be read side by side with their simulation twins and differ only
where the hardware genuinely differs. Everything that talks to a robot, a gripper or a camera lives
here; nothing here decides anything.

Almost all of it is airo-mono underneath -- ``PositionManipulator`` for the arm, ``RGBDCamera`` for the
RealSense, ``ParallelPositionGripper`` for the Robotiq, ``SE3Container`` for every pose built by hand.
This module is the adapter that gives those the vocabulary the simulation already speaks.

**Where the bench and the simulator really differ**, and so where reading the two files side by side
will show something other than a copy:

* **The TCP is the fingertips, not the flange.** The UR controller carries the tool offset, so
  ``arm.get_tcp_pose()`` already reports the point the grasp is about, and
  :func:`top_down_tool_pose` needs no gripper geometry at all. The simulation has to add the flange to
  fingertip offset itself, and that offset depends on the opening, because a 2F-85's fingertips swing
  through an arc as the jaws close. Hence :meth:`GripperCalibration.tip_offset` returning zero here:
  the correction is real, it is just already applied by the controller.
* **Moves take a speed, not a duration.** A simulated move is integrated for a fixed number of
  seconds; a real one is commanded at a joint or linear speed and takes as long as it takes. The
  duration arguments are kept in the signatures so the two modules stay line-for-line comparable, and
  they are used only to pace the settling waits.
* **There is no ground truth.** :meth:`SimWorld.nearest_brick` has no counterpart, and neither does
  ``score_against_truth``. On the bench the only mark on the homework is whether the grasp worked.
* **The camera pose is measured, not exact.** ``X_base_camera`` is forward kinematics composed with
  the hand-eye calibration, and the calibration is wrong by millimetres. That single fact is why
  :mod:`m1.physical.submodule_1` prefers the ray-plane projection where the simulation prefers the
  triangulation -- see its module docstring.
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
from config import (  # noqa: E402
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
from m1.physical.submodule_3 import PileView, capture_pile_view  # noqa: E402

# =================================================================================================
# the table and the pile
# =================================================================================================

#: Where on the table the pile is tipped out, in the robot's base frame. **Measure this on your own
#: bench** -- it only has to be right to a few centimetres (it aims the viewpoints and nothing else),
#: but a value from somebody else's table points the camera at bare plywood.
PILE_CENTER: Tuple[float, float] = (0.30, 0.02)

#: The arm's parking configuration: elbow up, central, well clear of the table. Every cross-table leg
#: goes via here for the same reason the simulation does it -- a joint-space straight line between two
#: poses at opposite edges of the workspace sweeps the wrist through everything in between.
HOME_CONFIGURATION = np.array([-0.0834, -1.3199, 0.2621, -0.4055, -1.2062, -1.6360])

#: Known-good viewpoint joint configurations, measured on the bench. Preferred over solving IK for a
#: Cartesian eye position because they are known to be reachable, comfortable and to see the whole
#: pile -- none of which a fresh IK solution at the edge of a UR3e's workspace guarantees. Set an
#: entry to ``None`` to have :func:`m1.physical.submodule_1.observe` solve for the eye position
#: instead, exactly as the simulation does.
VIEWPOINT_JOINT_CONFIGURATIONS = {
    "view 1": np.array([-0.08343679, -1.31992237, 0.26209098, -0.40548201, -1.20620281, -1.63604099]),
    "view 2": np.array([0.81975543, -1.24165185, 0.23308164, -0.76548697, -1.72945053, -0.95741016]),
}

# =================================================================================================
# speeds
# =================================================================================================

#: Fraction of the arm's maximum joint speed. Ten percent is the value the standalone modules have
#: always run at: fast enough not to be tedious, slow enough that a mistimed move nudges the pile
#: rather than scattering it.
DEFAULT_SPEED_RATIO = 10
#: m/s for the short straight-line moves -- the descent onto a brick and the lift off the table.
DEFAULT_LINEAR_SPEED = 0.03
#: The arm has stopped commanding before the camera is asked for a frame. Real joints ring for a
#: moment after a move completes, and a frame grabbed during the ringing is paired with a TCP pose
#: that is no longer where the lens was.
ARM_SETTLE_DURATION = 0.35

GRIPPER_FORCE = 50.0  # newtons; ample for a 1 g brick, gentle enough not to mark it
GRIPPER_SPEED = 0.05  # m/s
GRIPPER_MOVE_TIMEOUT = 8.0
#: Finger travel under which a completed-but-timed-out move means "the gripper never moved" rather
#: than "it stopped early on something". Above the ~0.4 mm register quantisation, far below a grasp.
GRIPPER_STALL_TOLERANCE_M = 0.002


# =================================================================================================
# the gripper, described the way the simulation describes it
# =================================================================================================


@dataclass(frozen=True)
class GripperCalibration:
    """The opening range the jaws actually have, and where the fingertips are relative to the TCP."""

    max_width: float
    min_width: float

    def tip_offset(self, width: float) -> float:
        """Zero, on purpose. See the module docstring.

        The UR controller already carries the tool offset, so the pose ``arm.get_tcp_pose()`` reports
        *is* the fingertip plane. The simulation's version of this returns a real, opening-dependent
        distance because there it is measuring from the flange. Keeping the method means the two
        modules can share a line rather than fork around it.
        """
        return 0.0


def _gripper_calibration(gripper: Optional[ParallelPositionGripper]) -> GripperCalibration:
    if gripper is None:
        return GripperCalibration(max_width=0.085, min_width=0.0)
    specs = gripper.gripper_specs
    return GripperCalibration(max_width=float(specs.max_width), min_width=float(specs.min_width))


# =================================================================================================
# the cell
# =================================================================================================


@dataclass
class Cell:
    """A connected bench: arm, camera, optional gripper, and the calibrations that tie them together.

    Hold on to it; every other function in the physical m1 modules takes one. The counterpart of
    :class:`m1.simulation.world.SimWorld`, with the same verbs.
    """

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

    # --- time -------------------------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since the cell was built, where the simulation reports simulated time."""
        return time.monotonic() - self._started_at

    def advance(self, duration: float) -> None:
        """Wait. The bench's answer to stepping the integrator forward."""
        time.sleep(max(0.0, float(duration)))

    # --- arm --------------------------------------------------------------------------------------
    def arm_positions(self) -> JointConfigurationType:
        return np.asarray(self.arm.get_joint_configuration(), dtype=float)

    def tool_pose(self) -> HomogeneousMatrixType:
        return self.arm.get_tcp_pose()

    def tcp_pose(self, width: Optional[float] = None) -> HomogeneousMatrixType:
        """Where the fingertip plane is right now. Identical to :meth:`tool_pose` -- see the docstring."""
        return self.arm.get_tcp_pose()

    def X_tool_tcp(self, width: float) -> HomogeneousMatrixType:
        return np.eye(4)

    def move_arm_to(self, q_goal: Sequence[float], duration: Optional[float] = None) -> None:
        """Drive the arm to a joint configuration.

        ``duration`` is accepted and ignored: a real move runs at :attr:`joint_speed` and takes as long
        as the controller needs. It stays in the signature so this reads the same as the simulation's.
        """
        ensure_control_ready(self.arm)
        self.arm.move_to_joint_configuration(np.asarray(q_goal, float), joint_speed=self.joint_speed).wait()
        self.advance(ARM_SETTLE_DURATION)

    def move_tcp_to(self, pose: HomogeneousMatrixType, linear: bool = False) -> None:
        """Move the TCP to a pose, either as a joint move or as a straight line in the world.

        Straight-line for the short legs among the bricks -- a joint move between two poses only
        centimetres apart still swings the fingers sideways through the neighbours on the way.
        """
        ensure_control_ready(self.arm)
        if linear:
            self.arm.move_linear_to_tcp_pose(pose, linear_speed=self.linear_speed).wait()
        else:
            self.arm.move_to_tcp_pose(pose, joint_speed=self.joint_speed).wait()
        self.advance(ARM_SETTLE_DURATION)

    # --- gripper ----------------------------------------------------------------------------------
    def finger_width(self) -> float:
        """The clear distance between the pads, *measured*.

        Measured rather than commanded, which is what makes it a grasp check: told to close past a
        brick, the fingers stop where the brick is, and the gap between commanded and reached is the
        whole signal. Exactly what the simulation measures off the two pad frames.
        """
        if self.gripper is None:
            raise RuntimeError("No gripper is connected to this cell; build it with `with_gripper=True`.")
        return float(self.gripper.get_current_width())

    @property
    def commanded_gripper_width(self) -> float:
        return self._commanded_width

    def move_gripper_to_width(self, width: float, duration: Optional[float] = None) -> None:
        """Command an opening and confirm the fingers actually went somewhere.

        A Robotiq accepts every command whether or not it is in a state to execute one, so a gripper
        that never moves looks healthy right up until the grasp is blamed on the brick. Comparing the
        width before and after separates the two outright.
        """
        if self.gripper is None:
            raise RuntimeError("No gripper is connected to this cell; build it with `with_gripper=True`.")
        width = float(np.clip(width, self.gripper_calibration.min_width, self.gripper_calibration.max_width))
        before = self.finger_width()
        self._commanded_width = width
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
        """The Robotiq's own object-detection flag, read from motor current.

        The simulation has no equivalent and infers the same thing from the pad separation alone. Here
        both signals are available and both are used, because either one on its own lies: the flag
        also fires when the fingers stall against each other, and the width alone cannot tell a held
        brick from one wedged between the pads and a neighbour.
        """
        if self.gripper is None:
            return False
        try:
            return bool(self.gripper.is_an_object_grasped())
        except Exception as exception:  # noqa: BLE001 - a missing flag degrades to the width check
            logger.debug(f"Object-detection flag unavailable: {exception}")
            return True

    # --- camera -----------------------------------------------------------------------------------
    def capture(self, name: str = "pile view") -> PileView:
        """Grab colour, depth and the camera pose together.

        Byte-for-byte the same :class:`~m1.physical.submodule_3.PileView` the simulation renders, which
        is what lets one perception module serve both.
        """
        return capture_pile_view(self.arm, self.camera, self.X_tcp_camera, name=name)

    def table_z_at(self, x: float, y: float) -> float:
        a, b, c = self.table_plane
        return float(c + a * x + b * y)


# =================================================================================================
# building one
# =================================================================================================


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
    """Connect everything and yield a :class:`Cell`, closing it all again on the way out.

    The counterpart of ``world.build_world``, and the only place in the physical m1 modules that opens
    a connection. Everything it needs that is not a wire -- the hand-eye calibration, the touched-off
    table plane -- is loaded here too, so that a module downstream never has to wonder whether it has
    them.
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
    """The tabletop as ``z = a*x + b*y + c``, best source first.

    The touched-off plane wins: the arm measured it by touching, so unlike anything the camera says it
    carries no hand-eye calibration error -- and every brick height is measured from it, so an error
    here is an error in every brick at once. An explicit ``table_z`` overrides it with a level plane
    (the tilt is lost), and ``config.TABLE_Z`` is the last resort.
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
        "clump and every clump into nothing. Run `python src/calibrate_table.py` first."
    )
    return 0.0, 0.0, TABLE_Z


@contextlib.contextmanager
def connect_gripper(robot_ip: str) -> Iterator[ParallelPositionGripper]:
    """Yield a connected, activated Robotiq 2F-85 that is armed to move.

    Reached through the UR controller's URCap socket on the *robot's* IP, so it needs no address of its
    own. Deliberately no "open on exit": the point of a successful run is that the brick is still held
    when it ends.
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


def arm_gripper_for_motion(gripper: ParallelPositionGripper) -> None:
    """Make sure the gripper will actually *move* when it is told to, and say so if it will not.

    A Robotiq only moves when three things hold: it is activated (``ACT``/``STA``), it is faultless
    (``FLT``), and its go-to bit (``GTO``) is set. Register writes are accepted regardless, so a
    gripper with ``GTO`` clear -- which is how a stopped Polyscope program or an aborted run leaves it
    -- connects cleanly, reports its width, accepts every command, and never moves.
    """

    def read_register(name: str) -> Optional[int]:
        try:
            return int(gripper._communicate(f"GET {name}").split(" ")[-1])
        except Exception as exception:  # noqa: BLE001 - a diagnostic read is never worth aborting on
            logger.debug(f"Could not read the gripper's {name} register: {exception}")
            return None

    fault = read_register("FLT")
    if fault:
        raise RuntimeError(
            f"The Robotiq reports fault status FLT {fault}. It will accept commands but not move. Clear the "
            "fault from the Robotiq URCap toolbar on the teach pendant (re-activate the gripper), then re-run."
        )
    if not gripper.gripper_is_active():
        logger.warning("The gripper is not activated; activating it now (the fingers will open and close once).")
        gripper._activate_gripper()
    gripper._communicate("SET GTO 1")
    if read_register("GTO") == 0:
        raise RuntimeError(
            "Could not set the gripper's GTO (go-to) bit, so it would accept move commands without moving. "
            "Check that the Robotiq URCap is running and the robot is in remote control."
        )


# =================================================================================================
# poses -- the same three the simulation builds, built the same way
# =================================================================================================


def look_at_tool_pose(cell: Cell, eye: Sequence[float], target: Sequence[float]) -> HomogeneousMatrixType:
    """The TCP pose that puts the wrist camera at ``eye`` looking at ``target``.

    Built camera-first, because the camera is the thing with a job: pick where the lens should be and
    what it should see, then let the hand-eye calibration say where the TCP has to be for that. The
    camera frame follows the optical convention the depth back-projection assumes -- +z out of the
    lens, +y down -- so "down" in the image is world-down projected into the sensor plane.
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


#: Straight-down poses are tried at these yaw offsets before being called unreachable. A parallel-jaw
#: grasp is unchanged by flipping the fingers end for end, and a UR wrist can often reach the same
#: orientation a full turn away when it cannot reach it directly -- so all five are the same grasp.
EQUIVALENT_YAW_OFFSETS = (0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi)
#: The tool axis the fingers close along, set by how the gripper is coupled to the flange.
CLOSING_AXIS = np.array([0.0, 1.0, 0.0])


def top_down_tool_pose(
    cell: Optional[Cell], position: Sequence[float], closing_heading: float, width: float = 0.0
) -> HomogeneousMatrixType:
    """TCP pose at ``position``, tool straight down, fingers closing along ``closing_heading``.

    The yaw is *solved* rather than guessed: with yaw = 0 the tool frame is ``Ry(pi)``, which sends the
    finger axis to a known heading; ``Rz(yaw)`` then turns the whole thing about the vertical until the
    jaws line up square to the brick's long axis.

    ``width`` is accepted and unused -- the controller carries the tool offset, so the fingertip plane
    does not move with the opening the way it does in the simulation. See the module docstring.
    """
    reference = SE3Container.from_euler_angles_and_translation(np.array([0.0, np.pi, 0.0])).rotation_matrix @ CLOSING_AXIS
    yaw = float(closing_heading) - math.atan2(reference[1], reference[0])
    return SE3Container.from_euler_angles_and_translation(
        np.array([0.0, np.pi, yaw]), np.asarray(position, float)
    ).homogeneous_matrix


def pose_is_reachable(cell: Cell, pose: HomogeneousMatrixType) -> bool:
    """Whether the arm can reach ``pose``, checked through the safety limits *and* IK.

    ``is_tcp_pose_reachable`` only asks whether the pose is inside the safety planes, which passes for
    poses no joint configuration can produce; the IK call catches those. IK is queried without a seed
    on purpose: ur-rtde seeds from the current configuration itself, and passing a numpy seed trips a
    bug in its wrapper (``joint_configuration_guess or np.array([])`` raises "truth value of an array
    is ambiguous").
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
    """A joint configuration that reaches ``pose``, or ``None``. The controller's IK, not our own.

    ``q_seed`` is accepted for signature parity with the simulation and deliberately not forwarded:
    ur-rtde already seeds from the arm's current configuration, and handing it a numpy array trips a
    truthiness bug in its wrapper.
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
    """A reachable straight-down pose at ``position`` closing along ``closing_heading``, and its IK.

    Tries the equivalent yaws in order of how far the wrist has to travel. Returns
    ``(pose, joint_configuration, heading_used)``, or ``None`` if none of them are reachable.
    """
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
    """A sentence about the target being near the arm's limit, or ``None`` if it comfortably is not."""
    horizontal = float(np.hypot(position[0], position[1]))
    limit = APPROX_ARM_REACH.get(cell.robot_type, 0.5)
    if horizontal <= 0.9 * limit:
        return None
    return (
        f"{horizontal * 100:.0f} cm from the base horizontally is near the {cell.robot_type}'s "
        f"~{limit * 100:.0f} cm reach; the move may fail as unreachable."
    )
