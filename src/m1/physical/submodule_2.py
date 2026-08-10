"""M1 submodule 2 (physical): grasp the brick under the pregrasp pose and hold it up.

Second half of the two-step pick. :mod:`submodule_1` triangulates a hand-clicked point on a brick and
leaves the TCP hovering ``submodule_1.PREGRASP_HEIGHT`` straight above it; this module starts from
wherever the arm is standing *right now* and finishes the job::

    read the current TCP pose  ->  (optionally re-aim the wrist yaw)  ->  open the gripper
      ->  descend onto the brick  ->  close  ->  verify  ->  lift LIFT_HEIGHT straight up
      ->  verify again, and keep holding

Run order::

    python submodule_1.py            # click the brick in both views, arm ends at the pregrasp
    python submodule_2.py --yaw-deg 30   # grasp it and hold it up

Because the two run as separate processes, the *arm's own pose is the interface* between them -- there
is no file to pass and nothing to keep in sync but ``PREGRASP_HEIGHT``, which is imported from
submodule_1 rather than copied. That does mean submodule_2 must be run while the arm is still parked
where submodule_1 left it: move the arm in between (or freedrive it) and the geometry below is
measuring from the wrong place. :func:`describe_pregrasp_pose` checks what it can -- that the tool
still points straight down, and that the pose is in a plausible place -- and refuses to descend from a
pose that fails those, since a tilted or arbitrary pose means submodule_1's output was lost.

The depth to descend is known rather than measured: the clicked point is on the brick's *top face*, so
the top face is exactly ``PREGRASP_HEIGHT`` below the current TCP, and the grasp sits ``--grasp-depth``
below that. Nothing here uses the camera at all -- at 3 cm the RealSense is far inside its Min-Z blind
zone and its RGB is out of focus, so there is nothing useful left to see from the pregrasp; the
measurement was already made, in submodule_1, from where it could be made well.

Hardware is the UR3e with the Robotiq 2F-85 adaptive gripper on the URCap socket (TCP port 63352 of
the robot controller). Built on airo-mono only: ``airo-robots`` for the arm and the gripper,
``airo-spatial-algebra`` for the pose maths.

Wrist yaw
---------
``submodule_1`` takes whatever yaw ``find_reachable_hover_orientation`` hands back -- it only needs *a*
reachable straight-down pose, so the yaw it lands on has nothing to do with how the brick is lying. The
fingers of a 2F-85 have to close *across* a 1x3 brick's 7.8 mm width, so pass ``--yaw-deg`` with the
direction (in the base frame, degrees) you want the fingers to close along, or ``--yaw-offset-deg`` to
nudge the yaw submodule_1 left. With neither, the current yaw is kept and the run says what direction
that has the fingers closing along, so the number to pass next time is in the log.
"""

import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import click
import numpy as np
from airo_robots.exceptions import RobotConfigurationException
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from loguru import logger

# submodule_2 lives next to submodule_0 and submodule_1; config.py (the shared robot/camera/
# calibration constants) lives two levels up, at the top of src/.
_PHYSICAL_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_PHYSICAL_DIR, "..", ".."))
for _path in (_PHYSICAL_DIR, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from config import (
    APPROX_ARM_REACH,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    SUPPORTED_ROBOT_TYPES,
)
from submodule_0 import connect_arm
from submodule_1 import (
    PREGRASP_HEIGHT,
    ensure_control_ready,
)

# --- the brick, and how to pinch it ---------------------------------------------------------------

# BrickLink 3622 "Brick 1 x 3" in Dark Bluish Gray: 3 x 8 - 0.2 = 23.8 mm long, 1 x 8 - 0.2 = 7.8 mm
# wide, and 9.6 mm tall because it is a brick rather than a plate. The width is what the fingers close
# on and what the grasp is verified against; the height bounds how far it is safe to descend.
BRICK_WIDTH_MM = 7.8
BRICK_HEIGHT_MM = 9.6

# How far below the brick's top face the TCP is driven before closing. Roughly half a brick height puts
# the finger pads on the brick's walls -- not on the studs, which would slip, and not on the table,
# which would jam the fingers and can trip a protective stop.
GRASP_DEPTH_MM = 5.0
# Hard cap on that descent, so no combination of options can drive the fingertips into the table: the
# clicked point is on the top face, and the table is one brick height below it.
MIN_FINGERTIP_CLEARANCE_MM = 1.5

# Metres to lift straight up after the grasp, and hold there. Enough to prove the brick is held and to
# clear the rest of the pile; ``--lift-height`` raises it. If the requested height turns out to be
# unreachable the lift shrinks toward MIN_LIFT_HEIGHT rather than failing -- having grasped the brick,
# lifting it 3 cm is a far better outcome than refusing to lift it at all.
LIFT_HEIGHT = 0.03
MIN_LIFT_HEIGHT = 0.03
LIFT_SEARCH_STEP = 0.01  # metres knocked off the requested lift per attempt while looking for a reachable one

# --- gripper --------------------------------------------------------------------------------------

GRIPPER_APPROACH_MARGIN_MM = 14.0  # added to the brick width for the opening during the descent
GRIPPER_SQUEEZE_MM = 3.0  # commanded *below* the brick width, so the gripper stalls on the brick
GRIPPER_FORCE = 50.0  # newtons; ample for a 1 g brick, gentle enough not to mark it
GRIPPER_SPEED = 0.05  # m/s, slow enough that a mistimed close nudges the pile rather than flicking it
GRIPPER_MAX_STROKE = 0.085  # metres, the 2F-85's full opening

# Verification band on the finger width after closing. Under the lower bound the fingers met with
# nothing between them; over the upper bound something much thicker than the brick is in the grip.
WIDTH_TOLERANCE_BELOW_MM = 4.0
WIDTH_TOLERANCE_ABOVE_MM = 6.0
SETTLE_TIME = 1.0  # seconds to hold after the lift, so a slipping grasp has time to show itself

CLOSING_AXIS_VECTORS = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}
DEFAULT_CLOSING_AXIS = "y"

# --- sanity limits on the inherited pose ----------------------------------------------------------

MAX_TILT_FROM_VERTICAL_DEG = 5.0  # a straight-down pregrasp should be well inside this
MIN_BASE_DISTANCE = 0.15  # metres; closer to the base axis than this the arm is folded into itself
WORKSPACE_MARGIN = 0.90  # fraction of APPROX_ARM_REACH accepted as a horizontal target distance


# =================================================================================================
# reading and re-aiming the pose submodule_1 left behind
# =================================================================================================


@dataclass
class PregraspPose:
    """The pose submodule_1 parked the arm at, and what it implies about the brick."""

    pose: HomogeneousMatrixType
    tilt_deg: float  # angle between the tool's z-axis and straight down
    closing_angle: float  # base-frame heading the fingers currently close along, radians

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def brick_top_z(self) -> float:
        """Height of the brick's top face: the clicked point, PREGRASP_HEIGHT below the TCP."""
        return float(self.pose[2, 3] - PREGRASP_HEIGHT)


def finger_direction(pose: HomogeneousMatrixType, closing_axis: str) -> float:
    """Base-frame heading (radians) that the gripper's fingers close along at ``pose``."""
    direction = pose[:3, :3] @ CLOSING_AXIS_VECTORS[closing_axis]
    return float(math.atan2(direction[1], direction[0]))


def top_down_pose(position: np.ndarray, closing_angle: float, closing_axis: str) -> HomogeneousMatrixType:
    """A straight-down TCP pose at ``position`` whose fingers close along ``closing_angle``.

    The tool's z-axis points down (euler ``[0, pi, yaw]``, the convention submodule_0 and submodule_1
    hover with) and the yaw is *solved* so the configured finger axis ends up along ``closing_angle``
    in the base frame: with yaw = 0 the tool frame is ``Ry(pi)``, and ``Rz(yaw)`` then turns the whole
    thing about the vertical, so the finger axis' heading is simply its heading at yaw = 0 plus the
    yaw. Solving it means ``--closing-axis`` needs no other change to be correct.
    """
    axis = CLOSING_AXIS_VECTORS[closing_axis]
    reference = SE3Container.from_euler_angles_and_translation(np.array([0.0, np.pi, 0.0])).rotation_matrix @ axis
    yaw = closing_angle - math.atan2(reference[1], reference[0])
    return SE3Container.from_euler_angles_and_translation(np.array([0.0, np.pi, yaw]), position).homogeneous_matrix


def describe_pregrasp_pose(arm: PositionManipulator, robot_type: str, closing_axis: str) -> PregraspPose:
    """Read the current TCP pose and check it really is a pregrasp left by submodule_1.

    Everything this module does is measured from this pose, so a pose that is not the one submodule_1
    produced silently invalidates all of it -- the descent would start from the wrong height and end
    somewhere that is not the brick. The two things that are checkable are checked:

    * the tool points (near enough) straight down, which submodule_1's ``find_reachable_hover_orientation``
      guarantees and freedriving or an interrupted run does not;
    * the pose is inside the workspace the arm can actually grasp in.

    Raises:
        RuntimeError: if either check fails, naming what to do about it.
    """
    pose = arm.get_tcp_pose()
    tool_z = pose[:3, 2]
    tilt_deg = math.degrees(math.acos(float(np.clip(-tool_z[2], -1.0, 1.0))))

    if tilt_deg > MAX_TILT_FROM_VERTICAL_DEG:
        raise RuntimeError(
            f"The tool is tilted {tilt_deg:.1f} degrees from straight down (limit "
            f"{MAX_TILT_FROM_VERTICAL_DEG:.0f}). submodule_1 always leaves a straight-down pregrasp, so the "
            "arm is not where it left it -- re-run submodule_1 and then this module, without moving the arm "
            "in between."
        )

    horizontal = float(np.hypot(pose[0, 3], pose[1, 3]))
    reach = APPROX_ARM_REACH.get(robot_type, 0.5)
    if horizontal < MIN_BASE_DISTANCE:
        raise RuntimeError(
            f"The TCP is only {horizontal * 100:.0f} cm from the base axis (minimum "
            f"{MIN_BASE_DISTANCE * 100:.0f} cm); this is not a pregrasp over the table. Re-run submodule_1."
        )
    if horizontal > WORKSPACE_MARGIN * reach:
        raise RuntimeError(
            f"The TCP is {horizontal * 100:.0f} cm from the base horizontally, past the usable "
            f"{WORKSPACE_MARGIN * reach * 100:.0f} cm for a {robot_type}. Move the brick closer to the base "
            "and re-run submodule_1."
        )

    return PregraspPose(pose=pose, tilt_deg=tilt_deg, closing_angle=finger_direction(pose, closing_axis))


def requested_closing_angle(pregrasp: PregraspPose, yaw_deg: Optional[float], yaw_offset_deg: float) -> float:
    """The base-frame direction the fingers should close along, from the CLI options.

    ``submodule_1`` picks its yaw purely for reachability, so what it left is unrelated to how the
    brick lies. ``--yaw-deg`` sets the closing direction outright; ``--yaw-offset-deg`` turns whatever
    the starting point is. Only the *orientation* is ever changed -- the grasp position is rebuilt from
    the pregrasp's x and y, so re-aiming can never shift the TCP off the brick.
    """
    base_angle = math.radians(yaw_deg) if yaw_deg is not None else pregrasp.closing_angle
    return base_angle + math.radians(yaw_offset_deg)


def pose_is_reachable(arm: PositionManipulator, pose: HomogeneousMatrixType) -> bool:
    """Whether the arm can reach ``pose``, checked through the safety limits *and* IK.

    ``is_tcp_pose_reachable`` only asks whether the pose is inside the safety planes, which passes for
    poses no joint configuration can produce; the IK call catches those. IK is queried without a seed
    on purpose: ur-rtde seeds from the current configuration itself, and passing a numpy seed trips a
    bug in its wrapper (``joint_configuration_guess or np.array([])`` raises "truth value of an array
    is ambiguous").
    """
    try:
        if not arm.is_tcp_pose_reachable(pose):
            return False
    except Exception as exception:  # noqa: BLE001 - a driver without the check must not block us
        logger.debug(f"Safety-limit check unavailable: {exception}")

    try:
        solution = np.asarray(arm.inverse_kinematics(pose), dtype=float)
    except Exception as exception:  # noqa: BLE001 - ur-rtde raises on an unsolvable pose
        logger.debug(f"IK failed for a candidate pose: {exception}")
        return False
    if solution.size != arm.manipulator_specs.dof or not np.all(np.isfinite(solution)):
        return False

    joint_check = getattr(arm, "_is_joint_configuration_reachable", None)
    if joint_check is None:
        return True
    try:
        return bool(joint_check(solution))
    except Exception:  # noqa: BLE001 - an unavailable check is not a reason to reject the pose
        return True


def reachable_lift_pose(
    arm: PositionManipulator, grasp_pose: HomogeneousMatrixType, requested_height: float
) -> Optional[Tuple[HomogeneousMatrixType, float]]:
    """The highest reachable straight-up lift from ``grasp_pose``, at most ``requested_height``.

    Lifting is the last thing that happens and the arm is already stretched out over the table by the
    time it does, so the requested height is the first thing to become unreachable. Refusing the whole
    run over that would leave the brick grasped and sitting on the table, which is strictly worse than
    lifting it less far, so the height is walked down in :data:`LIFT_SEARCH_STEP` steps and the first
    reachable one is taken.

    Returns ``None`` only if even :data:`MIN_LIFT_HEIGHT` is out of reach, which means the grasp should
    not be attempted at all.
    """
    # Heights are computed off the step index and rounded, not by repeatedly subtracting: repeated
    # subtraction drifts (0.10 - 3 x 0.01 lands at 0.06999999999999999), which both prints badly and
    # can push a height that exactly fits the arm's limit just past it, losing a whole centimetre.
    n_steps = int(math.floor((requested_height - MIN_LIFT_HEIGHT) / LIFT_SEARCH_STEP + 1e-9))
    heights = [round(requested_height - step * LIFT_SEARCH_STEP, 6) for step in range(n_steps + 1)]
    if not heights or heights[-1] > MIN_LIFT_HEIGHT + 1e-9:
        heights.append(MIN_LIFT_HEIGHT)

    for height in heights:
        pose = grasp_pose.copy()
        pose[2, 3] = grasp_pose[2, 3] + height
        if pose_is_reachable(arm, pose):
            return pose, height
    return None


def reachable_yaw_pose(
    arm: PositionManipulator, position: np.ndarray, closing_angle: float, closing_axis: str
) -> Optional[Tuple[HomogeneousMatrixType, float]]:
    """A reachable straight-down pose at ``position`` with the requested closing direction.

    A parallel-jaw grasp is unchanged by flipping the fingers, and the UR wrist can reach the same
    orientation a full turn away as well, so the equivalent yaws are tried in order of how far the
    wrist has to travel. Returns ``None`` if none of them are reachable.
    """
    for delta in (0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi):
        pose = top_down_pose(position, closing_angle + delta, closing_axis)
        if pose_is_reachable(arm, pose):
            return pose, closing_angle + delta
    return None


# =================================================================================================
# gripper
# =================================================================================================


@contextlib.contextmanager
def connect_gripper(robot_ip: str) -> Iterator[ParallelPositionGripper]:
    """Yield a connected Robotiq 2F-85.

    The gripper is reached through the UR controller's URCap socket on the *robot's* IP, so it needs
    no address of its own. A failure here is almost always the URCap not running or the robot not
    being in remote control, which the raised message says outright instead of leaving a bare socket
    error.

    Deliberately no "open on exit" teardown: the point of a successful run is that the brick is still
    held when the process ends. Releasing after a *failed* grasp is :func:`release_and_retreat`'s job,
    where it is a decision rather than a side effect.
    """
    from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

    logger.info(f"Connecting to the Robotiq 2F-85 through the UR controller at {robot_ip}...")
    try:
        gripper = Robotiq2F85(robot_ip)
    except Exception as exception:
        raise RuntimeError(
            f"Could not talk to the Robotiq 2F-85 via the UR controller at {robot_ip}:63352. Check that the "
            "Robotiq URCap is installed and running on the teach pendant, that the gripper moves from "
            f"Polyscope, and that the robot is in remote control. Original error: {exception}"
        ) from exception
    logger.info(f"Gripper connected; currently {gripper.get_current_width() * 1000:.0f} mm open.")
    yield gripper


def check_grasp(gripper: ParallelPositionGripper, brick_width_mm: float) -> Tuple[bool, str]:
    """Whether the gripper is holding a brick of that width right now, and how we know.

    Two independent signals, because either one alone lies:

    * the Robotiq's own object-detection flag (from motor current) says "I stalled on something", but
      it also fires when the fingers simply stall against each other;
    * the finger width says *what* is between the pads -- near zero means they closed on air, much
      wider means something else came along (a neighbouring brick, or this brick gripped end-on).
    """
    width_mm = float(gripper.get_current_width()) * 1000.0
    detected = bool(gripper.is_an_object_grasped())

    if not detected:
        return False, f"the gripper reports no object between the fingers (width {width_mm:.1f} mm)"
    if width_mm < brick_width_mm - WIDTH_TOLERANCE_BELOW_MM:
        return False, (
            f"the fingers closed to {width_mm:.1f} mm, under the brick's {brick_width_mm:.1f} mm -- "
            "nothing in the grip"
        )
    if width_mm > brick_width_mm + WIDTH_TOLERANCE_ABOVE_MM:
        return False, (
            f"the fingers stopped at {width_mm:.1f} mm, too wide for a {brick_width_mm:.1f} mm brick -- "
            "something else is in the grip"
        )
    return True, f"holding a {width_mm:.1f} mm object (the brick is {brick_width_mm:.1f} mm)"


# =================================================================================================
# the pick
# =================================================================================================


def release_and_retreat(
    arm: PositionManipulator, gripper: ParallelPositionGripper, pregrasp_pose: HomogeneousMatrixType, linear_speed: float
) -> None:
    """Recover from a failed grasp: let go, then go back up to the pregrasp.

    Order matters: opening first drops whatever was half-grasped straight back where it came from,
    from millimetres up, rather than carrying it away and dropping it somewhere else.
    """
    try:
        gripper.move(GRIPPER_MAX_STROKE, speed=GRIPPER_SPEED).wait()
    except Exception as exception:  # noqa: BLE001 - recovery must not raise
        logger.warning(f"Could not open the gripper while recovering: {exception}")
    try:
        arm.move_linear_to_tcp_pose(pregrasp_pose, linear_speed=linear_speed).wait()
    except (RobotConfigurationException, RuntimeError) as exception:
        logger.warning(f"Could not retreat to the pregrasp after the failed grasp: {exception}")


def grasp_and_lift(
    arm: PositionManipulator,
    gripper: ParallelPositionGripper,
    pregrasp_pose: HomogeneousMatrixType,
    grasp_pose: HomogeneousMatrixType,
    lift_pose: HomogeneousMatrixType,
    brick_width_mm: float,
    linear_speed: float,
) -> Tuple[bool, str]:
    """Open, descend, close, verify, lift, verify again. Returns ``(success, reason)``.

    Every move here is a straight line rather than a joint move: the poses are only centimetres apart,
    and a joint move between two of them can still swing the fingers sideways through the neighbouring
    bricks on the way down.

    The grasp is verified *twice* -- right after closing, which catches "closed on nothing", and again
    after the lift and a settling pause, which catches the brick slipping out as the arm accelerates.
    That second check is the one that separates "the gripper is closed" from "the brick is held", and
    it is the whole point of this module.
    """
    open_width = min((brick_width_mm + GRIPPER_APPROACH_MARGIN_MM) / 1000.0, GRIPPER_MAX_STROKE)
    close_width = max((brick_width_mm - GRIPPER_SQUEEZE_MM) / 1000.0, 0.0)

    logger.info(f"Opening the gripper to {open_width * 1000:.0f} mm.")
    gripper.move(open_width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE).wait()

    descent_mm = (pregrasp_pose[2, 3] - grasp_pose[2, 3]) * 1000.0
    logger.info(f"Descending {descent_mm:.1f} mm onto the brick.")
    arm.move_linear_to_tcp_pose(grasp_pose, linear_speed=linear_speed).wait()

    logger.info(f"Closing the gripper to {close_width * 1000:.0f} mm.")
    gripper.move(close_width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE).wait()

    holding, reason = check_grasp(gripper, brick_width_mm)
    if not holding:
        return False, f"the grasp did not take at the table: {reason}"
    lift_mm = (lift_pose[2, 3] - grasp_pose[2, 3]) * 1000.0
    logger.info(f"Closed on the brick: {reason}. Lifting {lift_mm / 10:.0f} cm.")

    arm.move_linear_to_tcp_pose(lift_pose, linear_speed=linear_speed).wait()
    time.sleep(SETTLE_TIME)
    holding, reason = check_grasp(gripper, brick_width_mm)
    if not holding:
        return False, f"the brick was lost during the lift: {reason}"
    return True, reason


def stop_arm(arm: PositionManipulator) -> None:
    """Bring the arm to a controlled stop, for aborts and unexpected exceptions."""
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None:
        return
    with contextlib.suppress(Exception):
        rtde_control.servoStop()
    with contextlib.suppress(Exception):
        rtde_control.stopL(2.0)


# =================================================================================================
# CLI
# =================================================================================================


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm to control. Only the ur3e carries the Robotiq 2F-85 this module grasps with.",
)
@click.option(
    "--ip-address",
    default=None,
    help=f"Robot controller IP address. Defaults per robot type (ur3e: {DEFAULT_IP_ADDRESSES['ur3e']}, "
    f"realman: {DEFAULT_IP_ADDRESSES['realman']}).",
)
@click.option("--port", default=DEFAULT_REALMAN_PORT, show_default=True, help="Controller port (RealMan only).")
@click.option(
    "--speed-ratio",
    type=click.IntRange(1, 100),
    default=10,
    show_default=True,
    help="1..100, fraction of the arm's max joint speed, used for the wrist turn above the brick.",
)
@click.option(
    "--linear-speed",
    type=click.FloatRange(0.005, 0.25),
    default=0.03,
    show_default=True,
    help="m/s for the descent onto the brick and the lift off the table.",
)
@click.option(
    "--lift-height",
    type=click.FloatRange(MIN_LIFT_HEIGHT, 0.30),
    default=LIFT_HEIGHT,
    show_default=True,
    help="Metres to lift the brick straight up and hold it there. If the arm cannot reach that high at "
    f"this position the lift shrinks toward {MIN_LIFT_HEIGHT * 100:.0f} cm rather than failing.",
)
@click.option(
    "--yaw-deg",
    type=float,
    default=None,
    help="Base-frame direction (degrees) the fingers should close along, i.e. across the brick's long "
    "axis. Defaults to keeping the yaw submodule_1 left, which was chosen for reachability and has "
    "nothing to do with how the brick lies -- the run logs the resulting direction either way.",
)
@click.option(
    "--yaw-offset-deg",
    type=float,
    default=0.0,
    show_default=True,
    help="Degrees to turn the wrist from --yaw-deg, or from the inherited yaw if that is not given. "
    "Use 90 when the fingers are lined up along the brick instead of across it.",
)
@click.option(
    "--closing-axis",
    type=click.Choice(sorted(CLOSING_AXIS_VECTORS)),
    default=DEFAULT_CLOSING_AXIS,
    show_default=True,
    help="Tool axis the gripper's fingers close along, set by how the gripper is coupled to the flange.",
)
@click.option(
    "--grasp-depth-mm",
    type=click.FloatRange(0.0, BRICK_HEIGHT_MM - MIN_FINGERTIP_CLEARANCE_MM),
    default=GRASP_DEPTH_MM,
    show_default=True,
    help="How far below the brick's top face to descend before closing. Capped so the fingertips stay "
    f"{MIN_FINGERTIP_CLEARANCE_MM:.1f} mm above the table for a {BRICK_HEIGHT_MM:.1f} mm brick.",
)
@click.option(
    "--brick-width-mm",
    type=click.FloatRange(1.0, 80.0),
    default=BRICK_WIDTH_MM,
    show_default=True,
    help="Width the fingers close on, used to size the grip and to verify it. Default is a 1x3 brick's "
    "7.8 mm short side; pass 23.8 to grip the same brick end-on instead.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report the inherited pose and the planned grasp, then stop without moving or gripping. The "
    "way to check the descent and the finger direction before letting the arm touch the pile.",
)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt before descending.")
def main(
    robot_type: str,
    ip_address: Optional[str],
    port: int,
    speed_ratio: int,
    linear_speed: float,
    lift_height: float,
    yaw_deg: Optional[float],
    yaw_offset_deg: float,
    closing_axis: str,
    grasp_depth_mm: float,
    brick_width_mm: float,
    dry_run: bool,
    yes: bool,
) -> None:
    """Grasp the brick under submodule_1's pregrasp pose and hold it up (UR3e + Robotiq 2F-85)."""
    if robot_type != "ur3e":
        raise click.ClickException(
            f"--robot-type {robot_type} has no parallel gripper wired up in this module; only the ur3e "
            "carries the Robotiq 2F-85 that the grasp is executed with."
        )
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]

    with connect_arm(robot_type, ip_address, port) as arm:
        # submodule_1 ends with a long pause at the pregrasp and this module starts as a fresh process,
        # so the UR control script may well have stopped in between -- and while it is stopped ur-rtde
        # reports *every* pose as unreachable, which looks exactly like a kinematic rejection.
        ensure_control_ready(arm)

        try:
            pregrasp = describe_pregrasp_pose(arm, robot_type, closing_axis)
        except RuntimeError as exception:
            raise click.ClickException(str(exception)) from exception

        logger.info(
            f"Inherited pregrasp at {pregrasp.position.round(4)} m (base frame), tool {pregrasp.tilt_deg:.1f} deg "
            f"off vertical; the brick's top face is {PREGRASP_HEIGHT * 100:.0f} cm below it at "
            f"z={pregrasp.brick_top_z:.4f} m."
        )
        logger.info(
            f"submodule_1 left the fingers closing along {math.degrees(pregrasp.closing_angle):.0f} deg "
            f"(base frame, {closing_axis}-axis of the tool)."
        )

        requested_angle = requested_closing_angle(pregrasp, yaw_deg, yaw_offset_deg)
        grasp_position = np.array(
            [pregrasp.position[0], pregrasp.position[1], pregrasp.brick_top_z - grasp_depth_mm / 1000.0]
        )

        solved = reachable_yaw_pose(arm, grasp_position, requested_angle, closing_axis)
        if solved is None:
            raise click.ClickException(
                f"No reachable straight-down pose at {grasp_position.round(4)} m with the fingers closing "
                f"along {math.degrees(requested_angle):.0f} deg. The pregrasp itself is reachable, so this is "
                "the wrist running out of travel -- try --yaw-offset-deg 90, or re-run submodule_1 with the "
                "brick a little closer to the base."
            )
        grasp_pose, actual_angle = solved

        # The wrist is re-aimed at the pregrasp height, so the fingers turn in free space above the
        # brick rather than sweeping sideways through it on the way down.
        aimed_pose = grasp_pose.copy()
        aimed_pose[2, 3] = pregrasp.position[2]

        lift = reachable_lift_pose(arm, grasp_pose, lift_height)
        if lift is None:
            raise click.ClickException(
                f"The grasp is reachable but not even a {MIN_LIFT_HEIGHT * 100:.0f} cm lift above it is, so the "
                "brick could be grasped and never raised. Re-run submodule_1 with the brick closer to the base."
            )
        lift_pose, actual_lift = lift
        if actual_lift < lift_height - 1e-9:
            logger.warning(
                f"A {lift_height * 100:.0f} cm lift is out of reach here; lifting {actual_lift * 100:.0f} cm "
                "instead, which is as high as the arm goes at this position."
            )
        if not pose_is_reachable(arm, aimed_pose):
            raise click.ClickException(
                "The re-aimed pregrasp is not reachable; try a different --yaw-deg / --yaw-offset-deg."
            )

        logger.info(
            f"Plan: turn the wrist to close along {math.degrees(actual_angle):.0f} deg, descend "
            f"{(pregrasp.position[2] - grasp_position[2]) * 1000:.1f} mm to z={grasp_position[2]:.4f} m "
            f"({grasp_depth_mm:.1f} mm into the brick, leaving "
            f"{BRICK_HEIGHT_MM - grasp_depth_mm:.1f} mm of fingertip clearance above the table), close on "
            f"{brick_width_mm:.1f} mm, then lift {actual_lift * 100:.0f} cm to z={lift_pose[2, 3]:.4f} m."
        )

        if dry_run:
            logger.info("--dry-run: stopping here without moving or gripping.")
            return

        if not yes and not click.confirm(
            f"Descend onto the brick at {grasp_position.round(3)} m and grasp it?", default=True
        ):
            logger.info("Aborted by the user.")
            return

        with connect_gripper(ip_address) as gripper:
            arm.gripper = gripper
            try:
                if not np.allclose(aimed_pose[:3, :3], pregrasp.pose[:3, :3], atol=1e-3):
                    joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)
                    logger.info(
                        f"Turning the wrist to {math.degrees(actual_angle):.0f} deg at the pregrasp height, "
                        "clear of the brick."
                    )
                    arm.move_to_tcp_pose(aimed_pose, joint_speed=joint_speed).wait()

                success, reason = grasp_and_lift(
                    arm, gripper, aimed_pose, grasp_pose, lift_pose, brick_width_mm, linear_speed
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted; stopping the arm.")
                stop_arm(arm)
                raise
            except RobotConfigurationException as exception:
                stop_arm(arm)
                raise click.ClickException(
                    f"The controller refused a pose mid-grasp: {exception}. The target is "
                    f"{np.hypot(grasp_pose[0, 3], grasp_pose[1, 3]) * 100:.0f} cm from the base horizontally "
                    f"and a {robot_type} reaches ~{APPROX_ARM_REACH[robot_type] * 100:.0f} cm."
                ) from exception
            except Exception:
                stop_arm(arm)
                raise

            if not success:
                logger.error(f"Grasp failed: {reason}")
                release_and_retreat(arm, gripper, aimed_pose, linear_speed)
                sys.exit(1)

            logger.success(
                f"Grasped and holding the brick {actual_lift * 100:.0f} cm up: {reason}. "
                "The gripper stays closed -- the brick is still held."
            )
            logger.info(f"TCP now at:\n{arm.get_tcp_pose().round(4)}")


if __name__ == "__main__":
    main()
