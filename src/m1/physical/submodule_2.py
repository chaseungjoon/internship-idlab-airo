"""
m1 submodule 2 (physical): grasp the brick under the pregrasp pose and hold it up.
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
from airo_robots.awaitable_action import ACTION_STATUS_ENUM
from airo_robots.exceptions import RobotConfigurationException
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import (
    APPROX_ARM_REACH,
    BRICK_HANDOFF_MAX_AGE,
    BRICK_HANDOFF_PATH,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    FALLBACK_BRICK_HEIGHT,
    FALLBACK_BRICK_WIDTH,
    PREGRASP_HEIGHT,
    SUPPORTED_ROBOT_TYPES,
    connect_arm,
    ensure_control_ready,
    load_table_plane,
)
from m1.physical.brick_measure import read_handoff

# --- the brick, and how to pinch it ---------------------------------------------------------------

# Fallbacks for when submodule_1's handoff is missing or stale (see config.FALLBACK_BRICK_*). The
# real dimensions come from the handoff: submodule_1 measures the clicked brick's footprint from both
# viewpoints and identifies the part, because the pile holds ~66 part numbers and only one of them is
# the 1x3 these describe. Everything about the gripper below is in millimetres, so convert once here.
FALLBACK_BRICK_WIDTH_MM = FALLBACK_BRICK_WIDTH * 1000.0
FALLBACK_BRICK_HEIGHT_MM = FALLBACK_BRICK_HEIGHT * 1000.0

# How far below the brick's top face the TCP is driven before closing. Roughly half a brick height puts
# the finger pads on the brick's walls -- not on the studs, which would slip, and not on the table,
# which would jam the fingers and can trip a protective stop.
GRASP_DEPTH_MM = 3.0
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
# A 2F-85 crosses its whole 85 mm stroke in a couple of seconds even at the slow speed above, so a
# move that is still unfinished after this is stuck rather than slow, and there is no reason to sit
# through the driver's 30 s default before saying so.
GRIPPER_MOVE_TIMEOUT = 8.0
# Finger travel under which we call a move "the gripper never moved" rather than "it stopped early on
# something". Comfortably above the ~0.4 mm register quantisation, far below a real grasp's travel.
GRIPPER_STALL_TOLERANCE_MM = 2.0

# Verification band on the finger width after closing. Under the lower bound the fingers met with
# nothing between them; over the upper bound something much thicker than the brick is in the grip.
WIDTH_TOLERANCE_BELOW_MM = 4.0
WIDTH_TOLERANCE_ABOVE_MM = 6.0
SETTLE_TIME = 1.0  # seconds to hold after the lift, so a slipping grasp has time to show itself

CLOSING_AXIS_VECTORS = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}
DEFAULT_CLOSING_AXIS = "y"

# --- sanity limits on the inherited pose ----------------------------------------------------------

# Gap between the touched-off table and submodule_1's depth estimate worth flagging. It is the
# hand-eye calibration's vertical error, so it is a diagnostic rather than a fault.
TABLE_SOURCE_DISAGREEMENT = 0.003

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


def requested_closing_angle(
    pregrasp: PregraspPose,
    yaw_deg: Optional[float],
    yaw_offset_deg: float,
    measured_closing_deg: Optional[float] = None,
) -> float:
    """The base-frame direction the fingers should close along.

    ``submodule_1`` picks the pregrasp's yaw purely for reachability, so what it left is unrelated to
    how the brick lies -- which is why this used to need ``--yaw-deg`` supplied by hand, guessed by
    eye. It no longer does: submodule_1 now measures the brick's long axis, and the handoff carries
    the direction square to it, which is the direction a parallel jaw has to close along. Precedence
    is ``--yaw-deg`` (an explicit instruction wins), then the measurement, then the inherited yaw.

    Only the *orientation* is ever changed -- the grasp position is rebuilt from the pregrasp's x and
    y, so re-aiming can never shift the TCP off the brick.
    """
    if yaw_deg is not None:
        base_angle = math.radians(yaw_deg)
    elif measured_closing_deg is not None:
        base_angle = math.radians(measured_closing_deg)
        logger.info(f"Closing along {measured_closing_deg:.0f} deg, square to the brick's measured long axis.")
    else:
        base_angle = pregrasp.closing_angle
        logger.warning(
            "No measured brick orientation and no --yaw-deg, so the fingers keep the yaw submodule_1 happened "
            f"to land on ({math.degrees(pregrasp.closing_angle):.0f} deg), which was chosen for reachability and "
            "has nothing to do with how the brick lies. Expect to have to pass --yaw-deg."
        )
    return base_angle + math.radians(yaw_offset_deg)


def resolve_brick(
    handoff: Optional[dict], width_override_mm: Optional[float], height_override_mm: Optional[float]
) -> Tuple[float, float, Optional[float]]:
    """Settle on the brick's width, height and closing direction, in that order of authority.

    An explicit CLI value wins, then submodule_1's measurement, then the 1x3 fallback. Returns
    ``(width_mm, height_mm, closing_heading_deg or None)``.
    """
    closing_deg = None
    width_mm = FALLBACK_BRICK_WIDTH_MM
    height_mm = FALLBACK_BRICK_HEIGHT_MM

    if handoff is None:
        logger.warning(
            f"Falling back to the 1x3 brick's {FALLBACK_BRICK_WIDTH_MM:.1f} x {FALLBACK_BRICK_HEIGHT_MM:.1f} mm. "
            "If the brick on the table is anything else, pass --brick-width-mm and --brick-height-mm, or re-run "
            "submodule_1 so it can measure it."
        )
    else:
        width_from_handoff = handoff.get("width")
        height_from_handoff = handoff.get("height")
        if width_from_handoff is not None:
            width_mm = float(width_from_handoff) * 1000.0
        if height_from_handoff is not None:
            height_mm = float(height_from_handoff) * 1000.0

        if handoff.get("closing_heading") is not None:
            closing_deg = math.degrees(float(handoff["closing_heading"]))

        part = handoff.get("part_number")
        if part is not None and width_from_handoff is not None and height_from_handoff is not None:
            same_size = handoff.get("same_size_parts") or []
            also = f" (or {len(same_size)} other part(s) of the same size)" if same_size else ""
            logger.info(
                f"submodule_1 measured this brick as part {part}{also}: {width_mm:.1f} mm wide, "
                f"{height_mm:.1f} mm tall, long axis at {math.degrees(float(handoff['long_axis_heading'])):.0f} deg."
            )
            if handoff.get("obstruction", 0.0) > 0:
                logger.warning(
                    f"That part has {float(handoff['obstruction']) * 1000:.1f} mm of structure standing above the face "
                    "being grasped; the fingers may foul it on the way down."
                )
        elif width_from_handoff is not None or height_from_handoff is not None:
            logger.info(
                f"submodule_1 left dimensions for this run: width {width_mm:.1f} mm, height {height_mm:.1f} mm."
            )
        else:
            logger.warning(
                "submodule_1 left only the brick position and runtime table floor, not trustworthy dimensions; "
                f"falling back to {FALLBACK_BRICK_WIDTH_MM:.1f} x {FALLBACK_BRICK_HEIGHT_MM:.1f} mm unless "
                "overridden on the command line."
            )

    if width_override_mm is not None:
        logger.info(f"--brick-width-mm {width_override_mm:.1f} overrides the {width_mm:.1f} mm above.")
        width_mm = width_override_mm
    if height_override_mm is not None:
        logger.info(f"--brick-height-mm {height_override_mm:.1f} overrides the {height_mm:.1f} mm above.")
        height_mm = height_override_mm
    return width_mm, height_mm, closing_deg


def resolve_runtime_table_floor_z(pregrasp: PregraspPose, handoff: Optional[dict], brick_height_mm: float) -> float:
    """Table surface height to treat as a hard floor for this run, best source first.

    Three sources, in descending order of how much they can be trusted:

    1. **The touched-off plane** from ``calibrate_table.py``, evaluated under this grasp's x, y. The
       arm physically touched the tabletop to make it, so no camera, no hand-eye calibration and no
       brick-height assumption are involved -- only forward kinematics and the TCP offset.
    2. **submodule_1's depth estimate**, carried in the handoff. It sees the table directly but
       through the eye-in-hand camera pose, so it inherits the hand-eye calibration's error along the
       view direction. That error is what drove the fingertips into the table: the camera put the
       tabletop at z = -0.0240 m where touching it found z = -0.0044 m.
    3. **Inference** from the parked pregrasp minus the brick height, which assumes everything
       upstream was right and therefore protects against nothing.

    Where 1 and 2 are both available their disagreement is worth printing: it *is* the hand-eye
    calibration's vertical error, measured, and it is the number to watch after a re-calibration.
    """
    inferred_table_z = float(pregrasp.brick_top_z - brick_height_mm / 1000.0)
    depth_table_z = float(handoff["safe_table_z"]) if handoff and handoff.get("safe_table_z") is not None else None

    plane = load_table_plane()
    if plane is not None:
        touched_table_z = plane.z_at(float(pregrasp.position[0]), float(pregrasp.position[1]))
        logger.info(
            f"Table floor z={touched_table_z:.4f} m under this grasp, from the {plane.describe()}."
        )
        if depth_table_z is not None and abs(depth_table_z - touched_table_z) > TABLE_SOURCE_DISAGREEMENT:
            logger.warning(
                f"submodule_1's depth put the table at z={depth_table_z:.4f} m, "
                f"{(depth_table_z - touched_table_z) * 1000:+.1f} mm from where the arm touched it. That gap is the "
                "hand-eye calibration's vertical error; the touched value is used. Re-run the hand-eye "
                "calibration with more board poses to close it."
            )
        if touched_table_z > inferred_table_z + 1e-6:
            logger.warning(
                f"The touched table is {1000.0 * (touched_table_z - inferred_table_z):.1f} mm above what the parked "
                "pregrasp and the brick height imply, so it will tighten the allowed descent."
            )
        return touched_table_z

    logger.warning(
        "The table has never been touched off, so the floor below falls back to the camera. Run "
        "`python src/calibrate_table.py` -- it is the difference between a floor that is measured and one that "
        "inherits the hand-eye calibration's error."
    )
    if depth_table_z is None:
        logger.warning(
            f"No per-run table floor was handed off either, so the table is inferred at z={inferred_table_z:.4f} m "
            "from the parked pregrasp and the assumed brick height. That adds no protection at all."
        )
        return inferred_table_z

    configured_table_z = handoff.get("table_z_configured")
    measured_views = handoff.get("table_z_measured_views") or []
    logger.info(
        f"Using submodule_1's runtime table floor z={depth_table_z:.4f} m"
        + (
            f" (configured {float(configured_table_z):.4f} m, depth views {', '.join(f'{float(z):.4f}' for z in measured_views)} m)."
            if measured_views or configured_table_z is not None
            else "."
        )
    )
    if depth_table_z > inferred_table_z + 1e-6:
        logger.warning(
            f"That floor is {1000.0 * (depth_table_z - inferred_table_z):.1f} mm above what the parked pregrasp "
            "and the brick height imply, so it will tighten the allowed descent for safety."
        )
    return depth_table_z


def resolve_grasp_depth(
    requested_mm: Optional[float], brick_height_mm: float, brick_top_z: float, table_floor_z: float
) -> float:
    """How far below the top face to descend, capped for the part actually being grasped.

    The cap used to be a constant in the ``--grasp-depth-mm`` option's range, computed from the 1x3's
    9.6 mm. On a 3.2 mm plate that same 5 mm descent would drive the fingertips 1.8 mm *below* the
    tabletop -- into the table, which jams the fingers and trips a protective stop. Now there are two
    per-run caps and the tighter one wins:

    * the part geometry itself, which says how far the fingertips can descend into this brick;
    * the runtime table floor from submodule_1, which says how low the TCP is allowed to go in this
      run even if the parked pregrasp ended up lower than expected.
    """
    geometric_ceiling = brick_height_mm - MIN_FINGERTIP_CLEARANCE_MM
    runtime_ceiling = (brick_top_z - table_floor_z) * 1000.0 - MIN_FINGERTIP_CLEARANCE_MM
    ceiling = min(geometric_ceiling, runtime_ceiling)
    if ceiling <= 0:
        raise click.ClickException(
            f"The brick top at z={brick_top_z:.4f} m and the runtime table floor at z={table_floor_z:.4f} m leave "
            f"no room to descend while keeping {MIN_FINGERTIP_CLEARANCE_MM:.1f} mm of fingertip clearance above "
            "the table. Re-run submodule_1; the parked pregrasp is already too low for a safe top-down grasp."
        )
    if runtime_ceiling < geometric_ceiling - 1e-6:
        logger.warning(
            f"The runtime table floor at z={table_floor_z:.4f} m tightens the descent cap from "
            f"{geometric_ceiling:.1f} mm to {runtime_ceiling:.1f} mm."
        )

    depth = GRASP_DEPTH_MM if requested_mm is None else requested_mm
    if depth > ceiling:
        logger.warning(
            f"A {depth:.1f} mm descent would cross the runtime table limit; capping it at {ceiling:.1f} mm."
        )
        depth = ceiling
    logger.info(
        f"Descending {depth:.1f} mm into the part's {brick_height_mm:.1f} mm, leaving at least "
        f"{brick_top_z - table_floor_z - depth / 1000.0:.4f} m between the TCP and the runtime table floor."
    )
    return depth


def ensure_pose_above_table_floor(name: str, pose: HomogeneousMatrixType, table_floor_z: float) -> None:
    """Reject any commanded TCP pose that would cross the runtime table floor."""
    min_tcp_z = table_floor_z + MIN_FINGERTIP_CLEARANCE_MM / 1000.0
    if pose[2, 3] < min_tcp_z - 1e-6:
        raise click.ClickException(
            f"{name} would command the TCP to z={pose[2, 3]:.4f} m, below the runtime floor of "
            f"z={min_tcp_z:.4f} m (table {table_floor_z:.4f} m + {MIN_FINGERTIP_CLEARANCE_MM:.1f} mm "
            "clearance). Refusing to descend."
        )


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


def _arm_gripper_for_motion(gripper: ParallelPositionGripper) -> None:
    """Make sure the gripper will actually *move* when it is told to, and say so if it will not.

    A Robotiq only moves when three things hold: it is activated (``ACT``/``STA``), it is faultless
    (``FLT``), and its go-to bit (``GTO``) is set. Register writes are accepted regardless -- ``SET
    POS`` succeeds and ``GET PRE`` reads the new target back -- so a gripper with ``GTO`` clear looks
    completely healthy right up until the fingers silently don't move.

    ``Robotiq2F85.__init__`` only sets ``GTO`` inside ``_activate_gripper``, and only calls that when
    the gripper is *not* already activated. A gripper left activated but with ``GTO`` cleared -- which
    is how a stopped Polyscope program or an aborted previous run leaves it -- therefore connects
    cleanly, reports its width, accepts every command, and never moves: every ``move(...).wait()``
    burns its full 30 s timeout and the grasp is then reported as "no object between the fingers" at
    the original opening width. That is exactly the failure this function exists to prevent.
    """
    # Every register read below is parsed defensively: an unreadable or unexpected reply from an older
    # URCap must not become a hard failure on a gripper that would have worked fine.
    def read_register(name: str) -> Optional[int]:
        try:
            value = gripper._communicate(f"GET {name}").split(" ")[-1]
            return int(value)
        except Exception as exception:  # noqa: BLE001 - a diagnostic read is never worth aborting on
            logger.debug(f"Could not read the gripper's {name} register: {exception}")
            return None

    fault = read_register("FLT")
    if fault:
        raise RuntimeError(
            f"The Robotiq reports fault status FLT {fault}. It will accept commands but not move. Clear the "
            "fault from the Robotiq URCap toolbar on the teach pendant (re-activate the gripper), then re-run. "
            "Common causes: an activation that never completed, or the fingers being blocked at power-up."
        )

    if not gripper.gripper_is_active():
        logger.warning("The gripper is not activated; activating it now (the fingers will open and close once).")
        gripper._activate_gripper()

    # Set GTO unconditionally: it is the bit that turns an accepted target into motion, and the driver
    # only sets it on the activation path we may well have just skipped.
    gripper._communicate("SET GTO 1")
    if read_register("GTO") == 0:
        raise RuntimeError(
            "Could not set the gripper's GTO (go-to) bit, so it would accept move commands without moving. "
            "Check that the Robotiq URCap is running and the robot is in remote control."
        )


@contextlib.contextmanager
def connect_gripper(robot_ip: str) -> Iterator[ParallelPositionGripper]:
    """Yield a connected, activated Robotiq 2F-85 that is armed to move.

    The gripper is reached through the UR controller's URCap socket on the *robot's* IP, so it needs
    no address of its own. A failure here is almost always the URCap not running or the robot not
    being in remote control, which the raised message says outright instead of leaving a bare socket
    error. :func:`_arm_gripper_for_motion` then checks the things that make a *connected* gripper
    still refuse to move.

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
    _arm_gripper_for_motion(gripper)
    logger.info(f"Gripper connected and armed; currently {gripper.get_current_width() * 1000:.0f} mm open.")
    yield gripper


def move_gripper(
    gripper: ParallelPositionGripper, width: float, description: str, timeout: float = GRIPPER_MOVE_TIMEOUT
) -> None:
    """Command the gripper to ``width`` and confirm the fingers actually went somewhere.

    ``ParallelPositionGripper.move(...).wait()`` returns a status that nothing here used to look at,
    and it warns rather than raises on timeout -- so a gripper that never moved produced a 30 s pause,
    a ``UserWarning`` buried in the log, and then a grasp "failure" blamed on the brick. Comparing the
    width before and after separates the two cases outright: fingers that did not move at all are a
    gripper problem, fingers that moved and stopped early are (usually) an object.
    """
    width_before_mm = gripper.get_current_width() * 1000.0
    logger.info(f"{description}: {width_before_mm:.1f} mm -> {width * 1000:.0f} mm.")

    status = gripper.move(width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE).wait(timeout=timeout)

    width_after_mm = gripper.get_current_width() * 1000.0
    if status is ACTION_STATUS_ENUM.TIMEOUT and abs(width_after_mm - width_before_mm) < GRIPPER_STALL_TOLERANCE_MM:
        raise RuntimeError(
            f"The gripper did not move at all in {timeout:.0f} s: told to go to {width * 1000:.0f} mm, still at "
            f"{width_after_mm:.1f} mm. It is accepting commands but not executing them -- check the Robotiq URCap "
            "on the teach pendant (is the gripper activated and fault-free?), that the robot is in remote control, "
            "and that the fingers are not physically blocked."
        )
    logger.info(f"{description}: fingers now at {width_after_mm:.1f} mm.")


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
        gripper.move(GRIPPER_MAX_STROKE, speed=GRIPPER_SPEED).wait(timeout=GRIPPER_MOVE_TIMEOUT)
    except Exception as exception:  # noqa: BLE001 - recovery must not raise
        logger.warning(f"Could not open the gripper while recovering: {exception}")
    try:
        arm.move_linear_to_tcp_pose(pregrasp_pose, linear_speed=linear_speed).wait()
    except (RobotConfigurationException, RuntimeError) as exception:
        logger.warning(f"Could not retreat to the pregrasp after the failed grasp: {exception}")


def descend(
    arm: PositionManipulator, grasp_pose: HomogeneousMatrixType, linear_speed: float, contact_guard: bool
) -> None:
    """Move down to ``grasp_pose``, optionally stopping early if the tool touches something.

    The descent is short and the fingers are open around the brick, so nothing *should* be touched on
    the way down -- which makes contact a reliable signal that something is wrong: the table is higher
    than the calibration says, or the brick is sitting on another brick. With ``--contact-guard`` the
    UR's own contact detection is armed downward for the move, and the arm stops itself instead of
    leaning on whatever it found.

    Off by default. The measured table plane is what stops the arm reaching the tabletop at all, and
    this is a second line of defence rather than a substitute for it; force detection on a UR3e at
    these speeds can also fire on nothing, and a false trigger costs a grasp.
    """
    if not contact_guard:
        arm.move_linear_to_tcp_pose(grasp_pose, linear_speed=linear_speed).wait()
        return

    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None or not hasattr(rtde_control, "startContactDetection"):
        logger.warning("--contact-guard was asked for but this driver has no contact detection; descending without it.")
        arm.move_linear_to_tcp_pose(grasp_pose, linear_speed=linear_speed).wait()
        return

    target_z = float(grasp_pose[2, 3])
    rtde_control.startContactDetection([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    try:
        action = arm.move_linear_to_tcp_pose(grasp_pose, linear_speed=linear_speed)
        while not action.is_action_done():
            if rtde_control.readContactDetection():
                with contextlib.suppress(Exception):
                    rtde_control.stopL(2.0)
                stopped_z = float(arm.get_tcp_pose()[2, 3])
                raise RuntimeError(
                    f"The tool touched something at z={stopped_z:+.4f} m while descending to z={target_z:+.4f} m, "
                    f"{(stopped_z - target_z) * 1000:.1f} mm early. Nothing should be in the way over an open "
                    "gripper, so either the table is higher here than the touch-off says (re-run "
                    "`python src/calibrate_table.py`) or this brick is resting on another one."
                )
            time.sleep(0.02)
        action.wait()
    finally:
        with contextlib.suppress(Exception):
            rtde_control.stopContactDetection()


def grasp_and_lift(
    arm: PositionManipulator,
    gripper: ParallelPositionGripper,
    pregrasp_pose: HomogeneousMatrixType,
    grasp_pose: HomogeneousMatrixType,
    lift_pose: HomogeneousMatrixType,
    brick_width_mm: float,
    linear_speed: float,
    contact_guard: bool = False,
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

    # Opening happens before the descent, and it is also the first time the gripper is asked to do
    # anything: if it is going to refuse to move, this is where we find out, with the fingers still
    # clear of the brick.
    move_gripper(gripper, open_width, "Opening the gripper")

    descent_mm = (pregrasp_pose[2, 3] - grasp_pose[2, 3]) * 1000.0
    logger.info(f"Descending {descent_mm:.1f} mm onto the brick.")
    descend(arm, grasp_pose, linear_speed, contact_guard)

    # A close that stalls on the brick is a *success*, not a timeout: the Robotiq's own
    # object-detection flag ends the wait, and check_grasp below decides what it closed on.
    move_gripper(gripper, close_width, "Closing the gripper")

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
    type=click.FloatRange(0.0, 80.0),
    default=None,
    help="How far below the brick's top face to descend before closing. Defaults to "
    f"{GRASP_DEPTH_MM:.1f} mm, and is capped at runtime to leave the fingertips {MIN_FINGERTIP_CLEARANCE_MM:.1f} mm "
    "above the table for whatever part this actually is -- which is why the cap is not a fixed number here.",
)
@click.option(
    "--brick-width-mm",
    type=click.FloatRange(1.0, 80.0),
    default=None,
    help="Width the fingers close on, used to size the grip and to verify it. By default it is taken "
    "from submodule_1's handoff, which measured the brick; pass this to override it (e.g. the long "
    "side, to grip the same brick end-on).",
)
@click.option(
    "--brick-height-mm",
    type=click.FloatRange(0.5, 100.0),
    default=None,
    help="Height of the brick's top face above the table. By default from the handoff; the descent is "
    "measured down from the pregrasp, so this only sets how deep it is safe to go.",
)
@click.option(
    "--handoff-path",
    default=BRICK_HANDOFF_PATH,
    show_default=True,
    help="Where submodule_1 left what it measured about this brick. Ignored if it is stale, or if it "
    "describes a brick somewhere other than where the arm is standing.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report the inherited pose and the planned grasp, then stop without moving or gripping. The "
    "way to check the descent and the finger direction before letting the arm touch the pile.",
)
@click.option(
    "--contact-guard",
    is_flag=True,
    help="Arm the UR's contact detection during the descent, so the arm stops itself if the tool "
    "touches anything before reaching the grasp. A second line of defence behind the touched-off "
    "table plane; off by default because a false trigger costs a grasp.",
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
    grasp_depth_mm: Optional[float],
    brick_width_mm: Optional[float],
    brick_height_mm: Optional[float],
    handoff_path: str,
    contact_guard: bool,
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

        # What is being grasped, from submodule_1's measurement. read_handoff refuses a file that is
        # stale or that describes a brick somewhere other than where the arm is standing, so a
        # fallback here means "nothing trustworthy was measured", never "the wrong brick was used".
        handoff = read_handoff(handoff_path, BRICK_HANDOFF_MAX_AGE, expected_position=pregrasp.position[:2])
        brick_width_mm, brick_height_mm, handoff_closing_deg = resolve_brick(
            handoff, brick_width_mm, brick_height_mm
        )
        table_floor_z = resolve_runtime_table_floor_z(pregrasp, handoff, brick_height_mm)
        grasp_depth_mm = resolve_grasp_depth(
            grasp_depth_mm, brick_height_mm, brick_top_z=pregrasp.brick_top_z, table_floor_z=table_floor_z
        )

        requested_angle = requested_closing_angle(
            pregrasp, yaw_deg, yaw_offset_deg, measured_closing_deg=handoff_closing_deg
        )
        grasp_position = np.array(
            [pregrasp.position[0], pregrasp.position[1], pregrasp.brick_top_z - grasp_depth_mm / 1000.0]
        )
        if grasp_position[2] < table_floor_z + MIN_FINGERTIP_CLEARANCE_MM / 1000.0 - 1e-6:
            raise click.ClickException(
                f"The planned grasp at z={grasp_position[2]:.4f} m would cross the runtime table floor at "
                f"z={table_floor_z:.4f} m. Re-run submodule_1; the parked pregrasp is too low."
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
        ensure_pose_above_table_floor("The re-aimed pregrasp", aimed_pose, table_floor_z)
        ensure_pose_above_table_floor("The grasp pose", grasp_pose, table_floor_z)

        lift = reachable_lift_pose(arm, grasp_pose, lift_height)
        if lift is None:
            raise click.ClickException(
                f"The grasp is reachable but not even a {MIN_LIFT_HEIGHT * 100:.0f} cm lift above it is, so the "
                "brick could be grasped and never raised. Re-run submodule_1 with the brick closer to the base."
            )
        lift_pose, actual_lift = lift
        ensure_pose_above_table_floor("The lift pose", lift_pose, table_floor_z)
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
            f"{1000.0 * (grasp_position[2] - table_floor_z):.1f} mm above the runtime table floor), close on "
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
                    arm, gripper, aimed_pose, grasp_pose, lift_pose, brick_width_mm, linear_speed, contact_guard
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
            except RuntimeError as exception:
                # move_gripper's "the gripper never moved" -- already a complete explanation, so print
                # it as the error rather than a traceback. The arm is stopped either way.
                stop_arm(arm)
                raise click.ClickException(str(exception)) from exception
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
