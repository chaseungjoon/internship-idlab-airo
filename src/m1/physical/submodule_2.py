"""m1 submodule 2 (physical): from the pregrasp, close on the brick and lift it off the table.

The counterpart of :mod:`m1.simulation.submodule_2`, doing the same six things in the same order and
checking the same things between them: open, descend, close, verify, lift, verify again. It takes the
:class:`~m1.physical.submodule_1.GraspTarget` submodule_1 produced, rather than reading a pose off the
arm and a JSON file off the disk -- the two halves run in one process now. The command line at the
bottom rebuilds a target from the handoff file so the old two-terminal workflow still works.

Two of those steps carry the whole module.

**How far to descend.** The fingertips have to get low enough down the brick's side wall to have
something to pinch, and must not touch the table. On a 9.6 mm brick that is a wide target; on a 3.2 mm
plate the entire budget is three millimetres, and a descent sized for the brick would drive the pads
into the tabletop -- which on a real robot jams the fingers and trips a protective stop. So the descent
is capped by the part being grasped and by the table underneath it, and the tighter cap wins. Where
the simulation knows the table exactly, here it comes from the touched-off plane, and every millimetre
of margin the simulation does not need is margin this module has to leave.

**Whether it worked.** The gripper is commanded to a *position* slightly inside the brick's width, so
the fingers stall on the brick rather than meeting each other. Two independent signals then say
whether anything is held -- the Robotiq's own object-detection flag, read off motor current, and the
gap between the width commanded and the width reached -- because either alone lies: the flag also
fires when the fingers stall against each other, and the width alone cannot tell a held brick from one
wedged between a pad and a neighbour. Checked twice: on closing, which catches a miss, and again after
the lift and a pause, which catches the brick sliding out as the arm accelerates. The second check is
the one that means anything, and it is the whole point of the module.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from airo_robots.exceptions import RobotConfigurationException
from airo_typing import HomogeneousMatrixType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from m1.physical import cell as C  # noqa: E402
from m1.physical.submodule_1 import GraspTarget  # noqa: E402

# --- the descent ------------------------------------------------------------------------------------
#: How far below the brick's top face the fingertips are driven before closing. Roughly a third of a
#: brick puts the pads on the side walls rather than on the studs, which would slip.
GRASP_DEPTH_M = 0.003
#: Hard floor: the fingertips never come closer than this to the tabletop, whatever else is asked for.
MIN_FINGERTIP_CLEARANCE_M = 0.0015

# --- the gripper ------------------------------------------------------------------------------------
#: Commanded *below* the brick's width, so the position controller keeps pushing and the fingers stall
#: on the brick. Nothing about a position command alone makes a grip; this overshoot is the grip.
GRIPPER_SQUEEZE_M = 0.003
#: Verification band on the width reached. Under the lower bound the pads met with nothing between
#: them; over the upper bound something much thicker than the brick is in the jaws -- a neighbour
#: dragged in, or the brick caught end-on.
WIDTH_TOLERANCE_BELOW_M = 0.004
WIDTH_TOLERANCE_ABOVE_M = 0.006

# --- the lift ---------------------------------------------------------------------------------------
LIFT_HEIGHT_M = 0.12
MIN_LIFT_HEIGHT_M = 0.04
LIFT_SEARCH_STEP_M = 0.01
SETTLE_DURATION = 1.0  # after the lift, so a slipping grasp has time to show itself

#: Somewhere on the table well clear of the pile to put a brick down. Sorting bricks into categories is
#: Module 3's job; this exists so the cycle can be run more than once without the next look at the pile
#: finding the last brick still in the gripper. **Measure this on your own bench**, and keep it well
#: away from ``cell.PILE_CENTER`` -- submodule_1's keep-out is what stops the robot re-picking its own
#: output, and it is drawn around this point.
DROP_POSITION = np.array([0.16, -0.20, 0.06])


@dataclass
class GraspResult:
    """What happened. ``success`` is the answer; the rest is why."""

    success: bool
    reason: str
    grasp_depth: float
    commanded_width: float
    width_at_close: float
    width_after_lift: float
    lift_height: float
    part: Optional[str] = None

    def describe(self) -> str:
        verdict = "grasped and holding" if self.success else "FAILED"
        return (
            f"{verdict}: {self.reason}. Descended {self.grasp_depth * 1000:.1f} mm, commanded "
            f"{self.commanded_width * 1000:.1f} mm, pads stopped at {self.width_at_close * 1000:.1f} mm "
            f"then {self.width_after_lift * 1000:.1f} mm after a {self.lift_height * 100:.0f} cm lift"
        )


# =================================================================================================
# planning the descent
# =================================================================================================


def resolve_grasp_depth(target: GraspTarget, requested: Optional[float] = None) -> float:
    """How far below the top face to descend, capped for the part actually being grasped.

    Two caps, and the tighter one wins:

    * the part's own geometry, which says how far the pads can go down its side before there is no
      more side;
    * the table under it, which says how low the fingertips may go at all.

    On a 9.6 mm brick these agree and neither bites. On a 3.2 mm plate the second one is the only thing
    between a grasp and a fingertip in the tabletop.
    """
    geometric_ceiling = target.height - MIN_FINGERTIP_CLEARANCE_M
    table_ceiling = (target.top_face_z - target.table_z) - MIN_FINGERTIP_CLEARANCE_M
    ceiling = min(geometric_ceiling, table_ceiling)
    if ceiling <= 0:
        raise RuntimeError(
            f"A {target.height * 1000:.1f} mm part leaves no room to descend while keeping "
            f"{MIN_FINGERTIP_CLEARANCE_M * 1000:.1f} mm of fingertip clearance above the table. This part is "
            "too thin for a top-down pinch; submodule_1 should not have offered it."
        )

    depth = GRASP_DEPTH_M if requested is None else requested
    if depth > ceiling:
        logger.warning(
            f"A {depth * 1000:.1f} mm descent would put the fingertips through the table; capping it at "
            f"{ceiling * 1000:.1f} mm for this {target.height * 1000:.1f} mm part."
        )
        depth = ceiling
    logger.info(
        f"Descending {depth * 1000:.1f} mm into the part's {target.height * 1000:.1f} mm, leaving "
        f"{(target.height - depth) * 1000:.1f} mm of fingertip clearance above the table."
    )
    return depth


def reachable_lift(
    cell: C.Cell, target: GraspTarget, grasp_position: np.ndarray, requested: float, width: float
) -> Optional[Tuple[HomogeneousMatrixType, float]]:
    """The highest reachable straight-up lift from the grasp, at most ``requested`` metres.

    Lifting is the last thing that happens and the arm is already stretched over the table by then, so
    the requested height is the first thing to become unreachable. Refusing the run over that would
    leave the brick gripped and still sitting on the table, which is strictly worse than lifting it
    less far -- so the height is walked down until something is reachable.

    Heights are computed off the step index and rounded rather than by repeated subtraction, which
    drifts (0.10 - 3 x 0.01 lands at 0.06999999999999999) and can push a height that exactly fits the
    arm's limit just past it, losing a whole centimetre.
    """
    steps = int(math.floor((requested - MIN_LIFT_HEIGHT_M) / LIFT_SEARCH_STEP_M + 1e-9))
    heights = [round(requested - step * LIFT_SEARCH_STEP_M, 6) for step in range(max(steps, 0) + 1)]
    if not heights or heights[-1] > MIN_LIFT_HEIGHT_M + 1e-9:
        heights.append(MIN_LIFT_HEIGHT_M)

    for height in heights:
        position = grasp_position + np.array([0.0, 0.0, height])
        pose = C.top_down_tool_pose(cell, position, target.closing_heading, width)
        if C.pose_is_reachable(cell, pose):
            return pose, height
    return None


def ensure_above_table_floor(name: str, pose: HomogeneousMatrixType, table_z: float) -> None:
    """Reject any commanded pose that would cross the runtime table floor."""
    floor = table_z + MIN_FINGERTIP_CLEARANCE_M
    if pose[2, 3] < floor - 1e-6:
        raise RuntimeError(
            f"{name} would command the fingertips to z={pose[2, 3]:.4f} m, below the floor of z={floor:.4f} m "
            f"(table {table_z:.4f} m + {MIN_FINGERTIP_CLEARANCE_M * 1000:.1f} mm clearance). Refusing to descend."
        )


# =================================================================================================
# the pick
# =================================================================================================


def check_grasp(cell: C.Cell, target: GraspTarget) -> Tuple[bool, str]:
    """Whether the jaws are holding a brick of the right width right now, and how we know.

    Both signals, because either alone lies -- see the module docstring. The commanded width is
    deliberately inside the brick, so a gripper that reached its command is one that closed on nothing.
    """
    width = cell.finger_width()
    commanded = cell.commanded_gripper_width
    if not cell.is_an_object_grasped():
        return False, f"the gripper reports no object between the fingers (width {width * 1000:.1f} mm)"
    if width <= commanded + 0.0005:
        return False, f"the pads reached {width * 1000:.1f} mm, the width they were commanded to -- nothing between them"
    if width < target.width - WIDTH_TOLERANCE_BELOW_M:
        return False, f"the pads closed to {width * 1000:.1f} mm, under the brick's {target.width * 1000:.1f} mm"
    if width > target.width + WIDTH_TOLERANCE_ABOVE_M:
        return False, (
            f"the pads stopped at {width * 1000:.1f} mm, too wide for a {target.width * 1000:.1f} mm brick -- "
            "something else is in the grip"
        )
    return True, f"holding a {width * 1000:.1f} mm object (the brick measured {target.width * 1000:.1f} mm)"


def descend(cell: C.Cell, grasp_pose: HomogeneousMatrixType, contact_guard: bool) -> None:
    """Move down to the grasp, optionally stopping early if the tool touches something.

    The descent is short and the fingers are open around the brick, so nothing *should* be touched on
    the way down -- which makes contact a reliable signal that something is wrong: the table is higher
    than the calibration says, or the brick is sitting on another brick. With ``contact_guard`` the
    UR's own contact detection is armed downward for the move and the arm stops itself instead of
    leaning on whatever it found.

    Off by default. The measured table plane is what stops the arm reaching the tabletop at all, and
    this is a second line of defence rather than a substitute; force detection on a UR3e at these
    speeds can also fire on nothing, and a false trigger costs a grasp.
    """
    if not contact_guard:
        cell.move_tcp_to(grasp_pose, linear=True)
        return

    rtde_control = getattr(cell.arm, "rtde_control", None)
    if rtde_control is None or not hasattr(rtde_control, "startContactDetection"):
        logger.warning("contact_guard was asked for but this driver has no contact detection; descending without it.")
        cell.move_tcp_to(grasp_pose, linear=True)
        return

    target_z = float(grasp_pose[2, 3])
    rtde_control.startContactDetection([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    try:
        action = cell.arm.move_linear_to_tcp_pose(grasp_pose, linear_speed=cell.linear_speed)
        while not action.is_action_done():
            if rtde_control.readContactDetection():
                with contextlib.suppress(Exception):
                    rtde_control.stopL(2.0)
                stopped_z = float(cell.arm.get_tcp_pose()[2, 3])
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


def run(
    cell: C.Cell,
    target: GraspTarget,
    grasp_depth: Optional[float] = None,
    lift_height: float = LIFT_HEIGHT_M,
    contact_guard: bool = False,
) -> GraspResult:
    """Descend onto the brick under the pregrasp, close, lift, and say whether it is held.

    Every pose is solved and floor-checked before anything moves, and the lift is planned *before* the
    brick is touched: a brick that can be grasped but not raised is not worth grasping, and finding
    that out afterwards means finding it out with the brick in the jaws.
    """
    if cell.gripper is None:
        raise RuntimeError("This cell has no gripper connected; build it with `with_gripper=True`.")

    depth = resolve_grasp_depth(target, grasp_depth)
    width = target.approach_width

    grasp_position = np.array([target.position[0], target.position[1], target.top_face_z - depth])
    # Floor-checked from the position rather than a pose, because every equivalent yaw the IK might
    # come back with sits at the same height and the check is only ever about height.
    ensure_above_table_floor(
        "The grasp", C.top_down_tool_pose(cell, grasp_position, target.closing_heading, width), target.table_z
    )
    solved = C.solve_top_down_ik(cell, grasp_position, target.closing_heading, width)
    if solved is None:
        raise RuntimeError(
            f"No reachable straight-down pose at {np.round(grasp_position, 4)} m with the jaws along "
            f"{math.degrees(target.closing_heading):.0f} deg, even though the pregrasp above it was reachable. "
            "This is the wrist running out of travel; a half-turn of the closing direction would fix it."
        )
    grasp_pose, _, _ = solved

    lift = reachable_lift(cell, target, grasp_position, lift_height, width)
    if lift is None:
        raise RuntimeError(
            f"The grasp is reachable but not even a {MIN_LIFT_HEIGHT_M * 100:.0f} cm lift above it is, so the "
            "brick could be picked up and never raised. Pick a brick closer to the base."
        )
    lift_pose, actual_lift = lift
    ensure_above_table_floor("The lift pose", lift_pose, target.table_z)
    if actual_lift < lift_height - 1e-9:
        logger.warning(
            f"A {lift_height * 100:.0f} cm lift is out of reach here; lifting {actual_lift * 100:.0f} cm instead."
        )

    logger.info(f"Descending {depth * 1000:.1f} mm onto the brick, jaws open at {width * 1000:.1f} mm ...")
    try:
        descend(cell, grasp_pose, contact_guard)
    except (RuntimeError, RobotConfigurationException) as exception:
        C.stop_arm(cell)
        raise RuntimeError(f"The descent stopped: {exception}") from exception

    close_width = max(target.width - GRIPPER_SQUEEZE_M, cell.gripper_calibration.min_width)
    logger.info(f"Closing: commanding {close_width * 1000:.1f} mm on a {target.width * 1000:.1f} mm brick ...")
    cell.move_gripper_to_width(close_width)

    holding, reason = check_grasp(cell, target)
    width_at_close = cell.finger_width()
    if not holding:
        logger.error(f"The grasp did not take at the table: {reason}")
        _release_and_retreat(cell, target)
        return GraspResult(
            success=False,
            reason=f"the grasp did not take at the table: {reason}",
            grasp_depth=depth,
            commanded_width=close_width,
            width_at_close=width_at_close,
            width_after_lift=width_at_close,
            lift_height=0.0,
        )

    logger.info(f"Closed on the brick: {reason}. Lifting {actual_lift * 100:.0f} cm ...")
    cell.move_tcp_to(lift_pose, linear=True)
    cell.advance(SETTLE_DURATION)

    holding, reason = check_grasp(cell, target)
    width_after_lift = cell.finger_width()
    result = GraspResult(
        success=holding,
        reason=reason if holding else f"the brick was lost during the lift: {reason}",
        grasp_depth=depth,
        commanded_width=close_width,
        width_at_close=width_at_close,
        width_after_lift=width_after_lift,
        lift_height=actual_lift,
    )
    (logger.success if result.success else logger.error)(result.describe())
    if not result.success:
        _release_and_retreat(cell, target)
    return result


def _release_and_retreat(cell: C.Cell, target: GraspTarget) -> None:
    """Recover from a failed grasp: let go first, then go back up.

    Order matters. Opening first drops whatever was half-caught straight back where it came from, from
    millimetres up, instead of carrying it somewhere else and dropping it there.
    """
    logger.info("Opening the jaws and retreating to the pregrasp.")
    with contextlib.suppress(Exception):
        cell.move_gripper_to_width(cell.gripper_calibration.max_width)
    if target.pregrasp_configuration is not None:
        with contextlib.suppress(RobotConfigurationException, RuntimeError):
            cell.move_arm_to(target.pregrasp_configuration)


def park(cell: C.Cell) -> None:
    """Take the arm back to its home configuration, holding whatever it is holding."""
    logger.info("Returning to the home configuration.")
    cell.move_arm_to(C.HOME_CONFIGURATION)


def place(cell: C.Cell, target: GraspTarget, position: np.ndarray = DROP_POSITION) -> bool:
    """Carry the brick clear of the pile and let it go. Returns whether the arm got there."""
    solved = C.solve_top_down_ik(cell, np.asarray(position, float), target.closing_heading, cell.finger_width())
    if solved is None:
        logger.warning(f"Nowhere to put the brick down: {np.round(position, 3)} m is out of reach. Keeping hold of it.")
        return False
    pose, q, _ = solved
    logger.info(f"Carrying the brick to {np.round(position, 3)} m and releasing it.")
    cell.move_arm_to(q)
    cell.move_gripper_to_width(cell.gripper_calibration.max_width)
    cell.advance(0.4)
    # Stand back up before handing control back. The next thing to happen is a move to a viewpoint at
    # the far side of the table, and starting that from a pose centimetres above the tabletop is asking
    # the arm to drag its wrist across everything in between.
    park(cell)
    return True


# =================================================================================================
# CLI
# =================================================================================================


def main() -> None:
    """Grasp the brick submodule_1 left the arm standing over. The standalone half of the pipeline.

    Rebuilds a :class:`GraspTarget` from the handoff file submodule_1 wrote and from the pose the arm
    is currently parked at, then runs the same :func:`run` the notebook does. The handoff is refused if
    it is stale or describes a brick somewhere other than where the arm is standing, which is exactly
    the check that catches the arm having been moved between the two commands.
    """
    import click

    from config import BRICK_HANDOFF_MAX_AGE, BRICK_HANDOFF_PATH, DEFAULT_CALIBRATION_DIR, DEFAULT_CAMERA_RESOLUTION, SUPPORTED_ROBOT_TYPES

    @click.command()
    @click.option("--robot-type", type=click.Choice(SUPPORTED_ROBOT_TYPES), default="ur3e", show_default=True)
    @click.option("--ip-address", default=None, help="Robot controller IP. Defaults per robot type.")
    @click.option("--calibration-path", default=DEFAULT_CALIBRATION_DIR, show_default=True)
    @click.option("--camera-resolution", type=click.Choice(list(C.CAMERA_RESOLUTIONS)), default=DEFAULT_CAMERA_RESOLUTION, show_default=True)
    @click.option("--linear-speed", type=click.FloatRange(0.005, 0.25), default=C.DEFAULT_LINEAR_SPEED, show_default=True)
    @click.option("--lift-height", type=click.FloatRange(MIN_LIFT_HEIGHT_M, 0.30), default=LIFT_HEIGHT_M, show_default=True)
    @click.option("--grasp-depth-mm", type=click.FloatRange(0.0, 80.0), default=None, help="Overrides the default descent, still capped by the table.")
    @click.option("--handoff-path", default=BRICK_HANDOFF_PATH, show_default=True)
    @click.option("--contact-guard", is_flag=True, help="Arm the UR's contact detection during the descent.")
    @click.option("--place/--no-place", "place_it", default=False, show_default=True, help="Carry the brick to the drop corner afterwards.")
    @click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt before descending.")
    def command(
        robot_type: str,
        ip_address: Optional[str],
        calibration_path: str,
        camera_resolution: str,
        linear_speed: float,
        lift_height: float,
        grasp_depth_mm: Optional[float],
        handoff_path: str,
        contact_guard: bool,
        place_it: bool,
        yes: bool,
    ) -> None:
        if robot_type != "ur3e":
            raise click.ClickException(
                f"--robot-type {robot_type} has no parallel gripper wired up; only the ur3e carries the "
                "Robotiq 2F-85 the grasp is executed with."
            )
        with C.build_cell(
            robot_type=robot_type,
            ip_address=ip_address,
            calibration_path=calibration_path,
            camera_resolution=camera_resolution,
            linear_speed=linear_speed,
            with_gripper=True,
        ) as cell:
            pose = cell.tcp_pose()
            handoff = read_handoff(handoff_path, BRICK_HANDOFF_MAX_AGE, expected_position=pose[:3, 3][:2])
            if handoff is None:
                raise click.ClickException(
                    "No usable handoff from submodule_1, so there is nothing trustworthy to grasp. Re-run "
                    "`python src/m1/physical/submodule_1.py` without moving the arm afterwards."
                )
            target = _target_from_handoff(handoff, pose, cell)
            logger.info(f"Inherited pregrasp over: {target.describe()}")

            if not yes and not click.confirm(
                f"Descend onto the brick at {np.round(target.position[:2], 3)} m and grasp it?", default=True
            ):
                logger.info("Aborted by the user.")
                return

            try:
                result = run(
                    cell,
                    target,
                    grasp_depth=None if grasp_depth_mm is None else grasp_depth_mm / 1000.0,
                    lift_height=lift_height,
                    contact_guard=contact_guard,
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted; stopping the arm.")
                C.stop_arm(cell)
                raise
            except Exception:
                C.stop_arm(cell)
                raise

            if not result.success:
                raise click.ClickException(f"Grasp failed: {result.reason}")
            logger.success(result.describe())
            if place_it:
                place(cell, target)

    command()


#: How far the handoff's brick may be from where the arm is standing and still be believed. Wider than
#: this and the arm has been moved, freedriven or re-run since submodule_1 wrote the file, which is
#: exactly when trusting its dimensions would be wrong.
HANDOFF_POSITION_TOLERANCE_M = 0.01


def read_handoff(path: str, max_age: float, expected_position: Optional[np.ndarray] = None) -> Optional[dict]:
    """Load submodule_1's handoff, or ``None`` with a reason logged if it should not be trusted.

    Three ways it is refused, all meaning "this describes a different brick than the one under the
    gripper": the file is missing, it is older than ``max_age``, or the brick it recorded is not where
    the arm is now standing. The last one is the useful one.
    """
    import json

    if not os.path.exists(path):
        logger.warning(f"No brick handoff at {path}.")
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exception:
        logger.warning(f"Could not read the brick handoff at {path}: {exception}")
        return None

    age = time.time() - float(payload.get("written_at", 0.0))
    if age > max_age:
        logger.warning(
            f"The brick handoff at {path} is {age / 60:.0f} min old (limit {max_age / 60:.0f} min), so it "
            "describes an earlier brick. Re-run submodule_1."
        )
        return None

    if expected_position is not None and payload.get("brick_position"):
        recorded = np.asarray(payload["brick_position"][:2], dtype=float)
        distance = float(np.linalg.norm(np.asarray(expected_position, float)[:2] - recorded))
        if distance > HANDOFF_POSITION_TOLERANCE_M:
            logger.warning(
                f"The handoff describes a brick at {recorded.round(3)} m but the arm is standing over "
                f"{np.asarray(expected_position)[:2].round(3)} m, {distance * 100:.1f} cm away. The arm has "
                "moved since submodule_1 ran."
            )
            return None
    return payload


def _target_from_handoff(handoff: dict, pregrasp_pose: HomogeneousMatrixType, cell: C.Cell) -> GraspTarget:
    """Rebuild the target submodule_1 measured, from its handoff plus the pose the arm is parked at.

    The x and y come from the *pregrasp pose*, not the file: whatever the file says, the descent has to
    happen straight down from where the arm actually is, or the fingers move sideways on the way in.
    ``read_handoff`` has already refused the file if those two disagree by more than a few millimetres.
    """
    x, y = float(pregrasp_pose[0, 3]), float(pregrasp_pose[1, 3])
    height = float(handoff.get("height") or 0.0096)
    table_z = float(handoff.get("safe_table_z") or cell.table_z_at(x, y))
    long_axis = float(handoff.get("long_axis_heading") or 0.0)
    width = float(handoff.get("width") or 0.0078)
    return GraspTarget(
        position=np.array([x, y, table_z + height]),
        width=width,
        length=float(handoff.get("length") or 0.0238),
        height=height,
        long_axis_heading=long_axis,
        table_z=table_z,
        colour=str(handoff.get("colour") or "unknown"),
        score=0.0,
        confidence=0.0,
        triangulated=np.array([x, y, table_z + height]),
        plane_projected=np.array([x, y, table_z + height]),
        triangulation_gap=float("nan"),
        method_disagreement=0.0,
        view_disagreement=float(handoff.get("view_disagreement") or 0.0),
        position_source="handoff",
        pregrasp_pose=pregrasp_pose,
        pregrasp_configuration=cell.arm_positions(),
        approach_width=min(width + 0.014, cell.gripper_calibration.max_width),
    )


if __name__ == "__main__":
    main()
