"""m1 submodule 2 (physical): from the pregrasp, close on the brick and lift it off the table.

Counterpart of :mod:`m1.simulation.submodule_2`: open, descend, close, verify, lift, verify again.
It takes submodule_1's :class:`~m1.physical.submodule_1.GraspTarget` directly; the CLI at the bottom
rebuilds one from the handoff file so the old two-terminal workflow still works.

**Where the position comes from.** submodule_1 picks the brick and stands over it from viewpoints
across the table, where its lateral error is at its worst. Rather than inherit that, this module
re-measures the brick from the pregrasp before planning anything
(:func:`relocate_from_pregrasp`), through the *same* colour pipeline the survey uses -- one perception,
called from both halves. The tool is vertical at the pregrasp, which zeroes both the
``height x tan(tilt)`` lever arm and the lateral part of a hand-eye translation error, so this look is
the precise one and the survey's only has to be good enough to find the brick again. The grasp height
still comes from the touched-off plane: measured by touching, not seen. Pass ``relook=False`` to
descend on the survey's position instead.

Two steps carry the module. **How far to descend** is capped both by the part's own height and by the
table under it, tighter cap winning -- on a 3.2 mm plate a descent sized for the brick would drive the
pads into the tabletop. **Whether it worked** is checked from two signals, the Robotiq's
object-detection flag and the gap between commanded and reached width, twice: on closing, and again
after the lift and a pause. The second check is the one that means anything.
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
from m1.physical import submodule_1  # noqa: E402
from m1.physical.submodule_1 import GRIPPER_APPROACH_MARGIN_M, GraspTarget  # noqa: E402

# --- the descent ----------------------------------------------------------------------------------
#: How far below the top face the fingertips go before closing -- about a third of a brick, so the pads
#: land on the side walls rather than the studs.
GRASP_DEPTH_M = 0.003
#: Hard floor on fingertip-to-tabletop distance, whatever else is asked for.
MIN_FINGERTIP_CLEARANCE_M = 0.0015

# --- the gripper ----------------------------------------------------------------------------------
#: The jaws are commanded shut, not to a width derived from the brick.
#:
#: Sizing the close from the measured width made the grasp only as good as that measurement, and it is
#: the weakest number in the pipeline: a colour footprint that reads a couple of millimetres wide leaves
#: the fingers reaching their commanded opening without ever touching the brick, which reports as "no
#: object" on a grasp that was never attempted. Commanding the jaws fully shut removes the measurement
#: from the loop entirely -- whatever stops the fingers is the brick, and the width they stop at is the
#: measurement rather than the test. The force limit, not the position target, is what protects the part.
CLOSE_FULLY = True
#: Finger travel above the fully-closed opening that counts as something being held. Above the ~0.4 mm
#: register quantisation and well under the thinnest part, so a plate on edge still registers.
HELD_MIN_OPENING_M = 0.0015

# --- the lift -------------------------------------------------------------------------------------
LIFT_HEIGHT_M = 0.12
MIN_LIFT_HEIGHT_M = 0.04
LIFT_SEARCH_STEP_M = 0.01
SETTLE_DURATION = 1.0  # after the lift, so a slipping grasp has time to show itself

#: Where to put a brick down, clear of the pile, so the cycle can run more than once. Measure on your
#: own bench and keep it well away from ``cell.PILE_CENTER`` -- submodule_1's keep-out is drawn around
#: this point and is what stops the robot re-picking its own output.
DROP_POSITION = np.array([0.16, -0.20, 0.06])


# --- the re-look from the pregrasp -----------------------------------------------------------------
#: Re-measure the brick from the pregrasp before descending, instead of trusting the survey's position.
#: The survey sees the pile from across the table, 20-30 deg off vertical, where a height error slides
#: the answer sideways by ``height error x tan(tilt)``. From the pregrasp the tool points straight down,
#: which zeroes that lever arm -- and also zeroes the lateral part of a hand-eye translation error,
#: since that error is fixed in the tool frame and the tool axis is now vertical. What is left of it is
#: pure height, and height is the one thing here that does not come from the camera at all.
RELOOK_FROM_PREGRASP = True
#: How far from the survey's position a region may be and still be taken for the same brick. It has to
#: be wider than the correction it exists to make; the size check below is what stops it locking onto a
#: neighbour instead.
RELOOK_MATCH_TOLERANCE_M = 0.025
#: ...and it has to be about the same size, which is what catches a neighbour inside that radius.
RELOOK_WIDTH_TOLERANCE_M = 0.006
#: Above this correction the pregrasp is re-flown over the new position, so the descent stays a pure
#: vertical drop rather than sliding sideways through whatever is beside the brick.
RELOOK_REPOSITION_M = 0.002
#: Above this the correction is wider than the jaws' slack: the survey would have missed outright.
RELOOK_WARN_M = 0.010
#: Board coverage below which the frame has too little bare table for the colour model to fit to.
RELOOK_MIN_BOARD_FRACTION = 0.20
#: ...and the camera height above the brick's top face to rise to and retry from. The camera sits about
#: 18 cm behind the fingertips, so a 3 cm pregrasp puts it 21 cm up -- at a D435's minimum range, where
#: patches of the frame come back empty. 26 cm is inside the band submodule_1's nadir look uses.
RELOOK_RETRY_CAMERA_HEIGHT_M = 0.26

def _look_at_brick(cell: C.Cell, target: GraspTarget) -> Optional[submodule_1.ViewResult]:
    """A colour look at the brick from above, rising off the pregrasp only if it is too close to see it.

    Depth coverage used to be the test for whether the frame was usable, which no longer applies: the
    colour pipeline needs no depth at all. What it does need is the *board* in frame, because the table
    model is fitted to it -- and at a 3 cm pregrasp the camera is only ~21 cm up, where the frame may be
    filled by the pile with no bare wood to fit to.
    """
    result = submodule_1.analyse_view(cell, cell.capture("pregrasp re-look"), "pregrasp")
    board_fraction = float(result.analysis["board"].area_px) / max(result.analysis["fg"].size, 1)
    if board_fraction >= RELOOK_MIN_BOARD_FRACTION:
        return result

    logger.info(
        f"Only {board_fraction * 100:.0f}% of the pregrasp frame is board (want "
        f"{RELOOK_MIN_BOARD_FRACTION * 100:.0f}%); the camera is too close to see the table it fits its "
        "colour model to. Rising to look again."
    )
    eye_z = target.top_face_z + RELOOK_RETRY_CAMERA_HEIGHT_M
    pose = C.look_at_tool_pose(cell, [target.position[0], target.position[1], eye_z], target.position)
    q = C.solve_tool_ik(cell, pose)
    if q is None:
        logger.warning("No reachable pose to rise to; keeping the close frame such as it is.")
        return result
    cell.move_arm_to(q)
    cell.advance(0.3)
    return submodule_1.analyse_view(cell, cell.capture("pregrasp re-look (risen)"), "pregrasp")


def relocate_from_pregrasp(cell: C.Cell, target: GraspTarget) -> GraspTarget:
    """Re-measure the brick from where the arm is standing and move ``target`` onto it, in place.

    The same colour pipeline the survey uses, run from straight above instead of from across the table.
    That is the whole point: the survey sees the pile 20-30 deg off vertical, where the assumed height
    slides the answer sideways by ``height error x tan(tilt)``, and a hand-eye translation error --
    fixed in the tool frame -- lands mostly sideways too. With the tool vertical both of those go to
    zero, so this look is the precise one and the survey's is only good enough to find the brick again.

    Best-effort throughout: a look that finds nothing, or finds something the wrong size, keeps the
    survey's position and says why. The grasp height stays ``cell.table_plane`` -- touched, not seen --
    plus the height measured here.
    """
    original = np.array(target.position[:2], float)
    result = _look_at_brick(cell, target)
    if result is None:
        return target

    measured = submodule_1.project_all(result, cell.table_plane)
    candidates = [
        (float(np.linalg.norm(measured[region.index][0] - original)), region)
        for region in result.bricks
        if region.index in measured
    ]
    within = [(d, r) for d, r in candidates if d <= RELOOK_MATCH_TOLERANCE_M]
    if not within:
        nearest = min(candidates, key=lambda item: item[0])[0] if candidates else float("inf")
        logger.warning(
            f"The re-look found nothing within {RELOOK_MATCH_TOLERANCE_M * 1000:.0f} mm of the pregrasp "
            f"(nearest region {nearest * 1000:.0f} mm away, {len(result.bricks)} found). Descending on the "
            "survey's position."
        )
        return target

    distance, region = min(within, key=lambda item: item[0])
    centre, width, length, heading, height, part = measured[region.index]
    if abs(width - target.width) > RELOOK_WIDTH_TOLERANCE_M:
        logger.warning(
            f"The region {distance * 1000:.0f} mm away is {width * 1000:.1f} mm wide where the survey measured "
            f"{target.width * 1000:.1f} mm -- a different part, not this one seen closer. Descending on the "
            "survey's position."
        )
        return target

    x, y = float(centre[0]), float(centre[1])
    table_z = cell.table_z_at(x, y)  # touched-off: the only trustworthy absolute height
    correction = float(np.linalg.norm(np.array([x, y]) - original))

    target.position = np.array([x, y, table_z + height])
    target.table_z = table_z
    target.width, target.length, target.height = width, length, height
    target.long_axis_heading = heading
    target.height_measured = part is not None
    target.nadir_correction = correction
    target.position_source = "colour/pregrasp_relook"
    target.per_view["pregrasp"] = np.array([x, y])
    target.approach_width = min(width + GRIPPER_APPROACH_MARGIN_M, cell.gripper_calibration.max_width)

    logger.success(
        f"Re-look from the pregrasp: the brick is at ({x:.4f}, {y:.4f}) m, {correction * 1000:.1f} mm from "
        f"where the survey put it; {width * 1000:.1f} x {length * 1000:.1f} mm, top face "
        f"z={table_z + height:+.4f} m ({height * 1000:.1f} mm over the touched-off table)."
    )
    if correction > RELOOK_WARN_M:
        logger.warning(
            f"A {correction * 1000:.0f} mm correction is wider than the jaws' slack, so the survey's position "
            "would have missed this brick. The grasp uses the corrected one."
        )
    if not target.height_measured:
        logger.warning(
            f"The footprint matched no catalog part, so the height is still the {height * 1000:.1f} mm fallback "
            "-- but from straight above that costs almost nothing sideways, which is the point of looking again."
        )
    return target


def reposition_over(cell: C.Cell, target: GraspTarget) -> None:
    """Re-fly the pregrasp over the corrected position, so the descent is a pure vertical drop.

    Without this the corrected grasp is reached by a slanted linear move from the old pregrasp, which
    walks the open fingers sideways through whatever is next to the brick on the way in.
    """
    if target.pregrasp_pose is None:
        return
    correction = float(target.nadir_correction or 0.0)
    if correction <= RELOOK_REPOSITION_M:
        return
    above = np.array([target.position[0], target.position[1], float(target.pregrasp_pose[2, 3])])
    solved = C.solve_top_down_ik(cell, above, target.closing_heading, target.approach_width)
    if solved is None:
        logger.warning(
            f"Cannot re-fly the pregrasp {correction * 1000:.1f} mm across to the corrected position; the "
            "descent will slant instead. Watch the neighbours."
        )
        return
    pose, q, _ = solved
    logger.info(f"Shifting the pregrasp {correction * 1000:.1f} mm onto the corrected position.")
    cell.move_tcp_to(pose, linear=True)
    target.pregrasp_pose = pose
    target.pregrasp_configuration = cell.arm_positions()


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
        commanded = (
            "shut" if self.commanded_width <= 1e-6 else f"to {self.commanded_width * 1000:.1f} mm"
        )
        return (
            f"{verdict}: {self.reason}. Descended {self.grasp_depth * 1000:.1f} mm, jaws commanded "
            f"{commanded}, pads stopped at {self.width_at_close * 1000:.1f} mm "
            f"then {self.width_after_lift * 1000:.1f} mm after a {self.lift_height * 100:.0f} cm lift"
        )


# --- planning the descent -------------------------------------------------------------------------


def table_referenced_depth(target: GraspTarget) -> float:
    """Descend until the fingertips are just clear of the *table*, whatever the part's height is said to be.

    The alternative, :func:`resolve_grasp_depth`, measures down from the top face -- so it inherits the
    height, which on a colour-only pipeline is a catalog guess. Overestimate the height and the fingers
    close in the air above the brick; underestimate it and they stop short of a real grip.

    Measuring *up* from the touched-off table removes that: the part is sitting on the table, so the
    lowest safe fingertip position is the same regardless of how tall the part is, and it engages the
    whole side wall instead of the top third.

    The risk this takes on is a brick resting on another brick, where the tabletop is not what is
    underneath it -- which is what ``--contact-guard`` is for, and why this is not the default.
    """
    depth = (target.top_face_z - target.table_z) - MIN_FINGERTIP_CLEARANCE_M
    logger.info(
        f"Descending {depth * 1000:.1f} mm, measured up from the touched-off table rather than down from "
        f"the part's {target.height * 1000:.1f} mm height, leaving "
        f"{MIN_FINGERTIP_CLEARANCE_M * 1000:.1f} mm of fingertip clearance."
    )
    return max(depth, 0.0)


def resolve_grasp_depth(target: GraspTarget, requested: Optional[float] = None) -> float:
    """How far below the top face to descend: the tighter of the part's height and the table under it."""
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

    The arm is stretched over the table by lift time, so the requested height is the first thing to go
    unreachable -- and lifting less far beats leaving the brick gripped on the table. Heights come from
    the step index rather than repeated subtraction, which drifts and can lose a whole centimetre.
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


# --- the pick -------------------------------------------------------------------------------------


def check_grasp(cell: C.Cell, target: GraspTarget) -> Tuple[bool, str]:
    """Whether the jaws are holding *anything*, and how we know. The brick's size does not come into it.

    Only two questions are asked, and neither needs to know how big the part is: does the Robotiq's own
    object-detection flag fire, and did the fingers stop short of shut? Either answer is enough --
    something is between the pads or it is not.

    This deliberately no longer checks the reached width against ``target.width``. That band compared a
    measurement to a measurement: the reached opening is good to a few tenths of a millimetre, and the
    expected width is a colour footprint good to a few millimetres, so the band rejected sound grasps
    far more often than it caught bad ones. A part that is not the one submodule_1 named is still a part
    in the gripper, and sorting is Module 3's job -- lifting it is strictly better than dropping it to
    protect a number we do not trust.

    The width is still reported, because it is the honest measurement of what is actually held.
    """
    width = cell.finger_width()
    floor = cell.gripper_calibration.min_width
    stopped_short = width > floor + HELD_MIN_OPENING_M
    flagged = cell.is_an_object_grasped()

    if flagged or stopped_short:
        how = "the object-detection flag" if flagged else "the pads stopping short of shut"
        expected = (
            f", where submodule_1 measured {target.width * 1000:.1f} mm" if target.width > 0 else ""
        )
        return True, f"holding a {width * 1000:.1f} mm object on {how}{expected}"
    return False, (
        f"the pads closed to {width * 1000:.1f} mm, effectively shut, and the object-detection flag is "
        "clear -- there is nothing between them"
    )


def descend(cell: C.Cell, grasp_pose: HomogeneousMatrixType, contact_guard: bool) -> None:
    """Move down to the grasp, optionally stopping early on contact.

    Nothing should be touched on the way down, so contact reliably means the table is higher than the
    calibration says or the brick sits on another. Off by default: the measured table plane is the
    real defence, and force detection on a UR3e at these speeds can fire on nothing.
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
                    "`python src/tools/calibrate_table.py`) or this brick is resting on another one."
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
    relook: bool = RELOOK_FROM_PREGRASP,
    deep_grasp: bool = False,
) -> GraspResult:
    """Descend onto the brick under the pregrasp, close, lift, and say whether it is held.

    With ``relook`` the brick is re-measured from where the arm is standing before anything is
    planned. That has to happen first: every pose below is built from ``target.position``, so
    correcting the position afterwards would send the arm to poses computed for somewhere else.

    Every pose is solved and floor-checked before anything moves, and the lift is planned before the
    brick is touched: finding out afterwards means finding out with the brick in the jaws.
    """
    if cell.gripper is None:
        raise RuntimeError("This cell has no gripper connected; build it with `with_gripper=True`.")

    if relook:
        relocate_from_pregrasp(cell, target)
        reposition_over(cell, target)

    if deep_grasp and grasp_depth is None:
        depth = table_referenced_depth(target)
    else:
        depth = resolve_grasp_depth(target, grasp_depth)
    width = target.approach_width

    grasp_position = np.array([target.position[0], target.position[1], target.top_face_z - depth])
    # From the position rather than a pose: every equivalent yaw sits at the same height.
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

    close_width = cell.gripper_calibration.min_width if CLOSE_FULLY else target.width
    logger.info(
        f"Closing the jaws shut on whatever is between them (submodule_1 measured "
        f"{target.width * 1000:.1f} mm, which the close does not rely on) ..."
    )
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
    """Recover from a failed grasp. Open first, then retreat: that drops whatever was half-caught back
    where it came from instead of carrying it somewhere else.
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
    # Stand back up: the next move is to a viewpoint across the table, and starting that from
    # centimetres above the tabletop drags the wrist through everything in between.
    park(cell)
    return True


# --- CLI ------------------------------------------------------------------------------------------


def main() -> None:
    """Grasp the brick submodule_1 left the arm standing over.

    Rebuilds a :class:`GraspTarget` from the handoff file and the arm's current pose, then runs
    :func:`run`. The handoff is refused if stale or if it describes a brick somewhere other than where
    the arm is standing.
    """
    import click

    from common.config import BRICK_HANDOFF_MAX_AGE, BRICK_HANDOFF_PATH, DEFAULT_CALIBRATION_DIR, DEFAULT_CAMERA_RESOLUTION, SUPPORTED_ROBOT_TYPES

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
    @click.option(
        "--deep-grasp/--no-deep-grasp",
        default=False,
        show_default=True,
        help="Descend until the fingertips are just clear of the touched-off table instead of measuring "
        "down from the part's assumed height. More grip and no dependence on the height, at the cost of "
        "driving into whatever is under a brick that is not sitting on the table.",
    )
    @click.option(
        "--relook/--no-relook",
        default=RELOOK_FROM_PREGRASP,
        show_default=True,
        help="Re-measure the brick from the pregrasp before descending, instead of trusting the survey's "
        "position. Costs one frame and no arm motion unless the camera is too close for depth.",
    )
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
        deep_grasp: bool,
        relook: bool,
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
                    relook=relook,
                    deep_grasp=deep_grasp,
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


#: How far the handoff's brick may be from where the arm stands and still be believed. Wider than this
#: and the arm has been moved since submodule_1 wrote the file.
HANDOFF_POSITION_TOLERANCE_M = 0.01


def read_handoff(path: str, max_age: float, expected_position: Optional[np.ndarray] = None) -> Optional[dict]:
    """Load submodule_1's handoff, or ``None`` with a reason logged if it should not be trusted.

    Refused if missing, older than ``max_age``, or recording a brick that is not where the arm now
    stands -- all meaning it describes a different brick than the one under the gripper.
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
    """Rebuild submodule_1's target from its handoff plus the pose the arm is parked at.

    x and y come from the pregrasp pose, not the file: the descent has to go straight down from where
    the arm actually is. ``read_handoff`` has already refused the file if the two disagree.
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
