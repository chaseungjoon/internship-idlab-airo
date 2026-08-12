"""m1 submodule 2 (simulation): from the pregrasp, close on the brick and lift it off the table.

The counterpart of ``m1/physical/submodule_2.py``, doing the same six things in the same order and
checking the same things between them: open, descend, close, verify, lift, verify again.

Two of those steps carry the whole module.

**How far to descend.** The fingertips have to get low enough down the brick's side wall to have
something to pinch, and must not touch the table. On a 9.6 mm brick that is a wide target; on a 3.2 mm
plate the entire budget is three millimetres, and a descent sized for the brick would drive the pads
into the tabletop -- which on the real robot jams the fingers and trips a protective stop. So the
descent is capped by the part actually being grasped and by the table underneath it, and the tighter
cap wins. That is the same arithmetic ``resolve_grasp_depth`` does on the bench, with one difference
worth noticing: here the table height is exact, so the cap is exactly right, and every millimetre of
margin the physical version has to leave for calibration error is margin this one does not need.

**Whether it worked.** The gripper is commanded to a *position* slightly inside the brick's width, so
the fingers stall on the brick rather than meeting each other. What separates "holding a brick" from
"closed on nothing" is then the gap between the width commanded and the width reached -- exactly the
signal the real Robotiq's object-detection flag reports, available here by measuring the pads. It is
checked twice: once on closing, which catches a miss, and again after the lift and a pause, which
catches the brick sliding back out as the arm accelerates. The second check is the one that means
anything, and it is the whole point of the module.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from m1.simulation import world as W  # noqa: E402
from m1.simulation.submodule_1 import GraspTarget  # noqa: E402

# --- the descent ------------------------------------------------------------------------------------
#: How far below the brick's top face the fingertips are driven before closing. Roughly a third of a
#: brick puts the pads on the side walls rather than on the studs, which would slip.
GRASP_DEPTH_M = 0.003
#: Hard floor: the fingertips never come closer than this to the tabletop, whatever else is asked for.
MIN_FINGERTIP_CLEARANCE_M = 0.0015
DESCEND_DURATION = 1.2

# --- the gripper ------------------------------------------------------------------------------------
#: Commanded *below* the brick's width, so the position controller keeps pushing and the fingers stall
#: on the brick. Nothing about a position command alone makes a grip; this overshoot is the grip.
GRIPPER_SQUEEZE_M = 0.003
GRIPPER_CLOSE_DURATION = 1.0
#: Verification band on the width reached. Under the lower bound the pads met with nothing between
#: them; over the upper bound something much thicker than the brick is in the jaws -- a neighbour
#: dragged in, or the brick caught end-on.
WIDTH_TOLERANCE_BELOW_M = 0.004
WIDTH_TOLERANCE_ABOVE_M = 0.006

# --- the lift ---------------------------------------------------------------------------------------
LIFT_HEIGHT_M = 0.12
MIN_LIFT_HEIGHT_M = 0.04
LIFT_SEARCH_STEP_M = 0.01
LIFT_DURATION = 2.0
SETTLE_DURATION = 0.8  # after the lift, so a slipping grasp has time to show itself


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
    brick_rise: Optional[float] = None
    part: Optional[str] = None

    def describe(self) -> str:
        verdict = "grasped and holding" if self.success else "FAILED"
        rise = f", the brick rose {self.brick_rise * 1000:.0f} mm" if self.brick_rise is not None else ""
        return (
            f"{verdict}: {self.reason}. Descended {self.grasp_depth * 1000:.1f} mm, commanded "
            f"{self.commanded_width * 1000:.1f} mm, pads stopped at {self.width_at_close * 1000:.1f} mm "
            f"then {self.width_after_lift * 1000:.1f} mm after a {self.lift_height * 100:.0f} cm lift{rise}"
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
    world: W.SimWorld, target: GraspTarget, grasp_position: np.ndarray, requested: float, width: float
) -> Optional[Tuple[np.ndarray, float]]:
    """The highest reachable straight-up lift from the grasp, at most ``requested`` metres.

    Lifting is the last thing that happens and the arm is already stretched over the table by then, so
    the requested height is the first thing to become unreachable. Refusing the run over that would
    leave the brick gripped and still sitting on the table, which is strictly worse than lifting it
    less far -- so the height is walked down until something is reachable.
    """
    steps = int(math.floor((requested - MIN_LIFT_HEIGHT_M) / LIFT_SEARCH_STEP_M + 1e-9))
    heights = [round(requested - step * LIFT_SEARCH_STEP_M, 6) for step in range(max(steps, 0) + 1)]
    if not heights or heights[-1] > MIN_LIFT_HEIGHT_M + 1e-9:
        heights.append(MIN_LIFT_HEIGHT_M)

    for height in heights:
        position = grasp_position + np.array([0.0, 0.0, height])
        pose = W.top_down_tool_pose(world, position, target.closing_heading, width)
        q = W.solve_tool_ik(world, pose)
        if q is not None:
            return q, height
    return None


# =================================================================================================
# the pick
# =================================================================================================


def check_grasp(world: W.SimWorld, target: GraspTarget) -> Tuple[bool, str]:
    """Whether the jaws are holding a brick of the right width right now, and how we know.

    The commanded width is deliberately inside the brick, so a gripper that reached its command is one
    that closed on nothing. Everything else is the band: near zero means empty jaws, much wider than
    the brick means something else came along with it.
    """
    width = world.finger_width()
    commanded = world.commanded_gripper_width
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


def run(
    world: W.SimWorld,
    target: GraspTarget,
    grasp_depth: Optional[float] = None,
    lift_height: float = LIFT_HEIGHT_M,
) -> GraspResult:
    """Descend onto the brick under the pregrasp, close, lift, and say whether it is held."""
    depth = resolve_grasp_depth(target, grasp_depth)
    width = target.approach_width

    grasp_position = np.array(
        [target.position[0], target.position[1], target.top_face_z - depth]
    )
    floor = target.table_z + MIN_FINGERTIP_CLEARANCE_M
    if grasp_position[2] < floor - 1e-9:
        raise RuntimeError(
            f"The planned grasp at z={grasp_position[2]:.4f} m is below the table floor at z={floor:.4f} m. "
            "Refusing to descend."
        )

    X_W_grasp = W.top_down_tool_pose(world, grasp_position, target.closing_heading, width)
    q_grasp = W.solve_tool_ik(world, X_W_grasp, q_seed=target.pregrasp_configuration)
    if q_grasp is None:
        raise RuntimeError(
            f"No reachable straight-down pose at {np.round(grasp_position, 4)} m with the jaws along "
            f"{math.degrees(target.closing_heading):.0f} deg, even though the pregrasp above it was reachable. "
            "This is the wrist running out of travel; a half-turn of the closing direction would fix it."
        )

    # The lift is planned *before* anything is touched: a brick that can be grasped but not raised is
    # not worth grasping, and finding that out afterwards means finding it out with the brick in the jaws.
    lift = reachable_lift(world, target, grasp_position, lift_height, width)
    if lift is None:
        raise RuntimeError(
            f"The grasp is reachable but not even a {MIN_LIFT_HEIGHT_M * 100:.0f} cm lift above it is, so the "
            "brick could be picked up and never raised. Pick a brick closer to the base."
        )
    q_lift, actual_lift = lift
    if actual_lift < lift_height - 1e-9:
        logger.warning(
            f"A {lift_height * 100:.0f} cm lift is out of reach here; lifting {actual_lift * 100:.0f} cm instead."
        )

    # Identify the real brick under the target *now*, and track that one body from here on. Asking
    # "what is nearest the target x, y" again after the lift would be a different question: the brick
    # this is about would be twelve centimetres in the air, and its former neighbour the new answer.
    tracked, tracked_distance = world.nearest_brick(target.position[:2])
    if tracked is not None and tracked_distance > 0.05:
        tracked = None
    brick_before = None if tracked is None else float(tracked.pose(world).translation()[2])

    logger.info(f"Descending {depth * 1000:.1f} mm onto the brick, jaws open at {width * 1000:.1f} mm ...")
    world.move_arm_to(q_grasp, DESCEND_DURATION)

    close_width = max(target.width - GRIPPER_SQUEEZE_M, world.gripper_calibration.min_width)
    logger.info(f"Closing: commanding {close_width * 1000:.1f} mm on a {target.width * 1000:.1f} mm brick ...")
    world.move_gripper_to_width(close_width, GRIPPER_CLOSE_DURATION)

    holding, reason = check_grasp(world, target)
    width_at_close = world.finger_width()
    if not holding:
        logger.error(f"The grasp did not take at the table: {reason}")
        _release_and_retreat(world, target)
        return GraspResult(
            success=False,
            reason=f"the grasp did not take at the table: {reason}",
            grasp_depth=depth,
            commanded_width=close_width,
            width_at_close=width_at_close,
            width_after_lift=width_at_close,
            lift_height=0.0,
            part=None if tracked is None else tracked.part,
        )

    logger.info(f"Closed on the brick: {reason}. Lifting {actual_lift * 100:.0f} cm ...")
    world.move_arm_to(q_lift, LIFT_DURATION)
    world.advance(SETTLE_DURATION)

    holding, reason = check_grasp(world, target)
    width_after_lift = world.finger_width()
    brick_after = None if tracked is None else float(tracked.pose(world).translation()[2])
    rise = None if brick_before is None or brick_after is None else brick_after - brick_before

    # Two independent verdicts, and they have to agree. The fingers can stay apart on a brick that was
    # never picked up -- caught against a neighbour, say -- and the simulator can see a brick rise that
    # is only being dragged along by one. Both, or it did not work.
    if holding and rise is not None and rise < 0.5 * actual_lift:
        holding = False
        reason = f"the jaws still hold something, but the brick only rose {rise * 1000:.0f} mm of {actual_lift * 1000:.0f} mm"

    result = GraspResult(
        success=holding,
        reason=reason if holding else f"the brick was lost during the lift: {reason}",
        grasp_depth=depth,
        commanded_width=close_width,
        width_at_close=width_at_close,
        width_after_lift=width_after_lift,
        lift_height=actual_lift,
        brick_rise=rise,
        part=None if tracked is None else tracked.part,
    )
    (logger.success if result.success else logger.error)(result.describe())
    if not result.success:
        _release_and_retreat(world, target)
    return result


def _release_and_retreat(world: W.SimWorld, target: GraspTarget) -> None:
    """Recover from a failed grasp: let go first, then go back up.

    Order matters. Opening first drops whatever was half-caught straight back where it came from, from
    millimetres up, instead of carrying it somewhere else and dropping it there.
    """
    logger.info("Opening the jaws and retreating to the pregrasp.")
    world.move_gripper_to_width(world.gripper_calibration.max_width, 0.6)
    if target.pregrasp_configuration is not None:
        world.move_arm_to(target.pregrasp_configuration, 1.2)


def park(world: W.SimWorld, duration: float = 2.0) -> None:
    """Take the arm back to its home configuration, holding whatever it is holding."""
    logger.info("Returning to the home configuration with the brick.")
    world.move_arm_to(W.HOME_CONFIGURATION, duration)


#: Somewhere on the table well clear of the pile to put a brick down. Sorting bricks into categories
#: is Module 3's job; this exists so the cycle can be run more than once without the second look at the
#: pile finding the first brick still in the gripper.
DROP_POSITION = np.array([0.16, -0.20, 0.06])


def place(
    world: W.SimWorld, target: GraspTarget, position: np.ndarray = DROP_POSITION, duration: float = 2.5
) -> bool:
    """Carry the brick clear of the pile and let it go. Returns whether the arm got there."""
    pose = W.top_down_tool_pose(world, np.asarray(position, float), target.closing_heading, world.finger_width())
    q = W.solve_tool_ik(world, pose)
    if q is None:
        logger.warning(f"Nowhere to put the brick down: {np.round(position, 3)} m is out of reach. Keeping hold of it.")
        return False
    logger.info(f"Carrying the brick to {np.round(position, 3)} m and releasing it.")
    world.move_arm_to(q, duration)
    world.move_gripper_to_width(world.gripper_calibration.max_width, 0.6)
    world.advance(0.4)
    # Stand back up before handing control back. The next thing to happen is a move to a viewpoint at
    # the far side of the table, and starting that from a pose centimetres above the tabletop is asking
    # the arm to drag its wrist across everything in between.
    park(world, duration=1.5)
    return True
