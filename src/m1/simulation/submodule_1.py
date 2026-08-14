"""m1 submodule 1 (simulation): look at the pile from two viewpoints, pick a brick, stand over it.

This is the physical pipeline's submodule_3 and submodule_1 fused into the one step they are going to
become: perceive the pile, choose which brick to grasp, work out where that brick actually is, and
park the gripper above it ready for submodule_2 to close on it.

**The perception is not simulated.** The rendered colour and depth go straight into
``m1.physical.submodule_3.analyse_pile`` -- the same function, the same thresholds, the same scoring
that runs against the RealSense on the bench. What this module adds is the part that needs two views:

* **Agreement.** Each viewpoint ranks the pile on its own. A brick both views find, in the same place,
  and both rank as graspable is a much better bet than the top of either view's list alone -- a brick
  half-hidden behind another one from one side usually is not from the other, and a region that only
  one view believes in is usually a segmentation accident.
* **Triangulation.** Having agreed which brick, its position comes from intersecting the two lines of
  sight through its centre, which is what ``physical/submodule_1.triangulate_rays`` does. On the bench
  that answer is dominated by hand-eye calibration error, which is why the physical module computes it
  for the record and then uses a ray-plane projection instead. Here the hand-eye transform is exact,
  so triangulation is the better estimate and the plane projection is the cross-check -- and the gap
  between the two is a direct, honest measure of how much the physical rig's calibration is costing.

Nothing about the choice consults the simulator's ground truth. :func:`score_against_truth` compares
the two afterwards, which is the whole reason for having a simulator.

**One survey, many picks.** The two viewpoints cost four seconds of arm travel and two full pile
analyses, and paying that for every single brick is most of the cycle time. So :func:`survey`
triangulates *every* brick the two views agree on, not just the winner, and hands back a
:class:`PileMap` ranked best-first; :class:`PileSession` then serves picks out of it without moving the
camera again. The pile is looked at again only when the map runs low -- which is also the moment the
bricks that were occluded at the start have become visible, and get their triangulation then. See
:class:`PileSession` for what makes a cached position go stale and how that is handled.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from pydrake.math import RigidTransform

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import PREGRASP_HEIGHT  # noqa: E402
from m1.physical.submodule_3 import Brick, PileAnalysis, analyse_pile, assign_priorities  # noqa: E402
from m1.simulation import world as W  # noqa: E402

#: Where the camera is put to look at the pile, and what it looks at. Two viewpoints roughly 30 cm
#: apart across the pile: far enough for real parallax (the lines of sight cross at about 45 degrees,
#: so a millimetre of pixel error is a millimetre of position error rather than a centimetre), close
#: enough that both are comfortably inside a UR3e's reach and both see the whole pile.
VIEWPOINTS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("view 1", (0.24, -0.13, 0.33)),
    ("view 2", (0.24, 0.17, 0.33)),
)
VIEW_TARGET = (W.PILE_CENTER[0], W.PILE_CENTER[1], 0.0)
VIEW_MOVE_DURATION = 2.0
VIEW_SETTLE_DURATION = 0.4

#: Two views have found the same brick if their base-frame centres are within this of each other. A
#: lego stud is 8 mm apart from the next one, so anything looser than half that could pair a brick
#: with its neighbour; anything tighter would reject a genuine match on ordinary measurement scatter.
MATCH_TOLERANCE_M = 0.006
#: ...and if they agree which way it points. Position alone is not enough: a region that merged two
#: touching bricks can sit within a millimetre of a real one and still be a completely different
#: rectangle, and averaging two long axes that are tens of degrees apart produces a direction that
#: belongs to neither -- which the jaws then close along, missing the brick entirely.
MATCH_HEADING_TOLERANCE_DEG = 15.0
#: Under this aspect ratio a footprint is square enough that its "long axis" is whichever side the
#: measurement noise favoured, so it is neither compared across views nor averaged between them.
SQUARE_ASPECT_RATIO = 1.25

#: Above this, the two lines of sight are missing each other by more than the width of the brick they
#: are supposed to be pointing at, and the triangulated point is not to be trusted.
MAX_TRIANGULATION_GAP_M = 0.006
#: Above this, triangulation and the ray-plane projection disagree about where the brick is -- measured
#: across the table, the only direction either of them decides -- by more than the jaws have slack, and
#: the more conservative of the two is used.
MAX_METHOD_DISAGREEMENT_M = 0.004

#: The approach is a staircase straight down over the brick: high, lower, pregrasp. Every move here is
#: a straight line in *joint* space, which is not a straight line in the world -- so the one leg that
#: crosses the table is made to end high above the pile, and the legs that come down are pure vertical
#: descents over the same point with the same wrist angle, where joint space and the world agree.
# Kept low on purpose. ``top_down_tool_pose`` places the *fingertips*, and the flange is another 160 mm
# of gripper above them -- so a retract 22 cm over the brick puts the flange 39 cm up and 30 cm out,
# which is a UR3e at full stretch. IK still finds a solution there and the arm still cannot hold it.
RETRACT_HEIGHT_M = 0.12
APPROACH_HEIGHT_M = 0.06
HOME_MOVE_DURATION = 1.5
RETRACT_MOVE_DURATION = 1.5
APPROACH_MOVE_DURATION = 1.2
PREGRASP_MOVE_DURATION = 1.0
#: The arm has to actually arrive. Past this the pose it reached is not the pose everything downstream
#: was computed for, and descending from it would put the fingers somewhere nobody planned.
MAX_PREGRASP_ERROR_M = 0.008

#: The jaws are opened this much wider than the brick before the arm goes anywhere near it -- the same
#: margin submodule_2 descends with, so the gripper is already at its approach opening on arrival.
GRIPPER_APPROACH_MARGIN_M = 0.014
GRIPPER_MOVE_DURATION = 0.6


# =================================================================================================
# the triangulation, as physical/submodule_1 does it
# =================================================================================================


def pixel_to_base_ray(
    u: float, v: float, intrinsics_matrix: np.ndarray, X_base_camera: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project pixel ``(u, v)`` to a ray ``(origin, unit direction)`` in the robot base frame.

    Pinhole model: the camera-frame ray is ``K^-1 [u, v, 1]`` in the optical convention (+z forward,
    +y down, the one both the RealSense point cloud and Drake's ``RgbdSensor`` use), rotated into the
    base frame by the eye-in-hand camera pose. The ray's origin is the camera centre.
    """
    direction_camera = np.linalg.inv(np.asarray(intrinsics_matrix, float)) @ np.array([u, v, 1.0])
    X = np.asarray(X_base_camera, float)
    direction_base = X[:3, :3] @ direction_camera
    return X[:3, 3].copy(), direction_base / np.linalg.norm(direction_base)


def triangulate_rays(
    ray_a: Tuple[np.ndarray, np.ndarray], ray_b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Midpoint triangulation of two rays, each an ``(origin, unit direction)`` pair.

    Two lines of sight in three dimensions almost never meet, so what is returned is the midpoint of
    the mutually closest points on them, and the distance between those points -- which is the quality
    metric: zero would mean the two views agree perfectly about the direction to the brick, and
    anything much larger than the brick means they are not looking at the same thing.

    Raises:
        RuntimeError: if the rays are near-parallel, i.e. the viewpoints give too little parallax.
    """
    origin_a, direction_a = ray_a
    origin_b, direction_b = ray_b

    b = float(direction_a @ direction_b)
    denominator = 1.0 - b * b  # (da.da)(db.db) - (da.db)^2, with unit directions
    if abs(denominator) < 1e-6:
        raise RuntimeError(
            "The two lines of sight are almost parallel; these viewpoints have too little parallax to "
            "triangulate. Move them further apart across the pile."
        )

    between = origin_a - origin_b
    d = float(direction_a @ between)
    e = float(direction_b @ between)
    closest_a = origin_a + ((b * e - d) / denominator) * direction_a
    closest_b = origin_b + ((e - b * d) / denominator) * direction_b
    return 0.5 * (closest_a + closest_b), float(np.linalg.norm(closest_a - closest_b))


def project_ray_onto_plane(
    ray: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float
) -> np.ndarray:
    """Intersect a base-frame ray with the plane ``z = a*x + b*y + c``.

    The other way to turn one line of sight into a point: instead of a second view, use the fact that
    the height is already known -- the table, raised by one brick. It fixes the answer's z outright and
    leaves only x and y carrying any error, which on the bench is the trade worth making. Here it is
    the cross-check on the triangulation rather than the answer.

    Raises:
        RuntimeError: if the ray runs parallel to the plane or the plane is behind the camera.
    """
    origin, direction = ray
    denominator = direction[2] - a * direction[0] - b * direction[1]
    if abs(denominator) < 1e-9:
        raise RuntimeError("The line of sight runs parallel to the table plane; it never crosses it.")
    distance = (a * origin[0] + b * origin[1] + c - origin[2]) / denominator
    if distance <= 0:
        raise RuntimeError("The table plane lies behind the camera; check the camera pose.")
    return origin + distance * direction


# =================================================================================================
# what one look at the pile leaves behind
# =================================================================================================


@dataclass
class ViewResult:
    """One viewpoint: where it looked from, what it saw, and what it made of it."""

    name: str
    eye: np.ndarray
    joint_configuration: np.ndarray
    analysis: PileAnalysis

    @property
    def graspable(self) -> List[Brick]:
        return [b for b in self.analysis.ordered if b.graspable and b.confidence >= 0.7]


@dataclass
class GraspTarget:
    """The brick submodule_2 is to grasp, and everything it needs to know to do it.

    This is the simulation's stand-in for the handoff file the physical modules pass through
    ``run/brick_handoff.json`` -- same contents, minus the staleness checks, because here the two
    halves run in one process and the pile cannot have been disturbed in between.
    """

    position: np.ndarray  # base frame, centre of the brick's top face
    width: float  # metres, the short side -- what the jaws close on
    length: float
    height: float  # metres, top face above the table
    long_axis_heading: float  # radians, base frame
    table_z: float
    colour: str
    score: float
    confidence: float

    # how the position was arrived at, kept for the record
    triangulated: np.ndarray
    plane_projected: np.ndarray
    triangulation_gap: float
    method_disagreement: float
    position_source: str

    # where the arm ended up
    pregrasp_pose: Optional[RigidTransform] = None
    pregrasp_configuration: Optional[np.ndarray] = None
    approach_width: float = 0.05
    per_view: Dict[str, np.ndarray] = field(default_factory=dict)

    # which look at the pile produced it, and how far apart the two views placed it. A target carries
    # its survey with it because a cached position is only as good as the pile it was measured on: the
    # session compares this against the current survey to know whether a failure is the brick's fault
    # or the map's age.
    survey_round: int = 0
    match_distance: float = 0.0

    @property
    def closing_heading(self) -> float:
        """Base-frame direction the fingers must close along: square to the brick's long axis."""
        return self.long_axis_heading + math.pi / 2

    @property
    def top_face_z(self) -> float:
        return float(self.position[2])

    def describe(self) -> str:
        return (
            f"{self.colour} {self.width * 1000:.1f} x {self.length * 1000:.1f} x {self.height * 1000:.1f} mm "
            f"at ({self.position[0]:.4f}, {self.position[1]:.4f}) m, top face z={self.top_face_z:.4f} m, "
            f"jaws close along {math.degrees(self.closing_heading):.0f} deg"
        )

    def to_dict(self) -> Dict:
        return {
            "position": np.asarray(self.position, float).round(5).tolist(),
            "width": round(self.width, 5),
            "length": round(self.length, 5),
            "height": round(self.height, 5),
            "long_axis_heading": round(self.long_axis_heading, 5),
            "closing_heading": round(self.closing_heading, 5),
            "table_z": round(self.table_z, 5),
            "colour": self.colour,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 3),
            "position_source": self.position_source,
            "triangulated": np.asarray(self.triangulated, float).round(5).tolist(),
            "plane_projected": np.asarray(self.plane_projected, float).round(5).tolist(),
            "triangulation_gap_mm": round(self.triangulation_gap * 1000, 2),
            "method_disagreement_mm": round(self.method_disagreement * 1000, 2),
            "approach_width": round(self.approach_width, 4),
            "survey_round": self.survey_round,
            "match_distance_mm": round(self.match_distance * 1000, 2),
        }


# =================================================================================================
# looking
# =================================================================================================


def observe(world: W.SimWorld, name: str, eye: Sequence[float], target: Sequence[float] = VIEW_TARGET) -> ViewResult:
    """Move the camera to ``eye``, look at ``target``, and run the pile perception on what it sees."""
    X_W_tool = W.look_at_tool_pose(world, np.asarray(eye, float), np.asarray(target, float))
    q = W.solve_tool_ik(world, X_W_tool)
    if q is None:
        raise RuntimeError(
            f"{name} at {np.round(eye, 3)} m is not reachable. Move the viewpoint closer to the base "
            "or lower, and remember a UR3e only reaches about half a metre."
        )
    logger.info(f"{name}: moving the camera to {np.round(eye, 3)} m, looking at {np.round(target, 3)} m ...")
    world.move_arm_to(q, VIEW_MOVE_DURATION)
    world.advance(VIEW_SETTLE_DURATION)

    view = world.capture(name)
    analysis = analyse_pile(view, world.table_plane, "ur3e")
    # No arm is passed: reachability is checked later, once, against the brick that actually wins, and
    # with Drake's IK rather than the physical module's driver call.
    assign_priorities(analysis.ordered, None, PREGRASP_HEIGHT)
    logger.info(
        f"{name}: {len(analysis.bricks)} brick(s), "
        f"{sum(1 for b in analysis.bricks if b.graspable)} graspable, "
        f"{len(analysis.rejected)} region(s) dropped."
    )
    return ViewResult(name=name, eye=np.asarray(eye, float), joint_configuration=world.arm_positions(), analysis=analysis)


def match_across_views(first: ViewResult, second: ViewResult) -> List[Tuple[Brick, Brick, float]]:
    """Pair up the bricks the two views both found, nearest-centre first, one pairing each.

    Both views report positions in the robot's base frame already -- that is what makes the pairing a
    two-line problem rather than a feature-matching one. Greedy nearest-neighbour over the pairs sorted
    by distance, so the closest, least ambiguous pairs claim their partners before the doubtful ones
    get a say.
    """
    candidates = []
    for a in first.graspable:
        for b in second.graspable:
            distance = float(np.linalg.norm(np.array(a.center_m) - np.array(b.center_m)))
            if distance > MATCH_TOLERANCE_M:
                continue
            # Elongated bricks have to agree about which way they lie as well as where they are. A
            # pair that agrees on the centre to a millimetre and disagrees on the axis by forty
            # degrees is not one brick seen twice; it is one brick and one segmentation accident that
            # happens to be centred on top of it.
            if min(a.aspect_ratio, b.aspect_ratio) >= SQUARE_ASPECT_RATIO:
                gap = abs(_heading_difference(a.long_axis_heading, b.long_axis_heading))
                if math.degrees(gap) > MATCH_HEADING_TOLERANCE_DEG:
                    continue
            candidates.append((distance, a, b))
    candidates.sort(key=lambda item: item[0])

    used_first, used_second = set(), set()
    pairs: List[Tuple[Brick, Brick, float]] = []
    for distance, a, b in candidates:
        if a.index in used_first or b.index in used_second:
            continue
        used_first.add(a.index)
        used_second.add(b.index)
        pairs.append((a, b, distance))
    return pairs


def locate(first: ViewResult, second: ViewResult, a: Brick, b: Brick, plane: Tuple[float, float, float]) -> Dict:
    """Where the brick is, by triangulation, with the ray-plane projection as the cross-check.

    Both estimates are computed and both are reported. Triangulation wins unless it fails its own
    quality test -- the two lines of sight missing each other by more than the brick is wide, or the
    two methods disagreeing by more than the jaws have slack -- in which case the projection is used,
    because its z is not an estimate at all: it is the table, plus a brick.
    """
    height = 0.5 * (a.height_m + b.height_m)
    ray_a = pixel_to_base_ray(*a.grasp_pixel, first.analysis.view.intrinsics_matrix, first.analysis.view.X_base_camera)
    ray_b = pixel_to_base_ray(*b.grasp_pixel, second.analysis.view.intrinsics_matrix, second.analysis.view.X_base_camera)

    plane_a, plane_b, plane_c = plane
    top_face = (plane_a, plane_b, plane_c + height)
    projected = 0.5 * (
        project_ray_onto_plane(ray_a, *top_face) + project_ray_onto_plane(ray_b, *top_face)
    )

    try:
        triangulated, gap = triangulate_rays(ray_a, ray_b)
    except RuntimeError as exception:
        logger.warning(f"{exception} Falling back to the ray-plane projection.")
        return {
            "position": projected,
            "triangulated": projected,
            "plane_projected": projected,
            "gap": float("nan"),
            "disagreement": 0.0,
            "source": "plane_projection",
            "height": height,
        }

    # Compared in the horizontal plane only, because that is the only part of the triangulated point
    # that is ever used: the grasp's z comes from the table plus the measured brick height either way.
    # Their z's differ by a few millimetres by construction -- the pixel being back-projected is the
    # centroid of the brick's *silhouette*, which from a tilted view includes the near side wall, so
    # the two lines of sight cross a little below the top face. That is not an error, and letting it
    # into the disagreement would have it condemning a triangulation that is millimetre-accurate.
    disagreement = float(np.linalg.norm(triangulated[:2] - projected[:2]))
    if gap > MAX_TRIANGULATION_GAP_M:
        source = "plane_projection"
        logger.warning(
            f"The two lines of sight miss each other by {gap * 1000:.1f} mm, wider than the "
            f"{min(a.width_mm, b.width_mm):.1f} mm brick they are aimed at, so the triangulated point is not "
            "trustworthy; using the ray-plane projection instead."
        )
    elif disagreement > MAX_METHOD_DISAGREEMENT_M:
        source = "plane_projection"
        logger.warning(
            f"Triangulation and the ray-plane projection disagree by {disagreement * 1000:.1f} mm. The "
            "projection's height is known rather than estimated, so it is the safer of the two here."
        )
    else:
        source = "triangulation"

    return {
        "position": triangulated if source == "triangulation" else projected,
        "triangulated": triangulated,
        "plane_projected": projected,
        "gap": gap,
        "disagreement": disagreement,
        "source": source,
        "height": height,
    }


#: A brick this close to one that has already been tried and failed is treated as that brick. Half a
#: stud: close enough to catch the same part after the failed grasp nudged it, far enough not to
#: condemn its neighbour.
AVOID_RADIUS_M = 0.004


def _matched_pairs(
    first: ViewResult,
    second: ViewResult,
    avoid: Sequence[np.ndarray] = (),
) -> List[Tuple[Brick, Brick, float]]:
    """The bricks both views found, minus any at a position already tried and failed."""
    pairs = match_across_views(first, second)
    if avoid:
        before = len(pairs)
        pairs = [
            pair
            for pair in pairs
            if all(np.linalg.norm(np.array(pair[0].center_m) - np.asarray(p, float)[:2]) > AVOID_RADIUS_M for p in avoid)
        ]
        if len(pairs) < before:
            logger.info(f"Skipping {before - len(pairs)} brick(s) that were already tried and not picked up.")
    return pairs


def build_target(
    first: ViewResult,
    second: ViewResult,
    a: Brick,
    b: Brick,
    match_distance: float,
    plane: Tuple[float, float, float],
    survey_round: int = 0,
) -> GraspTarget:
    """Triangulate one matched pair into everything submodule_2 needs to grasp it."""
    located = locate(first, second, a, b, plane)
    height = located["height"]
    position = np.array([located["position"][0], located["position"][1], plane[2] + plane[0] * located["position"][0] + plane[1] * located["position"][1] + height])

    # Averaged across the views: two independent measurements of the same rectangle, and the jaws want
    # the more conservative width anyway, which is why the wider of the two is taken for the opening.
    long_axis = _combine_headings(a, b)
    return GraspTarget(
        position=position,
        width=max(a.width_mm, b.width_mm) / 1000.0,
        length=0.5 * (a.length_mm + b.length_mm) / 1000.0,
        height=height,
        long_axis_heading=long_axis,
        table_z=float(plane[2] + plane[0] * position[0] + plane[1] * position[1]),
        colour=a.colour_name,
        score=0.5 * (a.score + b.score),
        confidence=0.5 * (a.confidence + b.confidence),
        triangulated=located["triangulated"],
        plane_projected=located["plane_projected"],
        triangulation_gap=located["gap"],
        method_disagreement=located["disagreement"],
        position_source=located["source"],
        per_view={first.name: np.array(a.center_m), second.name: np.array(b.center_m)},
        survey_round=survey_round,
        match_distance=match_distance,
    )


def choose_target(
    first: ViewResult,
    second: ViewResult,
    plane: Tuple[float, float, float],
    avoid: Sequence[np.ndarray] = (),
) -> Optional[GraspTarget]:
    """The best brick the two views agree on, located by triangulation.

    Ranked on the mean of the two views' scores rather than either alone. The scores already fold in
    fingertip clearance, isolation, how much of the brick's outline borders bare table and whether
    anything stands over it -- all of which are viewpoint-dependent, and a brick that scores well from
    two directions at once is one that really is out on its own.

    ``avoid`` lists positions already tried and failed. Without it the pile is unchanged after a failed
    grasp, so the same brick scores best again and the cycle picks it forever; a brick that has just
    refused to be picked up is exactly the brick to leave alone.

    This is the one-brick-at-a-time path, kept because it is the smallest thing that demonstrates the
    module. The loop uses :func:`survey`, which triangulates all of them in the same two looks.
    """
    pairs = _matched_pairs(first, second, avoid)
    if not pairs:
        logger.error(
            "The two views agree on no graspable brick at all. Either the pile is out of frame from one "
            "of them, or nothing in it is currently a safe grasp."
        )
        return None
    logger.info(f"The two views agree on {len(pairs)} graspable brick(s).")

    a, b, match_distance = max(pairs, key=lambda pair: 0.5 * (pair[0].score + pair[1].score))
    target = build_target(first, second, a, b, match_distance, plane)
    logger.info(
        f"Chosen: {target.describe()} (mean score {target.score:.3f}, the two views placed it "
        f"{match_distance * 1000:.1f} mm apart)."
    )
    logger.info(
        f"  triangulated {np.round(target.triangulated, 4)} m, rays missing by "
        f"{target.triangulation_gap * 1000:.2f} mm; ray-plane projection "
        f"{np.round(target.plane_projected, 4)} m, {target.method_disagreement * 1000:.2f} mm away. "
        f"Using the {target.position_source.replace('_', ' ')}."
    )
    return target


def _heading_difference(first: float, second: float) -> float:
    """Angle between two *axes*, in (-pi/2, pi/2]: a rectangle's long side has no head or tail."""
    return (first - second + math.pi / 2) % math.pi - math.pi / 2


def _average_heading(headings: Sequence[float]) -> float:
    """Mean of angles that name an axis, not a direction, so they are averaged modulo 180 degrees."""
    doubled = [2 * h for h in headings]
    return float(np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))) / 2)


def _combine_headings(a: Brick, b: Brick) -> float:
    """The direction the brick's long axis points, from whichever view can actually see it.

    Averaging only makes sense when both views are measuring the same axis. On a near-square footprint
    -- a 2x2 plate, a 1x1 brick -- there is no long axis to measure and the two views will name
    perpendicular sides as often as not, so the more elongated measurement is simply taken instead:
    it is the one with an axis worth having, and on a square part either answer grasps equally well.
    """
    if min(a.aspect_ratio, b.aspect_ratio) >= SQUARE_ASPECT_RATIO:
        return _average_heading([a.long_axis_heading, b.long_axis_heading])
    return (a if a.aspect_ratio >= b.aspect_ratio else b).long_axis_heading


# =================================================================================================
# surveying the whole pile at once
# =================================================================================================


@dataclass
class PileMap:
    """Every brick one pair of looks agreed on, triangulated, ranked best-first.

    The unit of work is the *survey*, not the brick: the two viewpoints are the expensive part of a
    pick and they see the whole pile, so everything they agree on is located in the same breath and
    kept here. Targets are served out of :attr:`targets` in score order and are removed as they go, so
    :attr:`remaining` is the count that decides when the pile is worth looking at again.
    """

    targets: List[GraspTarget]
    views: List[ViewResult]
    survey_round: int
    surveyed_at: float  # world.elapsed when the two looks were taken
    #: How many bricks have been *attempted* out of this map. Zero means the map still describes the
    #: pile exactly as the camera saw it; anything above zero means the pile has been reached into.
    picks_since_survey: int = 0

    @property
    def remaining(self) -> int:
        return len(self.targets)

    def take_best(self) -> Optional[GraspTarget]:
        """Remove and return the highest-scoring target left, or None if the map is empty."""
        return self.targets.pop(0) if self.targets else None

    def discard_near(self, position_xy: Sequence[float], radius: float) -> List[GraspTarget]:
        """Forget the targets within ``radius`` of a point, returning them. Used after each pick.

        A grasp is not a surgical operation: the jaws come down open around the brick and go back up
        with it, and anything close enough to be inside that sweep may have been nudged. Its cached
        position was measured before the nudge, so it is no longer worth what it was -- dropping it
        here costs one brick's place in the queue and buys it a fresh triangulation at the next survey.
        """
        centre = np.asarray(position_xy, float)[:2]
        keep, dropped = [], []
        for target in self.targets:
            (dropped if np.linalg.norm(target.position[:2] - centre) <= radius else keep).append(target)
        self.targets = keep
        return dropped

    def describe(self) -> str:
        return (
            f"survey {self.survey_round}: {self.remaining} brick(s) triangulated and queued, taken at "
            f"t={self.surveyed_at:.1f} s, {self.picks_since_survey} pick(s) made since"
        )


def build_pile_map(
    first: ViewResult,
    second: ViewResult,
    plane: Tuple[float, float, float],
    avoid: Sequence[np.ndarray] = (),
    keep_out: Sequence[Tuple[Sequence[float], float]] = (),
    survey_round: int = 0,
    surveyed_at: float = 0.0,
) -> PileMap:
    """Triangulate every brick the two views agree on, ranked by mean score.

    The same matching and the same triangulation :func:`choose_target` does, applied to all the pairs
    instead of only the winner. A pair whose lines of sight cannot be turned into a point at all -- one
    running parallel to the table, say -- is dropped with a warning rather than taking the survey down
    with it: on a full pile there is always a marginal region at the edge of the frame, and the other
    bricks are still perfectly good.

    ``keep_out`` is a list of ``(centre_xy, radius)`` circles that are not part of the pile -- the
    corner the picked bricks are stacked in, above all. Once the heap of already-picked bricks grows
    bigger than what is left of the pile, the perception's own "largest blob is the pile" heuristic
    starts pointing the wrong way, and without this the robot would cheerfully re-pick its own output.
    """
    pairs = _matched_pairs(first, second, avoid)
    if keep_out:
        before = len(pairs)
        pairs = [
            pair
            for pair in pairs
            if all(
                np.linalg.norm(np.array(pair[0].center_m) - np.asarray(centre, float)[:2]) > radius
                for centre, radius in keep_out
            )
        ]
        if len(pairs) < before:
            logger.info(f"Ignoring {before - len(pairs)} brick(s) sitting in a keep-out area, not in the pile.")

    targets: List[GraspTarget] = []
    for a, b, match_distance in pairs:
        try:
            targets.append(build_target(first, second, a, b, match_distance, plane, survey_round))
        except RuntimeError as exception:
            logger.warning(f"Could not locate the brick at {np.round(a.center_m, 3)} m: {exception} Skipping it.")
    targets.sort(key=lambda t: t.score, reverse=True)

    pile_map = PileMap(
        targets=targets, views=[first, second], survey_round=survey_round, surveyed_at=surveyed_at
    )
    if not targets:
        logger.error(
            "The two views agree on no graspable brick at all. Either the pile is empty, it is out of "
            "frame from one of them, or nothing left in it is a safe grasp."
        )
        return pile_map

    logger.success(
        f"Survey {survey_round}: {len(targets)} brick(s) triangulated in one pair of looks -- the next "
        f"{len(targets)} pick(s) need no camera move at all."
    )
    for rank, target in enumerate(targets, start=1):
        logger.info(
            f"  {rank:2d}. {target.describe()} | score {target.score:.3f}, conf {target.confidence:.2f}, "
            f"views {target.match_distance * 1000:.1f} mm apart, rays missing by "
            f"{target.triangulation_gap * 1000:.2f} mm, {target.position_source.replace('_', ' ')}"
        )
    return pile_map


def survey(
    world: W.SimWorld,
    avoid: Sequence[np.ndarray] = (),
    keep_out: Sequence[Tuple[Sequence[float], float]] = (),
    survey_round: int = 0,
) -> PileMap:
    """Two looks at the pile, and a triangulated position for every brick they agree on."""
    views = [observe(world, name, eye) for name, eye in VIEWPOINTS]
    return build_pile_map(
        views[0],
        views[1],
        world.table_plane,
        avoid=avoid,
        keep_out=keep_out,
        survey_round=survey_round,
        surveyed_at=world.elapsed,
    )


# =================================================================================================
# going there
# =================================================================================================


def go_to_pregrasp(world: W.SimWorld, target: GraspTarget, pregrasp_height: float = PREGRASP_HEIGHT) -> GraspTarget:
    """Open the jaws, swing over the brick well clear of the pile, then drop to the pregrasp.

    In two moves, not one. A joint-space interpolation from a viewpoint at the edge of the table to a
    pose three centimetres above one brick in the middle of the pile is a straight line in joint space
    and something quite different in the world -- typically a diagonal sweep through everything between
    the two. Going up and over first costs a second and keeps the pile where the perception found it.

    The wrist is already turned so the jaws are square to the brick's long axis: submodule_1 knows the
    orientation the moment it knows the brick, and turning the wrist here means it happens high above
    the pile rather than centimetres over it.
    """
    approach_width = min(
        target.width + GRIPPER_APPROACH_MARGIN_M, world.gripper_calibration.max_width
    )
    target.approach_width = approach_width
    world.move_gripper_to_width(approach_width, GRIPPER_MOVE_DURATION)
    logger.info(f"Jaws opened to {approach_width * 1000:.1f} mm for a {target.width * 1000:.1f} mm brick.")

    heights = (
        ("retract", RETRACT_HEIGHT_M, RETRACT_MOVE_DURATION),
        ("approach", APPROACH_HEIGHT_M, APPROACH_MOVE_DURATION),
        ("pregrasp", pregrasp_height, PREGRASP_MOVE_DURATION),
    )
    poses, configurations = [], []
    seed = None
    for name, height, _ in heights:
        position = np.array([target.position[0], target.position[1], target.top_face_z + height])
        pose = W.top_down_tool_pose(world, position, target.closing_heading, approach_width)
        q = W.solve_tool_ik(world, pose, q_seed=seed)
        if q is None:
            raise RuntimeError(
                f"No reachable straight-down {name} pose at {np.round(position, 3)} m with the jaws along "
                f"{math.degrees(target.closing_heading):.0f} deg. This brick is at the edge of the arm's "
                "workspace; the next-best candidate would have to be tried instead."
            )
        poses.append(pose)
        configurations.append(q)
        seed = q  # each step seeded from the one above it, so the whole descent stays on one branch

    # Via home first. The leg that crosses the table is the only one that can sweep the arm through
    # the pile, and it starts from a viewpoint pose out at the edge of the workspace with the wrist
    # tilted over -- the worst possible starting point for a joint-space straight line. Home is elbow
    # up and central, so both halves of the journey stay high and neither has far to go.
    logger.info("Retracting to the home configuration before crossing the table ...")
    world.move_arm_to(W.HOME_CONFIGURATION, HOME_MOVE_DURATION)
    logger.info(
        f"Swinging over the brick at {RETRACT_HEIGHT_M * 100:.0f} cm, jaws already square to its long axis, "
        f"then straight down to the pregrasp {pregrasp_height * 100:.0f} cm above its top face ..."
    )
    for (name, height, duration), q in zip(heights, configurations):
        world.move_arm_to(q, duration)

    world.advance(0.2)
    target.pregrasp_pose = poses[-1]
    target.pregrasp_configuration = world.arm_positions()

    commanded = poses[-1].translation() + poses[-1].rotation().matrix() @ np.array(
        [0.0, 0.0, world.gripper_calibration.tip_offset(approach_width)]
    )
    reached = world.tcp_pose(approach_width).translation()
    error = float(np.linalg.norm(reached - commanded))
    if error > MAX_PREGRASP_ERROR_M:
        raise RuntimeError(
            f"The arm stopped {error * 1000:.0f} mm from the pregrasp it was sent to -- fingertips at "
            f"{np.round(reached, 4)} m, commanded {np.round(commanded, 4)} m. Something is in the way (the "
            "table, or another brick), so the descent below would not land where it was planned."
        )
    logger.success(
        f"At the pregrasp: fingertips at {np.round(reached, 4)} m, {error * 1000:.1f} mm from commanded."
    )
    return target


def run(
    world: W.SimWorld,
    pregrasp_height: float = PREGRASP_HEIGHT,
    avoid: Sequence[np.ndarray] = (),
) -> Tuple[GraspTarget, List[ViewResult]]:
    """The whole of submodule_1: two looks, one decision, one pregrasp.

    Returns the target for submodule_2 and both views, so the notebook can draw what was seen. Pass
    the positions of bricks that have already been tried and dropped as ``avoid``.
    """
    views = [observe(world, name, eye) for name, eye in VIEWPOINTS]
    target = choose_target(views[0], views[1], world.table_plane, avoid)
    if target is None:
        raise RuntimeError(
            "No brick to grasp. Every region either failed the confidence test, was too tightly packed "
            "for a fingertip, or was seen by only one of the two views."
        )
    return go_to_pregrasp(world, target, pregrasp_height), views


# =================================================================================================
# emptying the pile: one survey, many picks
# =================================================================================================

#: Look at the pile again once fewer than this many triangulated bricks are left queued. At 2 the last
#: brick of a survey is never picked from a map that has nothing behind it, so the arm is never left
#: standing over an empty queue -- and the fresh look happens while there is still a known-good target
#: in hand to fall back on. Raise it to re-survey more often (safer positions, slower); drop it to 1 to
#: squeeze every last brick out of each survey.
RESURVEY_WHEN_REMAINING_BELOW = 2
#: On top of the jaws' own half-width: how far past the open fingers a brick can be and still be
#: counted as possibly disturbed by the pick. The pads are 37.5 mm tall and the lifted brick swings a
#: little as the arm accelerates away, so a centimetre past the fingertips is the honest margin.
FINGER_DISTURBANCE_MARGIN_M = 0.012
#: This many failed grasps in a row and the map is not believed any more, whatever it still has queued.
#: One failure is a brick; three in a row is a map that no longer describes the table in front of it.
MAX_CONSECUTIVE_FAILURES = 3


class PileSession:
    """Empties the pile, looking at it as few times as possible.

    The loop this replaces re-ran both viewpoints before every single pick, which is honest -- each
    pick disturbs the pile -- but pays for a whole survey to use one brick out of it. This one pays
    the same price and uses the whole survey:

    * The first survey triangulates every brick both views can see and queues them by score.
    * Each :meth:`next_target` serves the best one left, straight to the pregrasp, no camera move.
    * Bricks close enough to the one just picked to have been knocked by the jaws are dropped from
      the queue rather than trusted (:meth:`PileMap.discard_near`).
    * When the queue falls below :data:`RESURVEY_WHEN_REMAINING_BELOW`, the pile is looked at again.
      That is also when the bricks that were buried at the start are finally on top and visible, so
      the occluded ones get their triangulation exactly when it is worth doing.

    **What a cached position risks, and what happens then.** A stale position means the fingers close
    where the brick used to be. submodule_2 catches that on the width check, opens, and retreats -- so
    the failure mode is a wasted pick, not a collision. The session then treats the failure according
    to how much it can blame the map: a brick that failed on a *fresh* target (nothing had been touched
    since the survey) is a genuinely bad grasp and is avoided for good, while one that failed on a
    target the pile had since been reached into gets its position remeasured at the next survey and
    another chance. Three failures in a row force a survey outright.

    Usage::

        session = PileSession(world)
        while (target := session.next_target()) is not None:
            result = submodule_2.run(world, target)
            if result.success:
                submodule_2.place(world, target)
            session.record(target, result.success)
    """

    def __init__(
        self,
        world: W.SimWorld,
        pregrasp_height: float = PREGRASP_HEIGHT,
        resurvey_below: int = RESURVEY_WHEN_REMAINING_BELOW,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        keep_out: Optional[Sequence[Tuple[Sequence[float], float]]] = None,
    ) -> None:
        self.world = world
        self.pregrasp_height = pregrasp_height
        self.resurvey_below = max(1, int(resurvey_below))
        self.max_consecutive_failures = max_consecutive_failures
        self.keep_out = list(default_keep_out() if keep_out is None else keep_out)

        self.map: Optional[PileMap] = None
        self.surveys = 0
        self.attempts = 0
        self.picked: List[np.ndarray] = []
        #: Bricks that failed a grasp the map cannot be blamed for. Never retried.
        self.avoid: List[np.ndarray] = []
        #: Bricks that failed while the map was already out of date. Cleared by the next survey, which
        #: measures them again -- the position they were grasped at was not the position they were in.
        self.provisional_avoid: List[np.ndarray] = []
        self._consecutive_failures = 0
        self._force_survey = False
        self._last_target_was_fresh = True

    # --- picking ----------------------------------------------------------------------------------
    def next_target(self) -> Optional[GraspTarget]:
        """The next brick to grasp, with the arm already standing over it. None when the pile is done.

        Surveys only when the map is empty, low, or discredited. Targets whose pregrasp turns out to be
        unreachable are skipped over rather than raising: with a whole map in hand there is always a
        next-best candidate, which is exactly what the single-shot :func:`run` has to give up and ask
        the caller for.
        """
        while True:
            surveyed_now = False
            if self._needs_survey():
                self._survey()
                surveyed_now = True

            target = self._take_reachable()
            if target is not None:
                return target

            # The map is empty. If it was taken with the pile untouched since, that is the real answer:
            # nothing left is graspable. Otherwise the pile has moved under it, and it deserves a look.
            if surveyed_now or self.map is None or self.map.picks_since_survey == 0:
                logger.info(
                    f"Nothing left to grasp: {len(self.picked)} brick(s) picked over {self.surveys} survey(s) "
                    f"and {self.attempts} attempt(s)."
                )
                return None
            logger.info("The map is used up but the pile has been disturbed since it was made; looking again.")
            self._force_survey = True

    def record(self, target: GraspTarget, success: bool) -> None:
        """Tell the session how the grasp went. Call it after every attempt, successful or not."""
        centre = np.asarray(target.position, float)[:2]
        # Whatever happened to the brick, the jaws were down among its neighbours. Anything they could
        # have reached is remeasured rather than trusted.
        radius = 0.5 * target.approach_width + FINGER_DISTURBANCE_MARGIN_M
        disturbed = self.map.discard_near(centre, radius) if self.map is not None else []
        if disturbed:
            logger.info(
                f"Dropping {len(disturbed)} queued brick(s) within {radius * 1000:.0f} mm of the jaws; they will "
                "be triangulated again at the next survey."
            )

        if success:
            self.picked.append(centre)
            self._consecutive_failures = 0
            return

        self._consecutive_failures += 1
        if self._last_target_was_fresh:
            # Nothing had been touched since this brick was measured, so the position was as good as the
            # perception gets and the grasp still failed. That is the brick, not the map.
            self.avoid.append(centre)
            logger.info("The grasp failed on a freshly surveyed position; leaving that brick alone from now on.")
        else:
            self.provisional_avoid.append(centre)
            logger.info(
                "The grasp failed on a position measured before the pile was last reached into; it will be "
                "remeasured at the next survey rather than written off."
            )
        if self._consecutive_failures >= self.max_consecutive_failures:
            logger.warning(
                f"{self._consecutive_failures} failed grasps in a row -- the map no longer describes the table. "
                "Forcing a fresh survey."
            )
            self._force_survey = True

    # --- the map ----------------------------------------------------------------------------------
    def look(self) -> PileMap:
        """Survey the pile now and return the map, instead of waiting for :meth:`next_target` to.

        Only for looking at what the survey produced -- the picking that follows uses the same map, so
        calling this before the loop costs nothing.
        """
        self._survey()
        assert self.map is not None
        return self.map

    def _needs_survey(self) -> bool:
        return self.map is None or self._force_survey or self.map.remaining < self.resurvey_below

    def _survey(self) -> None:
        self.surveys += 1
        if self.map is not None:
            logger.info(
                f"Re-surveying: {self.map.remaining} triangulated brick(s) left, below the {self.resurvey_below} "
                "threshold. Bricks that were occluded at the start get located now."
            )
        # The provisional list only ever meant "measured on a pile that has since moved", and this is
        # the survey that moves it back.
        self.provisional_avoid.clear()
        self._consecutive_failures = 0
        self._force_survey = False
        self.map = survey(
            self.world,
            avoid=self.avoid,
            keep_out=self.keep_out,
            survey_round=self.surveys,
        )

    def _take_reachable(self) -> Optional[GraspTarget]:
        """Pop targets until one of them can actually be stood over."""
        assert self.map is not None
        while self.map.remaining:
            target = self.map.take_best()
            if self._is_avoided(target):
                logger.info(f"Skipping the queued brick at {np.round(target.position[:2], 3)} m; it failed earlier.")
                continue
            # Recorded before the counter moves: a target served off a map nothing has been picked from
            # is measuring the pile that is actually there.
            self._last_target_was_fresh = self.map.picks_since_survey == 0
            try:
                go_to_pregrasp(self.world, target, self.pregrasp_height)
            except RuntimeError as exception:
                logger.warning(f"Cannot stand over the brick at {np.round(target.position[:2], 3)} m: {exception}")
                continue
            self.map.picks_since_survey += 1
            self.attempts += 1
            logger.info(
                f"Pick {self.attempts} from survey {target.survey_round} "
                f"({self.map.remaining} triangulated brick(s) still queued, no camera move needed)."
            )
            return target
        return None

    def _is_avoided(self, target: GraspTarget) -> bool:
        centre = np.asarray(target.position, float)[:2]
        return any(
            np.linalg.norm(centre - np.asarray(p, float)[:2]) <= AVOID_RADIUS_M
            for p in (*self.avoid, *self.provisional_avoid)
        )

    # --- for the record ---------------------------------------------------------------------------
    def summary(self) -> Dict:
        """What the survey-once approach actually saved, in the only units that matter: looks taken."""
        return {
            "surveys": self.surveys,
            "camera_moves": self.surveys * len(VIEWPOINTS),
            "attempts": self.attempts,
            "picked": len(self.picked),
            "camera_moves_one_survey_per_pick": self.attempts * len(VIEWPOINTS),
            "queued": 0 if self.map is None else self.map.remaining,
            "given_up_on": len(self.avoid),
        }


def default_keep_out() -> List[Tuple[Sequence[float], float]]:
    """The corner the picked bricks are stacked in, which is not part of the pile.

    Imported late on purpose: submodule_2 imports :class:`GraspTarget` from here, so naming it at
    module level would close the circle. Where the bricks are put down belongs to the module that puts
    them down, and this is the one place that has to know about it.
    """
    from m1.simulation.submodule_2 import DROP_POSITION  # noqa: PLC0415

    # Generous, because the bricks are released from six centimetres up and bounce: the heap spreads.
    # There is room to be generous -- the drop point is a quarter of a metre from the pile's centre and
    # the pile is 85 mm across, so a 110 mm circle round the drop still leaves 65 mm of clear table
    # between the two, well beyond anything that could put a real pile brick inside it.
    return [(np.asarray(DROP_POSITION, float)[:2], 0.11)]


# =================================================================================================
# marking the homework
# =================================================================================================


def score_against_truth(world: W.SimWorld, target: GraspTarget) -> Dict:
    """Compare the chosen brick with the simulator's ground truth. Never used to decide anything.

    Position error is the honest headline number, and the one the physical rig cannot measure at all:
    on the bench there is nothing to compare the perception against except whether the grasp worked.
    """
    brick, distance = world.nearest_brick(target.position[:2])
    if brick is None:
        return {"matched": False}

    pose = brick.pose(world)
    # The part's long axis is its own x or y, whichever the mesh is longer along; the body's yaw then
    # says where that axis points in the world. Compared modulo 180 degrees, because a rectangle's long
    # axis has no head or tail and either end of it is the same grasp.
    x_extent, y_extent = brick.footprint_m
    truth_yaw = float(pose.rotation().ToRollPitchYaw().yaw_angle()) + (0.0 if x_extent >= y_extent else math.pi / 2)
    heading_error = (target.long_axis_heading - truth_yaw + math.pi / 2) % math.pi - math.pi / 2
    true_footprint = sorted(brick.footprint_m)
    report = {
        "matched": True,
        "part": brick.part,
        "true_position": pose.translation().round(5).tolist(),
        "position_error_mm": round(distance * 1000, 2),
        # These lego meshes are modelled with their origin on the part's top face (the studs stand
        # above it), so the body's own z *is* the height the perception is trying to measure.
        "height_error_mm": round((target.top_face_z - float(pose.translation()[2])) * 1000, 2),
        "heading_error_deg": round(math.degrees(heading_error), 1),
        "true_footprint_mm": [round(v * 1000, 1) for v in true_footprint],
        "measured_footprint_mm": [round(target.width * 1000, 1), round(target.length * 1000, 1)],
    }
    logger.info(
        f"Ground truth: the nearest real brick is {brick.part}, {report['position_error_mm']:.1f} mm from where "
        f"the perception put it; its long axis is {abs(report['heading_error_deg']):.1f} deg off, and its "
        f"footprint is {report['true_footprint_mm']} mm against the measured {report['measured_footprint_mm']} mm."
    )
    return report
