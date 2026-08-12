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
    """
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
    if not pairs:
        logger.error(
            "The two views agree on no graspable brick at all. Either the pile is out of frame from one "
            "of them, or nothing in it is currently a safe grasp."
        )
        return None
    logger.info(f"The two views agree on {len(pairs)} graspable brick(s).")

    a, b, match_distance = max(pairs, key=lambda pair: 0.5 * (pair[0].score + pair[1].score))
    located = locate(first, second, a, b, plane)
    height = located["height"]
    position = np.array([located["position"][0], located["position"][1], plane[2] + plane[0] * located["position"][0] + plane[1] * located["position"][1] + height])

    # Averaged across the views: two independent measurements of the same rectangle, and the jaws want
    # the more conservative width anyway, which is why the wider of the two is taken for the opening.
    long_axis = _combine_headings(a, b)
    target = GraspTarget(
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
    )
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
