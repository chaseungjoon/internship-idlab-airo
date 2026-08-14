"""m1 submodule 1 (physical): look at the pile from two viewpoints, pick a brick, stand over it.

This is what used to be submodule_3 (*which* brick) and submodule_1 (*where* it is, and go there) run
as two processes passing a JSON file between them. They are one step now, exactly as they already are
in :mod:`m1.simulation.submodule_1` -- read the two files side by side and they say the same things in
the same order. That is the point: the simulation is where this logic is exercised a hundred times an
afternoon, and it is only worth trusting on the bench if it is the same logic.

**The perception is shared, not mirrored.** Both stacks call
:func:`m1.physical.submodule_3.analyse_pile` -- the same function, thresholds and scoring. What this
module adds is the part that needs two views:

* **Agreement.** Each viewpoint ranks the pile on its own. A brick both views find, in the same place,
  and both rank as graspable is a much better bet than the top of either view's list alone -- a brick
  half-hidden behind another from one side usually is not from the other, and a region only one view
  believes in is usually a segmentation accident.
* **A position.** Two ways to turn lines of sight into a point, and both are computed every time:
  triangulating the two rays against each other, and projecting each onto the table plane raised by
  one brick height.

**Which of the two is used is the one real difference from the simulation.** There, the hand-eye
transform is exact, so triangulation is the better estimate and the plane projection is the
cross-check. Here it is the other way round, and not by a little: both rays start at the camera
centre, which the hand-eye calibration gets wrong by centimetres, so their intersection is wrong by
centimetres *in every axis including z*. The ray-plane projection instead takes its z from the table
the arm physically touched, and leaves only x and y carrying the calibration error -- a few
millimetres sideways still grasps a 7.8 mm brick, nine centimetres too high grasps nothing at all.
So :data:`POSITION_SOURCE_PREFERENCE` is ``"plane_projection"`` here and ``triangulation`` there, the
triangulated point is computed anyway, and the gap between the two is printed every run: it is a
direct, honest measure of what the rig's calibration is costing, and the number to watch after a
re-calibration.

**One survey, many picks.** The two viewpoints cost the better part of a minute of arm travel and two
full pile analyses, and paying that for every single brick is most of the cycle time. So
:func:`survey` triangulates *every* brick the two views agree on, not just the winner, and hands back a
:class:`PileMap` ranked best-first; :class:`PileSession` then serves picks out of it without moving the
camera again. The pile is looked at again only when the map runs low -- which is also the moment the
bricks that were occluded at the start have become visible, and get their position measured then.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from airo_camera_toolkit.pinhole_operations import (
    calculate_triangulation_errors,
    multiview_triangulation_midpoint,
)
from airo_camera_toolkit.pinhole_operations.unprojection import unproject_onto_world_z_plane
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import BRICK_HANDOFF_PATH, PREGRASP_HEIGHT  # noqa: E402
from m1.physical import cell as C  # noqa: E402
from m1.physical.submodule_3 import Brick, PileAnalysis, analyse_pile, assign_priorities  # noqa: E402

#: Where the camera is put to look at the pile, and what it looks at. Two viewpoints roughly 30 cm
#: apart across the pile: far enough for real parallax, close enough that both are comfortably inside
#: a UR3e's reach and both see the whole pile. Only used when the matching entry in
#: :data:`m1.physical.cell.VIEWPOINT_JOINT_CONFIGURATIONS` is ``None`` -- on the bench the measured
#: joint configurations are preferred, because they are known to be reachable and comfortable, which a
#: fresh IK solution at the edge of the workspace is not.
VIEWPOINTS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("view 1", (0.24, -0.13, 0.33)),
    ("view 2", (0.24, 0.17, 0.33)),
)
VIEW_TARGET = (C.PILE_CENTER[0], C.PILE_CENTER[1], 0.0)
VIEW_SETTLE_DURATION = 0.4

#: Two views have found the same brick if their base-frame centres are within this of each other. A
#: lego stud is 8 mm from the next one, so anything looser could pair a brick with its neighbour;
#: anything tighter would reject a genuine match on ordinary measurement scatter. Kept a shade looser
#: than the simulation's 6 mm, because here both centres carry the hand-eye calibration's lateral
#: error and the two views do not carry it in the same direction.
MATCH_TOLERANCE_M = 0.008
#: ...and if they agree which way it points. Position alone is not enough: a region that merged two
#: touching bricks can sit within a millimetre of a real one and still be a completely different
#: rectangle, and averaging two long axes tens of degrees apart produces a direction that belongs to
#: neither -- which the jaws then close along, missing the brick entirely.
MATCH_HEADING_TOLERANCE_DEG = 15.0
#: Under this aspect ratio a footprint is square enough that its "long axis" is whichever side the
#: measurement noise favoured, so it is neither compared across views nor averaged between them.
SQUARE_ASPECT_RATIO = 1.25

#: Which of the two position estimates is used. See the module docstring -- this is the one place the
#: bench deliberately disagrees with the simulator, and the reason is the hand-eye calibration.
POSITION_SOURCE_PREFERENCE = "plane_projection"
#: Above this the two lines of sight are missing each other by more than the brick they are aimed at.
#: It does not change the answer here (the projection is used regardless) but it is worth saying out
#: loud, because it means the two views are not looking at the same thing.
MAX_TRIANGULATION_GAP_M = 0.020
#: Above this, the two views' own ray-plane projections disagree about where the brick is by more than
#: the jaws have slack. Their midpoint is still used -- there is nothing better to use -- but a grasp
#: aimed at it should be expected to be off sideways.
MAX_VIEW_DISAGREEMENT_M = 0.010

#: The approach is a staircase straight down over the brick: high, lower, pregrasp. Every leg is a
#: joint-space move, which is not a straight line in the world -- so the one leg that crosses the
#: table is made to end high above the pile, and the legs that come down are pure vertical descents
#: over the same point with the same wrist angle, where joint space and the world agree.
RETRACT_HEIGHT_M = 0.12
APPROACH_HEIGHT_M = 0.06
#: The arm has to actually arrive. Past this the pose it reached is not the pose everything downstream
#: was computed for, and descending from it would put the fingers somewhere nobody planned.
MAX_PREGRASP_ERROR_M = 0.008

#: The jaws are opened this much wider than the brick before the arm goes anywhere near it -- the same
#: margin submodule_2 descends with, so the gripper is already at its approach opening on arrival.
GRIPPER_APPROACH_MARGIN_M = 0.014


# =================================================================================================
# turning pixels into places
# =================================================================================================


def pixel_to_base_ray(
    u: float, v: float, intrinsics_matrix: CameraIntrinsicsMatrixType, X_base_camera: HomogeneousMatrixType
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project pixel ``(u, v)`` to a ray ``(origin, unit direction)`` in the robot base frame.

    Pinhole model: the camera-frame ray is ``K^-1 [u, v, 1]`` in the optical convention (+z forward,
    +y down, the one the RealSense point cloud and airo-mono's unprojection both use), rotated into the
    base frame by the eye-in-hand camera pose. The ray's origin is the camera centre.
    """
    direction_camera = np.linalg.inv(np.asarray(intrinsics_matrix, float)) @ np.array([u, v, 1.0])
    camera_in_base = SE3Container.from_homogeneous_matrix(np.asarray(X_base_camera, float))
    direction_base = camera_in_base.rotation_matrix @ direction_camera
    return camera_in_base.translation, direction_base / np.linalg.norm(direction_base)


#: Below this angle between the two lines of sight there is not enough parallax to triangulate at all
#: and the midpoint solve is numerically meaningless. Two degrees; the viewpoints give about forty-five.
MIN_PARALLAX_DEG = 2.0


def triangulate_pixels(
    pixels: Sequence[Tuple[float, float]],
    intrinsics_matrices: Sequence[CameraIntrinsicsMatrixType],
    camera_poses: Sequence[HomogeneousMatrixType],
) -> Tuple[np.ndarray, float]:
    """Midpoint triangulation of one point seen from several views, and how badly the rays miss.

    airo-mono's :func:`multiview_triangulation_midpoint` does the solve -- it minimises the distance to
    every line of sight at once, which for two views is the midpoint of their mutually closest points.
    The gap returned is the sum of airo-mono's per-ray perpendicular errors, which for two rays *is*
    the distance between those closest points: zero would mean the views agree perfectly about the
    direction to the brick, and anything much larger than the brick means they are not looking at the
    same thing.

    Raises:
        RuntimeError: if the rays are near-parallel, i.e. the viewpoints give too little parallax. The
            midpoint solve inverts a matrix that is singular in exactly that case, so this has to be
            caught before rather than after.
    """
    poses = [np.asarray(pose, float) for pose in camera_poses]
    intrinsics = [np.asarray(matrix, float) for matrix in intrinsics_matrices]
    image_coordinates = np.asarray(pixels, float)

    directions = [
        pixel_to_base_ray(u, v, matrix, pose)[1]
        for (u, v), matrix, pose in zip(image_coordinates, intrinsics, poses)
    ]
    widest = max(
        math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))
        for i, a in enumerate(directions)
        for b in directions[i + 1 :]
    )
    if widest < MIN_PARALLAX_DEG:
        raise RuntimeError(
            f"The lines of sight are only {widest:.2f} degrees apart; these viewpoints have too little "
            "parallax to triangulate. Move them further apart across the pile."
        )

    point = multiview_triangulation_midpoint(poses, intrinsics, image_coordinates)
    errors = calculate_triangulation_errors(poses, intrinsics, image_coordinates, point)
    return np.asarray(point, float), float(sum(errors))


def project_ray_onto_plane(
    ray: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float
) -> np.ndarray:
    """Intersect a base-frame ray with the plane ``z = a*x + b*y + c``.

    The other way to turn one line of sight into a point: instead of a second view, use the fact that
    the height is already known -- the table, raised by one brick. It fixes the answer's z outright and
    leaves only x and y carrying any error, which on the bench is the trade worth making.

    A *tilted* plane, because the tabletop is not exactly square to the robot's base and a tilt of one
    degree is 7 mm across a 40 cm workspace -- more than the fingertip clearance a plate leaves. The
    closed form below is the tilted generalisation of airo-mono's ``unproject_onto_world_z_plane``,
    and reduces to it exactly when ``a = b = 0``; :func:`project_pixel_onto_plane` calls the airo-mono
    version directly in that case.

    Raises:
        RuntimeError: if the ray is near-horizontal, runs parallel to the plane, or the plane is
            behind the camera.
    """
    origin, direction = ray
    if abs(direction[2]) < 0.2:  # more than ~78 degrees off vertical
        raise RuntimeError(
            "The line of sight is almost horizontal, so it barely crosses the table plane and the "
            "projected position would be meaningless. Use viewpoints that look down at the pile."
        )

    denominator = direction[2] - a * direction[0] - b * direction[1]
    if abs(denominator) < 1e-9:
        raise RuntimeError("The line of sight runs parallel to the table plane; it never crosses it.")
    distance = (a * origin[0] + b * origin[1] + c - origin[2]) / denominator
    if distance <= 0:
        raise RuntimeError(
            "The table plane lies behind the camera. Check the table calibration and the hand-eye calibration."
        )
    return origin + distance * direction


def project_pixel_onto_plane(
    u: float,
    v: float,
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    plane: Tuple[float, float, float],
) -> np.ndarray:
    """:func:`project_ray_onto_plane` straight from a pixel, via airo-mono for the level case."""
    a, b, c = plane
    if a == 0.0 and b == 0.0:
        points = unproject_onto_world_z_plane(
            np.asarray([[u, v]], float), np.asarray(intrinsics_matrix, float), np.asarray(X_base_camera, float), c
        )
        return np.asarray(points[0], float)
    return project_ray_onto_plane(pixel_to_base_ray(u, v, intrinsics_matrix, X_base_camera), a, b, c)


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

    What used to travel between the two processes as ``run/brick_handoff.json``, now passed directly
    because the two halves run in one process. :meth:`to_handoff` still writes that file, so the
    standalone ``submodule_2`` command-line path keeps working unchanged.
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
    method_disagreement: float  # between the two methods, across the table
    view_disagreement: float  # between the two views' own projections
    position_source: str

    # where the arm ended up
    pregrasp_pose: Optional[HomogeneousMatrixType] = None
    pregrasp_configuration: Optional[np.ndarray] = None
    approach_width: float = 0.05
    per_view: Dict[str, np.ndarray] = field(default_factory=dict)

    # which look at the pile produced it, and how far apart the two views placed it
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
            "view_disagreement_mm": round(self.view_disagreement * 1000, 2),
            "approach_width": round(self.approach_width, 4),
            "survey_round": self.survey_round,
            "match_distance_mm": round(self.match_distance * 1000, 2),
        }

    def to_handoff(self, path: str = BRICK_HANDOFF_PATH) -> str:
        """Write the handoff file the standalone ``submodule_2`` command line reads.

        Redundant when both halves run in one process -- submodule_2 is handed this object directly --
        and kept because the two-terminal bench workflow is still the one to use when something is
        going wrong and you want to stop between the halves and look.
        """
        payload = {
            "written_at": time.time(),
            "brick_position": [float(self.position[0]), float(self.position[1])],
            "width": float(self.width),
            "length": float(self.length),
            "height": float(self.height),
            "long_axis_heading": float(self.long_axis_heading),
            "closing_heading": float(self.closing_heading),
            "table_z_configured": float(self.table_z),
            "safe_table_z": float(self.table_z),
            "table_z_measured_views": [],
            "colour": self.colour,
            "part_number": None,
            "same_size_parts": [],
            "obstruction": 0.0,
            "view_disagreement": float(self.view_disagreement),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote the brick handoff to {path}.")
        return path


# =================================================================================================
# looking
# =================================================================================================


def observe(
    cell: C.Cell, name: str, eye: Sequence[float], target: Sequence[float] = VIEW_TARGET
) -> ViewResult:
    """Move the camera to the viewpoint and run the pile perception on what it sees.

    Prefers the measured joint configuration in :data:`m1.physical.cell.VIEWPOINT_JOINT_CONFIGURATIONS`
    over solving IK for ``eye``: on the bench a known-good configuration beats a fresh solution at the
    edge of the workspace, and the triangulation does not care how the arm got there -- the camera pose
    it uses is forward kinematics composed with the hand-eye calibration either way.
    """
    configuration = C.VIEWPOINT_JOINT_CONFIGURATIONS.get(name)
    if configuration is not None:
        logger.info(f"{name}: moving to the measured viewpoint configuration {np.round(configuration, 3)} rad ...")
        cell.move_arm_to(configuration)
    else:
        pose = C.look_at_tool_pose(cell, np.asarray(eye, float), np.asarray(target, float))
        q = C.solve_tool_ik(cell, pose)
        if q is None:
            raise RuntimeError(
                f"{name} at {np.round(eye, 3)} m is not reachable. Move the viewpoint closer to the base or "
                "lower, and remember a UR3e only reaches about 66 cm. Alternatively, freedrive the arm to a "
                "good viewpoint and record its joint configuration in cell.VIEWPOINT_JOINT_CONFIGURATIONS."
            )
        logger.info(f"{name}: moving the camera to {np.round(eye, 3)} m, looking at {np.round(target, 3)} m ...")
        cell.move_arm_to(q)
    cell.advance(VIEW_SETTLE_DURATION)

    view = cell.capture(name)
    analysis = analyse_pile(view, cell.table_plane, cell.robot_type)
    # No arm is passed: eight IK calls per brick over the network is a poor way to spend a survey, and
    # reachability is settled anyway by go_to_pregrasp, which solves every pose in the descent before
    # the arm moves at all. A brick that turns out to be unreachable costs the session one skip.
    assign_priorities(analysis.ordered, None, PREGRASP_HEIGHT)
    logger.info(
        f"{name}: {len(analysis.bricks)} brick(s), "
        f"{sum(1 for b in analysis.bricks if b.graspable)} graspable, "
        f"{len(analysis.rejected)} region(s) dropped."
    )
    eye_position = np.asarray(view.X_base_camera, float)[:3, 3]
    return ViewResult(
        name=name, eye=eye_position, joint_configuration=cell.arm_positions(), analysis=analysis
    )


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
            # Elongated bricks have to agree about which way they lie as well as where they are. A pair
            # that agrees on the centre to a millimetre and disagrees on the axis by forty degrees is
            # not one brick seen twice; it is one brick and one segmentation accident centred on it.
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
    """Where the brick is. Both estimates computed, the ray-plane projection used.

    See the module docstring for why that is the other way round from the simulation. Everything the
    triangulation produces is still reported: the point, how badly the rays missed each other, and how
    far the two answers ended up apart across the table. Those numbers are the running measurement of
    the hand-eye calibration's error, and they cost nothing to keep.
    """
    height = 0.5 * (a.height_m + b.height_m)
    plane_a, plane_b, plane_c = plane
    top_face = (plane_a, plane_b, plane_c + height)

    ray_a = pixel_to_base_ray(*a.grasp_pixel, first.analysis.view.intrinsics_matrix, first.analysis.view.X_base_camera)
    ray_b = pixel_to_base_ray(*b.grasp_pixel, second.analysis.view.intrinsics_matrix, second.analysis.view.X_base_camera)
    point_a = project_ray_onto_plane(ray_a, *top_face)
    point_b = project_ray_onto_plane(ray_b, *top_face)
    projected = 0.5 * (point_a + point_b)
    # How far apart the two views put the *brick*, in the plane the grasp happens in. On the bench this
    # is the more useful quality metric of the two: it is measured in the same units and the same
    # direction as the error a grasp actually suffers from.
    view_disagreement = float(np.linalg.norm(point_a[:2] - point_b[:2]))

    try:
        triangulated, gap = triangulate_pixels(
            [a.grasp_pixel, b.grasp_pixel],
            [first.analysis.view.intrinsics_matrix, second.analysis.view.intrinsics_matrix],
            [first.analysis.view.X_base_camera, second.analysis.view.X_base_camera],
        )
    except RuntimeError as exception:
        logger.debug(f"Reference triangulation unavailable: {exception}")
        triangulated, gap = projected, float("nan")

    # Compared in the horizontal plane only, because that is the only part of either point that is
    # ever used: the grasp's z comes from the table plus the measured brick height either way.
    disagreement = float(np.linalg.norm(triangulated[:2] - projected[:2]))

    source = POSITION_SOURCE_PREFERENCE
    if source == "triangulation" and (not math.isfinite(gap) or gap > MAX_TRIANGULATION_GAP_M):
        logger.warning(
            f"The two lines of sight miss each other by {gap * 1000:.1f} mm, so the triangulated point is not "
            "trustworthy; using the ray-plane projection instead."
        )
        source = "plane_projection"

    return {
        "position": triangulated if source == "triangulation" else projected,
        "triangulated": triangulated,
        "plane_projected": projected,
        "gap": gap,
        "disagreement": disagreement,
        "view_disagreement": view_disagreement,
        "source": source,
        "height": height,
    }


#: A brick this close to one that has already been tried and failed is treated as that brick. Half a
#: stud: close enough to catch the same part after the failed grasp nudged it, far enough not to
#: condemn its neighbour.
AVOID_RADIUS_M = 0.004


def _matched_pairs(
    first: ViewResult, second: ViewResult, avoid: Sequence[np.ndarray] = ()
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
    """Turn one matched pair into everything submodule_2 needs to grasp it."""
    located = locate(first, second, a, b, plane)
    height = located["height"]
    x, y = float(located["position"][0]), float(located["position"][1])
    table_z = float(plane[2] + plane[0] * x + plane[1] * y)
    position = np.array([x, y, table_z + height])

    # Averaged across the views: two independent measurements of the same rectangle, and the jaws want
    # the more conservative width anyway, which is why the wider of the two is taken for the opening.
    return GraspTarget(
        position=position,
        width=max(a.width_mm, b.width_mm) / 1000.0,
        length=0.5 * (a.length_mm + b.length_mm) / 1000.0,
        height=height,
        long_axis_heading=_combine_headings(a, b),
        table_z=table_z,
        colour=a.colour_name,
        score=0.5 * (a.score + b.score),
        confidence=0.5 * (a.confidence + b.confidence),
        triangulated=located["triangulated"],
        plane_projected=located["plane_projected"],
        triangulation_gap=located["gap"],
        method_disagreement=located["disagreement"],
        view_disagreement=located["view_disagreement"],
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
    """The best brick the two views agree on, located as :func:`locate` describes.

    Ranked on the mean of the two views' scores rather than either alone. The scores already fold in
    fingertip clearance, isolation, how much of the brick's outline borders bare table and whether
    anything stands over it -- all viewpoint-dependent, and a brick that scores well from two
    directions at once is one that really is out on its own.

    This is the one-brick-at-a-time path, kept because it is the smallest thing that demonstrates the
    module. The loop uses :func:`survey`, which locates all of them in the same two looks.
    """
    pairs = _matched_pairs(first, second, avoid)
    if not pairs:
        logger.error(
            "The two views agree on no graspable brick at all. Either the pile is out of frame from one of "
            "them, or nothing in it is currently a safe grasp."
        )
        return None
    logger.info(f"The two views agree on {len(pairs)} graspable brick(s).")

    a, b, match_distance = max(pairs, key=lambda pair: 0.5 * (pair[0].score + pair[1].score))
    target = build_target(first, second, a, b, match_distance, plane)
    _report_target(target, rank=None)
    return target


def _report_target(target: GraspTarget, rank: Optional[int] = None) -> None:
    prefix = "Chosen" if rank is None else f"  {rank:2d}."
    logger.info(
        f"{prefix}: {target.describe()} (mean score {target.score:.3f}, the two views placed it "
        f"{target.match_distance * 1000:.1f} mm apart)."
    )
    logger.info(
        f"      ray-plane projection {np.round(target.plane_projected, 4)} m, the two views "
        f"{target.view_disagreement * 1000:.1f} mm apart; triangulation {np.round(target.triangulated, 4)} m, "
        f"rays missing by {target.triangulation_gap * 1000:.1f} mm and "
        f"{target.method_disagreement * 1000:.1f} mm from the projection across the table. "
        f"Using the {target.position_source.replace('_', ' ')}."
    )
    if target.view_disagreement > MAX_VIEW_DISAGREEMENT_M:
        logger.warning(
            f"The two views disagree by {target.view_disagreement * 1000:.1f} mm, wider than the "
            f"{target.width * 1000:.1f} mm brick. Either they are not looking at the same part, or the "
            "hand-eye calibration's lateral error is large. Expect the grasp to be off sideways."
        )


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
    perpendicular sides as often as not, so the more elongated measurement is simply taken instead: it
    is the one with an axis worth having, and on a square part either answer grasps equally well.
    """
    if min(a.aspect_ratio, b.aspect_ratio) >= SQUARE_ASPECT_RATIO:
        return _average_heading([a.long_axis_heading, b.long_axis_heading])
    return (a if a.aspect_ratio >= b.aspect_ratio else b).long_axis_heading


# =================================================================================================
# surveying the whole pile at once
# =================================================================================================


@dataclass
class PileMap:
    """Every brick one pair of looks agreed on, located, ranked best-first.

    The unit of work is the *survey*, not the brick: the two viewpoints are the expensive part of a
    pick and they see the whole pile, so everything they agree on is located in the same breath and
    kept here. Targets are served out of :attr:`targets` in score order and removed as they go, so
    :attr:`remaining` is the count that decides when the pile is worth looking at again.
    """

    targets: List[GraspTarget]
    views: List[ViewResult]
    survey_round: int
    surveyed_at: float
    #: How many bricks have been *attempted* out of this map. Zero means it still describes the pile
    #: exactly as the camera saw it; anything above zero means the pile has been reached into.
    picks_since_survey: int = 0

    @property
    def remaining(self) -> int:
        return len(self.targets)

    def take_best(self) -> Optional[GraspTarget]:
        return self.targets.pop(0) if self.targets else None

    def discard_near(self, position_xy: Sequence[float], radius: float) -> List[GraspTarget]:
        """Forget the targets within ``radius`` of a point, returning them. Used after each pick.

        A grasp is not a surgical operation: the jaws come down open around the brick and go back up
        with it, and anything close enough to be inside that sweep may have been nudged. Its position
        was measured before the nudge, so dropping it here costs one brick's place in the queue and
        buys it a fresh measurement at the next survey.
        """
        centre = np.asarray(position_xy, float)[:2]
        keep, dropped = [], []
        for target in self.targets:
            (dropped if np.linalg.norm(target.position[:2] - centre) <= radius else keep).append(target)
        self.targets = keep
        return dropped

    def describe(self) -> str:
        return (
            f"survey {self.survey_round}: {self.remaining} brick(s) located and queued, taken at "
            f"t={self.surveyed_at:.0f} s, {self.picks_since_survey} pick(s) made since"
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
    """Locate every brick the two views agree on, ranked by mean score.

    The same matching and the same positioning :func:`choose_target` does, applied to all the pairs
    instead of only the winner. A pair whose lines of sight cannot be turned into a point at all -- one
    running near-horizontal at the edge of the frame, say -- is dropped with a warning rather than
    taking the whole survey down with it.

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
            "The two views agree on no graspable brick at all. Either the pile is empty, it is out of frame "
            "from one of them, or nothing left in it is a safe grasp."
        )
        return pile_map

    logger.success(
        f"Survey {survey_round}: {len(targets)} brick(s) located in one pair of looks -- the next "
        f"{len(targets)} pick(s) need no camera move at all."
    )
    for rank, target in enumerate(targets, start=1):
        _report_target(target, rank)
    return pile_map


def survey(
    cell: C.Cell,
    avoid: Sequence[np.ndarray] = (),
    keep_out: Sequence[Tuple[Sequence[float], float]] = (),
    survey_round: int = 0,
) -> PileMap:
    """Two looks at the pile, and a position for every brick they agree on."""
    views = [observe(cell, name, eye) for name, eye in VIEWPOINTS]
    return build_pile_map(
        views[0],
        views[1],
        cell.table_plane,
        avoid=avoid,
        keep_out=keep_out,
        survey_round=survey_round,
        surveyed_at=cell.elapsed,
    )


# =================================================================================================
# going there
# =================================================================================================


def go_to_pregrasp(cell: C.Cell, target: GraspTarget, pregrasp_height: float = PREGRASP_HEIGHT) -> GraspTarget:
    """Open the jaws, swing over the brick well clear of the pile, then drop to the pregrasp.

    In two moves, not one. A joint-space interpolation from a viewpoint at the edge of the table to a
    pose three centimetres above one brick in the middle of the pile is a straight line in joint space
    and something quite different in the world -- typically a diagonal sweep through everything between
    the two. Going up and over first costs a few seconds and keeps the pile where the perception found
    it.

    The wrist is already turned so the jaws are square to the brick's long axis: this module knows the
    orientation the moment it knows the brick, and turning the wrist here means it happens high above
    the pile rather than centimetres over it.

    Every pose in the descent is solved *before* the arm moves, so a brick that turns out to be
    unreachable costs nothing but the IK calls -- which is what lets :class:`PileSession` skip it and
    take the next-best one instead.
    """
    approach_width = min(target.width + GRIPPER_APPROACH_MARGIN_M, cell.gripper_calibration.max_width)
    target.approach_width = approach_width

    heights = (
        ("retract", RETRACT_HEIGHT_M),
        ("approach", APPROACH_HEIGHT_M),
        ("pregrasp", pregrasp_height),
    )
    poses, configurations = [], []
    heading = target.closing_heading
    for name, height in heights:
        position = np.array([target.position[0], target.position[1], target.top_face_z + height])
        solved = C.solve_top_down_ik(cell, position, heading, approach_width)
        if solved is None:
            raise RuntimeError(
                f"No reachable straight-down {name} pose at {np.round(position, 3)} m with the jaws along "
                f"{math.degrees(heading):.0f} deg. This brick is at the edge of the arm's workspace; the "
                "next-best candidate would have to be tried instead."
            )
        pose, q, heading = solved  # the whole descent keeps the yaw the first leg found reachable
        poses.append(pose)
        configurations.append(q)

    warning = C.reach_warning(cell, target.position)
    if warning:
        logger.warning(warning)

    if cell.gripper is not None:
        cell.move_gripper_to_width(approach_width)
        logger.info(f"Jaws opened to {approach_width * 1000:.1f} mm for a {target.width * 1000:.1f} mm brick.")

    # Via home first. The leg that crosses the table is the only one that can sweep the arm through the
    # pile, and it starts from a viewpoint pose out at the edge of the workspace with the wrist tilted
    # over -- the worst possible starting point for a joint-space straight line. Home is elbow up and
    # central, so both halves of the journey stay high and neither has far to go.
    logger.info("Retracting to the home configuration before crossing the table ...")
    cell.move_arm_to(C.HOME_CONFIGURATION)
    logger.info(
        f"Swinging over the brick at {RETRACT_HEIGHT_M * 100:.0f} cm, jaws already square to its long axis, "
        f"then straight down to the pregrasp {pregrasp_height * 100:.0f} cm above its top face ..."
    )
    for q in configurations:
        cell.move_arm_to(q)

    cell.advance(0.2)
    target.pregrasp_pose = poses[-1]
    target.pregrasp_configuration = cell.arm_positions()

    commanded = np.asarray(poses[-1], float)[:3, 3]
    reached = np.asarray(cell.tcp_pose(approach_width), float)[:3, 3]
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
    cell: C.Cell,
    pregrasp_height: float = PREGRASP_HEIGHT,
    avoid: Sequence[np.ndarray] = (),
) -> Tuple[GraspTarget, List[ViewResult]]:
    """The whole of submodule_1: two looks, one decision, one pregrasp.

    Returns the target for submodule_2 and both views, so a notebook can draw what was seen. Pass the
    positions of bricks that have already been tried and dropped as ``avoid``.
    """
    views = [observe(cell, name, eye) for name, eye in VIEWPOINTS]
    target = choose_target(views[0], views[1], cell.table_plane, avoid)
    if target is None:
        raise RuntimeError(
            "No brick to grasp. Every region either failed the confidence test, was too tightly packed for "
            "a fingertip, or was seen by only one of the two views."
        )
    return go_to_pregrasp(cell, target, pregrasp_height), views


# =================================================================================================
# emptying the pile: one survey, many picks
# =================================================================================================

#: Look at the pile again once fewer than this many located bricks are left queued. At 2 the last
#: brick of a survey is never picked from a map with nothing behind it, so the arm is never left
#: standing over an empty queue -- and the fresh look happens while there is still a known-good target
#: in hand to fall back on. Raise it to re-survey more often (safer positions, slower); drop it to 1
#: to squeeze every last brick out of each survey.
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

    Line for line the simulation's :class:`m1.simulation.submodule_1.PileSession`, and deliberately so
    -- it is the same decision procedure, and the simulator is where it gets exercised.

    * The first survey locates every brick both views can see and queues them by score.
    * Each :meth:`next_target` serves the best one left, straight to the pregrasp, no camera move.
    * Bricks close enough to the one just picked to have been knocked by the jaws are dropped from the
      queue rather than trusted (:meth:`PileMap.discard_near`).
    * When the queue falls below :data:`RESURVEY_WHEN_REMAINING_BELOW`, the pile is looked at again.
      That is also when the bricks that were buried at the start are finally on top and visible, so the
      occluded ones get located exactly when it is worth doing.

    **What a cached position risks, and what happens then.** A stale position means the fingers close
    where the brick used to be. submodule_2 catches that on the width check and the Robotiq's own
    object-detection flag, opens, and retreats -- so the failure mode is a wasted pick, not a
    collision. The session then treats the failure according to how much it can blame the map: a brick
    that failed on a *fresh* target (nothing had been touched since the survey) is a genuinely bad
    grasp and is avoided for good, while one that failed on a target the pile had since been reached
    into gets its position remeasured at the next survey and another chance. Three failures in a row
    force a survey outright.

    Usage::

        session = PileSession(cell)
        while (target := session.next_target()) is not None:
            result = submodule_2.run(cell, target)
            if result.success:
                submodule_2.place(cell, target)
            session.record(target, result.success)
    """

    def __init__(
        self,
        cell: C.Cell,
        pregrasp_height: float = PREGRASP_HEIGHT,
        resurvey_below: int = RESURVEY_WHEN_REMAINING_BELOW,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        keep_out: Optional[Sequence[Tuple[Sequence[float], float]]] = None,
    ) -> None:
        self.cell = cell
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
        radius = 0.5 * target.approach_width + FINGER_DISTURBANCE_MARGIN_M
        disturbed = self.map.discard_near(centre, radius) if self.map is not None else []
        if disturbed:
            logger.info(
                f"Dropping {len(disturbed)} queued brick(s) within {radius * 1000:.0f} mm of the jaws; they will "
                "be located again at the next survey."
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
                f"Re-surveying: {self.map.remaining} located brick(s) left, below the {self.resurvey_below} "
                "threshold. Bricks that were occluded at the start get located now."
            )
        # The provisional list only ever meant "measured on a pile that has since moved", and this is
        # the survey that moves it back.
        self.provisional_avoid.clear()
        self._consecutive_failures = 0
        self._force_survey = False
        self.map = survey(self.cell, avoid=self.avoid, keep_out=self.keep_out, survey_round=self.surveys)

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
                go_to_pregrasp(self.cell, target, self.pregrasp_height)
            except RuntimeError as exception:
                logger.warning(f"Cannot stand over the brick at {np.round(target.position[:2], 3)} m: {exception}")
                continue
            self.map.picks_since_survey += 1
            self.attempts += 1
            logger.info(
                f"Pick {self.attempts} from survey {target.survey_round} "
                f"({self.map.remaining} located brick(s) still queued, no camera move needed)."
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
    from m1.physical.submodule_2 import DROP_POSITION  # noqa: PLC0415

    # Generous, because the bricks are released from a few centimetres up and bounce: the heap spreads.
    return [(np.asarray(DROP_POSITION, float)[:2], 0.11)]


# =================================================================================================
# CLI
# =================================================================================================


def main() -> None:
    """Perceive the pile, choose a brick, and stand over it. The standalone half of the pipeline.

    Kept for the two-terminal bench workflow: run this, look at what it chose, then run submodule_2 in
    the other terminal. The notebook (``main.ipynb``) runs both halves in one process and does not go
    through here or through the handoff file.
    """
    import click

    from config import DEFAULT_CALIBRATION_DIR, DEFAULT_CAMERA_RESOLUTION, SUPPORTED_ROBOT_TYPES

    @click.command()
    @click.option("--robot-type", type=click.Choice(SUPPORTED_ROBOT_TYPES), default="ur3e", show_default=True)
    @click.option("--ip-address", default=None, help="Robot controller IP. Defaults per robot type.")
    @click.option("--calibration-path", default=DEFAULT_CALIBRATION_DIR, show_default=True)
    @click.option("--camera-resolution", type=click.Choice(list(C.CAMERA_RESOLUTIONS)), default=DEFAULT_CAMERA_RESOLUTION, show_default=True)
    @click.option("--speed-ratio", type=click.IntRange(1, 100), default=C.DEFAULT_SPEED_RATIO, show_default=True)
    @click.option("--pregrasp-height", type=click.FloatRange(0.0, 0.05, min_open=True), default=PREGRASP_HEIGHT, show_default=True)
    @click.option("--table-z", type=float, default=None, help="Level table height, overriding the touched-off plane.")
    @click.option("--handoff-path", default=BRICK_HANDOFF_PATH, show_default=True)
    def command(
        robot_type: str,
        ip_address: Optional[str],
        calibration_path: str,
        camera_resolution: str,
        speed_ratio: int,
        pregrasp_height: float,
        table_z: Optional[float],
        handoff_path: str,
    ) -> None:
        with C.build_cell(
            robot_type=robot_type,
            ip_address=ip_address,
            calibration_path=calibration_path,
            camera_resolution=camera_resolution,
            speed_ratio=speed_ratio,
            table_z=table_z,
            with_gripper=True,
        ) as cell:
            target, _ = run(cell, pregrasp_height)
            target.to_handoff(handoff_path)
            logger.success(f"Standing over: {target.describe()}")
            logger.info("Now run `python src/m1/physical/submodule_2.py` to grasp it.")

    command()


if __name__ == "__main__":
    main()
