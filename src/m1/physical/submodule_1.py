"""m1 submodule 1 (physical): look at the pile from two viewpoints, pick a brick, stand over it.

**Colour, not depth.** Which regions are bricks comes from :mod:`prototypes.pile_perception`, the
robot-free RGB pipeline. The depth stream is captured with each frame and deliberately ignored.

That is a deliberate reversal. The depth path measures every height against a plane, so that one
assumption decides everything: where the camera's idea of the tabletop and the arm's disagree by a
constant offset -- a board standing on another table, a hand-eye calibration still settling -- bare
wood reads as standing proud, the table model has nothing to seed from, and the regions reported are
wood rather than bricks. Colour asks only whether a pixel looks like the tabletop, which no
calibration error can change.

**What it gives up, and buys back.** Height. There is no depth to measure it from, so a brick's height
comes from :mod:`common.lego_catalog` when its footprint matches a part and from
``config.FALLBACK_BRICK_HEIGHT`` when it does not. That matters more than it sounds: the height sets
the plane the outline is projected onto, so it feeds back into the position -- which is why
:func:`measure` measures the footprint twice, once at the assumed height and once at the matched one.

**How two views become one position.** Without depth neither view knows where anything is on its own,
so each view's ray is crossed with the table plane first and the *matching* runs on those base-frame
positions. Triangulation comes afterwards, on the pairs that produces.

* **View 1 decides.** Which brick is grasped and everything measured about it comes from view 1 alone.
  On this small table the two viewpoints look across the pile at shallow, very different angles, so
  with hand-eye error a brick's two centres routinely land further apart than the brick is wide.
  Demanding agreement filters out the whole pile, not the segmentation accidents.
* **Both views measure.** Triangulation is the one thing a single view cannot do, so each of view 1's
  bricks is paired where possible with view 2's region in the same place.

:data:`POSITION_SOURCE_PREFERENCE` picks which of the two the arm is actually sent to. Both are always
computed and the gap between them printed every run, because that gap is a direct measure of what the
rig's calibration is costing. Be warned which way the evidence points: both rays start at the camera
centre, so a hand-eye translation error moves their intersection in *every* axis including z, while
the ray-plane projection takes z from the table the arm physically touched and leaves only x and y
carrying it. On a rig whose hand-eye is still settling, ``"plane_projection"`` is usually the better
call -- it is one word to change here, and ``--position-source`` on the command line.

**One survey, many picks.** The two viewpoints cost most of the cycle time, so :func:`survey` locates
*every* brick view 1 found graspable and returns a :class:`PileMap` ranked best-first;
:class:`PileSession` serves picks from it without moving the camera. The pile is looked at again only
when the map runs low -- which is also when the bricks occluded at the start have become visible.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
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
from common.config import BRICK_HANDOFF_PATH, PREGRASP_HEIGHT  # noqa: E402
from m1.physical import cell as C  # noqa: E402
import prototypes.pile_perception as colour  # noqa: E402
from common.config import FALLBACK_BRICK_HEIGHT  # noqa: E402
from common.lego_catalog import load_catalog, match_footprint  # noqa: E402
from m1.physical.submodule_3 import PileView, project_pixels_onto_plane  # noqa: E402

Brick = colour.Brick

#: Where the camera looks from, and at what. Two viewpoints ~30 cm apart: enough parallax, both
#: inside a UR3e's reach, both seeing the whole pile. Only used when the matching entry in
#: :data:`m1.physical.cell.VIEWPOINT_JOINT_CONFIGURATIONS` is ``None``.
VIEWPOINTS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("view 1", (0.24, -0.13, 0.33)),
    ("view 2", (0.24, 0.17, 0.33)),
)
VIEW_TARGET = (C.PILE_CENTER[0], C.PILE_CENTER[1], 0.0)
VIEW_SETTLE_DURATION = 0.4

#: Two views are looking at the same brick if their base-frame centres are within this. A stud is
#: 8 mm, so anything looser could pair a brick with its neighbour. Looser than the simulation's 6 mm
#: because both centres carry hand-eye error, in different directions. This no longer decides whether
#: a brick is grasped -- a failed match only costs the triangulated cross-check.
MATCH_TOLERANCE_M = 0.008
#: ...and if they agree which way it points. A region that merged two touching bricks can sit within
#: a millimetre of a real one and still be a completely different rectangle.
MATCH_HEADING_TOLERANCE_DEG = 15.0
#: Under this aspect ratio the "long axis" is whichever side the noise favoured, so it is not compared.
SQUARE_ASPECT_RATIO = 1.25

#: Which position estimate is used. See the module docstring: the one place the bench deliberately
#: disagrees with the simulator.
POSITION_SOURCE_PREFERENCE = "triangulation"
#: Confidence below which a colour region is never offered to the arm. The colour pipeline's own bar.
MIN_CONFIDENCE = colour.PRIORITY_MIN_CONFIDENCE
#: Height assumed on the first projection pass, before the catalog has had a look. Only the footprint
#: measured at this height matters, and a 6 mm error there moves it by well under a millimetre.
ASSUMED_HEIGHT_M = FALLBACK_BRICK_HEIGHT
#: How far a measured footprint may sit from a catalog part's and still be called that part.
CATALOG_TOLERANCE_M = 0.004

#: Fingertip engagement -- height less the clearance kept above the table -- at which the grip-depth
#: preference saturates. Roughly two thirds of a full brick's side wall.
GRIP_DEPTH_SATURATION_M = 0.006
#: How much of the final ordering is grip depth rather than the colour pipeline's own score.
#:
#: This term exists because the colour score cannot contain it. ``SCORE_WEIGHTS`` rates clearance,
#: isolation, exposure, visibility, width fit, size and confidence -- and a plate lying alone on bare
#: table scores full marks on every one of them. It is also 3.2 mm tall, which leaves the pads under
#: two millimetres of purchase and turns the close into a shove, and nothing colour can see says so.
#: The depth pipeline had a ``grip_depth`` term for exactly this; here the height comes from the
#: catalog instead, but the preference has to be applied just the same or the ranking sends the arm at
#: the hardest grasps in the pile first.
GRIP_DEPTH_WEIGHT = 0.35
#: Clearance the fingertips keep above the table, mirroring submodule_2's descent cap. Subtracted from
#: the height to get the side wall actually reachable.
FINGERTIP_CLEARANCE_M = 0.0015

_UNLOADED = object()
_CATALOG: Optional[Dict] = _UNLOADED  # type: ignore[assignment]
#: Above this the two lines of sight miss each other by more than the brick. Does not change the
#: answer, but means the views are not looking at the same thing.
MAX_TRIANGULATION_GAP_M = 0.020
#: Above this the two views' own projections disagree by more than the jaws have slack. Changes
#: nothing -- view 1's projection is used either way -- but it is one of the only calibration
#: measurements this rig can make without ground truth.
MAX_VIEW_DISAGREEMENT_M = 0.010

#: The approach is a staircase straight down over the brick: high, lower, pregrasp. Joint-space moves
#: are not straight lines in the world, so the leg that crosses the table ends high above the pile and
#: the rest are pure vertical descents, where joint space and the world agree.
RETRACT_HEIGHT_M = 0.12
APPROACH_HEIGHT_M = 0.06
#: Retract heights to try, highest first. **Standing higher over a brick means reaching less far**:
#: the wrist sits ~32 cm above the fingertips, so every centimetre they rise pushes it a centimetre
#: further from a shoulder that only reaches 46 cm. Taking the highest that solves keeps the intent --
#: staying clear of the pile while crossing -- without throwing bricks away for it. The last entry
#: equals :data:`APPROACH_HEIGHT_M`, below which a retract has stopped being one.
RETRACT_HEIGHTS_M = (RETRACT_HEIGHT_M, 0.10, 0.08, APPROACH_HEIGHT_M)
#: The arm has to actually arrive; past this, downstream poses were computed for somewhere else.
MAX_PREGRASP_ERROR_M = 0.008

#: Jaws opened this much wider than the brick before approaching -- the margin submodule_2 descends
#: with, so the gripper arrives already at its approach opening. Half of it is the grasp's entire
#: lateral error budget, per side.
#:
#: At 14 mm a 7 mm sideways error put a fingertip *on* the brick rather than around it, which is a
#: collision and not a miss -- and 7 mm is inside what the survey's viewing angle alone can cost. 24 mm
#: gives 12 mm a side. It is not free: wider jaws need more room from the neighbours, so bricks that
#: were graspable at 14 mm can fail the clearance test at 24 mm. That is the right trade while the
#: position is the weak link -- a brick skipped for lack of room costs a pick, a fingertip driven into
#: a brick costs the pile.
GRIPPER_APPROACH_MARGIN_M = 0.024

# --- looking again, from straight above -----------------------------------------------------------

#: Camera heights above the brick's top face for the re-look, first that solves wins.
#:
#: Every position here is an outline projected onto the plane of its own top face, so a height error
#: slides the answer sideways by ``height error x tan(angle off vertical)``. From a survey viewpoint
#: that tangent is 0.4-0.6; from directly above it is zero. That is the one error term geometry
#: removes for free. With the camera ~18 cm behind the fingertips, 30 cm of camera height puts the TCP
#: ~12 cm over the brick; the lower entries are for bricks near the edge of the workspace.
NADIR_CAMERA_HEIGHTS_M = (0.30, 0.26, 0.22)
#: A part found this far from the survey's position is taken to be the same part. It must be wider
#: than the error being corrected -- the point of the re-look -- and so is necessarily wider than the
#: 8 mm gap to a neighbour. The size check below carries the other half of the burden.
NADIR_MATCH_TOLERANCE_M = 0.025
#: ...and it has to be about the same size, which is what catches a neighbour inside the tolerance.
NADIR_WIDTH_TOLERANCE_M = 0.006
#: Above this, the re-look and the survey disagree by more than a grasp can absorb. The re-look wins
#: (no lever arm), but this is the number that says the survey geometry is still wrong.
NADIR_CORRECTION_WARNING_M = 0.015


# --- turning pixels into places -------------------------------------------------------------------


def pixel_to_base_ray(
    u: float, v: float, intrinsics_matrix: CameraIntrinsicsMatrixType, X_base_camera: HomogeneousMatrixType
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project pixel ``(u, v)`` to a ray ``(origin, unit direction)`` in the robot base frame.

    Pinhole, optical convention (+z forward, +y down), rotated into the base frame by the eye-in-hand
    camera pose. The origin is the camera centre.
    """
    direction_camera = np.linalg.inv(np.asarray(intrinsics_matrix, float)) @ np.array([u, v, 1.0])
    camera_in_base = SE3Container.from_homogeneous_matrix(np.asarray(X_base_camera, float))
    direction_base = camera_in_base.rotation_matrix @ direction_camera
    return camera_in_base.translation, direction_base / np.linalg.norm(direction_base)


#: Below this angle there is too little parallax to triangulate and the midpoint solve is meaningless.
#: Two degrees; the viewpoints give about forty-five.
MIN_PARALLAX_DEG = 2.0


def triangulate_pixels(
    pixels: Sequence[Tuple[float, float]],
    intrinsics_matrices: Sequence[CameraIntrinsicsMatrixType],
    camera_poses: Sequence[HomogeneousMatrixType],
) -> Tuple[np.ndarray, float]:
    """Midpoint triangulation of one point seen from several views, and how badly the rays miss.

    The gap is the sum of airo-mono's per-ray perpendicular errors -- for two rays, the distance between
    their closest points. Much larger than the brick means the views disagree.

    Raises:
        RuntimeError: if the rays are near-parallel, which makes the midpoint solve singular.
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

    Uses the known height -- the table, raised by one brick -- to fix z outright, leaving only x and y
    carrying error. Tilted, because one degree is 7 mm across a 40 cm workspace; reduces to airo-mono's
    ``unproject_onto_world_z_plane`` when ``a = b = 0``.

    Raises:
        RuntimeError: if the ray is near-horizontal, parallel to the plane, or the plane is behind it.
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


# --- what one look at the pile leaves behind ------------------------------------------------------


@dataclass
class ViewResult:
    """One viewpoint: where it looked from, and what the colour pipeline made of it."""

    name: str
    eye: np.ndarray
    joint_configuration: np.ndarray
    view: PileView
    analysis: Dict
    projected: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def bricks(self) -> List[Brick]:
        return list(self.analysis["bricks"])

    @property
    def graspable(self) -> List[Brick]:
        """View 1's shortlist: ranked, confident, and a grasp on its own geometry."""
        return [b for b in self.analysis["ordered"] if b.graspable and b.confidence >= MIN_CONFIDENCE]


@dataclass
class GraspTarget:
    """The brick submodule_2 is to grasp, and everything it needs to do it.

    Passed directly now that both halves run in one process. :meth:`to_handoff` still writes
    ``run/brick_handoff.json`` so the standalone ``submodule_2`` command line keeps working.
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
    view_disagreement: float  # between the two views' own projections; nan when view 2 never saw it
    position_source: str
    #: Whether view 2 found this brick too. False means no triangulation to compare against -- not that
    #: the brick is a worse grasp.
    matched_in_second_view: bool = True
    #: Whether the height came from the depth stream or from ``config.FALLBACK_BRICK_HEIGHT``.
    #:
    #: This decides how much :attr:`position` is worth, not just :attr:`height`: the position is the
    #: outline projected onto the top face's plane, so a height wrong by h puts x and y wrong by h times
    #: the view angle's tangent. A *measured* height also cancels an error in the touched-off plane
    #: exactly, where a guessed one compounds with it. Parts standing on edge are the usual case.
    height_measured: bool = True
    #: How far the straight-down re-look moved this brick from the survey's position, or ``nan`` if it
    #: was never re-looked at. The survey's own lateral error, measured.
    nadir_correction: float = float("nan")

    # where the arm ended up
    pregrasp_pose: Optional[HomogeneousMatrixType] = None
    pregrasp_configuration: Optional[np.ndarray] = None
    approach_width: float = 0.05
    per_view: Dict[str, np.ndarray] = field(default_factory=dict)

    # which look at the pile produced it, and how far apart the two views placed it
    survey_round: int = 0
    match_distance: float = float("nan")  # nan when view 2 has no counterpart to measure against

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
            "matched_in_second_view": self.matched_in_second_view,
            "height_measured": self.height_measured,
            "nadir_correction_mm": round(self.nadir_correction * 1000, 2),
            "approach_width": round(self.approach_width, 4),
            "survey_round": self.survey_round,
            "match_distance_mm": round(self.match_distance * 1000, 2),
        }

    def to_handoff(self, path: str = BRICK_HANDOFF_PATH) -> str:
        """Write the handoff file the standalone ``submodule_2`` command line reads.

        Redundant when both halves run in one process, and kept because the two-terminal workflow is the one
        to use when you want to stop between the halves and look.
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
            # null rather than NaN: NaN is not JSON, and the reader already treats a missing value as
            # "not measured".
            "view_disagreement": float(self.view_disagreement) if math.isfinite(self.view_disagreement) else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote the brick handoff to {path}.")
        return path


# --- looking --------------------------------------------------------------------------------------


def analyse_view(cell: C.Cell, view: PileView, name: str) -> ViewResult:
    """Run the colour pipeline on a frame already captured, and wrap it as a :class:`ViewResult`.

    Split out of :func:`observe` so submodule_2 can re-measure a brick from the pregrasp through the
    same code the survey uses. One perception, called from both halves.
    """
    return ViewResult(
        name=name,
        eye=np.asarray(view.X_base_camera, float)[:3, 3],
        joint_configuration=cell.arm_positions(),
        view=view,
        analysis=colour.analyse(cv2.cvtColor(np.asarray(view.image_rgb), cv2.COLOR_RGB2BGR)),
    )


def observe(cell: C.Cell, name: str, eye: Sequence[float], target: Sequence[float] = VIEW_TARGET) -> ViewResult:
    """Move the camera to the viewpoint and run the colour pipeline on what it sees.

    The depth map is captured with the frame and deliberately ignored. Colour asks only whether a pixel
    looks like the tabletop, which no calibration error can change -- where a height, measured against a
    plane, is only ever as good as that plane.
    """
    configuration = C.VIEWPOINT_JOINT_CONFIGURATIONS.get(name)
    if configuration is not None:
        logger.info(f"{name}: moving to the measured viewpoint configuration ...")
        cell.move_arm_to(configuration)
    else:
        pose = C.look_at_tool_pose(cell, np.asarray(eye, float), np.asarray(target, float))
        q = C.solve_tool_ik(cell, pose)
        if q is None:
            raise RuntimeError(
                f"{name} at {np.round(eye, 3)} m is not reachable. Move the viewpoint closer to the base or "
                "lower, or freedrive the arm to a good one and record its joint configuration in "
                "cell.VIEWPOINT_JOINT_CONFIGURATIONS."
            )
        cell.move_arm_to(q)
    cell.advance(VIEW_SETTLE_DURATION)

    result = analyse_view(cell, cell.capture(name), name)
    analysis = result.analysis
    logger.info(
        f"{name}: {len(result.bricks)} region(s), {len(result.graspable)} graspable and confident, "
        f"{len(analysis['rejected'])} dropped."
    )
    return result


# --- turning a colour region into a place -----------------------------------------------------------


def catalog() -> Optional[Dict]:
    """The lego catalog, loaded once. ``None`` if it cannot be built, which costs heights, not picks."""
    global _CATALOG
    if _CATALOG is _UNLOADED:
        try:
            _CATALOG = load_catalog()
        except Exception as exception:  # noqa: BLE001 - a missing catalog must not stop a pick
            logger.warning(
                f"No lego catalog ({exception}); every height falls back to "
                f"{FALLBACK_BRICK_HEIGHT * 1000:.1f} mm."
            )
            _CATALOG = None
    return _CATALOG


def grasp_pixel(brick: Brick, scale: float) -> Tuple[float, float]:
    """The region's centre in the *source* frame's pixels.

    The colour pipeline works at its own working resolution and the intrinsics describe the frame as
    captured; mixing the two silently puts every ray in the wrong place.
    """
    return float(brick.obb_center_px[0] * scale), float(brick.obb_center_px[1] * scale)


def obb_corners(brick: Brick, scale: float) -> np.ndarray:
    """The region's oriented bounding box as four source-frame pixels."""
    box = cv2.boxPoints(
        ((brick.obb_center_px[0], brick.obb_center_px[1]),
         (brick.obb_size_px[0], brick.obb_size_px[1]),
         float(brick.obb_angle_deg))
    )
    return np.asarray(box, dtype=float) * scale


def footprint_on_plane(
    brick: Brick, result: ViewResult, plane: Tuple[float, float, float], height: float
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """Project the region's box onto the plane of its own top face and measure it there.

    Returns ``(centre_xy, width_m, length_m, long_axis_heading)``, or ``None`` when the box does not
    cross the plane -- which means the region is not on the table.

    Fitted in millimetres, not metres: ``cv2.minAreaRect`` works in float32 and a table-frame coordinate
    is around 0.3, so an 8 mm side would be quantised by the format rather than by the pixels.
    """
    a, b, c = plane
    corners = obb_corners(brick, result.analysis["scale"])
    projected = project_pixels_onto_plane(
        corners, result.view.intrinsics_matrix, result.view.X_base_camera, (a, b, c + height)
    )
    if projected is None or len(projected) < 3:
        return None

    (cx, cy), (side_a, side_b), angle_deg = cv2.minAreaRect((projected[:, :2] * 1000.0).astype(np.float32))
    angle = math.radians(angle_deg)
    if side_a >= side_b:
        length, width, heading = side_a, side_b, angle
    else:
        length, width, heading = side_b, side_a, angle + math.pi / 2
    return np.array([cx / 1000.0, cy / 1000.0]), width / 1000.0, length / 1000.0, heading


def measure(brick: Brick, result: ViewResult, plane: Tuple[float, float, float]) -> Optional[Tuple]:
    """Footprint, catalog part and height for one region, refined once through the matched height.

    This is what colour gives up and has to buy back. The outline is projected onto the plane at
    ``table + height`` and the height is what the footprint is matched against, so the two are mutually
    dependent: the first pass runs at :data:`ASSUMED_HEIGHT_M` and the second at whatever the catalog
    said. It converges immediately -- a 6 mm error in the assumed height moves the footprint by well
    under a millimetre.
    """
    parts = catalog()
    height = ASSUMED_HEIGHT_M
    measured = footprint_on_plane(brick, result, plane, height)
    if measured is None:
        return None
    part = None
    for _ in range(2):
        centre, width, length, heading = measured
        part = None
        if parts:
            candidates = match_footprint(parts, width, length, CATALOG_TOLERANCE_M)
            part = candidates[0].part if candidates else None
        new_height = part.height if part is not None else FALLBACK_BRICK_HEIGHT
        if abs(new_height - height) < 1e-6:
            break
        height = new_height
        refined = footprint_on_plane(brick, result, plane, height)
        if refined is None:
            break
        measured = refined
    centre, width, length, heading = measured
    return centre, width, length, heading, height, part


def project_all(result: ViewResult, plane: Tuple[float, float, float]) -> Dict[int, Tuple]:
    """Every region this view found, measured on the table plane, keyed by region index.

    This is what makes the two views comparable at all. Without depth neither view knows where anything
    is on its own, so each one's ray is crossed with the table plane first -- and that is what the
    matching runs on. The triangulation comes afterwards, on the pairs it produces.
    """
    measured: Dict[int, Tuple] = {}
    for brick in result.bricks:
        outcome = measure(brick, result, plane)
        if outcome is None:
            logger.debug(f"{result.name}: region #{brick.index} does not cross the table plane; skipping it.")
            continue
        measured[brick.index] = outcome
        result.projected[brick.index] = outcome[0]
    return measured


def match_across_views(
    first: ViewResult,
    second: ViewResult,
    first_measured: Dict[int, Tuple],
    second_measured: Dict[int, Tuple],
) -> List[Tuple[Brick, Optional[Brick], float]]:
    """Every graspable brick *view 1* found, each with view 2's counterpart where it has one.

    View 1's list is the list; nothing view 2 does adds to or removes from it. On this small table the
    two views disagree far too often for agreement to be a usable filter. View 2 is asked only *where
    else is this brick on your sensor*, so it is matched against every region it accepted rather than
    only the ones it would grasp. Greedy nearest-neighbour over the pairs sorted by distance, so the
    least ambiguous claim their partners first; a leftover comes back with ``None`` and a ``nan``.
    """
    candidates = []
    for a in first.graspable:
        if a.index not in first_measured:
            continue
        a_centre, a_width, a_length, a_heading, _, _ = first_measured[a.index]
        for b in second.bricks:
            if b.index not in second_measured:
                continue
            b_centre, b_width, b_length, b_heading, _, _ = second_measured[b.index]
            distance = float(np.linalg.norm(a_centre - b_centre))
            if distance > MATCH_TOLERANCE_M:
                continue
            if min(a_length / max(a_width, 1e-6), b_length / max(b_width, 1e-6)) > SQUARE_ASPECT_RATIO:
                # Elongated bricks have to agree which way they lie as well as where they are.
                if abs(math.degrees(_heading_difference(a_heading, b_heading))) > MATCH_HEADING_TOLERANCE_DEG:
                    continue
            candidates.append((distance, a, b))

    pairs: List[Tuple[Brick, Optional[Brick], float]] = []
    taken_a, taken_b = set(), set()
    for distance, a, b in sorted(candidates, key=lambda item: item[0]):
        if a.index in taken_a or b.index in taken_b:
            continue
        taken_a.add(a.index)
        taken_b.add(b.index)
        pairs.append((a, b, distance))
    for a in first.graspable:
        if a.index in first_measured and a.index not in taken_a:
            pairs.append((a, None, float("nan")))
    return pairs


def locate(
    first: ViewResult,
    second: ViewResult,
    a: Brick,
    b: Optional[Brick],
    first_measured: Dict[int, Tuple],
    second_measured: Dict[int, Tuple],
    plane: Tuple[float, float, float],
) -> Dict:
    """Where the brick is, by triangulation and by ray-plane projection, with both reported.

    ``b`` is ``None`` when view 2 has no counterpart: there is then no second ray, the two-view numbers
    come back ``nan``, and the position falls back to view 1's projection whatever
    :data:`POSITION_SOURCE_PREFERENCE` says -- one ray cannot be triangulated.
    """
    a_centre, width, length, heading, height, part = first_measured[a.index]
    top_face = (plane[0], plane[1], plane[2] + height)

    a_pixel = grasp_pixel(a, first.analysis["scale"])
    a_ray = pixel_to_base_ray(*a_pixel, first.view.intrinsics_matrix, first.view.X_base_camera)
    projected = project_ray_onto_plane(a_ray, *top_face)

    triangulated = np.array([np.nan] * 3)
    gap = float("nan")
    view_disagreement = float("nan")
    if b is not None:
        b_pixel = grasp_pixel(b, second.analysis["scale"])
        try:
            triangulated, gap = triangulate_pixels(
                [a_pixel, b_pixel],
                [first.view.intrinsics_matrix, second.view.intrinsics_matrix],
                [first.view.X_base_camera, second.view.X_base_camera],
            )
        except RuntimeError as exception:
            logger.warning(f"Could not triangulate region #{a.index}: {exception}")
        view_disagreement = float(np.linalg.norm(a_centre[:2] - second_measured[b.index][0][:2]))

    source = POSITION_SOURCE_PREFERENCE
    if source == "triangulation" and not np.all(np.isfinite(triangulated)):
        source = "plane_projection"
    position = triangulated if source == "triangulation" else projected
    disagreement = (
        float(np.linalg.norm(triangulated[:2] - projected[:2])) if np.all(np.isfinite(triangulated)) else float("nan")
    )
    return {
        "position": np.asarray(position, float),
        "triangulated": triangulated,
        "plane_projected": np.asarray(projected, float),
        "gap": gap,
        "disagreement": disagreement,
        "view_disagreement": view_disagreement,
        "source": source,
        "width": width,
        "length": length,
        "heading": heading,
        "height": height,
        "part": part,
        "centre": a_centre,
        "table_z": float(plane[2] + plane[0] * a_centre[0] + plane[1] * a_centre[1]),
    }


#: A brick this close to one already tried and failed is treated as that brick. Half a stud: close
#: enough to catch the same part after a failed grasp nudged it, far enough not to condemn a neighbour.
AVOID_RADIUS_M = 0.004


def _matched_pairs(
    first: ViewResult,
    second: ViewResult,
    plane: Tuple[float, float, float],
    avoid: Sequence[np.ndarray] = (),
    keep_out: Sequence[Tuple[Sequence[float], float]] = (),
) -> Tuple[List[Tuple[Brick, Optional[Brick], float]], Dict[int, Tuple], Dict[int, Tuple]]:
    """The bricks view 1 found paired with view 2's, minus anything already tried or out of bounds.

    Filtering happens on the *measured* base-frame centre rather than on anything in pixels, because
    ``avoid`` and ``keep_out`` are both statements about the table, not about the frame.
    """
    first_measured = project_all(first, plane)
    second_measured = project_all(second, plane)
    pairs = match_across_views(first, second, first_measured, second_measured)

    def centre_of(pair) -> np.ndarray:
        return first_measured[pair[0].index][0]

    if avoid:
        before = len(pairs)
        pairs = [
            p for p in pairs
            if all(np.linalg.norm(centre_of(p) - np.asarray(a, float)[:2]) > AVOID_RADIUS_M for a in avoid)
        ]
        if len(pairs) < before:
            logger.info(f"Skipping {before - len(pairs)} brick(s) that were already tried and not picked up.")
    if keep_out:
        before = len(pairs)
        pairs = [
            p for p in pairs
            if all(np.linalg.norm(centre_of(p) - np.asarray(c, float)[:2]) > r for c, r in keep_out)
        ]
        if len(pairs) < before:
            logger.info(f"Ignoring {before - len(pairs)} brick(s) sitting in a keep-out area, not in the pile.")
    return pairs, first_measured, second_measured


def build_target(
    first: ViewResult,
    second: ViewResult,
    a: Brick,
    b: Optional[Brick],
    match_distance: float,
    located: Dict,
    survey_round: int = 0,
) -> GraspTarget:
    """Turn one of view 1's regions into everything submodule_2 needs to grasp it.

    Every measurement is view 1's. Nothing is averaged with view 2, because a view whose centre for
    this brick can be a brick's width away is not a second measurement of the same rectangle -- view
    2's contribution is the second line of sight inside :func:`locate`.
    """
    position = located["position"]
    per_view = {first.name: np.asarray(located["plane_projected"], float)[:2]}
    if b is not None:
        per_view[second.name] = np.asarray(second.projected.get(b.index, position[:2]), float)[:2]
    return GraspTarget(
        position=np.array([position[0], position[1], located["table_z"] + located["height"]]),
        width=located["width"],
        length=located["length"],
        height=located["height"],
        long_axis_heading=located["heading"],
        table_z=located["table_z"],
        colour=str(a.colour_name),
        score=float(a.score),
        confidence=float(a.confidence),
        triangulated=located["triangulated"],
        plane_projected=located["plane_projected"],
        triangulation_gap=located["gap"],
        method_disagreement=located["disagreement"],
        view_disagreement=located["view_disagreement"],
        position_source=f"colour/{located['source']}",
        matched_in_second_view=b is not None,
        height_measured=located["part"] is not None,
        match_distance=match_distance,
        survey_round=survey_round,
        per_view=per_view,
        approach_width=min(located["width"] + GRIPPER_APPROACH_MARGIN_M, 0.085),
    )


def choose_target(
    first: ViewResult,
    second: ViewResult,
    plane: Tuple[float, float, float],
    avoid: Sequence[np.ndarray] = (),
) -> Optional[GraspTarget]:
    """The best brick *view 1* found, located as :func:`locate` describes.

    Ranked on view 1's colour score alone, which already folds in fingertip clearance, isolation and
    how much of the outline borders bare table. Nothing calls this any more -- :func:`run` walks the
    whole ranked map -- but it is the smallest statement of the decision.
    """
    pairs, first_measured, second_measured = _matched_pairs(first, second, plane, avoid)
    if not pairs:
        logger.error(
            f"{first.name} found no graspable brick at all. Either the pile is out of its frame, or nothing "
            "in it is currently a safe grasp."
        )
        return None
    logger.info(f"{first.name} found {len(pairs)} graspable brick(s); {_pairing_summary(pairs, second)}.")

    a, b, match_distance = max(pairs, key=lambda pair: pair[0].score)
    located = locate(first, second, a, b, first_measured, second_measured, plane)
    target = build_target(first, second, a, b, match_distance, located)
    _report_target(target, rank=None)
    return target


def _pairing_summary(pairs: Sequence[Tuple[Brick, Optional[Brick], float]], second: ViewResult) -> str:
    matched = sum(1 for _, b, _ in pairs if b is not None)
    return f"{second.name} has a counterpart for {matched} of them to triangulate against"


def _report_target(target: GraspTarget, rank: Optional[int] = None) -> None:
    prefix = "Chosen" if rank is None else f"  {rank:2d}."
    engagement = max(float(target.height) - FINGERTIP_CLEARANCE_M, 0.0)
    logger.info(
        f"{prefix}: {target.describe()} (colour score {target.score:.3f}, "
        f"{engagement * 1000:.1f} mm of side wall to pinch, ordered on {grasp_quality(target):.3f})."
    )
    if not target.height_measured:
        logger.warning(
            f"      This brick's height is the {target.height * 1000:.1f} mm fallback: its footprint matched no "
            "catalog part, so nothing measured it. Its x and y are the outline projected onto that assumed "
            "height, so if the part is not that tall the position is off sideways by the difference times the "
            "tangent of the view angle, and the descent stops at the wrong height too."
        )
    if not target.matched_in_second_view:
        logger.info(
            f"      ray-plane projection {np.round(target.plane_projected, 4)} m, from view 1's ray alone -- "
            "view 2 found nothing at this position, so there is no triangulation to compare it against."
        )
        return
    logger.info(
        f"      ray-plane projection {np.round(target.plane_projected, 4)} m, the two views "
        f"{target.view_disagreement * 1000:.1f} mm apart (their centres {target.match_distance * 1000:.1f} mm); "
        f"triangulation {np.round(target.triangulated, 4)} m, rays missing by "
        f"{target.triangulation_gap * 1000:.1f} mm and {target.method_disagreement * 1000:.1f} mm from the "
        f"projection across the table. Using the {target.position_source.replace('_', ' ')}."
    )
    if target.view_disagreement > MAX_VIEW_DISAGREEMENT_M:
        logger.warning(
            f"The two views disagree by {target.view_disagreement * 1000:.1f} mm, wider than the "
            f"{target.width * 1000:.1f} mm brick. Either they are not looking at the same part, or the "
            "hand-eye calibration's lateral error is large. The grasp goes to view 1's position regardless, "
            "so this is a measurement of the rig, not a warning about this brick."
        )


def _heading_difference(first: float, second: float) -> float:
    """Angle between two *axes*, in (-pi/2, pi/2]: a rectangle's long side has no head or tail."""
    return (first - second + math.pi / 2) % math.pi - math.pi / 2


# --- surveying the whole pile at once -------------------------------------------------------------


@dataclass
class PileMap:
    """Every brick one pair of looks located, ranked best-first.

    The unit of work is the survey, not the brick: the viewpoints are the expensive part and they see the
    whole pile. Targets are served from :attr:`targets` in score order and removed as they go, so
    :attr:`remaining` decides when the pile is worth looking at again.
    """

    targets: List[GraspTarget]
    views: List[ViewResult]
    survey_round: int
    surveyed_at: float
    #: How many bricks have been *attempted* out of this map. Above zero means the pile has been reached
    #: into since it was made.
    picks_since_survey: int = 0

    @property
    def remaining(self) -> int:
        return len(self.targets)

    def take_best(self) -> Optional[GraspTarget]:
        return self.targets.pop(0) if self.targets else None

    def discard_near(self, position_xy: Sequence[float], radius: float) -> List[GraspTarget]:
        """Forget the targets within ``radius`` of a point, returning them. Used after each pick.

        The jaws come down open around the brick and go back up with it, so anything inside that sweep may
        have been nudged since its position was measured. Dropping it buys a fresh measurement next survey.
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


def grasp_quality(target: GraspTarget) -> float:
    """How good a grasp this is, as the colour score plus the grip depth the score cannot see.

    Ordering on the colour score alone puts a 3.2 mm plate ahead of a 9.6 mm brick whenever the plate
    is a little cleaner or better isolated, which it usually is -- a plate sitting by itself on bare
    table is the best-looking grasp in the pile and one of the worst actual ones. Weighting in the
    fingertip engagement fixes the order without touching the perception.

    Ordering on ``height_measured`` first, as this did while it meant "depth resolved this", is worse
    than useless now that it means "the catalog recognised this footprint": the parts the catalog
    recognises are the small simple ones, so it promoted exactly the plates it should have demoted.
    """
    engagement = max(float(target.height) - FINGERTIP_CLEARANCE_M, 0.0)
    depth_term = min(engagement / GRIP_DEPTH_SATURATION_M, 1.0)
    return (1.0 - GRIP_DEPTH_WEIGHT) * float(target.score) + GRIP_DEPTH_WEIGHT * depth_term


def build_pile_map(
    first: ViewResult,
    second: ViewResult,
    plane: Tuple[float, float, float],
    avoid: Sequence[np.ndarray] = (),
    keep_out: Sequence[Tuple[Sequence[float], float]] = (),
    survey_round: int = 0,
    surveyed_at: float = 0.0,
) -> PileMap:
    """Locate every brick view 1 found graspable, ranked by score.

    :func:`choose_target`'s pairing and positioning applied to all of view 1's bricks; one whose ray
    cannot be turned into a point is dropped with a warning. ``keep_out`` is ``(centre_xy, radius)``
    circles that are not pile -- the corner picked bricks are stacked in, which once it outgrows the
    pile would otherwise capture the colour pipeline's "largest blob is the pile" crop.
    """
    pairs, first_measured, second_measured = _matched_pairs(first, second, plane, avoid, keep_out)

    targets: List[GraspTarget] = []
    for a, b, match_distance in pairs:
        try:
            located = locate(first, second, a, b, first_measured, second_measured, plane)
        except RuntimeError as exception:
            logger.warning(f"Could not locate region #{a.index}: {exception} Skipping it.")
            continue
        if not np.all(np.isfinite(located["position"])):
            continue
        targets.append(build_target(first, second, a, b, match_distance, located, survey_round))
    targets.sort(key=grasp_quality, reverse=True)

    pile_map = PileMap(
        targets=targets, views=[first, second], survey_round=survey_round, surveyed_at=surveyed_at
    )
    if not targets:
        logger.error(
            f"{first.name} found no graspable brick at all. Either the pile is empty, it is out of its frame, "
            "or nothing left in it is a safe grasp."
        )
        return pile_map

    triangulated = sum(1 for target in targets if target.matched_in_second_view)
    logger.success(
        f"Survey {survey_round}: {len(targets)} brick(s) located in one pair of looks -- the next "
        f"{len(targets)} pick(s) need no camera move at all. {triangulated} of them {second.name} saw too "
        f"and could be triangulated; the other {len(targets) - triangulated} rest on {first.name}'s ray and "
        "the table plane."
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
    """Two looks at the pile: view 1 says which bricks, both views measure where."""
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


# --- looking again, from straight above -----------------------------------------------------------


def look_straight_down(
    cell: C.Cell,
    position_xy: Sequence[float],
    top_face_z: float,
    heights: Sequence[float] = NADIR_CAMERA_HEIGHTS_M,
) -> Optional[ViewResult]:
    """Put the camera directly over a point, looking down, and run the colour pipeline on what it sees.

    ``None`` when no pose in :data:`NADIR_CAMERA_HEIGHTS_M` is reachable -- a failure of the improvement,
    not of the pick, so the caller keeps the survey's answer.
    """
    x, y = float(position_xy[0]), float(position_xy[1])
    target = np.array([x, y, top_face_z])
    for height in heights:
        eye = np.array([x, y, top_face_z + height])
        q = C.solve_tool_ik(cell, C.look_at_tool_pose(cell, eye, target))
        if q is None:
            continue
        logger.info(f"Looking again from {height * 100:.0f} cm straight above the brick ...")
        cell.move_arm_to(q)
        cell.advance(VIEW_SETTLE_DURATION)
        view = cell.capture("nadir")
        analysis = colour.analyse(cv2.cvtColor(np.asarray(view.image_rgb), cv2.COLOR_RGB2BGR))
        return ViewResult(
            name="nadir",
            eye=np.asarray(view.X_base_camera, float)[:3, 3],
            joint_configuration=cell.arm_positions(),
            view=view,
            analysis=analysis,
        )
    logger.warning(
        f"No reachable pose above the brick at ({x:.3f}, {y:.3f}) m to look straight down from, at any of "
        f"{[round(h * 100) for h in heights]} cm. Descending on the survey's position instead, which carries "
        "whatever lateral error the view angle gave it."
    )
    return None


def refine_over_brick(cell: C.Cell, target: GraspTarget) -> GraspTarget:
    """Re-measure ``target`` from straight above it, and move it there. Modified in place.

    Only the geometry is taken from the new look -- position, size, height, long axis -- because the
    choice was view 1's and is not being reopened. Best-effort throughout: an unreachable overhead pose,
    a look that finds nothing, or a mismatched size keeps the survey's numbers and says so.

    From straight above the projection has no lever arm, so the height the footprint is projected onto
    barely matters -- which is exactly what makes this worth the camera move on a colour-only pipeline
    that has to assume the height in the first place.
    """
    original = np.array(target.position[:2], float)
    nadir = look_straight_down(cell, original, target.top_face_z)
    if nadir is None:
        return target

    measured = project_all(nadir, cell.table_plane)
    candidates = [
        (float(np.linalg.norm(measured[b.index][0] - original)), b)
        for b in nadir.bricks if b.index in measured
    ]
    within = [(d, b) for d, b in candidates if d <= NADIR_MATCH_TOLERANCE_M]
    if not within:
        nearest = min(candidates, key=lambda item: item[0])[0] if candidates else float("inf")
        logger.warning(
            f"The re-look found nothing within {NADIR_MATCH_TOLERANCE_M * 1000:.0f} mm of where the survey put "
            f"this brick (nearest region {nearest * 1000:.0f} mm away). Either it was knocked, or the survey is "
            "further out than the search radius. Keeping the survey's position."
        )
        return target

    distance, brick = min(within, key=lambda item: item[0])
    centre, width, length, heading, height, part = measured[brick.index]
    if abs(width - target.width) > NADIR_WIDTH_TOLERANCE_M:
        logger.warning(
            f"The region {distance * 1000:.0f} mm from the survey's position is {width * 1000:.1f} mm wide "
            f"where the survey measured {target.width * 1000:.1f} mm. That is a different part, not this one "
            "seen better. Keeping the survey's position."
        )
        return target

    x, y = float(centre[0]), float(centre[1])
    table_z = cell.table_z_at(x, y)
    target.nadir_correction = float(np.linalg.norm(np.array([x, y]) - original))
    target.position = np.array([x, y, table_z + height])
    target.table_z = table_z
    target.width, target.length, target.height = width, length, height
    target.long_axis_heading = heading
    target.height_measured = part is not None
    target.position_source = "colour/nadir_projection"
    target.per_view["nadir"] = np.array([x, y])

    logger.success(
        f"Re-look from overhead: the brick is at ({x:.4f}, {y:.4f}) m, "
        f"{target.nadir_correction * 1000:.1f} mm from where the survey put it. {target.describe()}"
    )
    if target.nadir_correction > NADIR_CORRECTION_WARNING_M:
        logger.warning(
            f"That is a {target.nadir_correction * 1000:.0f} mm correction -- wider than the jaws' slack, so the "
            "survey's position would have missed. The grasp uses the corrected one, but a survey that wrong is "
            "the hand-eye calibration talking: see src/tools/verify_pick_accuracy.py."
        )
    return target


# --- going there ----------------------------------------------------------------------------------


def go_to_pregrasp(cell: C.Cell, target: GraspTarget, pregrasp_height: float = PREGRASP_HEIGHT) -> GraspTarget:
    """Open the jaws, swing over the brick well clear of the pile, then drop to the pregrasp.

    In two moves, not one: a joint-space line from the table's edge to a pose above the pile sweeps
    diagonally through everything between. The wrist turns square to the brick up high. Every pose is
    solved before the arm moves, so an unreachable brick costs only IK calls. The retract is the one leg
    allowed to give ground -- the highest of :data:`RETRACT_HEIGHTS_M` that solves wins, since it exists
    only to clear the pile while crossing.
    """
    approach_width = min(target.width + GRIPPER_APPROACH_MARGIN_M, cell.gripper_calibration.max_width)
    target.approach_width = approach_width

    def solve(height: float, heading: float):
        position = np.array([target.position[0], target.position[1], target.top_face_z + height])
        return position, C.solve_top_down_ik(cell, position, heading, approach_width)

    poses, configurations = [], []
    heading = target.closing_heading

    retract_height, retract = None, None
    for candidate in RETRACT_HEIGHTS_M:
        position, solved = solve(candidate, heading)
        if solved is not None:
            retract_height, retract = candidate, solved
            break
    if retract is None:
        horizontal = float(np.hypot(target.position[0], target.position[1]))
        raise RuntimeError(
            f"No reachable straight-down retract pose over the brick at {np.round(target.position[:2], 3)} m, "
            f"{horizontal * 100:.1f} cm from the base, at any height from {RETRACT_HEIGHTS_M[0] * 100:.0f} down "
            f"to {RETRACT_HEIGHTS_M[-1] * 100:.0f} cm above it, jaws along {math.degrees(heading):.0f} deg. "
            "Freedriving to this brick does not contradict that: by hand the tool arrives at whatever angle "
            "suits, where its 231 mm buys horizontal reach, while a grasp needs the tool vertical, where the "
            "same 231 mm buys none and costs 32 cm of the wrist's vertical budget. The brick has to come "
            "closer to the base; the next-best candidate is tried instead."
        )
    pose, q, heading = retract  # the whole descent keeps the yaw the retract found reachable
    poses.append(pose)
    configurations.append(q)
    if retract_height != RETRACT_HEIGHT_M:
        logger.info(
            f"Crossing at {retract_height * 100:.0f} cm over the brick rather than "
            f"{RETRACT_HEIGHT_M * 100:.0f} cm -- there is no reachable pose that high over a brick this far "
            "out. Lower over the pile than intended, so watch the swing across."
        )

    for name, height in (("approach", APPROACH_HEIGHT_M), ("pregrasp", pregrasp_height)):
        position, solved = solve(height, heading)
        if solved is None:
            raise RuntimeError(
                f"No reachable straight-down {name} pose at {np.round(position, 3)} m with the jaws along "
                f"{math.degrees(heading):.0f} deg, though the retract {retract_height * 100:.0f} cm up solved. "
                "The next-best candidate is tried instead."
            )
        pose, q, heading = solved
        poses.append(pose)
        configurations.append(q)

    warning = C.reach_warning(cell, target.position)
    if warning:
        logger.warning(warning)

    if cell.gripper is not None:
        cell.move_gripper_to_width(approach_width)
        logger.info(f"Jaws opened to {approach_width * 1000:.1f} mm for a {target.width * 1000:.1f} mm brick.")

    # Via home first. The cross-table leg starts from a viewpoint at the edge of the workspace with the
    # wrist tilted over -- the worst start for a joint-space line. Home is elbow up and central.
    logger.info("Retracting to the home configuration before crossing the table ...")
    cell.move_arm_to(C.HOME_CONFIGURATION)
    logger.info(
        f"Swinging over the brick at {retract_height * 100:.0f} cm, jaws already square to its long axis, "
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
    refine: bool = True,
) -> Tuple[GraspTarget, List[ViewResult]]:
    """The whole of submodule_1: two colour looks, one decision, one pregrasp.

    Returns the target and both views, so a notebook can draw what was seen; pass already-tried
    positions as ``avoid``. Walks the ranked list until one target can be stood over -- top-down poses
    run out of reach before the arm does -- and gives up only when it is exhausted.
    """
    views = [observe(cell, name, eye) for name, eye in VIEWPOINTS]
    pile_map = build_pile_map(views[0], views[1], cell.table_plane, avoid=avoid, surveyed_at=cell.elapsed)
    if not pile_map.remaining:
        raise RuntimeError(
            f"No brick to grasp. Every region {VIEWPOINTS[0][0]} found either failed the colour pipeline's "
            "confidence test, was too tightly packed for a fingertip, or did not cross the table plane. The "
            "overlay the colour pipeline writes shows what it outlined; start there."
        )

    unreachable = 0
    while (target := pile_map.take_best()) is not None:
        try:
            # Re-looked at before the descent is planned, because it moves the brick: every pose
            # go_to_pregrasp solves is built from the position.
            if refine:
                refine_over_brick(cell, target)
            go_to_pregrasp(cell, target, pregrasp_height)
        except RuntimeError as exception:
            unreachable += 1
            logger.warning(
                f"Skipping the {target.colour} brick at {np.round(target.position[:2], 3)} m "
                f"({pile_map.remaining} candidate(s) left): {exception}"
            )
            continue
        if unreachable:
            logger.info(f"Took candidate {unreachable + 1} of the list; the {unreachable} above it were out of reach.")
        return target, views

    raise RuntimeError(
        f"All {unreachable} located brick(s) were out of reach for a straight-down grasp. They are visible and "
        "the arm can touch them by hand, which is not the same thing: the gripper's 231 mm adds horizontal "
        "reach only when the tool is tilted, and a grasp needs it vertical. Move the pile closer to the base."
    )


# --- emptying the pile: one survey, many picks ----------------------------------------------------

#: Re-survey once fewer than this many located bricks are queued. At 2 the last brick of a survey is
#: never picked from an empty map, and the fresh look happens while a known-good target is still in
#: hand. Raise it to re-survey more often; drop to 1 to squeeze every brick out of each survey.
RESURVEY_WHEN_REMAINING_BELOW = 2
#: On top of the jaws' half-width: how far past the open fingers a brick can be and still count as
#: possibly disturbed. The pads are 37.5 mm tall and the lifted brick swings, so a centimetre.
FINGER_DISTURBANCE_MARGIN_M = 0.012
#: This many failed grasps in a row and the map is not believed any more. One failure is a brick;
#: three in a row is a map that no longer describes the table.
MAX_CONSECUTIVE_FAILURES = 3


class PileSession:
    """Empties the pile, looking at it as few times as possible.

    Line for line the simulation's :class:`m1.simulation.submodule_1.PileSession`; only who votes on what
    enters the queue differs (view 1 alone here, both views there). One survey queues every graspable
    brick by score; each :meth:`next_target` serves the best one left with no camera move; bricks near the
    one just picked are dropped as possibly knocked; below
    :data:`RESURVEY_WHEN_REMAINING_BELOW` the pile is looked at again.

    A stale position just means a wasted pick -- submodule_2 catches it on the width check and retreats.
    A brick that failed on a *fresh* target is a bad grasp and is avoided for good; one that failed on a
    map the pile had since been reached into is remeasured next survey. Three failures force a survey.

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
        refine: bool = True,
    ) -> None:
        self.cell = cell
        self.pregrasp_height = pregrasp_height
        self.resurvey_below = max(1, int(resurvey_below))
        self.max_consecutive_failures = max_consecutive_failures
        self.keep_out = list(default_keep_out() if keep_out is None else keep_out)
        #: Re-measure each brick from straight above before descending. One camera move and one pile analysis
        #: per pick, in exchange for dropping the survey's viewing-angle error -- see :func:`refine_over_brick`.
        #: The two viewpoint moves are still paid once per survey, not once per pick.
        self.refine = refine

        self.map: Optional[PileMap] = None
        self.surveys = 0
        self.attempts = 0
        #: How many overhead re-looks were taken, for :meth:`summary` to price.
        self.refinements = 0
        self.picked: List[np.ndarray] = []
        #: Bricks that failed a grasp the map cannot be blamed for. Never retried.
        self.avoid: List[np.ndarray] = []
        #: Bricks that failed while the map was already out of date. Cleared by the next survey, which
        #: measures them again.
        self.provisional_avoid: List[np.ndarray] = []
        self._consecutive_failures = 0
        self._force_survey = False
        self._last_target_was_fresh = True

    # --- picking ----------------------------------------------------------------------------------
    def next_target(self) -> Optional[GraspTarget]:
        """The next brick to grasp, with the arm already standing over it. None when the pile is done.

        Surveys only when the map is empty, low, or discredited. Unreachable pregrasps are skipped rather
        than raised on: with a whole map in hand there is always a next-best candidate.
        """
        while True:
            surveyed_now = False
            if self._needs_survey():
                self._survey()
                surveyed_now = True

            target = self._take_reachable()
            if target is not None:
                return target

            # The map is empty. Untouched since it was taken means nothing left is graspable; otherwise the
            # pile has moved under it and deserves a look.
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
            # Nothing had been touched since this brick was measured, so the grasp failed on the best
            # position the perception gets. That is the brick, not the map.
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

        The picking that follows uses the same map, so calling this before the loop costs nothing.
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
        # The provisional list only ever meant "measured on a pile that has since moved".
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
            # Recorded before the counter moves: a target served off an untouched map is measuring the pile
            # that is actually there.
            self._last_target_was_fresh = self.map.picks_since_survey == 0
            try:
                # Before the descent is planned, not after: the poses are all built from the position.
                if self.refine:
                    refine_over_brick(self.cell, target)
                    self.refinements += 1
                go_to_pregrasp(self.cell, target, self.pregrasp_height)
            except RuntimeError as exception:
                logger.warning(f"Cannot stand over the brick at {np.round(target.position[:2], 3)} m: {exception}")
                continue
            self.map.picks_since_survey += 1
            self.attempts += 1
            logger.info(
                f"Pick {self.attempts} from survey {target.survey_round} "
                f"({self.map.remaining} located brick(s) still queued, "
                f"{'one overhead look taken' if self.refine else 'no camera move needed'})."
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
            "camera_moves": self.surveys * len(VIEWPOINTS) + self.refinements,
            "survey_moves": self.surveys * len(VIEWPOINTS),
            "overhead_relooks": self.refinements,
            "attempts": self.attempts,
            "picked": len(self.picked),
            "camera_moves_one_survey_per_pick": self.attempts * len(VIEWPOINTS),
            "queued": 0 if self.map is None else self.map.remaining,
            "given_up_on": len(self.avoid),
        }


def default_keep_out() -> List[Tuple[Sequence[float], float]]:
    """The corner the picked bricks are stacked in, which is not part of the pile.

    Imported late on purpose: submodule_2 imports :class:`GraspTarget` from here, so naming it at module
    level would close the circle.
    """
    from m1.physical.submodule_2 import DROP_POSITION  # noqa: PLC0415

    # Generous: the bricks are released from a few centimetres up and bounce, so the heap spreads.
    return [(np.asarray(DROP_POSITION, float)[:2], 0.11)]


# --- CLI ------------------------------------------------------------------------------------------


def main() -> None:
    """Perceive the pile, choose a brick, and stand over it. The standalone half of the pipeline.

    Kept for the two-terminal bench workflow; the notebook runs both halves in one process and does not
    go through here or the handoff file.
    """
    import click

    from common.config import DEFAULT_CALIBRATION_DIR, DEFAULT_CAMERA_RESOLUTION, SUPPORTED_ROBOT_TYPES

    @click.command()
    @click.option("--robot-type", type=click.Choice(SUPPORTED_ROBOT_TYPES), default="ur3e", show_default=True)
    @click.option("--ip-address", default=None, help="Robot controller IP. Defaults per robot type.")
    @click.option("--calibration-path", default=DEFAULT_CALIBRATION_DIR, show_default=True)
    @click.option("--camera-resolution", type=click.Choice(list(C.CAMERA_RESOLUTIONS)), default=DEFAULT_CAMERA_RESOLUTION, show_default=True)
    @click.option("--speed-ratio", type=click.IntRange(1, 100), default=C.DEFAULT_SPEED_RATIO, show_default=True)
    @click.option("--pregrasp-height", type=click.FloatRange(0.0, 0.05, min_open=True), default=PREGRASP_HEIGHT, show_default=True)
    @click.option("--table-z", type=float, default=None, help="Level table height, overriding the touched-off plane.")
    @click.option(
        "--position-source",
        type=click.Choice(["triangulation", "plane_projection"]),
        default=POSITION_SOURCE_PREFERENCE,
        show_default=True,
        help="Which of the two estimates the arm is sent to. Both are computed and compared either way.",
    )
    @click.option("--handoff-path", default=BRICK_HANDOFF_PATH, show_default=True)
    def command(
        robot_type: str,
        ip_address: Optional[str],
        calibration_path: str,
        camera_resolution: str,
        speed_ratio: int,
        pregrasp_height: float,
        table_z: Optional[float],
        position_source: str,
        handoff_path: str,
    ) -> None:
        global POSITION_SOURCE_PREFERENCE
        POSITION_SOURCE_PREFERENCE = position_source
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
