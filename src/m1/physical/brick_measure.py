"""
Measure the clicked brick's footprint from a camera frame and identify which part it is.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, NumpyIntImageType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.lego_catalog import FootprintMatch, LegoPart, match_footprint  # noqa: E402
from m1.pile_perception import resize_to_working_resolution, segment_bricks, split_instances  # noqa: E402

MIN_MEASURABLE_FOOTPRINT = 0.002
MAX_VIEW_FOOTPRINT_DISAGREEMENT = 0.003
MIN_TRUSTWORTHY_HEIGHT_SEPARATION = 0.004
HANDOFF_POSITION_TOLERANCE = 0.02


@dataclass(frozen=True)
class ViewObservation:
    """One viewpoint's raw material: what was seen, where the camera was, and what was clicked."""

    image_rgb: NumpyIntImageType
    click_uv: Tuple[int, int]
    intrinsics_matrix: CameraIntrinsicsMatrixType
    X_base_camera: HomogeneousMatrixType
    depth_map: Optional[np.ndarray] = None  # metres, aligned to image_rgb; only used to break ties
    name: str = "view"


@dataclass(frozen=True)
class ViewFootprint:
    """One view's measurement of the clicked brick, in metres in the robot base frame."""

    width: float  # shorter side
    length: float  # longer side
    long_axis_heading: float  # radians, direction the long side points in the base frame
    center: np.ndarray  # (2,) x, y of the rectangle's centre
    pixel_area: int  # size of the segmented instance, for judging how much was actually seen
    tilt_deg: float  # how far off vertical this view's line of sight to the brick was
    mask: np.ndarray  # bool, the segmented instance at working resolution
    working_scale: float  # working pixels per source pixel, to map ``mask`` back onto the frame


def _instance_containing(instances: Sequence[np.ndarray], click_xy: Tuple[float, float]) -> Optional[np.ndarray]:
    """The segmented instance the clicked pixel lands on, or the nearest one within a few pixels.

    A click a pixel or two off the brick's edge -- onto the seam the instance splitter cut away, or
    onto a stud's shadow -- should still select the brick the user obviously meant, so a miss falls
    back to the closest instance centroid rather than failing.
    """
    x, y = int(round(click_xy[0])), int(round(click_xy[1]))
    for instance in instances:
        if 0 <= y < instance.shape[0] and 0 <= x < instance.shape[1] and instance[y, x]:
            return instance

    best: Optional[np.ndarray] = None
    best_distance = float("inf")
    for instance in instances:
        rows, columns = np.nonzero(instance)
        if rows.size == 0:
            continue
        distance = float(np.hypot(columns.mean() - x, rows.mean() - y))
        if distance < best_distance:
            best, best_distance = instance, distance
    if best is not None:
        logger.warning(
            f"The click at ({x}, {y}) did not land on any segmented brick; using the nearest instance, "
            f"{best_distance:.0f} px away. Check the debug overlay if the measurement looks wrong."
        )
    return best


def _project_pixels_onto_plane(
    pixels: np.ndarray,
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    plane_z: float,
) -> Optional[np.ndarray]:
    """Back-project ``(N, 2)`` pixels onto the horizontal plane at ``plane_z``, in the base frame.

    The vectorised twin of ``submodule_1.pixel_to_base_ray`` + ``project_ray_onto_height``: rays
    through the camera centre, rotated into the base frame, scaled until they cross the plane.
    Returns ``(N, 2)`` x, y points, or ``None`` if the rays run away from the plane.
    """
    homogeneous = np.column_stack([pixels.astype(float), np.ones(len(pixels))])
    directions_camera = homogeneous @ np.linalg.inv(intrinsics_matrix).T
    directions_base = directions_camera @ X_base_camera[:3, :3].T
    origin = X_base_camera[:3, 3]

    vertical = directions_base[:, 2]
    if np.any(np.abs(vertical) < 1e-9):
        return None
    distances = (plane_z - origin[2]) / vertical
    if np.any(distances <= 0):
        return None
    return origin[:2] + distances[:, None] * directions_base[:, :2]


def _shrink_for_view_tilt(
    dimensions: Tuple[float, float],
    axes: Tuple[np.ndarray, np.ndarray],
    view_direction: np.ndarray,
    height: float,
) -> Tuple[float, float]:
    """Remove the side walls a tilted view adds to a silhouette (see the module docstring).

    The inflation is ``height * tan(tilt)`` along the camera's azimuth, and a rectangle axis only
    picks up the component of that along itself.
    """
    horizontal = view_direction[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-9 or abs(view_direction[2]) < 1e-9:
        return dimensions  # looking straight down: the silhouette is the top face
    azimuth = horizontal / horizontal_norm
    inflation = height * horizontal_norm / abs(view_direction[2])
    return tuple(  # type: ignore[return-value]
        max(dimension - inflation * abs(float(axis @ azimuth)), 0.0)
        for dimension, axis in zip(dimensions, axes)
    )


def save_debug_overlay(
    debug_dir: str,
    name: str,
    working_image: NumpyIntImageType,
    instances: Sequence[np.ndarray],
    chosen: np.ndarray,
    click_working: Tuple[float, float],
    caption: str,
) -> str:
    """Draw what was segmented and which instance the click selected, and save it.

    The one thing that can go quietly wrong in this measurement is the segmentation picking the wrong
    pixels -- two touching bricks merged into one instance measures as one big brick, and the match
    then reports a part that is not on the table. There is no live display during a run, so this is
    the only way to see it. Every instance is outlined faintly; the chosen one is drawn bright.
    """
    os.makedirs(debug_dir, exist_ok=True)
    overlay = cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR).copy()
    for instance in instances:
        contours, _ = cv2.findContours(instance.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (90, 90, 90), 1)
    contours, _ = cv2.findContours(chosen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.drawMarker(overlay, (int(click_working[0]), int(click_working[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(overlay, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    path = os.path.join(debug_dir, f"measure_{name.replace(' ', '_')}_{time.time():.0f}.png")
    cv2.imwrite(path, overlay)
    return path


def measure_footprint_in_view(
    image_rgb: NumpyIntImageType,
    click_uv: Tuple[int, int],
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    table_z: float,
    assumed_height: float,
    debug_dir: Optional[str] = None,
    name: str = "view",
) -> Optional[ViewFootprint]:
    """Measure the clicked brick's footprint in one frame.

    Args:
        image_rgb: the frame the user clicked in.
        click_uv: the confirmed pixel, in ``image_rgb``'s full resolution.
        intrinsics_matrix: colour intrinsics for that resolution.
        X_base_camera: camera pose in the base frame at the moment the frame was grabbed.
        table_z: the tabletop's height in the base frame.
        assumed_height: current best guess at the brick's height, used both to place the projection
            plane and to size the tilt correction. The caller iterates this (see
            :func:`measure_brick`), so it only has to be roughly right.

    Returns ``None`` if the brick could not be segmented or projected, which is a reason to fall back
    rather than to abort -- the pick still works with the default dimensions.
    """
    working_image, working_scale = resize_to_working_resolution(image_rgb)
    foreground, _, _ = segment_bricks(working_image)
    instances = split_instances(working_image, foreground)
    if not instances:
        logger.warning("No brick instances segmented in this view; cannot measure the footprint here.")
        return None

    click_working = (click_uv[0] * working_scale, click_uv[1] * working_scale)
    instance = _instance_containing(instances, click_working)
    if instance is None:
        return None

    contours, _ = cv2.findContours(instance.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2) / working_scale  # back to source pixels

    plane_z = table_z + assumed_height
    projected = _project_pixels_onto_plane(contour, intrinsics_matrix, X_base_camera, plane_z)
    if projected is None:
        logger.warning("The brick's outline does not cross the table plane in this view; skipping it.")
        return None

    (center_x, center_y), (side_a, side_b), angle_deg = cv2.minAreaRect(projected.astype(np.float32))
    angle = np.radians(angle_deg)
    # minAreaRect's angle names the first side; build both axes from it so the tilt correction can be
    # applied to each side along its own direction.
    axis_a = np.array([np.cos(angle), np.sin(angle)])
    axis_b = np.array([-np.sin(angle), np.cos(angle)])

    brick_center = np.array([center_x, center_y])
    view_direction = np.array([brick_center[0], brick_center[1], plane_z]) - X_base_camera[:3, 3]
    view_direction = view_direction / np.linalg.norm(view_direction)
    side_a, side_b = _shrink_for_view_tilt((side_a, side_b), (axis_a, axis_b), view_direction, assumed_height)

    if side_a >= side_b:
        length, width, long_axis = side_a, side_b, axis_a
    else:
        length, width, long_axis = side_b, side_a, axis_b

    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(view_direction[2])))))
    if width < MIN_MEASURABLE_FOOTPRINT:
        logger.warning(f"Measured a {width * 1000:.1f} mm footprint here, too small to be a part; skipping it.")
        return None

    if debug_dir is not None:
        path = save_debug_overlay(
            debug_dir,
            name,
            working_image,
            instances,
            instance,
            click_working,
            f"{width * 1000:.1f} x {length * 1000:.1f} mm, {tilt_deg:.0f} deg off vertical",
        )
        logger.debug(f"Saved the segmentation overlay for {name} to {path}.")

    return ViewFootprint(
        width=float(width),
        length=float(length),
        long_axis_heading=float(np.arctan2(long_axis[1], long_axis[0])),
        center=brick_center,
        pixel_area=int(instance.sum()),
        tilt_deg=tilt_deg,
        mask=instance,
        working_scale=working_scale,
    )


def measure_height_above_table(
    depth_map: np.ndarray,
    footprint: ViewFootprint,
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    table_z: float,
) -> Optional[float]:
    """How far the brick's top surface stands above ``table_z``, from the depth stream.

    Only ever used to break a tie between catalog parts that share a footprint -- typically "3.2 mm
    plate or 9.6 mm brick", a 6.4 mm question. That is a fair ask of a RealSense at 40 cm; measuring
    a brick's height in absolute terms is not, which is why nothing else depends on this.

    The median of the brick's own pixels is taken (not the mean) so the studs, which stand 1.6 mm
    proud over a minority of the top face, cannot pull it up.
    """
    mask = footprint.mask
    if footprint.working_scale != 1.0:
        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    if mask.shape != depth_map.shape:
        logger.debug(f"Depth map {depth_map.shape} does not match the brick mask {mask.shape}; skipping the tie-break.")
        return None

    rows, columns = np.nonzero(mask & np.isfinite(depth_map) & (depth_map > 0.1))
    if rows.size < 30:
        logger.debug(f"Only {rows.size} depth pixel(s) on the brick; not enough to judge its height.")
        return None

    depth = depth_map[rows, columns]
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    points_camera = np.stack([(columns - cx) * depth / fx, (rows - cy) * depth / fy, depth], axis=-1)
    heights = points_camera @ X_base_camera[:3, :3].T[:, 2] + X_base_camera[2, 3]
    return float(np.median(heights) - table_z)


@dataclass(frozen=True)
class MeasuredBrick:
    """What the two views, the catalog and the depth tie-break jointly concluded about the brick."""

    width: float  # metres, what the fingers close on
    length: float
    height: float  # metres, top face above the table
    long_axis_heading: float  # radians in the base frame
    part: Optional[LegoPart]  # None when nothing in the catalog fits the measurement
    measured_width: float  # what the camera actually saw, before any catalog snapping
    measured_length: float
    view_disagreement: float  # metres, largest per-dimension gap between the two views
    alternatives: Tuple[str, ...] = ()  # other catalog parts that fit the same footprint

    @property
    def closing_heading(self) -> float:
        """Base-frame direction the fingers must close along: square to the brick's long axis."""
        return self.long_axis_heading + np.pi / 2

    def describe(self) -> str:
        """A one-line summary. Deliberately does not claim to have identified the part.

        Several part numbers routinely share a footprint *and* a height -- the 1x2 plates 3023, 3069,
        1748, 15573 and 3794b are all 7.8 x 15.8 x 3.2 mm -- and telling them apart needs their
        topside detail, which this does not look at. It does not need to: the grasp is decided by the
        dimensions, and on those the answer is the same whichever of them it is.
        """
        if self.part is None:
            identity = "no catalog match"
        elif self.alternatives:
            identity = f"{self.part.describe()} (or {len(self.alternatives)} other part(s) of the same size)"
        else:
            identity = self.part.describe()
        return (
            f"{identity}; measured {self.measured_width * 1000:.1f} x {self.measured_length * 1000:.1f} mm, "
            f"long axis at {np.degrees(self.long_axis_heading):.0f} deg (fingers close along "
            f"{np.degrees(self.closing_heading):.0f} deg)"
        )


def identify_part(
    catalog: dict,
    width: float,
    length: float,
    tolerance: float,
    measured_height: Optional[float] = None,
    restrict_to: Optional[Sequence[str]] = None,
) -> Tuple[Optional[FootprintMatch], List[FootprintMatch]]:
    """Pick the catalog part the measured footprint is, and return the runners-up.

    :func:`lego_catalog.match_footprint` already orders candidates shortest-first, which is the safe
    default when a footprint is shared by a plate and a brick. ``measured_height`` overrides that
    ordering, but only when the candidates' heights are far enough apart for depth to have a real say
    (see :data:`MIN_TRUSTWORTHY_HEIGHT_SEPARATION`) -- otherwise a couple of millimetres of RealSense
    noise would be deciding it.
    """
    candidates = match_footprint(catalog, width, length, tolerance, restrict_to=restrict_to)
    if not candidates:
        return None, []

    heights = [candidate.part.height for candidate in candidates]
    separation = max(heights) - min(heights)
    if measured_height is not None and separation >= MIN_TRUSTWORTHY_HEIGHT_SEPARATION:
        best = min(candidates, key=lambda candidate: abs(candidate.part.height - measured_height))
        logger.info(
            f"Candidate heights span {separation * 1000:.1f} mm and depth measured "
            f"{measured_height * 1000:.1f} mm, so it picks {best.part.number} "
            f"({best.part.height * 1000:.1f} mm) over the shortest candidate."
        )
        return best, candidates

    if separation >= MIN_TRUSTWORTHY_HEIGHT_SEPARATION:
        logger.info(
            f"{len(candidates)} parts share this footprint with heights {separation * 1000:.1f} mm apart and no "
            f"usable depth to separate them; taking the shortest ({candidates[0].part.height * 1000:.1f} mm), "
            "which grips lower down a taller brick rather than above a shorter one."
        )
    return candidates[0], candidates


def combine_views(views: Sequence[ViewFootprint]) -> Tuple[float, float, float, float]:
    """Average the per-view footprints and report how far apart they were.

    Returns ``(width, length, long_axis_heading, disagreement)``. Headings are averaged modulo 180
    degrees, because a rectangle's long axis has no head or tail and the two views may well name it
    in opposite directions.
    """
    widths = [view.width for view in views]
    lengths = [view.length for view in views]
    disagreement = max(max(widths) - min(widths), max(lengths) - min(lengths))

    doubled = [2 * view.long_axis_heading for view in views]
    heading = float(np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))) / 2)
    return float(np.mean(widths)), float(np.mean(lengths)), heading, float(disagreement)


def measure_brick(
    observations: Sequence[ViewObservation],
    table_z: float,
    catalog: dict,
    tolerance: float,
    initial_height: float,
    restrict_to: Optional[Sequence[str]] = None,
    passes: int = 2,
    debug_dir: Optional[str] = None,
) -> Optional[MeasuredBrick]:
    """Measure and identify the clicked brick from every view that can see it.

    Runs twice by default because the measurement and the height are mutually dependent: the
    silhouette is projected onto the plane at ``table_z + height`` and corrected for a tilt whose
    effect scales with ``height``, but ``height`` is what the match is trying to find. Starting from
    ``initial_height`` and re-running once with the matched height converges immediately -- a 6 mm
    error in the assumed height moves the measured footprint by well under a millimetre, so the
    second pass confirms the first far more often than it changes it.

    Returns ``None`` if no view could be measured at all; a measurement that finds no catalog match
    still comes back, with ``part=None``, so the caller can decide whether to fall back or stop.
    """
    height = initial_height
    result: Optional[MeasuredBrick] = None

    for pass_index in range(max(1, passes)):
        footprints = []
        for observation in observations:
            footprint = measure_footprint_in_view(
                observation.image_rgb,
                observation.click_uv,
                observation.intrinsics_matrix,
                observation.X_base_camera,
                table_z,
                assumed_height=height,
                debug_dir=debug_dir if pass_index == 0 else None,  # the first pass is the one to inspect
                name=observation.name,
            )
            if footprint is None:
                logger.warning(f"Could not measure the brick in {observation.name}.")
                continue
            footprints.append((observation, footprint))
            logger.info(
                f"{observation.name}: {footprint.width * 1000:.1f} x {footprint.length * 1000:.1f} mm "
                f"(seen {footprint.tilt_deg:.0f} deg off vertical, {footprint.pixel_area} px)."
            )

        if not footprints:
            return None

        width, length, heading, disagreement = combine_views([footprint for _, footprint in footprints])
        if disagreement > MAX_VIEW_FOOTPRINT_DISAGREEMENT and len(footprints) > 1:
            logger.warning(
                f"The views' footprints differ by {disagreement * 1000:.1f} mm (> "
                f"{MAX_VIEW_FOOTPRINT_DISAGREEMENT * 1000:.0f} mm). They are probably not outlining the same "
                "brick -- the clicks may be on different parts, or one view merged two touching bricks."
            )

        measured_heights = [
            measured
            for observation, footprint in footprints
            if observation.depth_map is not None
            for measured in [
                measure_height_above_table(
                    observation.depth_map, footprint, observation.intrinsics_matrix, observation.X_base_camera, table_z
                )
            ]
            if measured is not None
        ]
        depth_height = float(np.mean(measured_heights)) if measured_heights else None

        best, candidates = identify_part(
            catalog, width, length, tolerance, measured_height=depth_height, restrict_to=restrict_to
        )
        if best is None:
            logger.warning(
                f"Measured {width * 1000:.1f} x {length * 1000:.1f} mm, which matches no part in the catalog "
                f"within {tolerance * 1000:.1f} mm. Either two touching bricks were segmented as one, or this "
                "part has no mesh in lego_3d/."
            )
            return MeasuredBrick(
                width=width,
                length=length,
                height=height,
                long_axis_heading=heading,
                part=None,
                measured_width=width,
                measured_length=length,
                view_disagreement=disagreement,
            )

        same_size = tuple(
            candidate.part.number
            for candidate in candidates
            if candidate.part.number != best.part.number
            and abs(candidate.part.height - best.part.height) < 1e-6
            and abs(candidate.part.width - best.part.width) < 1e-6
        )
        result = MeasuredBrick(
            width=best.part.width,
            length=best.part.length,
            height=best.part.height,
            long_axis_heading=heading,
            part=best.part,
            measured_width=width,
            measured_length=length,
            view_disagreement=disagreement,
            alternatives=same_size,
        )

        if abs(best.part.height - height) < 1e-6:
            break  # the assumed height was already the matched one; a further pass changes nothing
        logger.debug(
            f"Pass {pass_index + 1}: height {height * 1000:.1f} -> {best.part.height * 1000:.1f} mm; re-measuring."
        )
        height = best.part.height

    return result


# =================================================================================================
# handoff between submodule_1 (measures) and submodule_2 (grasps)
# =================================================================================================


def write_handoff(
    path: str,
    brick_position: Sequence[float],
    brick: Optional[MeasuredBrick] = None,
    *,
    width: Optional[float] = None,
    height: Optional[float] = None,
    configured_table_z: Optional[float] = None,
    measured_table_zs: Sequence[float] = (),
    safe_table_z: Optional[float] = None,
) -> None:
    
    if brick is not None:
        width = brick.width
        height = brick.height

    payload = {
        "written_at": time.time(),
        "brick_position": [float(value) for value in brick_position],
        "table_z_measured_views": [float(value) for value in measured_table_zs],
    }
    if configured_table_z is not None:
        payload["table_z_configured"] = float(configured_table_z)
    if safe_table_z is not None:
        payload["safe_table_z"] = float(safe_table_z)
    if width is not None:
        payload["width"] = float(width)
    if height is not None:
        payload["height"] = float(height)
    if brick is not None:
        payload.update(
            {
                "part_number": brick.part.number if brick.part else None,
                "same_size_parts": list(brick.alternatives),
                "length": brick.length,
                "obstruction": brick.part.obstruction if brick.part else 0.0,
                "long_axis_heading": brick.long_axis_heading,
                "closing_heading": brick.closing_heading,
                "measured_width": brick.measured_width,
                "measured_length": brick.measured_length,
                "view_disagreement": brick.view_disagreement,
            }
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote the brick handoff to {path}.")


def read_handoff(path: str, max_age: float, expected_position: Optional[Sequence[float]] = None) -> Optional[dict]:
    """Load submodule_1's handoff, or ``None`` with a reason logged if it should not be trusted.

    Three ways it is refused, all of which mean "this describes a different brick than the one under
    the gripper": the file is missing, it is older than ``max_age``, or the brick it recorded is not
    where the arm is now standing. The last one is the useful one -- it catches the arm having been
    moved, freedriven or re-run between the two scripts, which is exactly when using these dimensions
    would be wrong.
    """
    if not os.path.exists(path):
        logger.warning(f"No brick handoff at {path}; falling back to the default brick dimensions.")
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
            "describes an earlier brick. Re-run submodule_1; falling back to the default dimensions."
        )
        return None

    if expected_position is not None and payload.get("brick_position"):
        recorded = np.asarray(payload["brick_position"][:2], dtype=float)
        distance = float(np.linalg.norm(np.asarray(expected_position[:2], dtype=float) - recorded))
        if distance > HANDOFF_POSITION_TOLERANCE:
            logger.warning(
                f"The brick handoff describes a brick at {recorded.round(3)} m but the arm is standing over "
                f"{np.asarray(expected_position[:2]).round(3)} m, {distance * 100:.1f} cm away. The arm has moved "
                "since submodule_1 ran; falling back to the default dimensions."
            )
            return None

    return payload
