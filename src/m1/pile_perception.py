"""M1 submodule 2: find every lego brick in a pile frame and rank them by graspability.

Submodule 0 grasps a lone brick; submodule 1 grasps the brick a human clicked on. This module is
the piece that replaces that click: given one frame of the pile it decides *which* brick to grasp
and with what gripper yaw, so the arm can pick the pile apart on its own.

Pipeline:
  1. :class:`TableColorModel` / :func:`segment_bricks` -- brick pixels vs. table pixels.
  2. :func:`split_instances` -- cut that foreground into one mask per brick.
  3. :func:`score_candidates` -- rank the instances by how safely a top-down parallel-jaw grasp
     would work on each.

Deliberately camera- and robot-free: it takes an RGB image (plus, once the physical stack wires up
depth, a point cloud and camera pose) and returns pixel/metre geometry, nothing more. Tuning a
detector like this is only practical against saved frames you can re-run a hundred times, so the
whole thing stays importable and runnable offline -- see the ``__main__`` entry point at the
bottom, which annotates a still image and writes an overlay PNG.

Sim and physical share this module the way m0's two stacks share ``hand_model.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from loguru import logger

# Long side the frame is resized to before any processing. Every ``*_px`` quantity in this module
# is in these working pixels, not the source image's. Fixing it keeps the pixel-space thresholds
# below meaningful across a 5712-px phone photo and a 1280-px RealSense frame alike.
WORKING_LONG_SIDE = 1280

# --- table colour model -------------------------------------------------------------------------
# Fraction of the frame's width/height, measured inward from each edge, sampled as "known table".
# The pile must sit clear of this ring; widen the workspace or lower this if it ever overlaps.
TABLE_BORDER_FRACTION = 0.10
# Colour modes the table is modelled with. Wood needs several (light bands, dark bands, knots); a
# matte mat collapses to effectively one and the same model keeps working.
TABLE_COLOR_MODES = 4

# How far below the Otsu threshold the *occupancy* mask is taken. Measured on the sample frame,
# brick-vs-table distances overlap genuinely: wood's 99th percentile is 4.9 while a dark grey plate
# sits at 3.9 and a cream brick at 4.3, so no single threshold both finds every brick and keeps
# them apart. Otsu (~5.4 there) is the precise end -- clean, separable instances, but it misses
# roughly the darkest and the most wood-coloured third of the pile. This factor picks a second,
# deliberately over-inclusive threshold where the pile fuses into one blob: useless for identifying
# bricks, but exactly right for asking "is anything here at all". See segment_bricks.
PERMISSIVE_THRESHOLD_FACTOR = 0.6

# --- foreground cleanup -------------------------------------------------------------------------
OPENING_KERNEL_PX = 5  # removes grain speckle that survived thresholding
CLOSING_KERNEL_PX = 7  # closes the shading rings around studs so a brick top stays one blob
MIN_BRICK_AREA_PX = 300  # ~ a 1x2 plate at WORKING_LONG_SIDE; below this it's noise
MAX_BRICK_AREA_FRACTION = 0.25  # a blob covering more than this of the frame isn't a brick

# --- instance splitting -------------------------------------------------------------------------
# Watershed seeds come from cutting the mask along colour seams. The obvious alternative -- seeding
# from the peaks of a distance transform -- does not work on a pile: one merged blob contains both
# a chunky 2x4 brick and a 13-px-wide plate, so any single "distance greater than X" rule either
# floods the plate or misses it entirely. Every pair of touching bricks does have a visible seam
# between them, though, and cutting on that seeds thick and thin bricks alike.
SEAM_BLUR_SIGMA = 1.5  # smooths grain and stud shading so only real brick borders survive
SEAM_CANNY_LOW = 40
SEAM_CANNY_HIGH = 90
# A cut fragment smaller than this is a stud highlight or a sliver, not a brick's core. Measured
# on the sample frame: dropping to 60 over-splits into ~100 instances (studs seeding their own
# basins), while 250+ merges genuinely separate bricks back together.
SEAM_MIN_SEED_AREA_PX = 150

# Bricks cast a shadow that the permissive occupancy threshold picks up, so occupancy hugs each
# brick a few pixels wider than the brick really is. Eroded away before clearance is measured,
# otherwise even a completely isolated brick reads as having a neighbour pressed against it.
OCCUPANCY_SHADOW_MARGIN_PX = 5

# The pile is one compact cluster; wood knots and grain elsewhere in the frame also clear the
# permissive threshold and would otherwise register as obstacles. Restricting to the hull of the
# largest occupancy cluster is the offline stand-in for the workspace crop the physical stack does
# on the point cloud.
PILE_REGION_DILATION_PX = 25

# --- grasp scoring ------------------------------------------------------------------------------
# Robotiq 2F-85: 85 mm stroke, ~22 mm wide pads. The fingers close along the brick's *short* axis.
GRIPPER_MAX_APERTURE_MM = 85.0
GRIPPER_FINGER_CLEARANCE_MM = 14.0  # free table needed beside the brick for a finger to descend
# Used only when the frame's mm-per-pixel scale is unknown (no depth yet): require free space
# beside the brick proportional to the typical brick width in this frame, which self-scales.
CLEARANCE_FALLBACK_FACTOR = 0.6
CLEARANCE_MARCH_LIMIT_PX = 250  # give up looking for free space this far out

SCORE_WEIGHTS = {
    "clearance": 0.45,  # can the fingers actually get down beside it -- the dominant term
    "convexity": 0.25,  # a brick partly hidden under another reads as a bitten-off contour
    "rect_fill": 0.15,  # lego is rectangular; a poor rect fit means a bad split or an odd pose
    "elongation": 0.15,  # prefer a defined long axis, so the yaw we hand back is meaningful
}


@dataclass
class BrickCandidate:
    """One segmented brick and everything the grasp ranking is derived from.

    All pixel quantities are in working pixels (see :data:`WORKING_LONG_SIDE`).
    """

    mask: np.ndarray  # bool (H, W), True on this brick
    center_px: np.ndarray  # (2,) float, (x, y) of the fitted min-area rectangle
    long_axis_px: float
    short_axis_px: float
    grasp_axis: np.ndarray  # (2,) unit vector, direction the fingers close = rect's short axis
    clearance_px: float  # free table beside the brick along ``grasp_axis``, the tighter of the two sides
    convexity: float  # contour area / convex hull area, in [0, 1]
    rect_fill: float  # contour area / min-area-rect area, in [0, 1]
    score: float = 0.0

    @property
    def grasp_yaw_rad(self) -> float:
        """Yaw of the gripper's closing direction in image coordinates.

        The physical stack turns this into a TCP yaw about the base-frame vertical once the
        camera pose is known; on its own it is only meaningful in the image.
        """
        return float(np.arctan2(self.grasp_axis[1], self.grasp_axis[0]))


@dataclass
class PileDetection:
    """Everything one frame produced, ready to rank, visualize or hand to the robot stack."""

    image: np.ndarray  # RGB at working resolution
    foreground: np.ndarray  # bool (H, W), confidently-brick pixels; what instances are cut from
    occupancy: np.ndarray  # bool (H, W), possibly-brick pixels; what "free table" is judged against
    candidates: List[BrickCandidate]  # ranked best-first
    working_scale: float  # working pixels per source pixel, to map results back to the source frame

    @property
    def best(self) -> Optional[BrickCandidate]:
        return self.candidates[0] if self.candidates else None


@dataclass
class TableColorModel:
    """Colour model of the surface the bricks lie on, learned from pixels known to be table.

    This is the only background-specific piece, isolated on purpose so the rest of the detector
    is not. Wood is the hard case: it spans a wide, multi-modal range of warm tones, so the single
    median colour ``submodule_0.detect_brick_mask`` used labels half the grain as brick and the tan
    bricks as table. Several modes with a pooled covariance cover that spread instead. Swapping the
    wood for a matte mat later needs no code change -- the modes simply collapse together.
    """

    means: np.ndarray  # (k, 3) mode centres in CIELAB
    inverse_covariance: np.ndarray  # (3, 3), pooled across modes

    @classmethod
    def from_border(
        cls,
        image_lab: np.ndarray,
        border_fraction: float = TABLE_BORDER_FRACTION,
        n_modes: int = TABLE_COLOR_MODES,
    ) -> "TableColorModel":
        """Learn the table's colours from a ring of pixels around the frame's edge.

        Assumes the pile sits clear of that ring, which is what framing the workspace so the pile
        is centred already gives you -- and unlike "the median pixel is the table", it stays true
        even when bricks cover most of the middle of the frame.
        """
        border = _border_mask(image_lab.shape[:2], border_fraction)
        samples = image_lab[border].astype(np.float32)
        if samples.shape[0] < n_modes * 10:
            raise RuntimeError(
                f"Only {samples.shape[0]} border pixel(s) to learn the table colour from; raise "
                "border_fraction or use a larger frame."
            )

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, means = cv2.kmeans(samples, n_modes, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        # One pooled covariance rather than one per mode: the modes differ mainly in *where* they
        # sit (light vs. dark grain), not in how the colour scatters around them, and pooling keeps
        # a mode that caught few pixels from getting a wildly over- or under-confident shape.
        deviations = samples - means[labels]
        covariance = np.cov(deviations, rowvar=False) + np.eye(3) * 1e-6  # regularized: LAB channels
        return cls(means=means, inverse_covariance=np.linalg.inv(covariance))  # can be near-singular

    def distance(self, image_lab: np.ndarray) -> np.ndarray:
        """Per-pixel Mahalanobis distance to the *nearest* table mode, as float32 (H, W).

        Nearest rather than a mixture likelihood: a pixel only has to look like one kind of table
        to be table, and the minimum keeps this a plain distance map that Otsu can threshold.
        """
        pixels = image_lab.reshape(-1, 3).astype(np.float32)
        best = np.full(pixels.shape[0], np.inf, dtype=np.float32)
        for mean in self.means:
            deviation = pixels - mean
            # einsum evaluates the quadratic form per row without materializing an (N, 3) product
            # per mode, which matters at ~1.2 M pixels.
            squared = np.einsum("ij,jk,ik->i", deviation, self.inverse_covariance, deviation)
            np.minimum(best, squared, out=best)
        return np.sqrt(np.maximum(best, 0.0)).reshape(image_lab.shape[:2])


def _border_mask(shape: Tuple[int, int], border_fraction: float) -> np.ndarray:
    """Boolean mask (H, W) marking a ring ``border_fraction`` deep around the frame's edge."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    band_height = max(1, int(round(height * border_fraction)))
    band_width = max(1, int(round(width * border_fraction)))
    mask[:band_height, :] = True
    mask[-band_height:, :] = True
    mask[:, :band_width] = True
    mask[:, -band_width:] = True
    return mask


def resize_to_working_resolution(image_rgb: np.ndarray, long_side: int = WORKING_LONG_SIDE) -> Tuple[np.ndarray, float]:
    """Scale ``image_rgb`` so its long side is ``long_side``. Returns the image and the scale used.

    Never upscales: a frame already smaller than ``long_side`` is passed through untouched, since
    interpolating it would invent detail the detector would then threshold on.
    """
    height, width = image_rgb.shape[:2]
    scale = long_side / max(height, width)
    if scale >= 1.0:
        return image_rgb, 1.0
    resized = cv2.resize(image_rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill the interior of every blob in a uint8 0/255 mask.

    Stud shading and specular highlights punch holes in an otherwise solid brick top; left alone
    they wreck the distance transform the instance split depends on.
    """
    filled = binary.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _clean_mask(binary: np.ndarray) -> np.ndarray:
    """Despeckle, bridge stud shading, and solidify a raw threshold result."""
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((OPENING_KERNEL_PX, OPENING_KERNEL_PX), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((CLOSING_KERNEL_PX, CLOSING_KERNEL_PX), np.uint8))
    return _fill_holes(binary)


def segment_bricks(
    image_rgb: np.ndarray,
    table_model: Optional[TableColorModel] = None,
    border_fraction: float = TABLE_BORDER_FRACTION,
    permissive_factor: float = PERMISSIVE_THRESHOLD_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the frame into brick pixels and table pixels, at two levels of confidence.

    The distance-from-table map is thresholded with Otsu rather than a constant: the split between
    "looks like table" and "doesn't" moves with the table, the lighting and the white balance, and
    Otsu re-finds it per frame instead of needing a threshold re-tuned every time any of those
    change (which is exactly what ``COLOR_ANOMALY_THRESHOLD = 40`` needed).

    Otsu alone is not enough, though, because brick and table distances genuinely overlap on wood
    (see :data:`PERMISSIVE_THRESHOLD_FACTOR`) -- so two masks come out instead of one, because the
    two ways of being wrong hurt different things:

    * **foreground**, at Otsu: precise. Some bricks are missed, but the ones found are clean and
      separable, which is what instance splitting and grasping need.
    * **occupancy**, lower: over-inclusive to the point of fusing the pile into one blob. Useless
      for identifying bricks, but it means a brick this frame's colour model failed to see cannot
      be mistaken for free table beside the brick we do grasp -- which would otherwise send a
      finger down onto it. Only :func:`score_candidates` uses it.

    Once depth is available, occupancy is better taken from height-above-table, which does not care
    about colour at all; this keeps the same shape so it can be swapped in.

    Args:
        image_rgb: RGB uint8 image at working resolution.
        table_model: reuse a model learned elsewhere; by default one is learned from this frame's
            border ring.
        border_fraction: how deep the border ring is, when learning a model here.
        permissive_factor: occupancy threshold, as a fraction of the Otsu threshold.

    Returns:
        ``(foreground, occupancy, distance)`` -- two bool (H, W) masks and the raw
        distance-from-table map they were thresholded from (for debugging a bad segmentation).
    """
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    if table_model is None:
        table_model = TableColorModel.from_border(image_lab, border_fraction)
    distance = table_model.distance(image_lab)

    # Clip at the 99th percentile before quantizing: a handful of extreme pixels (a specular
    # highlight, a dark knot) would otherwise squash every real distance into the bottom few of
    # the 256 levels Otsu has to work with.
    upper = float(np.percentile(distance, 99.0))
    scaled = np.clip(distance / max(upper, 1e-6), 0.0, 1.0)
    distance_u8 = (scaled * 255).astype(np.uint8)

    otsu_level, strict = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, permissive = cv2.threshold(distance_u8, otsu_level * permissive_factor, 255, cv2.THRESH_BINARY)
    logger.debug(
        f"table distance p99={upper:.1f}; otsu at {otsu_level * upper / 255:.1f}, "
        f"occupancy at {otsu_level * permissive_factor * upper / 255:.1f}"
    )

    foreground = _clean_mask(strict) > 0
    # Occupancy is deliberately not hole-filled: it should keep every hint of something being
    # there, including the speckle a strict mask is right to throw away.
    occupancy = (permissive > 0) | foreground

    region = _pile_region(occupancy)
    return foreground & region, occupancy & region, distance


def _pile_region(occupancy: np.ndarray) -> np.ndarray:
    """Bool mask of the area the pile occupies, as the convex hull of its largest cluster.

    Wood knots and grain clear the permissive threshold too, and outside the pile they would be
    counted as obstacles that block grasps that are actually clear. The pile itself is one compact
    cluster (it held ~89% of all occupancy area on the sample frame), so taking its hull discards
    the rest. Falls back to the whole frame if there is nothing to cluster.
    """
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(occupancy.astype(np.uint8))
    if n_blobs <= 1:
        return np.ones_like(occupancy, dtype=bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    points = cv2.findNonZero((labels == largest).astype(np.uint8))
    hull = cv2.convexHull(points)

    region = np.zeros(occupancy.shape, dtype=np.uint8)
    cv2.drawContours(region, [hull], -1, 255, thickness=cv2.FILLED)
    # Dilate so a brick sitting just off the edge of the cluster, and the free table immediately
    # around the pile that clearance is measured into, both stay inside the region.
    kernel = np.ones((PILE_REGION_DILATION_PX, PILE_REGION_DILATION_PX), np.uint8)
    return cv2.dilate(region, kernel) > 0


def _seam_edges(image_rgb: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Edge map of the borders between touching bricks, as a uint8 0/255 mask.

    Takes the strongest gradient across the three CIELAB channels rather than luminance alone, so
    a seam between two equally-bright but differently-coloured bricks is found as readily as one
    between a light and a dark brick. The Canny hysteresis is normalized against the gradients
    inside the foreground, which keeps it working when a whole frame is low-contrast.
    """
    blurred = cv2.GaussianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), SEAM_BLUR_SIGMA)
    gradient = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    for channel in range(3):
        dx = cv2.Sobel(blurred[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.maximum(gradient, np.sqrt(dx * dx + dy * dy))

    reference = float(np.percentile(gradient[foreground], 98)) if foreground.any() else 1.0
    gradient_u8 = (np.clip(gradient / max(reference, 1e-6), 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.Canny(gradient_u8, SEAM_CANNY_LOW, SEAM_CANNY_HIGH)


def split_instances(
    image_rgb: np.ndarray,
    foreground: np.ndarray,
    min_area_px: int = MIN_BRICK_AREA_PX,
) -> List[np.ndarray]:
    """Cut the foreground into one mask per brick.

    Bricks that touch merge into one blob -- on the sample frame a single blob held 143 751 px,
    most of the pile -- so the blobs have to be broken up rather than taken as instances. Two
    steps:

    1. Cut the mask along colour seams (:func:`_seam_edges`). What is left is one disconnected
       core per brick, since the seam runs all the way between any two touching bricks.
    2. Watershed those cores back out to the mask's real borders, so each instance regains the
       few pixels the cut took off it.

    The cut is what makes this work on a pile. Seeding from distance-transform peaks instead --
    the textbook approach for touching objects -- fails here because one blob spans both chunky
    bricks and 13-px-wide plates, and the plates never clear any threshold the bricks set.

    Args:
        image_rgb: RGB uint8 image at working resolution.
        foreground: bool (H, W) brick mask from :func:`segment_bricks`.
        min_area_px: instances below this are dropped as noise.

    Returns:
        A list of bool (H, W) masks, one per brick instance.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    max_area_px = MAX_BRICK_AREA_FRACTION * foreground.size

    cores = (foreground & (_seam_edges(image_rgb, foreground) == 0)).astype(np.uint8)
    cores = cv2.morphologyEx(cores, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cores, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(cores)

    kept = [index for index in range(1, n_cores) if core_stats[index, cv2.CC_STAT_AREA] >= SEAM_MIN_SEED_AREA_PX]
    if not kept:
        logger.warning("No brick cores survived the seam cut; the frame may be out of focus or very low contrast.")
        return []

    # watershed wants >0 for known regions and 0 for the pixels it has to assign. Everything
    # outside the foreground is one known background region, so basins stop at the mask's border.
    markers = np.zeros(foreground.shape, dtype=np.int32)
    for new_label, core_label in enumerate(kept, start=1):
        markers[core_labels == core_label] = new_label
    markers[~foreground] = len(kept) + 1
    cv2.watershed(image_bgr, markers)

    instances = []
    for label in range(1, len(kept) + 1):
        instance = (markers == label) & foreground
        if min_area_px <= instance.sum() <= max_area_px:
            instances.append(instance)

    logger.debug(f"{n_cores - 1} seam-cut core(s) -> {len(kept)} seed(s) -> {len(instances)} brick instance(s).")
    return instances


def _measure_clearance(
    center: np.ndarray,
    direction: np.ndarray,
    own_mask: np.ndarray,
    others: np.ndarray,
    limit_px: int = CLEARANCE_MARCH_LIMIT_PX,
) -> float:
    """Free table beside a brick along ``direction``, in pixels, on the tighter of the two sides.

    Walks outward from the brick's centre in both directions, steps over the brick itself, then
    counts how far it gets before running into another brick. That run is the room a finger has to
    descend, which is what decides whether a grasp is physically possible -- a brick wedged between
    two neighbours scores near zero here however clean its own segmentation looks.

    Running off the edge of the frame counts as free space: it is unseen table, not a known
    obstacle, and treating it as blocked would penalize every brick at the frame's edge.
    """
    height, width = own_mask.shape
    gaps: List[float] = []

    for sign in (1.0, -1.0):
        gap = 0.0
        left_own = False
        for step in range(1, limit_px + 1):
            point = center + sign * direction * step
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < width and 0 <= y < height):
                gap = float(limit_px - step) if left_own else 0.0
                break
            if own_mask[y, x]:
                if left_own:
                    break  # re-entered our own mask (concave brick); stop counting here
                continue
            left_own = True
            if others[y, x]:
                break
            gap += 1.0
        gaps.append(gap)

    return float(min(gaps))


def score_candidates(
    instances: Sequence[np.ndarray],
    occupancy: np.ndarray,
    mm_per_px: Optional[float] = None,
) -> List[BrickCandidate]:
    """Turn instance masks into ranked :class:`BrickCandidate`s, best grasp first.

    Every term answers "would a top-down Robotiq 2F-85 pinch on this brick actually work":

    * **clearance** -- free table beside the brick where the fingers come down. Dominant, because
      it is the one that fails *physically* rather than just producing a worse grasp.
    * **convexity** -- a brick lying partly under another is segmented as a bitten-off shape, and
      its true centroid isn't where we think it is. Also a proxy for "something is on top of it".
    * **rect_fill** -- lego is rectangular. A blob that fits its own min-area rectangle badly is
      usually a bad split or a brick at an odd angle, and its short axis is not trustworthy.
    * **elongation** -- a near-square blob has no meaningful long axis, so the yaw handed back is
      arbitrary. Mildly preferring elongated bricks means the yaw we commit to is one we believe.

    Bricks too wide for the gripper are dropped outright rather than scored low, so they can never
    be reached by falling through the ranking.

    Args:
        instances: per-brick bool masks from :func:`split_instances`.
        occupancy: the permissive bool mask from :func:`segment_bricks`, used to tell "something is
            there" from "free table". Deliberately the permissive one and not the strict
            foreground: clearance is the term that decides whether a finger can physically come
            down, so it has to account for bricks the strict mask missed.
        mm_per_px: millimetres per working pixel. Without it (no depth yet) the aperture check is
            skipped and the clearance requirement falls back to a self-scaling fraction of this
            frame's typical brick width -- see :data:`CLEARANCE_FALLBACK_FACTOR`.

    Returns:
        Candidates sorted by descending :attr:`BrickCandidate.score`.
    """
    geometries = []
    for mask in instances:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        (center_x, center_y), (width, height), angle_deg = cv2.minAreaRect(contour)
        long_axis, short_axis = max(width, height), min(width, height)
        if short_axis <= 0:
            continue
        # minAreaRect's angle describes its *width* edge; the short axis is that edge rotated 90
        # degrees when width is the longer of the two.
        short_axis_angle = np.deg2rad(angle_deg + (90.0 if width >= height else 0.0))
        geometries.append(
            (mask, contour, area, np.array([center_x, center_y]), long_axis, short_axis, short_axis_angle)
        )

    if not geometries:
        return []

    # Shave the shadow halo off occupancy so an isolated brick doesn't read as hemmed in; the
    # instances themselves are unioned back so a brick's own body is never eroded away.
    kernel = np.ones((OCCUPANCY_SHADOW_MARGIN_PX, OCCUPANCY_SHADOW_MARGIN_PX), np.uint8)
    obstacles = cv2.erode(occupancy.astype(np.uint8), kernel) > 0
    for geometry in geometries:
        obstacles |= geometry[0]

    # Sets the scale for the clearance requirement when there is no metric scale to use instead.
    median_short_axis = float(np.median([geometry[5] for geometry in geometries]))
    if mm_per_px is not None:
        required_clearance_px = GRIPPER_FINGER_CLEARANCE_MM / mm_per_px
        max_short_axis_px = GRIPPER_MAX_APERTURE_MM / mm_per_px
    else:
        required_clearance_px = CLEARANCE_FALLBACK_FACTOR * median_short_axis
        max_short_axis_px = np.inf

    candidates: List[BrickCandidate] = []
    for mask, contour, area, center, long_axis, short_axis, short_axis_angle in geometries:
        if short_axis > max_short_axis_px:
            logger.debug(f"Dropped a candidate {short_axis * (mm_per_px or 1):.0f} mm wide; wider than the gripper opens.")
            continue

        grasp_axis = np.array([np.cos(short_axis_angle), np.sin(short_axis_angle)])
        clearance_px = _measure_clearance(center, grasp_axis, mask, obstacles & ~mask)

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        convexity = area / hull_area if hull_area > 0 else 0.0
        rect_fill = area / max(long_axis * short_axis, 1e-6)
        elongation = 1.0 - short_axis / long_axis

        terms = {
            "clearance": min(clearance_px / max(required_clearance_px, 1e-6), 1.0),
            "convexity": float(np.clip(convexity, 0.0, 1.0)),
            "rect_fill": float(np.clip(rect_fill, 0.0, 1.0)),
            "elongation": float(np.clip(elongation, 0.0, 1.0)),
        }
        score = sum(SCORE_WEIGHTS[name] * value for name, value in terms.items())

        candidates.append(
            BrickCandidate(
                mask=mask,
                center_px=center,
                long_axis_px=float(long_axis),
                short_axis_px=float(short_axis),
                grasp_axis=grasp_axis,
                clearance_px=clearance_px,
                convexity=terms["convexity"],
                rect_fill=terms["rect_fill"],
                score=float(score),
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def detect_pile(
    image_rgb: np.ndarray,
    mm_per_px: Optional[float] = None,
    border_fraction: float = TABLE_BORDER_FRACTION,
    min_area_px: int = MIN_BRICK_AREA_PX,
) -> PileDetection:
    """Run the whole pipeline on one RGB frame: segment, split, rank.

    Args:
        image_rgb: RGB uint8 image at any resolution; it is scaled to
            :data:`WORKING_LONG_SIDE` internally.
        mm_per_px: millimetres per *working* pixel, if known (see :func:`score_candidates`).
        border_fraction: depth of the known-table border ring.
        min_area_px: smallest blob accepted as a brick.
    """
    image, working_scale = resize_to_working_resolution(image_rgb)
    foreground, occupancy, _ = segment_bricks(image, border_fraction=border_fraction)
    instances = split_instances(image, foreground, min_area_px=min_area_px)
    candidates = score_candidates(instances, occupancy, mm_per_px=mm_per_px)
    logger.info(f"{len(candidates)} graspable candidate(s) from {len(instances)} instance(s).")
    return PileDetection(
        image=image,
        foreground=foreground,
        occupancy=occupancy,
        candidates=candidates,
        working_scale=working_scale,
    )


def render_debug_overlay(detection: PileDetection, top_n: int = 5) -> np.ndarray:
    """Draw the detection over the frame for inspection. Returns a BGR image ready for imwrite.

    Every instance gets its outline; the top ``top_n`` get their rank and score, and the winner
    additionally gets its grasp axis drawn as the line the fingers would close along -- which is
    the fastest way to see whether a plausible-looking ranking is actually gripping across the
    brick rather than along it.
    """
    canvas = cv2.cvtColor(detection.image, cv2.COLOR_RGB2BGR).copy()

    # Occupancy that no instance claimed: bricks the strict mask missed. Drawn faintly because it
    # is the thing to look at when a grasp is scored as clear but obviously isn't.
    unclaimed = detection.occupancy & ~detection.foreground
    canvas[unclaimed] = (0.6 * canvas[unclaimed] + 0.4 * np.array([180, 90, 200])).astype(np.uint8)

    shading = canvas.copy()

    for rank, candidate in enumerate(detection.candidates):
        is_top = rank < top_n
        color = (0, 200, 255) if is_top else (110, 110, 110)
        shading[candidate.mask] = color
        contours, _ = cv2.findContours(candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1)

    canvas = cv2.addWeighted(shading, 0.25, canvas, 0.75, 0)

    for rank, candidate in enumerate(detection.candidates[:top_n]):
        center = tuple(np.round(candidate.center_px).astype(int))
        cv2.putText(canvas, str(rank + 1), (center[0] - 8, center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(canvas, str(rank + 1), (center[0] - 8, center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    best = detection.best
    if best is not None:
        half = 0.5 * best.short_axis_px + best.clearance_px
        offset = best.grasp_axis * half
        start = np.round(best.center_px - offset).astype(int)
        end = np.round(best.center_px + offset).astype(int)
        cv2.line(canvas, tuple(start), tuple(end), (0, 255, 0), 2)
        for point in (start, end):
            cv2.circle(canvas, tuple(point), 5, (0, 255, 0), -1)

    return canvas


@click.command()
@click.option("--image", "image_path", required=True, help="Frame of the pile to analyse.")
@click.option(
    "--out",
    "output_path",
    default=None,
    help="Where to write the overlay. Defaults to the input image's path with a '_detection.png' "
    "suffix, so each frame's result lands next to the frame it came from.",
)
@click.option(
    "--mm-per-px",
    type=float,
    default=None,
    help="Millimetres per working pixel. Without it the gripper-aperture check is skipped and the "
    "clearance requirement self-scales to this frame; see score_candidates.",
)
@click.option(
    "--border-fraction",
    type=click.FloatRange(0.01, 0.45),
    default=TABLE_BORDER_FRACTION,
    show_default=True,
    help="Depth of the frame-edge ring sampled as known table. The pile must sit clear of it.",
)
@click.option(
    "--min-area-px",
    type=int,
    default=MIN_BRICK_AREA_PX,
    show_default=True,
    help=f"Smallest blob accepted as a brick, in working pixels (long side {WORKING_LONG_SIDE}).",
)
@click.option("--top-n", type=int, default=5, show_default=True, help="How many ranked candidates to label.")
def main(
    image_path: str,
    output_path: Optional[str],
    mm_per_px: Optional[float],
    border_fraction: float,
    min_area_px: int,
    top_n: int,
) -> None:
    """Rank the graspable bricks in a saved pile frame and write an annotated overlay.

    Offline counterpart to the live detection: no camera, no robot, so a change to the pipeline can
    be judged against a fixed frame in seconds.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise click.ClickException(f"Could not read an image from {image_path!r}.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection = detect_pile(image_rgb, mm_per_px=mm_per_px, border_fraction=border_fraction, min_area_px=min_area_px)

    for rank, candidate in enumerate(detection.candidates[:top_n]):
        logger.info(
            f"#{rank + 1} score={candidate.score:.3f} at ({candidate.center_px[0]:.0f}, {candidate.center_px[1]:.0f}) px, "
            f"{candidate.long_axis_px:.0f}x{candidate.short_axis_px:.0f} px, clearance={candidate.clearance_px:.0f} px, "
            f"convexity={candidate.convexity:.2f}, rect_fill={candidate.rect_fill:.2f}, "
            f"grasp yaw={np.rad2deg(candidate.grasp_yaw_rad):.0f} deg"
        )

    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}_detection.png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, render_debug_overlay(detection, top_n=top_n))
    logger.info(f"Overlay written to {output_path}")


if __name__ == "__main__":
    main()
