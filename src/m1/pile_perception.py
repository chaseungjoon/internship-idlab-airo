"""M1 submodule 2: find every lego brick in a pile frame and rank them by graspability.

Submodule 0 grasps a lone brick; submodule 1 grasps the brick a human clicked on. This module is
the piece that replaces that click: given one frame of the pile it decides *which* brick to grasp
and with what gripper yaw, so the arm can pick the pile apart on its own.

Pipeline:
  1. :class:`TableAppearanceModel` / :func:`segment_bricks` -- brick pixels vs. table pixels.
  2. :func:`split_instances` -- cut that foreground into one mask per brick.
  3. :func:`score_candidates` -- rank the instances by how safely a top-down parallel-jaw grasp
     would work on each, and pick the gripper yaw for each.

Deliberately camera- and robot-free: it takes an RGB image (and optionally a height-above-table map,
which is what the RealSense will supply) and returns pixel/millimetre geometry, nothing more. Tuning
a detector like this is only practical against saved frames you can re-run a hundred times, so the
whole thing stays importable and runnable offline -- see the ``__main__`` entry point at the bottom,
which annotates a still image and writes an overlay PNG: every brick gets a black detection number,
and the ones worth grasping get a blue priority number.

Sim and physical share this module the way m0's two stacks share ``hand_model.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from loguru import logger

# Long side the frame is resized to before any processing. Every ``*_px`` quantity in this module
# is in these working pixels, not the source image's. Fixing it keeps the pixel-space thresholds
# below meaningful across a 5712-px phone photo and a 1280-px RealSense frame alike.
WORKING_LONG_SIDE = 1280

# --- table appearance model ---------------------------------------------------------------------
# Fraction of the frame's width/height, measured inward from each edge, sampled as "known table".
# The pile must sit clear of this ring; widen the workspace or lower this if it ever overlaps.
TABLE_BORDER_FRACTION = 0.10
# Appearance modes the table is modelled with. Wood needs several (light bands, dark bands, knots);
# a matte mat collapses to effectively one and the same model keeps working.
TABLE_APPEARANCE_MODES = 5
# How much darker than a table mode a pixel may be before it stops counting as that mode in shadow,
# in natural-log intensity units: 0.45 is a pixel at e^-0.45 = 64% of the table's brightness. Every
# brick in the pile casts one of these, and the halo is what previously fused the whole pile into a
# single blob. See TableAppearanceModel for why darkening is treated differently from brightening.
TABLE_SHADOW_LOG_DROP = 0.45

# --- foreground thresholds ----------------------------------------------------------------------
# Deviations are in standard deviations of the table's own scatter, so these are frame-independent
# in a way an absolute colour distance never is. Hysteresis rather than one threshold: STRONG says
# "this is definitely not table" and seeds a blob, WEAK grows it out to the brick's true silhouette.
# A brick only has to be confidently visible *somewhere* to be recovered in full, and a patch of
# wood grain that merely drifts past WEAK is discarded because nothing in it ever clears STRONG.
FOREGROUND_STRONG_SIGMA = 4.0
FOREGROUND_WEAK_SIGMA = 2.0
# The same two levels for a height-above-table map, in millimetres. The thinnest lego plate is
# 3.2 mm tall, so half of that is a floor no real part falls under, and a full plate's height is
# something the table's own noise will not reach.
DEPTH_MIN_HEIGHT_MM = 1.6
DEPTH_CONFIDENT_HEIGHT_MM = 3.0

# --- foreground cleanup ---------------------------------------------------------------------------
OPENING_KERNEL_PX = 3  # removes grain speckle that survived thresholding
CLOSING_KERNEL_PX = 5  # closes the shading rings around studs so a brick top stays one blob
MIN_BRICK_AREA_PX = 200  # noise floor only; the real size test is MIN_BRICK_AREA_MM2, once scaled
MAX_BRICK_AREA_FRACTION = 0.15  # a blob covering more than this of the frame isn't one brick
# The smallest part in the pile is a 1x1 plate, 8x8 mm = 64 mm2. Half of that leaves room for one
# seen partly under a neighbour while still discarding the fragments an over-eager split leaves.
MIN_BRICK_AREA_MM2 = 32.0

# The pile is one compact cluster; wood knots and grain elsewhere in the frame also clear the
# thresholds and would otherwise register as bricks or as obstacles. Restricting to the hull of the
# largest occupancy cluster is the offline stand-in for the workspace crop the physical stack does
# on the point cloud.
PILE_CLUSTER_BRIDGE_PX = 21  # blobs closer than this are one cluster, for the purpose of finding it
PILE_REGION_DILATION_PX = 25  # keeps the free table just outside the pile inside the region

# --- instance splitting ---------------------------------------------------------------------------
# The split runs over-segment -> regrow -> re-split, because no single cue separates every pair of
# touching bricks: colour separates two differently-coloured ones, shape separates two of the same
# colour, and each is blind where the other works. The three stages are, in order:
#
#   1. _seam_watershed   -- cut the foreground along every colour seam, deliberately far too eagerly
#   2. _merge_by_colour  -- glue the fragments of each brick back together by their colour
#   3. _split_by_shape   -- cut apart what colour could not, using the fact that lego is rectangular
#
# Stage 1 alone is the classic approach and it over-splits badly: every stud rim and printed marking
# is a seam. Seeding a watershed from distance-transform peaks instead -- the textbook answer for
# touching objects -- fails differently, because one blob spans both chunky 2x4 bricks and 13-px-wide
# plates, and the plates never clear any threshold the bricks set.
SEAM_BLUR_SIGMA = 1.5  # smooths grain and stud shading so only real brick borders survive
# Canny hysteresis, as percentiles of the gradient magnitude *inside the foreground*. Percentiles
# rather than absolute levels so a dim or low-contrast frame cuts in the same places a bright one
# does; the pair is wide because a seam only has to be strong somewhere along its length.
SEAM_GRADIENT_HIGH_PERCENTILE = 88.0
SEAM_GRADIENT_LOW_PERCENTILE = 65.0
# Deliberately small: stage 1 is supposed to over-split, and a brick that gets no seed at all is a
# brick a neighbour's basin floods into and swallows, which stage 2 and 3 cannot undo.
SEAM_MIN_SEED_AREA_PX = 40

# Stage 2 tolerances, in the same normalized units the quantities below are divided by. Lego parts
# are moulded in a flat, uniform colour, so two fragments of one brick match closely while two
# different bricks rarely do; comparing shading-invariant chromaticity separately from brightness is
# what keeps a brick's lit top face and its shaded side wall recognisably the same part.
COLOUR_MERGE_CHROMA_TOL = 1.5
COLOUR_MERGE_INTENSITY_TOL = 1.0
CHROMA_SCALE = 0.06  # typical spread of log-chromaticity within one surface
INTENSITY_SCALE = 0.5  # typical spread of log intensity across one surface's shading
MERGE_MIN_SHARED_BORDER_PX = 4

# Stage 3. A lego brick's silhouette is a rectangle, so a region that fills its own minimum-area
# rectangle poorly is two bricks that stage 2 glued together (or a brick lying half under another).
# 0.72 rather than something near 1.0 because a real brick photographed at an angle shows its top
# face plus a side wall and its studs break the outline, so even a correct instance rarely exceeds
# ~0.85.
RECTANGULARITY_TARGET = 0.72
# A cut has to make things clearly better, not marginally: without this every ragged instance gets
# sliced in half for a rounding error's worth of improvement.
MIN_CUT_IMPROVEMENT = 0.10
# Cuts start at the deep notches in the outline, which is where two touching rectangles meet. A
# notch shallower than this is a stud or a segmentation artefact.
MIN_CONCAVITY_DEPTH_PX = 4.0
MIN_CONCAVITY_DEPTH_FRACTION = 0.10  # ...or this much of the region's short axis, whichever is more
MAX_SPLIT_DEPTH = 4  # a merged region is at most this many bricks deep

# --- gripper geometry -----------------------------------------------------------------------------
# Robotiq 2F-85: 85 mm stroke, ~22 mm wide pads. The fingers close along the brick's *short* axis.
GRIPPER_MAX_APERTURE_MM = 85.0
GRIPPER_FINGER_PAD_WIDTH_MM = 22.0  # pad extent along the brick's long axis
GRIPPER_FINGER_PAD_DEPTH_MM = 12.0  # how far out from the brick face the pad's footprint reaches
GRIPPER_APPROACH_GAP_MM = 4.0  # the jaws descend this much wider than the brick, then close
# Usable aperture, as a fraction of the stroke: closing onto a brick nearly as wide as the gripper
# opens leaves no room to approach around it.
GRIPPER_USABLE_APERTURE_FRACTION = 0.8
# Below this the brick is a 1x1 plate seen edge-on or a bad split; the pads have nothing to bite.
MIN_GRASPABLE_WIDTH_MM = 5.0
# Ring around a brick searched for neighbours pressing against it, beyond the finger pads.
NEIGHBOUR_RING_MM = 10.0

# --- scale ------------------------------------------------------------------------------------------
# Without depth there is no metric scale, so one is estimated from the bricks themselves: lego is
# quantised to an 8 mm module and a pile of mixed parts is dominated by 2-stud-wide ones, so the
# median short axis is ~16 mm. Good to maybe 20%, which is enough for the gripper checks to be
# meaningful; pass --mm-per-px (or a height map) to use the real thing instead.
ASSUMED_MEDIAN_SHORT_AXIS_MM = 16.0

# --- grasp scoring ------------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "pad_clearance": 0.40,  # can the fingers actually get down beside it -- the dominant term
    "isolation": 0.20,  # how much of the brick's outline is free of neighbours at all
    "exposure": 0.15,  # is it on top, or is something lying over it
    "convexity": 0.10,  # a brick partly hidden under another reads as a bitten-off contour
    "rect_fill": 0.05,  # lego is rectangular; a poor rect fit means a bad split or an odd pose
    "size_fit": 0.10,  # comfortably inside the gripper's stroke, not at either extreme
}
# A grasp whose finger pads are this blocked is not worth attempting at any score, so it is dropped
# from the ranking outright rather than left to be reached by falling through it.
MIN_VIABLE_PAD_CLEARANCE = 0.35
# How much clearer the end-on grasp has to be before it is taken over the across-the-brick one.
LONG_AXIS_GRASP_MARGIN = 0.25
# A brick standing this far above the ring around it counts as fully on top of the pile.
EXPOSURE_FULL_HEIGHT_MM = 6.0
# ...and this much brighter than the ring, in natural-log intensity, for the no-depth fallback.
EXPOSURE_FULL_LOG_CONTRAST = 0.6


@dataclass
class BrickCandidate:
    """One segmented brick, its chosen grasp, and everything the ranking is derived from.

    All pixel quantities are in working pixels (see :data:`WORKING_LONG_SIDE`); the ``*_mm``
    quantities use the frame's metric scale, which is estimated from the bricks when no real one is
    supplied (see :data:`ASSUMED_MEDIAN_SHORT_AXIS_MM`).
    """

    index: int  # stable detection id, assigned in raster order; the black label on the overlay
    mask: np.ndarray  # bool (H, W), True on this brick
    contour: np.ndarray  # (N, 1, 2) int, its outer contour
    center_px: np.ndarray  # (2,) float, (x, y) of the fitted min-area rectangle
    long_axis_px: float
    short_axis_px: float
    grasp_axis: np.ndarray  # (2,) unit vector, direction the fingers close
    grasp_width_px: float  # brick extent along ``grasp_axis``; what the jaws have to span
    mm_per_px: float

    pad_clearance: float  # free fraction of the tighter finger pad's footprint, in [0, 1]
    isolation: float  # free fraction of the ring around the brick, in [0, 1]
    exposure: float  # how much this brick sits above what surrounds it, in [0, 1]
    convexity: float  # contour area / convex hull area, in [0, 1]
    rect_fill: float  # contour area / min-area-rect area, in [0, 1]
    size_fit: float  # how comfortably the grasp width sits inside the gripper's stroke, in [0, 1]
    score: float = 0.0
    terms: Dict[str, float] = field(default_factory=dict)  # the weighted terms, for debugging

    @property
    def grasp_yaw_rad(self) -> float:
        """Yaw of the gripper's closing direction in image coordinates.

        The physical stack turns this into a TCP yaw about the base-frame vertical once the
        camera pose is known; on its own it is only meaningful in the image.
        """
        return float(np.arctan2(self.grasp_axis[1], self.grasp_axis[0]))

    @property
    def grasp_width_mm(self) -> float:
        return self.grasp_width_px * self.mm_per_px

    @property
    def commanded_aperture_mm(self) -> float:
        """How wide to open the jaws before descending."""
        return self.grasp_width_mm + GRIPPER_APPROACH_GAP_MM


@dataclass
class PileDetection:
    """Everything one frame produced, ready to rank, visualize or hand to the robot stack."""

    image: np.ndarray  # RGB at working resolution
    foreground: np.ndarray  # bool (H, W), confidently-brick pixels; what instances are cut from
    occupancy: np.ndarray  # bool (H, W), possibly-brick pixels; what "free table" is judged against
    deviation: np.ndarray  # float32 (H, W), how un-table-like each pixel is, in sigmas
    candidates: List[BrickCandidate]  # ranked best-first; `index` is the stable detection id
    rejected: List[BrickCandidate]  # detected, but not worth grasping this cycle
    working_scale: float  # working pixels per source pixel, to map results back to the source frame
    mm_per_px: float  # millimetres per working pixel
    mm_per_px_is_estimated: bool  # True when it came from the bricks, not from depth or the caller

    @property
    def best(self) -> Optional[BrickCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def all_bricks(self) -> List[BrickCandidate]:
        """Every brick found, graspable or not, in detection-id order."""
        return sorted(self.candidates + self.rejected, key=lambda candidate: candidate.index)


# --------------------------------------------------------------------------------------------------
# Table model
# --------------------------------------------------------------------------------------------------


def _log_appearance(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split an RGB frame into intensity-invariant colour and log intensity.

    Working in logs is what makes the shadow handling below possible: shading scales all three
    channels by roughly the same factor, which is a *shift* in log space, so subtracting each
    pixel's mean log channel leaves a chromaticity that a shadow does not move. The mean itself is
    the log intensity, and it is the only place the shadow went.

    Returns ``(chromaticity, log_intensity)`` -- float32 (H, W, 3) summing to zero per pixel, and
    float32 (H, W).
    """
    log_channels = np.log(image_rgb.astype(np.float32) + 1.0)
    log_intensity = log_channels.mean(axis=2)
    return log_channels - log_intensity[:, :, None], log_intensity


@dataclass
class TableAppearanceModel:
    """Appearance model of the surface the bricks lie on, learned from pixels known to be table.

    This is the only background-specific piece, isolated on purpose so the rest of the detector is
    not. Two things make it work where a plain colour distance does not:

    * **Chromaticity, not colour.** Wood is the hard case: it spans a wide, multi-modal range of
      warm tones at wildly different brightnesses, so a single median colour labels half the grain
      as brick. Several modes cover that spread, and each mode is a *chromaticity* plus a
      brightness, not one colour.
    * **Shadow is darkness, not a colour change.** Every brick casts one, and a symmetric distance
      reads that halo as "not table" -- which is how the pile ends up fused into one blob with no
      free table between any two bricks. Here a pixel that matches a mode's chromaticity is allowed
      to be up to :data:`TABLE_SHADOW_LOG_DROP` darker for free, while being *brighter* than the
      table costs immediately. Nothing casts a highlight, so the asymmetry is safe, and it is what
      lets a cream brick a shade lighter than the wood still register.

    Swapping the wood for a matte mat later needs no code change -- the modes simply collapse
    together. Once depth is available, prefer :func:`occupancy_from_height`, which does not care
    about the table's appearance at all.
    """

    means: np.ndarray  # (k, 4), each row [chromaticity (3), log intensity]
    chroma_inverse_covariance: np.ndarray  # (3, 3), pooled across modes
    luminance_sigma: float  # spread of log intensity within a mode
    shadow_log_drop: float = TABLE_SHADOW_LOG_DROP

    @classmethod
    def from_border(
        cls,
        image_rgb: np.ndarray,
        border_fraction: float = TABLE_BORDER_FRACTION,
        n_modes: int = TABLE_APPEARANCE_MODES,
        shadow_log_drop: float = TABLE_SHADOW_LOG_DROP,
    ) -> "TableAppearanceModel":
        """Learn the table from a ring of pixels around the frame's edge.

        Assumes the pile sits clear of that ring, which is what framing the workspace so the pile
        is centred already gives you -- and unlike "the median pixel is the table", it stays true
        even when bricks cover most of the middle of the frame.
        """
        chromaticity, log_intensity = _log_appearance(image_rgb)
        border = _border_mask(image_rgb.shape[:2], border_fraction)
        samples = np.concatenate(
            [chromaticity[border].reshape(-1, 3), log_intensity[border].reshape(-1, 1)], axis=1
        ).astype(np.float32)
        if samples.shape[0] < n_modes * 10:
            raise RuntimeError(
                f"Only {samples.shape[0]} border pixel(s) to learn the table from; raise "
                "border_fraction or use a larger frame."
            )

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        _, labels, means = cv2.kmeans(samples, n_modes, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        deviations = samples - means[labels.ravel()]

        # One pooled covariance rather than one per mode: the modes differ mainly in *where* they
        # sit (light bands vs. dark knots), not in how the colour scatters around them, and pooling
        # keeps a mode that caught few pixels from getting a wildly over- or under-confident shape.
        # Chromaticity sums to zero per pixel, so its covariance is rank 2 -- hence the pseudo-
        # inverse rather than a plain one.
        covariance = np.cov(deviations[:, :3], rowvar=False) + np.eye(3) * 1e-6
        return cls(
            means=means,
            chroma_inverse_covariance=np.linalg.pinv(covariance),
            luminance_sigma=max(float(deviations[:, 3].std()), 1e-3),
            shadow_log_drop=shadow_log_drop,
        )

    def deviation(self, image_rgb: np.ndarray) -> np.ndarray:
        """How un-table-like every pixel is, in standard deviations, as float32 (H, W).

        Distance to the *nearest* mode: a pixel only has to look like one kind of table to be
        table. Because the result is in units of the table's own scatter rather than raw colour
        units, the thresholds it is compared against mean the same thing on a different table,
        under different lighting, at a different exposure.
        """
        chromaticity, log_intensity = _log_appearance(image_rgb)
        flat_chroma = chromaticity.reshape(-1, 3)
        flat_intensity = log_intensity.reshape(-1)

        best = np.full(flat_chroma.shape[0], np.inf, dtype=np.float32)
        for mean in self.means:
            offset = flat_chroma - mean[:3]
            # einsum evaluates the quadratic form per row without materializing an (N, 3) product
            # per mode, which matters at ~1.2 M pixels.
            chroma_term = np.einsum("ij,jk,ik->i", offset, self.chroma_inverse_covariance, offset)
            brightness = (flat_intensity - mean[3]) / self.luminance_sigma
            shadow_allowance = self.shadow_log_drop / self.luminance_sigma
            # Brighter than the table costs from the first sigma; darker is free until the pixel is
            # darker than any shadow could plausibly make this mode.
            luminance_term = np.maximum(brightness, 0.0) + np.maximum(-brightness - shadow_allowance, 0.0)
            np.minimum(best, chroma_term + luminance_term ** 2, out=best)
        return np.sqrt(np.maximum(best, 0.0)).reshape(image_rgb.shape[:2]).astype(np.float32)


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
    resized = cv2.resize(
        image_rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


# --------------------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------------------


def _dilate(mask: np.ndarray, size_px: int) -> np.ndarray:
    if size_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size_px + 1, 2 * size_px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill the interior of every blob in a uint8 0/255 mask.

    Stud shading and specular highlights punch holes in an otherwise solid brick top; left alone
    they wreck the instance split that follows.
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


def _hysteresis(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    """Keep every connected component of ``weak`` that contains at least one ``strong`` pixel."""
    n_blobs, labels = cv2.connectedComponents(weak.astype(np.uint8))
    if n_blobs <= 1:
        return np.zeros_like(weak, dtype=bool)
    seeded = np.unique(labels[strong])
    keep = np.zeros(n_blobs, dtype=bool)
    keep[seeded[seeded > 0]] = True
    return keep[labels]


def segment_bricks(
    image_rgb: np.ndarray,
    table_model: Optional[TableAppearanceModel] = None,
    border_fraction: float = TABLE_BORDER_FRACTION,
    height_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the frame into brick pixels and table pixels, at two levels of confidence.

    Two masks come out, not one, because the two ways of being wrong hurt different things:

    * **foreground** -- what instances are cut from, so it wants clean, separable brick silhouettes.
    * **occupancy** -- what "free table" is judged against, so it wants to include anything that
      might be a brick. A brick this frame's model failed to see must not be mistaken for free
      table beside the brick we do grasp, which would send a finger down onto it.

    Both come from the same hysteresis (see :data:`FOREGROUND_STRONG_SIGMA`); occupancy is the raw
    weak mask, foreground is the seeded-and-cleaned one.

    Args:
        image_rgb: RGB uint8 image at working resolution.
        table_model: reuse a model learned elsewhere; by default one is learned from this frame's
            border ring.
        border_fraction: how deep the border ring is, when learning a model here.
        height_map: optional float (H, W) of millimetres above the table, from the RealSense. When
            given it *replaces* the appearance model for occupancy -- height does not care what
            colour the table is -- and still refines the foreground with it.

    Returns:
        ``(foreground, occupancy, deviation)`` -- two bool (H, W) masks and the deviation map they
        were thresholded from (for debugging a bad segmentation).
    """
    if table_model is None:
        table_model = TableAppearanceModel.from_border(image_rgb, border_fraction)
    deviation = table_model.deviation(image_rgb)

    weak = deviation > FOREGROUND_WEAK_SIGMA
    strong = deviation > FOREGROUND_STRONG_SIGMA
    if height_map is not None:
        # A millimetre above the table is a millimetre above the table whatever the colour, so
        # depth is folded in as extra evidence at both levels rather than as a tiebreak.
        weak |= occupancy_from_height(height_map)
        strong |= occupancy_from_height(height_map, DEPTH_CONFIDENT_HEIGHT_MM)

    foreground = _clean_mask(_hysteresis(weak, strong).astype(np.uint8) * 255) > 0
    occupancy = weak | foreground

    region = _pile_region(occupancy)
    logger.debug(
        f"deviation p50={np.percentile(deviation, 50):.2f} p99={np.percentile(deviation, 99):.2f} sigma; "
        f"foreground covers {100.0 * (foreground & region).mean():.1f}% of the frame"
    )
    return foreground & region, occupancy & region, deviation


def occupancy_from_height(height_map: np.ndarray, min_height_mm: float = DEPTH_MIN_HEIGHT_MM) -> np.ndarray:
    """Occupancy straight from a height-above-table map, for when the RealSense stack is wired up.

    Named rather than inlined so the physical stack has an obvious place to hand depth in, and so it
    is clear that the appearance model above is the *fallback*, not the design: this function needs
    to know nothing at all about what the table looks like.
    """
    return height_map > min_height_mm


def _pile_region(occupancy: np.ndarray) -> np.ndarray:
    """Bool mask of the area the pile occupies, as the convex hull of its largest cluster.

    Wood knots and grain clear the thresholds too, and outside the pile they would be counted as
    bricks that don't exist and as obstacles blocking grasps that are actually clear. The pile
    itself is one compact cluster, so taking its hull discards the rest. Falls back to the whole
    frame if there is nothing to cluster.
    """
    bridged = _dilate(occupancy, PILE_CLUSTER_BRIDGE_PX // 2)
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(bridged.astype(np.uint8))
    if n_blobs <= 1:
        return np.ones_like(occupancy, dtype=bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(cv2.findNonZero((labels == largest).astype(np.uint8)))

    region = np.zeros(occupancy.shape, dtype=np.uint8)
    cv2.drawContours(region, [hull], -1, 255, thickness=cv2.FILLED)
    # Dilate so a brick sitting just off the edge of the cluster, and the free table immediately
    # around the pile that clearance is measured into, both stay inside the region.
    return _dilate(region > 0, PILE_REGION_DILATION_PX)


# --------------------------------------------------------------------------------------------------
# Instance splitting
# --------------------------------------------------------------------------------------------------


def _colour_gradient(image_rgb: np.ndarray) -> np.ndarray:
    """Strongest gradient magnitude across the three CIELAB channels, as float32 (H, W).

    All three channels rather than luminance alone, so a seam between two equally-bright but
    differently-coloured bricks is found as readily as one between a light and a dark brick.
    """
    blurred = cv2.GaussianBlur(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), SEAM_BLUR_SIGMA
    )
    gradient = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    for channel in range(3):
        dx = cv2.Sobel(blurred[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)
        np.maximum(gradient, cv2.magnitude(dx, dy), out=gradient)
    return gradient


def _seam_edges(image_rgb: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Edge map of the borders between touching bricks, as a uint8 0/255 mask.

    The Canny hysteresis is set from percentiles of the gradient *inside the foreground* rather
    than from absolute levels, which keeps it cutting in the same places on a dim frame as on a
    bright one.
    """
    gradient = _colour_gradient(image_rgb)
    inside = gradient[foreground] if foreground.any() else gradient.ravel()
    high = float(np.percentile(inside, SEAM_GRADIENT_HIGH_PERCENTILE))
    low = float(np.percentile(inside, SEAM_GRADIENT_LOW_PERCENTILE))

    scale = 255.0 / max(high, 1e-6)
    gradient_u8 = np.clip(gradient * scale, 0, 255).astype(np.uint8)
    return cv2.Canny(gradient_u8, int(low * scale), 255)


def _seam_watershed(image_rgb: np.ndarray, foreground: np.ndarray) -> Tuple[np.ndarray, int]:
    """Stage 1: over-segment the foreground into basins that never straddle a colour seam.

    Cutting the mask along :func:`_seam_edges` leaves one disconnected core per brick -- and
    several per brick, since stud rims are seams too. A watershed grows those cores back out to the
    mask's real borders, so no pixel is lost to the cut.

    Returns ``(labels, n_labels)``, with 0 for background.
    """
    cores = (foreground & (_seam_edges(image_rgb, foreground) == 0)).astype(np.uint8)
    cores = cv2.morphologyEx(cores, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cores, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(cores)

    kept = [index for index in range(1, n_cores) if core_stats[index, cv2.CC_STAT_AREA] >= SEAM_MIN_SEED_AREA_PX]
    if not kept:
        return np.zeros(foreground.shape, dtype=np.int32), 0

    # watershed wants >0 for known regions and 0 for the pixels it has to assign. Everything
    # outside the foreground is one known background region, so basins stop at the mask's border.
    background_label = len(kept) + 1
    markers = np.zeros(foreground.shape, dtype=np.int32)
    for new_label, core_label in enumerate(kept, start=1):
        markers[core_labels == core_label] = new_label
    markers[~foreground] = background_label
    cv2.watershed(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), markers)

    basins = np.where((markers > 0) & (markers < background_label) & foreground, markers, 0)
    return basins.astype(np.int32), len(kept)


def _merge_by_colour(labels: np.ndarray, n_labels: int, image_rgb: np.ndarray) -> List[np.ndarray]:
    """Stage 2: glue the fragments of each brick back together by colour.

    Agglomerative, closest pair first, until nothing close enough is left to merge. Two things make
    it reliable on lego specifically:

    * Parts are moulded in a flat, uniform colour, so two fragments of one brick agree far more
      closely than two different bricks ever do.
    * Chromaticity and brightness are compared *separately*, against separate tolerances. A brick's
      lit top face and its shaded side wall differ a lot in brightness and hardly at all in
      chromaticity, so keeping the two apart is what lets them merge without also merging a light
      grey brick into a dark grey one.

    Colours are re-measured over the merged region after every merge, which is what stops a slow
    colour gradient from chaining a whole row of bricks into one.

    Note this deliberately fuses two same-coloured bricks that touch -- colour cannot see that
    boundary. :func:`_split_by_shape` is the stage that can.
    """
    chromaticity, log_intensity = _log_appearance(image_rgb)
    chromaticity = chromaticity / CHROMA_SCALE
    log_intensity = log_intensity / INTENSITY_SCALE

    masks = {label: labels == label for label in range(1, n_labels + 1)}
    masks = {label: mask for label, mask in masks.items() if mask.any()}
    chroma = {label: np.median(chromaticity[mask], axis=0) for label, mask in masks.items()}
    intensity = {label: float(np.median(log_intensity[mask])) for label, mask in masks.items()}

    working = labels.copy()
    neighbourhood = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while True:
        pairs = []
        for label, mask in masks.items():
            ring = (cv2.dilate(mask.astype(np.uint8), neighbourhood) > 0) & ~mask
            for other in np.unique(working[ring]):
                other = int(other)
                if other <= label or other not in masks:  # 0 is background; test each pair once
                    continue
                if int((ring & masks[other]).sum()) < MERGE_MIN_SHARED_BORDER_PX:
                    continue
                chroma_gap = float(np.linalg.norm(chroma[label] - chroma[other]))
                intensity_gap = abs(intensity[label] - intensity[other])
                if chroma_gap <= COLOUR_MERGE_CHROMA_TOL and intensity_gap <= COLOUR_MERGE_INTENSITY_TOL:
                    pairs.append((chroma_gap + intensity_gap, label, other))

        if not pairs:
            break
        # Closest first, and each region takes part in at most one merge per sweep, so the colours
        # every later decision is based on are always freshly measured.
        pairs.sort()
        merged_this_sweep = set()
        for _, label, other in pairs:
            if label in merged_this_sweep or other in merged_this_sweep:
                continue
            merged_this_sweep |= {label, other}
            union = masks[label] | masks[other]
            working[masks[other]] = label
            masks[label] = union
            del masks[other]
            chroma[label] = np.median(chromaticity[union], axis=0)
            intensity[label] = float(np.median(log_intensity[union]))

    return list(masks.values())


def _region_shape(mask: np.ndarray) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """``(contour, area, rectangularity, short_axis_px)`` of a mask's largest contour, or None."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    if area <= 0 or min(width, height) <= 0:
        return None
    return contour, area, float(area / (width * height)), float(min(width, height))


def _best_shape_cut(mask: np.ndarray, min_area_px: int) -> Optional[List[np.ndarray]]:
    """Find the straight cut that best turns one non-rectangular region into rectangular pieces.

    Where two bricks touch, their outlines meet at a notch. Every candidate cut therefore starts at
    a deep concavity in the region's outline and ends somewhere else on it; the winner is the one
    whose pieces fill their own minimum-area rectangles best, weighted by area so slicing a corner
    off never wins. Returns None when no cut clears :data:`MIN_CUT_IMPROVEMENT`, which is the usual
    answer for a region that was one brick all along.
    """
    shape = _region_shape(mask)
    if shape is None:
        return None
    contour, area, rectangularity, short_axis = shape
    if rectangularity >= RECTANGULARITY_TARGET or area < 2 * min_area_px:
        return None

    # Simplified first: convexityDefects on a raw pixel outline reports every staircase step.
    outline = cv2.approxPolyDP(contour, 2.0, True)
    if len(outline) < 4:
        return None
    hull = cv2.convexHull(outline, returnPoints=False)
    if len(hull) < 3:
        return None
    try:
        defects = cv2.convexityDefects(outline, hull)
    except cv2.error:
        return None
    if defects is None:
        return None

    min_depth = max(MIN_CONCAVITY_DEPTH_PX, MIN_CONCAVITY_DEPTH_FRACTION * short_axis)
    notches = [tuple(outline[defect[0][2]][0]) for defect in defects if defect[0][3] / 256.0 > min_depth]
    if not notches:
        return None

    # Work in the region's bounding box; the cut only ever affects pixels inside it.
    rows, columns = np.nonzero(mask)
    top, left = int(rows.min()), int(columns.min())
    window = mask[top : int(rows.max()) + 1, left : int(columns.max()) + 1].astype(np.uint8)

    # A cut runs from a notch to any other notch, or to a sampled point on the outline -- the latter
    # covers two bricks meeting in a T, where only one side of the junction makes a notch.
    stride = max(1, len(outline) // 40)
    endpoints = notches + [tuple(point[0]) for point in outline[::stride]]

    best_score, best_pieces = rectangularity + MIN_CUT_IMPROVEMENT, None
    for start in notches:
        for end in endpoints:
            if abs(start[0] - end[0]) + abs(start[1] - end[1]) < 8:
                continue
            knife = np.zeros_like(window)
            cv2.line(knife, (start[0] - left, start[1] - top), (end[0] - left, end[1] - top), 1, 3)
            n_pieces, piece_labels, piece_stats, _ = cv2.connectedComponentsWithStats(window & (1 - knife))
            pieces = [
                piece_labels == piece
                for piece in range(1, n_pieces)
                if piece_stats[piece, cv2.CC_STAT_AREA] >= 0.6 * min_area_px
            ]
            if len(pieces) < 2:
                continue
            shapes = [_region_shape(piece) for piece in pieces]
            if any(piece_shape is None for piece_shape in shapes):
                continue
            total = sum(piece_shape[1] for piece_shape in shapes)
            score = sum(piece_shape[1] * piece_shape[2] for piece_shape in shapes) / max(total, 1.0)
            if score > best_score:
                best_score, best_pieces = score, pieces

    if best_pieces is None:
        return None
    full_size = []
    for piece in best_pieces:
        restored = np.zeros_like(mask)
        restored[top : top + window.shape[0], left : left + window.shape[1]] = piece
        full_size.append(restored)
    return full_size


def _split_by_shape(regions: Sequence[np.ndarray], min_area_px: int) -> List[np.ndarray]:
    """Stage 3: cut apart regions that colour could not, using lego's rectangularity.

    Applied recursively, so a run of three same-coloured plates comes apart in three, but bounded by
    :data:`MAX_SPLIT_DEPTH` so a genuinely ragged region cannot be sliced indefinitely.
    """
    settled: List[np.ndarray] = []
    pending = list(regions)
    for _ in range(MAX_SPLIT_DEPTH):
        if not pending:
            break
        next_round: List[np.ndarray] = []
        for region in pending:
            pieces = _best_shape_cut(region, min_area_px)
            if pieces is None:
                settled.append(region)
            else:
                next_round.extend(pieces)
        pending = next_round
    return settled + pending


def _tidy_instance(mask: np.ndarray) -> np.ndarray:
    """Open away the tendrils a watershed leaves behind, keep the body, and fill the studs in."""
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n_blobs <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return _fill_holes((labels == largest).astype(np.uint8) * 255) > 0


def split_instances(
    image_rgb: np.ndarray,
    foreground: np.ndarray,
    min_area_px: int = MIN_BRICK_AREA_PX,
) -> List[np.ndarray]:
    """Cut the foreground into one mask per brick.

    Bricks that touch merge into one blob -- on the sample frame a single blob holds most of the
    pile -- so the blobs have to be broken up rather than taken as instances. Runs the three stages
    described at :data:`SEAM_BLUR_SIGMA`: over-segment on colour seams, regrow each brick from its
    fragments by colour, then split what colour could not by rectangularity.

    Args:
        image_rgb: RGB uint8 image at working resolution.
        foreground: bool (H, W) brick mask from :func:`segment_bricks`.
        min_area_px: instances below this are dropped as noise.

    Returns:
        A list of bool (H, W) masks, one per brick instance, in raster order (top-to-bottom).
    """
    basins, n_basins = _seam_watershed(image_rgb, foreground)
    if n_basins == 0:
        logger.warning("No brick cores survived the seam cut; the frame may be out of focus or very low contrast.")
        return []

    regions = _merge_by_colour(basins, n_basins, image_rgb)
    pieces = _split_by_shape(regions, min_area_px)

    max_area_px = MAX_BRICK_AREA_FRACTION * foreground.size
    instances = []
    for piece in pieces:
        instance = _tidy_instance(piece)
        if min_area_px <= int(instance.sum()) <= max_area_px:
            instances.append(instance)

    # Raster order, so the detection numbers on the overlay read top-to-bottom instead of following
    # the arbitrary order the watershed happened to label its basins in.
    instances.sort(key=lambda mask: tuple(np.argwhere(mask).mean(axis=0)))
    logger.debug(
        f"{n_basins} seam basin(s) -> {len(regions)} colour region(s) -> {len(pieces)} piece(s) "
        f"-> {len(instances)} brick instance(s)."
    )
    return instances


# --------------------------------------------------------------------------------------------------
# Grasp scoring
# --------------------------------------------------------------------------------------------------


def _rect_free_fraction(blocked: np.ndarray, center: np.ndarray, along: np.ndarray, extent: Tuple[float, float]) -> float:
    """Fraction of a rotated rectangle's footprint that is free of ``blocked``.

    ``along`` is the unit vector the rectangle's first extent runs along. Pixels outside the frame
    count as free: they are unseen table, not a known obstacle, and treating them as blocked would
    penalize every brick near the frame's edge.
    """
    perpendicular = np.array([-along[1], along[0]])
    half_a, half_b = 0.5 * extent[0] * along, 0.5 * extent[1] * perpendicular
    corners = np.array([center - half_a - half_b, center + half_a - half_b, center + half_a + half_b, center - half_a + half_b])

    height, width = blocked.shape
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 1
    crop_x0, crop_y0 = max(x0, 0), max(y0, 0)
    crop_x1, crop_y1 = min(x1, width), min(y1, height)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return 1.0  # entirely outside the frame

    footprint = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(footprint, np.round(corners - [x0, y0]).astype(np.int32), 1)
    total = int(footprint.sum())
    if total == 0:
        return 1.0

    visible = np.zeros_like(footprint, dtype=bool)
    visible[crop_y0 - y0 : crop_y1 - y0, crop_x0 - x0 : crop_x1 - x0] = blocked[crop_y0:crop_y1, crop_x0:crop_x1]
    return 1.0 - float((footprint.astype(bool) & visible).sum()) / total


def _pad_clearance(
    blocked: np.ndarray,
    center: np.ndarray,
    closing_axis: np.ndarray,
    grasp_width_px: float,
    long_extent_px: float,
    mm_per_px: float,
) -> float:
    """Free fraction of the *tighter* of the two finger pads' footprints, in [0, 1].

    This is the term that decides whether a grasp is physically possible rather than merely
    imperfect: the pads have to come down through the table plane on both sides of the brick, and a
    brick wedged between two neighbours scores near zero here however clean its segmentation looks.
    Measuring the pads' actual footprints, rather than marching a ray out from the centre, is what
    makes it sensitive to a neighbour that only clips the corner of where a finger has to go.
    """
    pad_depth_px = GRIPPER_FINGER_PAD_DEPTH_MM / mm_per_px
    pad_width_px = min(GRIPPER_FINGER_PAD_WIDTH_MM / mm_per_px, long_extent_px)
    gap_px = GRIPPER_APPROACH_GAP_MM / mm_per_px
    offset = 0.5 * grasp_width_px + 0.5 * gap_px + 0.5 * pad_depth_px

    return min(
        _rect_free_fraction(blocked, center + closing_axis * offset, closing_axis, (pad_depth_px, pad_width_px)),
        _rect_free_fraction(blocked, center - closing_axis * offset, closing_axis, (pad_depth_px, pad_width_px)),
    )


def _minimum_area_rect(contour: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """``(center, long_axis, short_axis, long_axis_direction)`` of a contour's min-area rectangle."""
    (center_x, center_y), (width, height), angle_deg = cv2.minAreaRect(contour)
    angle = np.deg2rad(angle_deg)
    width_direction = np.array([np.cos(angle), np.sin(angle)])
    height_direction = np.array([-np.sin(angle), np.cos(angle)])
    if width >= height:
        return np.array([center_x, center_y]), float(width), float(height), width_direction
    return np.array([center_x, center_y]), float(height), float(width), height_direction


def _instance_geometry(instances: Sequence[np.ndarray]) -> List[Tuple]:
    """``(index, mask, contour, area, center, long_axis, short_axis, long_direction)`` per instance."""
    geometries = []
    for index, mask in enumerate(instances):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        center, long_axis, short_axis, long_direction = _minimum_area_rect(contour)
        if area <= 0 or short_axis <= 0:
            continue
        geometries.append((index, mask, contour, area, center, long_axis, short_axis, long_direction))
    return geometries


def estimate_mm_per_px(short_axes_px: Sequence[float], areas_px: Sequence[float]) -> float:
    """Guess the frame's scale from the bricks themselves. See :data:`ASSUMED_MEDIAN_SHORT_AXIS_MM`.

    Area-weighted rather than a plain median, because the instance list always has a tail of small
    fragments -- half a brick the split cut in two, a stud that became its own instance -- and a
    plain median sits among them rather than among the bricks. Weighting by area puts the median on
    a pixel of brick rather than on an instance, which is what makes it land on a real part: on the
    sample frame the plain median reads 20 px (0.79 mm/px, three times too coarse) where the
    weighted one reads 65 px (0.25 mm/px, about right).
    """
    short_axes = np.asarray(list(short_axes_px), dtype=np.float64)
    areas = np.asarray(list(areas_px), dtype=np.float64)
    order = np.argsort(short_axes)
    cumulative = np.cumsum(areas[order])
    weighted_median = short_axes[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]
    return ASSUMED_MEDIAN_SHORT_AXIS_MM / max(float(weighted_median), 1e-6)


def score_candidates(
    instances: Sequence[np.ndarray],
    occupancy: np.ndarray,
    image_rgb: np.ndarray,
    mm_per_px: Optional[float] = None,
    height_map: Optional[np.ndarray] = None,
) -> Tuple[List[BrickCandidate], List[BrickCandidate], float, bool]:
    """Turn instance masks into scored :class:`BrickCandidate`s and pick each one's gripper yaw.

    Every term answers "would a top-down Robotiq 2F-85 pinch on this brick actually work":

    * **pad_clearance** -- is there room for the finger pads beside it. Dominant, because it is the
      one that fails *physically* rather than just producing a worse grasp.
    * **isolation** -- how much of the ring around the brick is free. Where pad_clearance asks
      about the two places the fingers land, this asks whether the brick is loose or buried, which
      also predicts whether lifting it will drag a neighbour along.
    * **exposure** -- is this brick on top. With a height map it is literally that; without one it
      falls back to whether the brick is brighter than the bricks around it, which is weak evidence
      and weighted as such.
    * **convexity** -- a brick lying partly under another is segmented as a bitten-off shape, and
      its true centroid isn't where we think it is.
    * **rect_fill** -- lego is rectangular. A blob that fits its own min-area rectangle badly is
      usually a bad split, and its axes are not trustworthy.
    * **size_fit** -- comfortably inside the gripper's stroke, at neither extreme.

    The yaw is chosen, not assumed: closing across the short axis is the default, but a brick whose
    long faces are both pressed against neighbours falls back to an end-on grasp when the gripper
    can span its length. That one check recovers a brick that would otherwise be unreachable.

    Args:
        instances: per-brick bool masks from :func:`split_instances`, in detection order.
        occupancy: the permissive bool mask from :func:`segment_bricks`. Deliberately the permissive
            one and not the strict foreground: clearance decides whether a finger can physically
            come down, so it has to account for bricks the strict mask missed.
        image_rgb: the working-resolution frame, used for the no-depth exposure fallback.
        mm_per_px: millimetres per working pixel. Estimated from the bricks when omitted.
        height_map: optional float (H, W) of millimetres above the table, at working resolution.

    Returns:
        ``(graspable, rejected, mm_per_px, mm_per_px_is_estimated)``. ``graspable`` is sorted
        best-first; ``rejected`` holds bricks that were found but are not worth attempting, and
        keeps them out of the ranking rather than at the bottom of it.
    """
    geometries = _instance_geometry(instances)
    if not geometries:
        return [], [], mm_per_px or 1.0, mm_per_px is None

    mm_per_px_is_estimated = mm_per_px is None
    if mm_per_px is None:
        mm_per_px = estimate_mm_per_px(
            [geometry[6] for geometry in geometries], [geometry[3] for geometry in geometries]
        )

    usable_aperture_mm = GRIPPER_USABLE_APERTURE_FRACTION * GRIPPER_MAX_APERTURE_MM
    ring_px = max(1, int(round(NEIGHBOUR_RING_MM / mm_per_px)))
    _, log_intensity = _log_appearance(image_rgb)

    # Everything that is or might be a brick. Each candidate's own body is subtracted per candidate
    # rather than here, so a brick is never treated as its own obstacle.
    all_bricks = occupancy.copy()
    for geometry in geometries:
        all_bricks |= geometry[1]

    graspable: List[BrickCandidate] = []
    rejected: List[BrickCandidate] = []
    for index, mask, contour, area, center, long_axis, short_axis, long_direction in geometries:
        short_direction = np.array([-long_direction[1], long_direction[0]])
        others = all_bricks & ~mask

        # Across the short axis by default -- the fingers meet the brick's two long faces, which is
        # the stable pinch and the one whose width the gripper is sure to span. The long axis is a
        # fallback, not an equal option: pads at the *ends* of a brick are usually in freer space,
        # so scoring the two evenly would pick the end grasp on nearly every plate in the pile.
        grasp_axis, grasp_width_px = short_direction, short_axis
        pad_clearance = _pad_clearance(others, center, short_direction, short_axis, long_axis, mm_per_px)
        if pad_clearance < MIN_VIABLE_PAD_CLEARANCE and long_axis * mm_per_px <= usable_aperture_mm:
            end_clearance = _pad_clearance(others, center, long_direction, long_axis, short_axis, mm_per_px)
            if end_clearance > pad_clearance + LONG_AXIS_GRASP_MARGIN:
                grasp_axis, grasp_width_px, pad_clearance = long_direction, long_axis, end_clearance

        ring = _dilate(mask, ring_px) & ~mask
        isolation = 1.0 - float((ring & others).sum()) / max(int(ring.sum()), 1)

        if height_map is not None:
            # Literally "how far above its surroundings is it": a brick on top of the pile stands
            # proud of the ring around it, a buried one does not.
            own_height = float(np.median(height_map[mask]))
            ring_height = float(np.median(height_map[ring])) if ring.any() else 0.0
            exposure = float(np.clip((own_height - ring_height) / EXPOSURE_FULL_HEIGHT_MM, 0.0, 1.0))
        else:
            # Fallback with the same shape and a much weaker cue: relative brightness. A brick on
            # top is lit and casts the shadow that falls on whatever it lies across, so being
            # brighter than its neighbours is evidence -- suggestive, not conclusive, since a white
            # brick beats a black one on this measure whichever is on top. Only depth settles it,
            # which is the main thing the RealSense buys this module.
            #
            # Compared against the *neighbouring bricks* rather than the whole ring: most of an
            # isolated brick's ring is bare table, and a dark grey plate alone on pale wood would
            # otherwise score as buried. Nothing adjacent at all means nothing can be on top of it.
            neighbours = ring & others
            if not neighbours.any():
                exposure = 1.0
            else:
                contrast = float(np.median(log_intensity[mask]) - np.median(log_intensity[neighbours]))
                exposure = float(np.clip(0.5 + contrast / EXPOSURE_FULL_LOG_CONTRAST, 0.0, 1.0))

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        convexity = float(np.clip(area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
        rect_fill = float(np.clip(area / max(long_axis * short_axis, 1e-6), 0.0, 1.0))

        grasp_width_mm = grasp_width_px * mm_per_px
        too_small = grasp_width_mm / MIN_GRASPABLE_WIDTH_MM
        too_large = (usable_aperture_mm - grasp_width_mm) / (0.3 * usable_aperture_mm)
        size_fit = float(np.clip(min(too_small, too_large), 0.0, 1.0))

        terms = {
            "pad_clearance": pad_clearance,
            "isolation": isolation,
            "exposure": exposure,
            "convexity": convexity,
            "rect_fill": rect_fill,
            "size_fit": size_fit,
        }
        candidate = BrickCandidate(
            index=index,
            mask=mask,
            contour=contour,
            center_px=center,
            long_axis_px=long_axis,
            short_axis_px=short_axis,
            grasp_axis=grasp_axis,
            grasp_width_px=grasp_width_px,
            mm_per_px=mm_per_px,
            pad_clearance=pad_clearance,
            isolation=isolation,
            exposure=exposure,
            convexity=convexity,
            rect_fill=rect_fill,
            size_fit=size_fit,
            score=float(sum(SCORE_WEIGHTS[name] * value for name, value in terms.items())),
            terms=terms,
        )

        # Hard vetoes, not low scores: a brick the gripper cannot span or cannot reach must never be
        # reachable by falling through the ranking on a thin frame.
        if grasp_width_mm > usable_aperture_mm:
            logger.debug(f"Brick {index}: {grasp_width_mm:.0f} mm across, wider than the gripper usefully opens.")
            rejected.append(candidate)
        elif grasp_width_mm < MIN_GRASPABLE_WIDTH_MM:
            logger.debug(f"Brick {index}: {grasp_width_mm:.0f} mm across, too thin for the pads to bite.")
            rejected.append(candidate)
        elif pad_clearance < MIN_VIABLE_PAD_CLEARANCE:
            logger.debug(f"Brick {index}: finger pads only {100 * pad_clearance:.0f}% clear; boxed in.")
            rejected.append(candidate)
        else:
            graspable.append(candidate)

    graspable.sort(key=lambda candidate: candidate.score, reverse=True)
    return graspable, rejected, mm_per_px, mm_per_px_is_estimated


def detect_pile(
    image_rgb: np.ndarray,
    mm_per_px: Optional[float] = None,
    height_map: Optional[np.ndarray] = None,
    border_fraction: float = TABLE_BORDER_FRACTION,
    min_area_px: int = MIN_BRICK_AREA_PX,
) -> PileDetection:
    """Run the whole pipeline on one RGB frame: segment, split, score, rank.

    Args:
        image_rgb: RGB uint8 image at any resolution; it is scaled to :data:`WORKING_LONG_SIDE`
            internally.
        mm_per_px: millimetres per *working* pixel, if known. Estimated from the bricks otherwise.
        height_map: optional millimetres above the table, at the *source* resolution; it is
            resized alongside the image. This is what the RealSense stack supplies.
        border_fraction: depth of the known-table border ring.
        min_area_px: smallest blob accepted as a brick.
    """
    image, working_scale = resize_to_working_resolution(image_rgb)
    if height_map is not None and height_map.shape[:2] != image.shape[:2]:
        height_map = cv2.resize(height_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    foreground, occupancy, deviation = segment_bricks(image, border_fraction=border_fraction, height_map=height_map)
    instances = split_instances(image, foreground, min_area_px=min_area_px)

    # ``min_area_px`` can only be a noise floor, because how many pixels the smallest real brick
    # covers is not knowable until the frame's scale is. Now that it is, drop everything below one
    # in *millimetres*: what is left is the list of things that could be bricks, and the leftovers
    # were fragments of a brick the split cut in two, not bricks the split found.
    scale_is_estimated = mm_per_px is None
    if scale_is_estimated and instances:
        geometries = _instance_geometry(instances)
        if geometries:
            mm_per_px = estimate_mm_per_px(
                [geometry[6] for geometry in geometries], [geometry[3] for geometry in geometries]
            )
            logger.info(
                f"No metric scale given; estimated {mm_per_px:.3f} mm/px from the bricks' own widths. "
                "Pass --mm-per-px (or a height map) for the real thing."
            )
    if mm_per_px is not None:
        min_pixels = MIN_BRICK_AREA_MM2 / (mm_per_px ** 2)
        kept = [instance for instance in instances if instance.sum() >= min_pixels]
        if len(kept) < len(instances):
            logger.debug(
                f"Dropped {len(instances) - len(kept)} fragment(s) under {MIN_BRICK_AREA_MM2:.0f} mm2 "
                f"({min_pixels:.0f} px at this scale)."
            )
        instances = kept

    graspable, rejected, mm_per_px, _ = score_candidates(
        instances, occupancy, image, mm_per_px=mm_per_px, height_map=height_map
    )

    logger.info(
        f"{len(instances)} brick(s) detected; {len(graspable)} graspable, {len(rejected)} skipped "
        f"(scale {mm_per_px:.3f} mm/px{', estimated' if scale_is_estimated else ''})."
    )
    return PileDetection(
        image=image,
        foreground=foreground,
        occupancy=occupancy,
        deviation=deviation,
        candidates=graspable,
        rejected=rejected,
        working_scale=working_scale,
        mm_per_px=mm_per_px,
        mm_per_px_is_estimated=scale_is_estimated,
    )


# --------------------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------------------

_LABEL_BLACK = (0, 0, 0)
_LABEL_BLUE = (220, 40, 0)  # BGR
_LABEL_HALO = (255, 255, 255)


def _instance_color(index: int) -> Tuple[int, int, int]:
    """A distinct, saturated BGR colour per instance, so a bad split is visible at a glance.

    Stepped by the golden ratio around the hue circle: consecutive detection numbers -- which are
    in raster order, hence usually adjacent bricks -- always land far apart in hue.
    """
    hue = int((index * 137.508) % 180)
    return tuple(int(channel) for channel in cv2.cvtColor(np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _draw_label(canvas: np.ndarray, text: str, anchor: Tuple[int, int], color: Tuple[int, int, int], scale: float) -> None:
    """Draw centred text with a white halo, so it stays readable on a black brick and a white one."""
    thickness = max(1, int(round(2 * scale)))
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (int(anchor[0] - text_width / 2), int(anchor[1] + text_height / 2))
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, _LABEL_HALO, thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_detection_overlay(detection: PileDetection, top_n: int = 10) -> np.ndarray:
    """Draw the detection over the frame for inspection. Returns a BGR image ready for imwrite.

    Two numberings, because they answer two different questions:

    * **black** on every brick -- its detection id. Reading these is how you check the segmentation:
      one number per brick and no brick without one means the split is right.
    * **blue** above the top ``top_n`` -- the grasp priority, 1 being the brick to pick next.

    The top-ranked brick additionally gets its finger pads drawn where they would come down, which
    is the fastest way to see whether a plausible-looking ranking is actually gripping across the
    brick rather than along it.
    """
    canvas = cv2.cvtColor(detection.image, cv2.COLOR_RGB2BGR).copy()
    bricks = detection.all_bricks
    ranks = {candidate.index: rank for rank, candidate in enumerate(detection.candidates, start=1)}

    # Occupancy no instance claimed: bricks the split missed. Drawn faintly because it is the thing
    # to look at when a grasp is scored as clear but obviously isn't.
    unclaimed = detection.occupancy & ~detection.foreground
    canvas[unclaimed] = (0.75 * canvas[unclaimed] + 0.25 * np.array([190, 190, 190])).astype(np.uint8)

    fill = canvas.copy()
    for candidate in bricks:
        fill[candidate.mask] = _instance_color(candidate.index)
    canvas = cv2.addWeighted(fill, 0.30, canvas, 0.70, 0)

    for candidate in bricks:
        color = _instance_color(candidate.index)
        cv2.drawContours(canvas, [candidate.contour], -1, color, 2, cv2.LINE_AA)
        if candidate.index not in ranks:
            # Detected but not worth grasping: hatch it so it reads as "seen, skipped".
            cv2.drawContours(canvas, [candidate.contour], -1, (60, 60, 60), 1, cv2.LINE_AA)

    _draw_grasps(canvas, detection, top_n)

    for candidate in bricks:
        scale = float(np.clip(min(candidate.long_axis_px, candidate.short_axis_px) / 42.0, 0.38, 0.75))
        center = np.round(candidate.center_px).astype(int)
        rank = ranks.get(candidate.index)
        if rank is not None and rank <= top_n:
            # Split the two numbers vertically so neither hides the other on a small brick.
            offset = int(round(13 * scale + 8))
            _draw_label(canvas, str(rank), (center[0], center[1] - offset), _LABEL_BLUE, scale * 1.15)
            _draw_label(canvas, str(candidate.index), (center[0], center[1] + offset), _LABEL_BLACK, scale)
        else:
            _draw_label(canvas, str(candidate.index), tuple(center), _LABEL_BLACK, scale)

    return _draw_legend(canvas, detection, top_n)


def _draw_grasps(canvas: np.ndarray, detection: PileDetection, top_n: int) -> None:
    """Draw the closing direction for the top-ranked bricks, and the finger pads for the winner."""
    for rank, candidate in enumerate(detection.candidates[:top_n], start=1):
        half = 0.5 * candidate.grasp_width_px + GRIPPER_APPROACH_GAP_MM / candidate.mm_per_px
        offset = candidate.grasp_axis * half
        start = np.round(candidate.center_px - offset).astype(int)
        end = np.round(candidate.center_px + offset).astype(int)
        color = (0, 255, 0) if rank == 1 else (0, 235, 235)
        cv2.line(canvas, tuple(start), tuple(end), color, 3 if rank == 1 else 1, cv2.LINE_AA)

        if rank == 1:
            pad_depth = GRIPPER_FINGER_PAD_DEPTH_MM / candidate.mm_per_px
            pad_width = min(GRIPPER_FINGER_PAD_WIDTH_MM / candidate.mm_per_px, candidate.long_axis_px)
            perpendicular = np.array([-candidate.grasp_axis[1], candidate.grasp_axis[0]])
            reach = half + 0.5 * pad_depth
            for sign in (1.0, -1.0):
                pad_center = candidate.center_px + sign * candidate.grasp_axis * reach
                corners = np.array(
                    [
                        pad_center + a * 0.5 * pad_depth * candidate.grasp_axis + b * 0.5 * pad_width * perpendicular
                        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                    ]
                )
                cv2.polylines(canvas, [np.round(corners).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)


def _draw_legend(canvas: np.ndarray, detection: PileDetection, top_n: int) -> np.ndarray:
    """Strip along the bottom naming what the two numberings mean and summarizing the frame."""
    best = detection.best
    lines = [
        f"BLACK = brick id, one per detected brick ({len(detection.all_bricks)} found)",
        f"BLUE  = grasp priority, 1 = pick this next "
        f"(top {min(top_n, len(detection.candidates))} of {len(detection.candidates)} graspable; "
        f"{len(detection.rejected)} found but skipped)",
        f"scale {detection.mm_per_px:.3f} mm/px"
        + (" (estimated from the bricks' own widths)" if detection.mm_per_px_is_estimated else ""),
    ]
    if best is not None:
        lines.append(
            f"next grasp: brick {best.index} at ({best.center_px[0]:.0f}, {best.center_px[1]:.0f}) px, "
            f"{best.grasp_width_mm:.0f} mm across, open to {best.commanded_aperture_mm:.0f} mm, "
            f"yaw {np.rad2deg(best.grasp_yaw_rad):+.0f} deg, pads {100 * best.pad_clearance:.0f}% clear"
        )
    else:
        lines.append("next grasp: nothing in this frame is worth attempting")

    scale = 0.46
    strip = np.full((24 * len(lines) + 16, canvas.shape[1], 3), 24, dtype=np.uint8)
    for row, line in enumerate(lines):
        # Shrink rather than clip: the last line carries the numbers the robot stack acts on.
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        fitted = scale * min(1.0, (canvas.shape[1] - 24) / max(width, 1))
        colour = (255, 190, 120) if row == 1 else (235, 235, 235)
        cv2.putText(strip, line, (12, 22 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, fitted, colour, 1, cv2.LINE_AA)
    return np.vstack([canvas, strip])



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
    help="Millimetres per working pixel. Estimated from the bricks' own widths when omitted; see "
    "estimate_mm_per_px.",
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
@click.option("--top-n", type=int, default=10, show_default=True, help="How many bricks get a blue priority number.")
@click.option(
    "--debug-masks",
    is_flag=True,
    help="Also write the deviation map and the foreground/occupancy masks next to the overlay.",
)
def main(
    image_path: str,
    output_path: Optional[str],
    mm_per_px: Optional[float],
    border_fraction: float,
    min_area_px: int,
    top_n: int,
    debug_masks: bool,
) -> None:
    """Rank the graspable bricks in a saved pile frame and write an annotated overlay.

    Offline counterpart to the live detection: no camera, no robot, so a change to the pipeline can
    be judged against a fixed frame in seconds.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise click.ClickException(f"Could not read an image from {image_path!r}.")

    detection = detect_pile(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        mm_per_px=mm_per_px,
        border_fraction=border_fraction,
        min_area_px=min_area_px,
    )

    for rank, candidate in enumerate(detection.candidates[:top_n], start=1):
        logger.info(
            f"priority {rank:>2}  brick {candidate.index:>3}  score={candidate.score:.3f}  "
            f"at ({candidate.center_px[0]:.0f}, {candidate.center_px[1]:.0f}) px  "
            f"{candidate.long_axis_px * candidate.mm_per_px:.0f}x{candidate.short_axis_px * candidate.mm_per_px:.0f} mm  "
            f"grasp {candidate.grasp_width_mm:.0f} mm @ {np.rad2deg(candidate.grasp_yaw_rad):+.0f} deg  "
            + "  ".join(f"{name}={value:.2f}" for name, value in candidate.terms.items())
        )

    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}_detection.png"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, render_detection_overlay(detection, top_n=top_n))
    logger.info(f"Overlay written to {output_path}")

    if debug_masks:
        stem = os.path.splitext(output_path)[0]
        normalized = np.clip(detection.deviation / (2.0 * FOREGROUND_STRONG_SIGMA), 0.0, 1.0)
        cv2.imwrite(f"{stem}_deviation.png", (normalized * 255).astype(np.uint8))
        cv2.imwrite(f"{stem}_foreground.png", detection.foreground.astype(np.uint8) * 255)
        cv2.imwrite(f"{stem}_occupancy.png", detection.occupancy.astype(np.uint8) * 255)
        logger.info(f"Debug masks written next to {output_path}")


if __name__ == "__main__":
    main()
