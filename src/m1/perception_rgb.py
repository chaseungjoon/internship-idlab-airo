"""Pile perception: find every lego brick in an RGB frame of the pile and rank the grasps.

The RGB-only half of M1's perception, and the one the bench pipeline currently runs -- see the
README for why. :mod:`m1.perception_rgbd` is the same pipeline with the RealSense depth map and the
hand-eye transform on top.

Run on its own it takes a still frame of the pile on the table and writes two files next to it:

  <stem>_perception.png    the frame with a solid outline around every brick and the numbers
                           1..5 on the five bricks that should be grasped first
  <stem>_manifest.json     every measurement behind that picture

Pipeline:
  1. detect_board        -- crop to the table
  2. fit_table_model     -- what the bare table looks like at every pixel, fitted from the frame
  3. segment_foreground  -- brick pixels vs. table pixels, inside the pile's workspace
  4. segment_instances   -- cut that foreground into one region per brick
  5. build_bricks        -- geometry and colour per region, with a confidence, and the junk dropped
  6. measure_clearance   -- how much bare table each brick has around it and beside its grasp faces
  7. estimate_mm_per_px  -- from the 8 mm grid lego is moulded on, so the report is in millimetres
  8. rank_bricks         -- order them by how safely a top-down parallel-jaw grasp would work

Robot-free, which is what makes it the one that can be tuned offline. Bricks are not identified as
catalogue parts; one frame of a pile does not carry enough of a brick to do that, and M2 identifies
the brick after it has been picked up anyway.

The one thing a single RGB frame cannot always settle is a tan brick against a knot in the plywood,
which is why every brick carries a confidence and only confident ones are offered as grasps -- see
the note above MIN_BRICK_CONFIDENCE. Depth collapses that ambiguity, which is what
:mod:`m1.perception_rgbd` replaces the confidence cues with.

Reads ``lego_pic/`` and writes back into it unless told otherwise:

    python src/m1/perception_rgb.py
    python src/m1/perception_rgb.py --debug
    python src/m1/perception_rgb.py lego_pic/lego_pile_20260810_140851.jpg --out-dir /tmp/out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

# Where the pile frames live. Absolute, off this file's own location, so the script runs the same
# from the repo root, from src/, or from anywhere else.
LEGO_PIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lego_pic")

# Every ``*_px`` threshold below is in pixels of a frame whose long side is this many pixels. The
# frame is resized to it before processing and all reported pixel geometry is scaled back to the
# source resolution, so a 1280-px RealSense frame and a larger still tune identically.
WORKING_LONG_SIDE = 1280

# --- table ---------------------------------------------------------------------------------------
BOARD_MARGIN_PX = 34  # trimmed off the detected table, where its lit bevel edge sits
# Radius of the table appearance model. Large enough to look past a brick to the wood beside it,
# small enough to follow the lighting falloff across the board.
TABLE_MODEL_SIGMA_FRAC = 0.06
TABLE_MODEL_ITERATIONS = 6
# The model is a blur this wide, so it is fitted on a copy shrunk until that blur is this many
# pixels across. Nothing the full-resolution fit would find survives a kernel that size anyway.
BACKGROUND_DOWNSAMPLE_SIGMA_PX = 8.0
TABLE_INLIER_SIGMA = 2.5

# --- foreground ----------------------------------------------------------------------------------
# In standard deviations of the table's own scatter. Hysteresis: a brick has to be unmistakable
# somewhere (STRONG) to be admitted, and is then grown to its full silhouette (WEAK). Wood grain
# drifts past WEAK all over the board and is discarded because it never reaches STRONG.
FOREGROUND_STRONG_SIGMA = 7.0
FOREGROUND_WEAK_SIGMA = 3.5
FOREGROUND_SEED_AREA_PX = 60
FOREGROUND_CLOSE_PX = 5
FOREGROUND_OPEN_PX = 3
FOREGROUND_MIN_AREA_PX = 150

# A knot in the plywood and a tan brick are the same colour, the same size and both have a crisp
# outline, so no appearance test tells them apart -- but the bricks were all tipped out in one
# place and the knots are wherever the wood put them. The workspace is the pile's own footprint
# grown by a margin, which keeps a brick that skidded clear of the pile and drops everything at
# the far edges of the board. The physical submodule gets this crop from the depth map instead.
PILE_BRIDGE_PX = 25  # blobs closer than this belong to the same pile
PILE_MARGIN_PX = 50  # how far outside the pile's hull a stray brick is still workspace

# --- instance splitting ----------------------------------------------------------------------------
EDGE_BLUR_SIGMA = 2.0  # smooths stud shading and grain so only brick borders survive as ridges
SEED_GRADIENT_FRACTION = 0.30  # a watershed seed is foreground flatter than this much of the 95th pct
SEED_MIN_AREA_PX = 40
MERGE_ROUNDS = 3
MERGE_COLOUR_TOL = 2.6  # region colour distance, in table-scatter sigmas
MERGE_EDGE_TOL = 0.42  # normalized gradient along the shared border below which it is not a border
MERGE_ABSORB_AREA_PX = 90  # fragments smaller than this always join their closest neighbour
MERGE_MIN_BORDER_PX = 5  # two bricks meeting at a corner are not one brick
MERGE_MIN_FILL = 0.62  # a merge that leaves an L or a T joined two bricks, not one brick's pieces
SPLIT_FILL_TARGET = 0.68  # a region filling its own box worse than this is more than one brick
SPLIT_MIN_SIDE_PX = 11  # no lego part is thinner than this, so no cut may leave a sliver
SPLIT_MIN_EDGE = 0.30  # a cut has to run along a real gradient ridge, not through a flat brick top
SPLIT_MAX_DEPTH = 3

# --- what counts as a brick ---------------------------------------------------------------------
MIN_BRICK_AREA_PX = 200
MAX_BRICK_AREA_FRACTION = 0.12  # of the table; anything larger is a segmentation failure
MIN_BRICK_SIDE_PX = 9  # a 1x1 plate seen edge-on is still wider than this
MAX_ASPECT_RATIO = 14.0
MIN_RECTANGULARITY = 0.42
MIN_SOLIDITY = 0.62

# A tan brick and a knot in the plywood are the same colour, the same size and both have a crisp
# outline. In a single RGB frame that question does not always have an answer -- the depth map the
# physical submodule adds settles it in one step, since one of them is 10 mm off the table -- so
# rather than a hard test that has to be wrong somewhere, three graded cues combine into a
# confidence, and everything downstream carries it:
#
#   deviation -- how far the region's colour is from the table's, in the table's own sigmas
#   edge support -- how much of a step, rather than a fade, its outline is
#   interior texture -- gradient inside the region over gradient just outside it. A brick is a
#       moulded surface laid over the wood: flat inside, and its own edge is the busiest thing
#       near it. A stain is the wood, so the grain runs through it as it runs around it, ~1.
#
# Only the clearly-hopeless are dropped outright (MIN_BRICK_CONFIDENCE); a region below
# PRIORITY_MIN_CONFIDENCE is still reported and still outlined, it just cannot be chosen to grasp.
CONFIDENCE_DEVIATION_SIGMA = 12.0
CONFIDENCE_EDGE_SUPPORT = 0.80
CONFIDENCE_INTERIOR_TEXTURE = 0.70
CONFIDENCE_WEIGHTS = {"deviation": 0.45, "edge_support": 0.30, "interior_texture": 0.25}
RING_MIN_TABLE_PX = 40  # below this much bare table beside it, the texture cue has nothing to say
MIN_BRICK_CONFIDENCE = 0.45
PRIORITY_MIN_CONFIDENCE = 0.70

# --- grasping ------------------------------------------------------------------------------------
# Robotiq 2F-85 fingertips. The jaws close on the brick's two long sides, so the brick's short side
# is the width the gripper has to span and the long sides are where the fingers need room.
GRIPPER_MIN_WIDTH_MM = 4.0
GRIPPER_MAX_WIDTH_MM = 46.0
GRIPPER_IDEAL_WIDTH_MM = 16.0
FINGER_HALF_THICKNESS_MM = 6.0  # room one fingertip needs beside the brick
CLEARANCE_GOOD_MM = 14.0  # fingertip room at which the clearance term saturates
ISOLATION_GOOD_MM = 26.0  # gap to the rest of the pile at which the isolation term saturates
CONFIDENT_AREA_MM2 = 260.0  # footprint of a 2x4 plate; the area term saturates there
# A 1x1 plate is 7.8 mm square, 61 mm2, and it is the smallest part in the catalogue. A region
# under this is not a part the gripper can be sent at whatever else it looks like -- in practice a
# screw hole or a chip in the wood, both of which are otherwise perfect grasps: small and alone.
MIN_PART_AREA_MM2 = 45.0
JAW_PROBE_MARGIN_PX = 3  # how far outside the brick the fingertip clearance is sampled

SCORE_WEIGHTS = {
    "clearance": 0.30,  # room for the fingertips beside the brick -- the thing that fails first
    "isolation": 0.16,  # how far the brick sits from the rest of the pile
    "exposure": 0.18,  # how much of its outline borders bare table rather than another brick
    "visibility": 0.16,  # a clean rectangle is a brick nothing is lying on top of
    "width_fit": 0.08,  # short side comfortably inside the gripper's range
    "size": 0.08,  # a whole brick rather than the corner of one, and a bigger target for the jaws
    "confidence": 0.14,  # how sure the detector is this is a brick at all and not the table
}

# --- scale ---------------------------------------------------------------------------------------
STUD_PITCH_MM = 8.0  # the grid every lego part is moulded on, and so the frame's built-in ruler
STUD_MIN_SAMPLES = 10
STUD_MIN_PITCH_RADII = 2.2  # below this many stud radii apart, two 'studs' are one stud twice
STUD_MAX_SPREAD = 0.25  # reject the estimate if the detected studs disagree by more than this

PRIORITY_COUNT = 5

# name -> RGB, for reporting the colour family only; the pile has no red or yellow parts but a
# nearest-neighbour palette needs the poles present to not drag other colours towards its members.
COLOUR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "white": (240, 240, 235),
    "light_grey": (175, 181, 182),
    "dark_grey": (89, 93, 96),
    "black": (33, 33, 33),
    "tan": (222, 198, 156),
    "dark_tan": (176, 148, 106),
    "brown": (152, 96, 62),
    "reddish_brown": (124, 72, 50),
    "sand_blue": (120, 144, 160),
    "blue": (48, 92, 160),
    "dark_green": (32, 82, 62),
    "green": (60, 140, 80),
    "red": (168, 46, 44),
    "yellow": (226, 188, 68),
}


@dataclass
class Board:
    mask: np.ndarray
    quad: np.ndarray
    area_px: int


@dataclass
class TableModel:
    """The bare table's colour at every pixel, and how much that colour normally wanders."""

    residual: np.ndarray  # observed minus predicted table, in opponent channels
    covariance: np.ndarray
    sigma: np.ndarray
    inliers: np.ndarray

    def deviation(self) -> np.ndarray:
        """Distance from the table, in sigmas, treating shadow as table and highlight as not.

        Chroma is used in both directions: a brick of any colour but the wood's own moves off the
        table's chroma axis. Brightness is used upwards only, because a pixel darker than the table
        is the shadow every brick casts, while a pixel brighter than it is a tan or white brick --
        the parts that share the wood's hue and would otherwise be invisible to the chroma term.
        """
        chroma = self.residual[..., :2]
        cinv = np.linalg.inv(self.covariance[:2, :2])
        d_chroma = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", chroma, cinv, chroma), 0.0))
        d_bright = np.clip(self.residual[..., 2] / self.sigma[2], 0.0, None)
        return np.sqrt(d_chroma**2 + d_bright**2)


@dataclass
class Brick:
    index: int
    mask: np.ndarray = field(repr=False)
    contour: np.ndarray = field(repr=False)
    centroid_px: Tuple[float, float] = (0.0, 0.0)
    area_px: float = 0.0
    obb_center_px: Tuple[float, float] = (0.0, 0.0)
    obb_size_px: Tuple[float, float] = (0.0, 0.0)  # (long, short)
    obb_angle_deg: float = 0.0  # of the long axis, image frame, x right / y down
    rectangularity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 1.0
    colour_name: str = "unknown"
    colour_rgb: Tuple[int, int, int] = (0, 0, 0)
    deviation: float = 0.0
    edge_support: float = 0.0
    interior_texture: float = 0.0
    confidence: float = 0.0
    isolation_px: float = 0.0
    jaw_clearance_px: float = 0.0
    exposed_ratio: float = 0.0
    stud_count: int = 0
    score: float = 0.0
    score_terms: Dict[str, float] = field(default_factory=dict)
    priority: Optional[int] = None
    is_clump: bool = False
    graspable: bool = True
    reject_reason: Optional[str] = None


# --------------------------------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------------------------------


def detect_board(bgr: np.ndarray) -> Board:
    """The lit table, as the largest bright blob's convex hull, trimmed in from its edge."""
    grey = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        mask = np.ones(bgr.shape[:2], bool)
        h, w = bgr.shape[:2]
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.int32)
        return Board(mask, quad, int(mask.sum()))

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillPoly(filled, [hull], 1)
    margin = max(3, BOARD_MARGIN_PX)
    mask = cv2.erode(filled, np.ones((margin, margin), np.uint8)).astype(bool)

    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    return Board(mask, quad, int(mask.sum()))


def opponent_image(bgr: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """(chroma_rg, chroma_yb, log intensity), so shading moves a pixel along one axis only."""
    smooth = cv2.GaussianBlur(bgr, (0, 0), sigma).astype(np.float32) + 6.0
    log = np.log(smooth)
    b, g, r = log[..., 0], log[..., 1], log[..., 2]
    return np.stack([(r - g) / np.sqrt(2.0), (r + g - 2 * b) / np.sqrt(6.0), (r + g + b) / 3.0], -1)


def _normalized_convolution(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Blur ``values`` over only the pixels ``weights`` selects, and renormalize.

    Done on a shrunk copy: the result is a blur this wide, so it holds nothing the small image
    cannot, and a kernel of a few hundred pixels costs a hundred times what one of a few does.
    """
    h, w = weights.shape
    step = max(1, int(sigma / BACKGROUND_DOWNSAMPLE_SIGMA_PX))
    small = (max(w // step, 8), max(h // step, 8))
    mask = cv2.resize(weights.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    data = cv2.resize(values * weights.astype(np.float32)[..., None], small, interpolation=cv2.INTER_AREA)
    scaled = sigma * small[0] / w
    blended = cv2.GaussianBlur(data, (0, 0), scaled) / (cv2.GaussianBlur(mask, (0, 0), scaled) + 1e-6)[..., None]
    return cv2.resize(blended, (w, h), interpolation=cv2.INTER_LINEAR)


def fit_table_model(opponent: np.ndarray, board: Board) -> TableModel:
    """Blur the table's own pixels into an estimate of it, re-deciding which pixels those are.

    A polynomial over the whole board cannot follow the lighting band along one edge, and a plain
    blur is dragged upward by the pile sitting in the middle of it. Weighting the blur by the
    current table mask does both: the estimate under the pile is extrapolated from the wood ringing
    it, and the mask sharpens each round as the bricks drop out of it.
    """
    sigma = TABLE_MODEL_SIGMA_FRAC * float(np.hypot(*opponent.shape[:2]))
    inliers = board.mask.copy()
    residual = np.zeros_like(opponent)
    cov = np.eye(3, dtype=np.float64)
    for _ in range(TABLE_MODEL_ITERATIONS):
        residual = opponent - _normalized_convolution(opponent, inliers, sigma)
        samples = residual[inliers].reshape(-1, 3)
        cov = np.cov(samples.T) + np.eye(3) * 1e-9
        d = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", residual, np.linalg.inv(cov), residual), 0.0))
        inliers = board.mask & (d < TABLE_INLIER_SIGMA)
    return TableModel(residual, cov, np.sqrt(np.diag(cov)), inliers)


def pile_workspace(strong: np.ndarray, board: Board) -> np.ndarray:
    """The part of the table the pile was tipped out onto, grown by a margin."""
    bridged = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((PILE_BRIDGE_PX,) * 2, np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged)
    if count < 2:
        return board.mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    region = np.zeros(strong.shape, np.uint8)
    cv2.fillPoly(region, [hull], 1)
    region = cv2.dilate(region, np.ones((2 * PILE_MARGIN_PX + 1,) * 2, np.uint8))
    return region.astype(bool) & board.mask


def segment_foreground(deviation: np.ndarray, board: Board) -> Tuple[np.ndarray, np.ndarray]:
    strong = (deviation > FOREGROUND_STRONG_SIGMA) & board.mask
    strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strong = _drop_small(strong, FOREGROUND_SEED_AREA_PX)
    workspace = pile_workspace(strong, board)
    strong = strong * workspace

    weak = ((deviation > FOREGROUND_WEAK_SIGMA) & workspace).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(weak)
    seeded = np.unique(labels[strong > 0])
    fg = np.isin(labels, seeded[seeded > 0]).astype(np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((FOREGROUND_CLOSE_PX,) * 2, np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((FOREGROUND_OPEN_PX,) * 2, np.uint8))
    fg = _drop_small(fg, FOREGROUND_MIN_AREA_PX)
    return ndi.binary_fill_holes(fg > 0).astype(np.uint8), workspace


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(max(count, 1), bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# instances
# --------------------------------------------------------------------------------------------------


def edge_strength(model: TableModel) -> Tuple[np.ndarray, np.ndarray]:
    """Border evidence, as the gradient of the frame with all three channels put on one scale.

    Dividing each channel by the table's own scatter in it is what lets one number stand for both
    "these two bricks are different colours" and "there is a shadow line between them"; in raw
    values the intensity channel would drown the chroma ones and same-brightness neighbours of
    different colours would have no border at all.
    """
    normalized = np.clip((model.residual / model.sigma) * 24.0 + 128.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), EDGE_BLUR_SIGMA)
    total = np.zeros(blurred.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(blurred[..., c], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(blurred[..., c], cv2.CV_32F, 0, 1, 3)
        total += gx * gx + gy * gy
    return np.sqrt(total), normalized


def _nearest_label(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Flood every unassigned pixel of ``region`` with its nearest assigned label."""
    holes = (labels <= 0) & (region > 0)
    if not holes.any():
        return labels
    _, (iy, ix) = ndi.distance_transform_edt(labels <= 0, return_indices=True)
    out = labels.copy()
    out[holes] = labels[iy[holes], ix[holes]]
    return out


def oversegment(fg: np.ndarray, normalized: np.ndarray, gradient_norm: np.ndarray) -> np.ndarray:
    """Watershed the foreground from its flat interiors, deliberately into too many pieces."""
    seeds = ((gradient_norm < SEED_GRADIENT_FRACTION) & (fg > 0)).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seeds = _drop_small(seeds, SEED_MIN_AREA_PX)

    count, seed_labels = cv2.connectedComponents(seeds)
    markers = np.where(seed_labels > 0, seed_labels + 1, 0).astype(np.int32)
    markers[fg == 0] = 1
    if count < 2:
        return (fg > 0).astype(np.int32)

    cv2.watershed(normalized, markers)
    labels = np.where((markers > 1) & (fg > 0), markers - 1, 0).astype(np.int32)
    return _nearest_label(labels, fg)


def _region_stats(labels: np.ndarray, feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    means = np.stack(
        [np.bincount(labels.ravel(), weights=feature[..., c].ravel(), minlength=n) for c in range(feature.shape[2])],
        axis=1,
    )
    means /= np.maximum(areas, 1)[:, None]
    return areas, means


def _adjacency(labels: np.ndarray, gradient_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every touching pair of regions, with the length of their shared border and its mean gradient."""
    stride = int(labels.max()) + 1
    keys: List[np.ndarray] = []
    values: List[np.ndarray] = []
    for da, db in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - da, : labels.shape[1] - db]
        b = labels[da:, db:]
        g = 0.5 * (gradient_norm[: labels.shape[0] - da, : labels.shape[1] - db] + gradient_norm[da:, db:])
        sel = (a > 0) & (b > 0) & (a != b)
        if not sel.any():
            continue
        keys.append(np.minimum(a[sel], b[sel]).astype(np.int64) * stride + np.maximum(a[sel], b[sel]))
        values.append(g[sel])
    if not keys:
        return np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(0)

    flat = np.concatenate(keys)
    grads = np.concatenate(values)
    # Median rather than mean along the border: a stud rim clipping the border, or the one corner
    # where two bricks really do touch, must not by itself make a border out of a non-border.
    order = np.lexsort((grads, flat))
    unique, starts, lengths = np.unique(flat[order], return_index=True, return_counts=True)
    strengths = grads[order][starts + lengths // 2]
    return np.stack([unique // stride, unique % stride], axis=1), lengths.astype(np.float64), strengths


def _union_fill_ratio(labels: np.ndarray, a: int, b: int) -> float:
    """How well the two regions together fill their common minimum-area box."""
    ys, xs = np.nonzero((labels == a) | (labels == b))
    if len(xs) == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
    crop[ys - y0, xs - x0] = 1
    return _fill_ratio(crop)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def merge_regions(labels: np.ndarray, model: TableModel, gradient_norm: np.ndarray) -> np.ndarray:
    """Glue the fragments of each brick back together: same colour, no border between them.

    Lego is moulded in flat uniform colour, so two pieces of one brick match closely while two
    bricks rarely do -- but two neighbouring bricks of the *same* colour also match, which is why
    the gradient along the shared border has to be weak as well. Colour alone would fuse the pile's
    many sand-blue parts into one blob; border strength alone would keep every stud rim.
    """
    feature = model.residual / model.sigma
    for _ in range(MERGE_ROUNDS):
        areas, means = _region_stats(labels, feature)
        pairs, lengths, edges = _adjacency(labels, gradient_norm)
        if not len(pairs):
            break
        colours = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
        small = np.minimum(areas[pairs[:, 0]], areas[pairs[:, 1]]) < MERGE_ABSORB_AREA_PX
        joins = (colours < MERGE_COLOUR_TOL) & (edges < MERGE_EDGE_TOL) & (lengths >= MERGE_MIN_BORDER_PX)
        absorbs = small & ((colours < MERGE_COLOUR_TOL * 1.6) | (edges < MERGE_EDGE_TOL * 1.6))

        uf = _UnionFind(len(areas))
        selected = np.nonzero(joins | absorbs)[0]
        for i in selected[np.argsort(colours[selected] + edges[selected])]:
            a, b = int(pairs[i, 0]), int(pairs[i, 1])
            # Two same-coloured bricks lying at an angle to each other pass the colour and border
            # tests and would fuse into an L, which no lego part is. Absorbing a fragment is exempt:
            # a fragment is part of a brick, so it can only make the shape better.
            if not small[i] and _union_fill_ratio(labels, a, b) < MERGE_MIN_FILL:
                continue
            uf.union(a, b)
        remap = np.array([uf.find(i) for i in range(len(areas))], np.int32)
        _, labels_flat = np.unique(remap[labels.ravel()], return_inverse=True)
        merged = labels_flat.reshape(labels.shape).astype(np.int32)
        if merged.max() == labels.max():
            labels = merged
            break
        labels = merged
    return labels


def _obb(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.minAreaRect(max(contours, key=cv2.contourArea))


def _fill_ratio(mask: np.ndarray) -> float:
    rect = _obb(mask)
    if rect is None:
        return 0.0
    w, h = rect[1]
    return float(mask.sum() / max(w * h, 1.0))


def _best_cut(mask: np.ndarray, gradient_norm: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split a region with one straight cut across a gradient ridge, if there is one to cut on.

    A lego silhouette is a rectangle, so a region that fills its own minimum-area box badly is two
    bricks the colour test could not tell apart. Both cut directions are tried, because two bricks
    lying side by side and two lying end to end need opposite cuts.
    """
    rect = _obb(mask)
    if rect is None:
        return None
    (cx, cy), (w, h), angle = rect
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    u = (xs - cx) * cos + (ys - cy) * sin  # along the box's w side
    v = -(xs - cx) * sin + (ys - cy) * cos
    g = gradient_norm[ys, xs]

    best = None
    for coords, extent in ((u, w), (v, h)):
        if extent < 2 * SPLIT_MIN_SIDE_PX:
            continue
        lo, hi = -extent / 2 + SPLIT_MIN_SIDE_PX, extent / 2 - SPLIT_MIN_SIDE_PX
        edges = np.arange(lo, hi, 1.0)
        if len(edges) < 3:
            continue
        bins = np.clip(((coords - lo) / 1.0).astype(int), 0, len(edges) - 1)
        inside = (coords >= lo) & (coords <= hi)
        counts = np.bincount(bins[inside], minlength=len(edges)).astype(np.float64)
        sums = np.bincount(bins[inside], weights=g[inside], minlength=len(edges))
        profile = sums / np.maximum(counts, 1)
        profile = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1), (0, 0), 1.5).ravel()
        profile[counts < 3] = 0.0
        k = int(np.argmax(profile))
        if profile[k] < SPLIT_MIN_EDGE:
            continue
        if best is None or profile[k] > best[0]:
            best = (float(profile[k]), float(edges[k]), coords)

    if best is None:
        return None
    _, offset, coords = best
    side_a = coords < offset
    part_a = np.zeros_like(mask)
    part_b = np.zeros_like(mask)
    part_a[ys[side_a], xs[side_a]] = 1
    part_b[ys[~side_a], xs[~side_a]] = 1
    part_a = _drop_small(part_a, SPLIT_MIN_SIDE_PX**2)
    part_b = _drop_small(part_b, SPLIT_MIN_SIDE_PX**2)
    if part_a.sum() == 0 or part_b.sum() == 0:
        return None
    if min(_fill_ratio(part_a), _fill_ratio(part_b)) <= _fill_ratio(mask):
        return None
    return part_a, part_b


def split_regions(labels: np.ndarray, gradient_norm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels)
    next_label = 1
    for value in range(1, int(labels.max()) + 1):
        mask = (labels == value).astype(np.uint8)
        if mask.sum() < MIN_BRICK_AREA_PX:
            if mask.any():
                out[mask > 0] = next_label
                next_label += 1
            continue
        stack = [(mask, 0)]
        while stack:
            part, depth = stack.pop()
            cut = _best_cut(part, gradient_norm) if depth < SPLIT_MAX_DEPTH and _fill_ratio(part) < SPLIT_FILL_TARGET else None
            if cut is None:
                out[part > 0] = next_label
                next_label += 1
            else:
                stack.extend((piece, depth + 1) for piece in cut)
    return out


def segment_instances(fg: np.ndarray, model: TableModel) -> Tuple[np.ndarray, np.ndarray]:
    gradient, normalized = edge_strength(model)
    scale = float(np.percentile(gradient[fg > 0], 95)) if (fg > 0).any() else 1.0
    gradient_norm = gradient / max(scale, 1e-6)
    labels = oversegment(fg, normalized, gradient_norm)
    labels = merge_regions(labels, model, gradient_norm)
    labels = split_regions(labels, gradient_norm)
    labels = _nearest_label(labels, fg)
    # A merge across a diagonal touch or a straight cut can leave one label in two pieces, and one
    # brick per label is what everything downstream assumes.
    return _split_disconnected(labels), gradient_norm


def _split_disconnected(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels)
    next_label = 1
    for value in range(1, int(labels.max()) + 1):
        mask = (labels == value).astype(np.uint8)
        if not mask.any():
            continue
        count, pieces = cv2.connectedComponents(mask)
        for piece in range(1, count):
            out[pieces == piece] = next_label
            next_label += 1
    return out


# --------------------------------------------------------------------------------------------------
# bricks
# --------------------------------------------------------------------------------------------------


def _colour_name(rgb: Sequence[float]) -> str:
    sample = np.array(rgb, np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB).astype(np.float32).ravel()
    best, best_d = "unknown", float("inf")
    for name, ref in COLOUR_PALETTE.items():
        ref_lab = cv2.cvtColor(np.array(ref, np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32).ravel()
        d = float(np.linalg.norm(lab - ref_lab))
        if d < best_d:
            best, best_d = name, d
    return best


def _brick_confidence(brick: Brick) -> float:
    cues = {
        "deviation": np.clip(brick.deviation / CONFIDENCE_DEVIATION_SIGMA, 0.0, 1.0),
        "edge_support": np.clip(brick.edge_support / CONFIDENCE_EDGE_SUPPORT, 0.0, 1.0),
        "interior_texture": np.clip((1.0 - brick.interior_texture) / CONFIDENCE_INTERIOR_TEXTURE, 0.0, 1.0),
    }
    return float(sum(CONFIDENCE_WEIGHTS[k] * v for k, v in cues.items()))


def build_bricks(
    labels: np.ndarray,
    bgr: np.ndarray,
    deviation: np.ndarray,
    gradient_norm: np.ndarray,
    foreground: np.ndarray,
    board: Board,
) -> Tuple[List[Brick], List[Brick]]:
    """One :class:`Brick` per region, with the regions that cannot be bricks split off."""
    max_area = MAX_BRICK_AREA_FRACTION * board.area_px
    kept: List[Brick] = []
    rejected: List[Brick] = []
    for value in range(1, int(labels.max()) + 1):
        mask = (labels == value).astype(np.uint8)
        area = float(mask.sum())
        if area < 1:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        long_side, short_side = (max(w, h), min(w, h))
        if long_side < 1 or short_side < 1:
            continue
        if h > w:
            angle += 90.0
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        moments = cv2.moments(mask, binaryImage=True)
        centroid = (moments["m10"] / max(moments["m00"], 1e-6), moments["m01"] / max(moments["m00"], 1e-6))
        bgr_median = np.median(bgr[mask > 0].reshape(-1, 3), axis=0)
        outline = np.zeros_like(mask)
        cv2.drawContours(outline, [contour], -1, 1, 2)
        inside = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if inside.sum() < 25:
            inside = mask
        # Measured against bare table only. A brick buried in the pile is ringed by other bricks,
        # whose tops are as flat as its own, and comparing it to those says nothing about whether
        # it is wood -- but a region with no table around it is inside the pile, so it is a brick.
        ring = cv2.dilate(mask, np.ones((13, 13), np.uint8)) - cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ring = ring * (foreground == 0)

        brick = Brick(
            index=len(kept) + len(rejected),
            mask=mask,
            contour=contour,
            centroid_px=(float(centroid[0]), float(centroid[1])),
            area_px=area,
            obb_center_px=(float(cx), float(cy)),
            obb_size_px=(float(long_side), float(short_side)),
            obb_angle_deg=float(angle % 180.0),
            rectangularity=float(area / max(long_side * short_side, 1.0)),
            solidity=float(area / max(hull_area, 1.0)),
            aspect_ratio=float(long_side / max(short_side, 1e-6)),
            colour_rgb=(int(bgr_median[2]), int(bgr_median[1]), int(bgr_median[0])),
            deviation=float(np.median(deviation[mask > 0])),
            edge_support=float(np.median(gradient_norm[outline > 0])),
            interior_texture=(
                float(np.median(gradient_norm[inside > 0]) / max(float(np.median(gradient_norm[ring > 0])), 1e-6))
                if ring.sum() >= RING_MIN_TABLE_PX
                else 0.0
            ),
        )
        brick.colour_name = _colour_name(brick.colour_rgb)

        brick.confidence = _brick_confidence(brick)

        # Failing the shape tests means the region is not one brick -- but on this much colour
        # evidence it is certainly bricks, so it is kept as an unresolved clump rather than thrown
        # away. Outlining it says "there is something here the detector could not take apart",
        # which is the truth, where dropping it would draw bare table over several real bricks.
        brick.is_clump = brick.rectangularity < MIN_RECTANGULARITY or brick.solidity < MIN_SOLIDITY

        reason = None
        if area < MIN_BRICK_AREA_PX:
            reason = "too_small"
        elif area > max_area:
            reason = "too_large"
        elif short_side < MIN_BRICK_SIDE_PX:
            reason = "too_thin"
        elif brick.aspect_ratio > MAX_ASPECT_RATIO:
            reason = "not_brick_shaped"
        elif brick.confidence < MIN_BRICK_CONFIDENCE:
            reason = "looks_like_table"
        elif brick.is_clump and brick.confidence < PRIORITY_MIN_CONFIDENCE:
            reason = "shapeless_and_unconfident"

        if reason is None:
            kept.append(brick)
        else:
            brick.reject_reason = reason
            rejected.append(brick)

    for i, brick in enumerate(kept):
        brick.index = i
    return kept, rejected


def measure_clearance(bricks: List[Brick], shape: Tuple[int, int]) -> None:
    """How much bare table each brick has around it, and beside its two grasp faces."""
    occupancy = np.zeros(shape, np.uint8)
    for brick in bricks:
        occupancy[brick.mask > 0] = 1
    for brick in bricks:
        others = occupancy.copy()
        others[brick.mask > 0] = 0
        free = cv2.distanceTransform((1 - others).astype(np.uint8), cv2.DIST_L2, 3)

        cx, cy = brick.obb_center_px
        theta = np.deg2rad(brick.obb_angle_deg)
        nx, ny = -np.sin(theta), np.cos(theta)  # across the long axis: where the fingertips go
        half = brick.obb_size_px[1] / 2 + JAW_PROBE_MARGIN_PX
        clearances = []
        for sign in (1, -1):
            probe_x = int(round(np.clip(cx + sign * nx * half, 0, shape[1] - 1)))
            probe_y = int(round(np.clip(cy + sign * ny * half, 0, shape[0] - 1)))
            clearances.append(float(free[probe_y, probe_x]))
        brick.jaw_clearance_px = float(min(clearances))

        centroid = (int(round(np.clip(brick.centroid_px[1], 0, shape[0] - 1))), int(round(np.clip(brick.centroid_px[0], 0, shape[1] - 1))))
        interior = cv2.distanceTransform(brick.mask, cv2.DIST_L2, 3)
        brick.isolation_px = float(max(free[centroid] - interior[centroid], 0.0))

        outline = brick.contour.reshape(-1, 2)
        brick.exposed_ratio = float(
            np.mean(free[np.clip(outline[:, 1], 0, shape[0] - 1), np.clip(outline[:, 0], 0, shape[1] - 1)] > 2.5)
        )


def find_studs(bricks: List[Brick], bgr: np.ndarray) -> Tuple[Dict[int, np.ndarray], float]:
    """Stud centres, grouped by the brick they sit on. Lego is the only ruler the frame contains."""
    grey = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=1.0, minDist=8, param1=110, param2=15, minRadius=3, maxRadius=13
    )
    found: Dict[int, List[Tuple[float, float]]] = {}
    radii: List[float] = []
    if circles is None:
        return {}, 0.0
    height, width = grey.shape
    for x, y, r in circles[0]:
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cy < height and 0 <= cx < width):
            continue
        for brick in bricks:
            if brick.mask[cy, cx]:
                brick.stud_count += 1
                found.setdefault(brick.index, []).append((float(x), float(y)))
                radii.append(float(r))
                break
    median_radius = float(np.median(radii)) if radii else 0.0
    return {k: np.array(v, np.float64) for k, v in found.items()}, median_radius


def estimate_mm_per_px(studs: Dict[int, np.ndarray], median_radius: float) -> Tuple[Optional[float], Dict[str, float]]:
    """Millimetres per pixel, from the 8 mm grid the studs on any one brick are moulded on.

    The spacing rather than the stud's own 4.8 mm diameter: a Hough radius is quantized to whole
    pixels and a stud is only about six of them across, so a radius is worth no better than +-15%,
    while two centres eight millimetres apart are located to a fraction of a pixel each. Only
    within one brick, since the gap between two bricks lying next to each other is arbitrary.
    """
    # Two studs 8 mm apart are 3.3 stud-radii apart, so anything under a couple of radii is one
    # stud the circle finder returned twice, and those are exactly the spacings a low percentile
    # would otherwise lock onto.
    floor = STUD_MIN_PITCH_RADII * median_radius
    spacings: List[float] = []
    for centres in studs.values():
        if len(centres) < 2:
            continue
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        spacings.extend(nearest[nearest > floor].tolist())

    info = {
        "stud_count": float(sum(len(v) for v in studs.values())),
        "stud_radius_px_median": median_radius,
        "stud_pitch_samples": float(len(spacings)),
    }
    if len(spacings) < STUD_MIN_SAMPLES:
        return None, info
    pitch = np.asarray(spacings, np.float64)
    # Two studs are never closer than one pitch, but a stud whose neighbour was missed reads as two
    # or three, so the error is one-sided and the median can sit above the true pitch. A low
    # percentile is under the tail wherever the tail is, and the median of what clusters around it
    # then uses every good sample rather than only the smallest ones.
    seed = float(np.percentile(pitch, 25))
    pitch = pitch[(pitch > 0.7 * seed) & (pitch < 1.3 * seed)]
    if len(pitch) < STUD_MIN_SAMPLES:
        return None, info
    median = float(np.median(pitch))
    spread = float(np.median(np.abs(pitch - median)) / max(median, 1e-6))
    info.update({"stud_pitch_px_median": median, "stud_pitch_relative_spread": spread})
    if spread > STUD_MAX_SPREAD or median <= 0:
        return None, info
    return STUD_PITCH_MM / median, info


# --------------------------------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------------------------------


def rank_bricks(bricks: List[Brick], mm_per_px: Optional[float]) -> List[Brick]:
    """Order the bricks by how safely a top-down parallel-jaw grasp would work on each.

    Nothing here needs depth, which is the point: the terms are the ones a single RGB frame can
    actually support. A brick that is isolated, whose outline is a clean rectangle bordering bare
    table, is a brick nothing is lying on top of -- which is the same brick that is highest in the
    pile, arrived at without measuring height. The physical submodule replaces ``exposure`` and
    ``visibility`` with the RealSense height map, and keeps the rest.
    """
    scale = mm_per_px if mm_per_px else 1.0
    for brick in bricks:
        width_mm = brick.obb_size_px[1] * scale
        clearance_mm = brick.jaw_clearance_px * scale
        isolation_mm = brick.isolation_px * scale
        area_mm2 = brick.area_px * scale * scale
        need_mm = FINGER_HALF_THICKNESS_MM if mm_per_px else 0.0

        clearance = float(np.clip((clearance_mm - need_mm * 0.5) / max(CLEARANCE_GOOD_MM, 1e-6), 0.0, 1.0))
        isolation = float(np.clip(isolation_mm / ISOLATION_GOOD_MM, 0.0, 1.0))
        exposure = float(np.clip(brick.exposed_ratio, 0.0, 1.0))
        visibility = float(np.clip((brick.rectangularity - MIN_RECTANGULARITY) / (0.92 - MIN_RECTANGULARITY), 0.0, 1.0))
        size = float(np.clip(area_mm2 / CONFIDENT_AREA_MM2, 0.0, 1.0)) if mm_per_px else float(np.clip(brick.area_px / 1600.0, 0.0, 1.0))

        if mm_per_px:
            if width_mm < GRIPPER_MIN_WIDTH_MM or width_mm > GRIPPER_MAX_WIDTH_MM:
                width_fit = 0.0
            else:
                width_fit = float(np.exp(-((width_mm - GRIPPER_IDEAL_WIDTH_MM) ** 2) / (2 * 12.0**2)))
            brick.graspable = GRIPPER_MIN_WIDTH_MM <= width_mm <= GRIPPER_MAX_WIDTH_MM and area_mm2 >= MIN_PART_AREA_MM2
        else:
            width_fit = 0.5
            brick.graspable = True
        # An unresolved clump has no single outline to aim at, so its geometry is not a grasp.
        brick.graspable = brick.graspable and not brick.is_clump

        brick.score_terms = {
            "clearance": clearance,
            "isolation": isolation,
            "exposure": exposure,
            "visibility": visibility,
            "width_fit": width_fit,
            "size": size,
            "confidence": brick.confidence,
        }
        brick.score = float(sum(SCORE_WEIGHTS[k] * v for k, v in brick.score_terms.items()))
        if not brick.graspable:
            brick.score *= 0.25

    ordered = sorted(bricks, key=lambda b: b.score, reverse=True)
    # A region the detector is unsure of stays in the report and keeps its outline, but the arm is
    # not sent at it: an unconfident region is most likely a mark in the wood, and the cost of
    # closing the gripper on the bare table is a wasted cycle plus a fingertip into the board.
    rank = 0
    for brick in ordered:
        if rank >= PRIORITY_COUNT:
            break
        if brick.confidence < PRIORITY_MIN_CONFIDENCE or not brick.graspable:
            continue
        rank += 1
        brick.priority = rank
    return ordered


# --------------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------------

OUTLINE_COLOUR = (80, 235, 90)
PRIORITY_COLOURS = [(60, 60, 245), (0, 150, 255), (0, 215, 255), (190, 220, 60), (230, 160, 60)]


def render_overlay(bgr: np.ndarray, bricks: List[Brick], ordered: List[Brick], scale: float, label_all: bool) -> np.ndarray:
    out = bgr.copy()
    for brick in bricks:
        contour = np.round(brick.contour.astype(np.float32) * scale).astype(np.int32)
        cv2.drawContours(out, [contour], -1, OUTLINE_COLOUR, 2, cv2.LINE_AA)

    for brick in sorted((b for b in ordered if b.priority), key=lambda b: b.priority or 0):
        colour = PRIORITY_COLOURS[(brick.priority or 1) - 1]
        contour = np.round(brick.contour.astype(np.float32) * scale).astype(np.int32)
        cv2.drawContours(out, [contour], -1, colour, 3, cv2.LINE_AA)

        cx, cy = brick.obb_center_px[0] * scale, brick.obb_center_px[1] * scale
        theta = np.deg2rad(brick.obb_angle_deg)
        nx, ny = -np.sin(theta), np.cos(theta)
        reach = (brick.obb_size_px[1] / 2 + JAW_PROBE_MARGIN_PX) * scale
        for sign in (1, -1):
            a = (int(round(cx + sign * nx * reach * 0.75)), int(round(cy + sign * ny * reach * 0.75)))
            b = (int(round(cx + sign * nx * reach * 1.6)), int(round(cy + sign * ny * reach * 1.6)))
            cv2.line(out, a, b, colour, 2, cv2.LINE_AA)

        label = str(brick.priority)
        radius = 15
        centre = (int(round(cx)), int(round(cy)))
        cv2.circle(out, centre, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, centre, radius, colour, 2, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.putText(out, label, (centre[0] - size[0] // 2, centre[1] + size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)

    if label_all:
        for brick in bricks:
            if brick.priority:
                continue
            p = (int(round(brick.centroid_px[0] * scale)) - 8, int(round(brick.centroid_px[1] * scale)) + 5)
            cv2.putText(out, str(brick.index), p, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(out, str(brick.index), p, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _brick_record(brick: Brick, scale: float, mm_per_px: Optional[float]) -> Dict:
    def mm(value_px: float) -> Optional[float]:
        return round(value_px * mm_per_px, 2) if mm_per_px else None

    long_px, short_px = brick.obb_size_px
    yaw = (brick.obb_angle_deg + 90.0) % 180.0  # jaws close across the long axis
    return {
        "id": brick.index,
        "priority": brick.priority,
        "confidence": round(brick.confidence, 3),
        "colour": {"name": brick.colour_name, "rgb": list(brick.colour_rgb)},
        "centroid_px": [round(brick.centroid_px[0] * scale, 1), round(brick.centroid_px[1] * scale, 1)],
        "area_px": round(brick.area_px * scale * scale, 1),
        "area_mm2": round(brick.area_px * mm_per_px**2, 1) if mm_per_px else None,
        "obb": {
            "center_px": [round(brick.obb_center_px[0] * scale, 1), round(brick.obb_center_px[1] * scale, 1)],
            "length_px": round(long_px * scale, 1),
            "width_px": round(short_px * scale, 1),
            "length_mm": mm(long_px),
            "width_mm": mm(short_px),
            "angle_deg": round(brick.obb_angle_deg, 1),
        },
        "shape": {
            "aspect_ratio": round(brick.aspect_ratio, 2),
            "rectangularity": round(brick.rectangularity, 3),
            "solidity": round(brick.solidity, 3),
            "studs_visible": brick.stud_count,
            "unresolved_clump": bool(brick.is_clump),
        },
        "context": {
            "table_deviation_sigma": round(brick.deviation, 1),
            "edge_support": round(brick.edge_support, 3),
            "interior_texture": round(brick.interior_texture, 3),
            "isolation_px": round(brick.isolation_px * scale, 1),
            "isolation_mm": mm(brick.isolation_px),
            "jaw_clearance_px": round(brick.jaw_clearance_px * scale, 1),
            "jaw_clearance_mm": mm(brick.jaw_clearance_px),
            "exposed_perimeter_ratio": round(brick.exposed_ratio, 3),
        },
        "grasp": {
            "point_px": [round(brick.obb_center_px[0] * scale, 1), round(brick.obb_center_px[1] * scale, 1)],
            "yaw_deg": round(yaw, 1),
            "jaw_opening_px": round(short_px * scale, 1),
            "jaw_opening_mm": mm(short_px),
            "feasible": bool(brick.graspable),
        },
        "score": round(brick.score, 4),
        "score_terms": {k: round(v, 3) for k, v in brick.score_terms.items()},
    }


def build_manifest(
    image_path: str,
    shape: Tuple[int, int],
    board: Board,
    bricks: List[Brick],
    ordered: List[Brick],
    rejected: List[Brick],
    scale: float,
    mm_per_px: Optional[float],
    scale_info: Dict[str, float],
    mm_per_px_source: str,
) -> Dict:
    centroids = np.array([b.centroid_px for b in bricks], np.float64) * scale if bricks else np.zeros((0, 2))
    areas = np.array([b.area_px for b in bricks], np.float64) * scale * scale if bricks else np.zeros(0)
    widths = np.array([b.obb_size_px[1] for b in bricks], np.float64) * scale if bricks else np.zeros(0)
    clearances = np.array([b.jaw_clearance_px for b in bricks], np.float64) * scale if bricks else np.zeros(0)

    colours: Dict[str, int] = {}
    for brick in bricks:
        colours[brick.colour_name] = colours.get(brick.colour_name, 0) + 1
    reasons: Dict[str, int] = {}
    for brick in rejected:
        reasons[brick.reject_reason or "unknown"] = reasons.get(brick.reject_reason or "unknown", 0) + 1

    if len(centroids):
        pile_centroid = centroids.mean(axis=0).round(1).tolist()
        bbox = [
            float(centroids[:, 0].min().round(1)),
            float(centroids[:, 1].min().round(1)),
            float(centroids[:, 0].max().round(1)),
            float(centroids[:, 1].max().round(1)),
        ]
    else:
        pile_centroid, bbox = None, None

    return {
        "schema": "pile_perception/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": {
            "path": os.path.abspath(image_path),
            "name": os.path.basename(image_path),
            "width_px": shape[1],
            "height_px": shape[0],
        },
        "scale": {
            # Per pixel of the source frame, so it applies directly to every *_px below.
            "mm_per_px": round(mm_per_px / scale, 5) if mm_per_px else None,
            "source": mm_per_px_source,
            **{k: round(v, 4) for k, v in scale_info.items()},
        },
        "table": {
            "area_px": int(board.area_px * scale * scale),
            "quad_px": (np.round(board.quad.astype(np.float64) * scale).astype(int)).tolist(),
        },
        "summary": {
            "brick_count": len(bricks),
            "graspable_count": int(sum(1 for b in bricks if b.graspable)),
            "unresolved_clump_count": int(sum(1 for b in bricks if b.is_clump)),
            "rejected_region_count": len(rejected),
            "rejected_reasons": reasons,
            "colour_histogram": dict(sorted(colours.items(), key=lambda kv: -kv[1])),
            "brick_area_px_total": float(areas.sum().round(1)) if len(areas) else 0.0,
            "table_coverage_fraction": round(float(areas.sum() / max(board.area_px * scale * scale, 1)), 4) if len(areas) else 0.0,
            "median_brick_width_px": round(float(np.median(widths)), 1) if len(widths) else None,
            "median_jaw_clearance_px": round(float(np.median(clearances)), 1) if len(clearances) else None,
            "isolated_brick_count": int(sum(1 for b in bricks if b.jaw_clearance_px * scale > 12)),
            "pile_centroid_px": pile_centroid,
            "pile_bbox_px": bbox,
        },
        "grasp_order": [b.index for b in sorted((x for x in ordered if x.priority), key=lambda x: x.priority or 0)],
        "bricks": [_brick_record(b, scale, mm_per_px) for b in ordered],
        "params": {
            "working_long_side": WORKING_LONG_SIDE,
            "foreground_strong_sigma": FOREGROUND_STRONG_SIGMA,
            "foreground_weak_sigma": FOREGROUND_WEAK_SIGMA,
            "merge_colour_tol": MERGE_COLOUR_TOL,
            "merge_edge_tol": MERGE_EDGE_TOL,
            "split_fill_target": SPLIT_FILL_TARGET,
            "min_brick_area_px": MIN_BRICK_AREA_PX,
            "min_rectangularity": MIN_RECTANGULARITY,
            "min_brick_confidence": MIN_BRICK_CONFIDENCE,
            "priority_min_confidence": PRIORITY_MIN_CONFIDENCE,
            "score_weights": SCORE_WEIGHTS,
            "gripper_width_mm": [GRIPPER_MIN_WIDTH_MM, GRIPPER_MAX_WIDTH_MM],
        },
    }


def _debug_panel(
    bgr: np.ndarray,
    board: Board,
    deviation: np.ndarray,
    fg: np.ndarray,
    workspace: np.ndarray,
    labels: np.ndarray,
    rejected: List[Brick],
) -> np.ndarray:
    dev = cv2.applyColorMap((np.clip(deviation / 8.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dev[~board.mask] = 0
    fgv = bgr.copy()
    fgv[fg == 0] = (fgv[fg == 0] * 0.25).astype(np.uint8)
    edges, _ = cv2.findContours(workspace.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fgv, edges, -1, (0, 220, 255), 2)
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 255, (int(labels.max()) + 2, 3)).astype(np.uint8)
    seg = np.where((labels > 0)[..., None], palette[labels], 0).astype(np.uint8)

    drops = bgr.copy()
    for brick in rejected:
        cv2.drawContours(drops, [brick.contour], -1, (60, 60, 245), 2)
        text = f"{brick.reject_reason} d{brick.deviation:.0f} e{brick.edge_support:.2f} t{brick.interior_texture:.2f}"
        cv2.putText(drops, text, (int(brick.centroid_px[0]) - 40, int(brick.centroid_px[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([dev, fgv, cv2.addWeighted(bgr, 0.45, seg, 0.55, 0), drops])


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------


def analyse(bgr: np.ndarray, mm_per_px_override: Optional[float] = None) -> Dict:
    """Run the whole pipeline on one frame. Returns everything the caller might want."""
    h, w = bgr.shape[:2]
    ratio = WORKING_LONG_SIDE / max(h, w)
    work = cv2.resize(bgr, (int(round(w * ratio)), int(round(h * ratio))), interpolation=cv2.INTER_AREA) if ratio < 0.999 else bgr
    scale = w / work.shape[1]

    board = detect_board(work)
    model = fit_table_model(opponent_image(work), board)
    deviation = model.deviation()
    fg, workspace = segment_foreground(deviation, board)
    labels, gradient_norm = segment_instances(fg, model)
    bricks, rejected = build_bricks(labels, work, deviation, gradient_norm, fg, board)
    measure_clearance(bricks, work.shape[:2])

    estimated, scale_info = estimate_mm_per_px(*find_studs(bricks, work))
    if mm_per_px_override:
        mm_per_px, source = mm_per_px_override * scale, "argument"
    elif estimated:
        mm_per_px, source = estimated, "lego_stud_pitch"
    else:
        mm_per_px, source = None, "unavailable"

    ordered = rank_bricks(bricks, mm_per_px)
    return {
        "work": work,
        "scale": scale,
        "board": board,
        "deviation": deviation,
        "fg": fg,
        "workspace": workspace,
        "labels": labels,
        "bricks": bricks,
        "rejected": rejected,
        "ordered": ordered,
        "mm_per_px": mm_per_px,
        "mm_per_px_source": source,
        "scale_info": scale_info,
    }


def process_image(path: str, out_dir: Optional[str], mm_per_px: Optional[float], label_all: bool, debug: bool) -> Dict:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    result = analyse(bgr, mm_per_px)
    stem = os.path.splitext(os.path.basename(path))[0]
    destination = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(destination, exist_ok=True)

    overlay = render_overlay(bgr, result["bricks"], result["ordered"], result["scale"], label_all)
    png_path = os.path.join(destination, f"{stem}_perception.png")
    cv2.imwrite(png_path, overlay)

    manifest = build_manifest(
        path,
        bgr.shape[:2],
        result["board"],
        result["bricks"],
        result["ordered"],
        result["rejected"],
        result["scale"],
        result["mm_per_px"],
        result["scale_info"],
        result["mm_per_px_source"],
    )
    json_path = os.path.join(destination, f"{stem}_manifest.json")
    with open(json_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    if debug:
        panel = _debug_panel(
            result["work"],
            result["board"],
            result["deviation"],
            result["fg"],
            result["workspace"],
            result["labels"],
            result["rejected"],
        )
        cv2.imwrite(os.path.join(destination, f"{stem}_debug.png"), panel)

    return {"png": png_path, "json": json_path, "manifest": manifest}


def _expand(inputs: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("jpg", "jpeg", "png", "bmp"):
                paths.extend(sorted(glob.glob(os.path.join(item, f"*.{ext}"))))
        else:
            paths.extend(sorted(glob.glob(item)) or [item])
    return [p for p in paths if "_perception" not in os.path.basename(p) and "_debug" not in os.path.basename(p)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[LEGO_PIC_DIR],
        help=f"image files, globs or directories (default: {LEGO_PIC_DIR})",
    )
    parser.add_argument("--out-dir", default=None, help="where to write the outputs (default: beside the input)")
    parser.add_argument("--mm-per-px", type=float, default=None, help="override the stud-based scale estimate")
    parser.add_argument("--label-all", action="store_true", help="also draw every brick's id, not just 1..5")
    parser.add_argument("--debug", action="store_true", help="also write <stem>_debug.png with the intermediate stages")
    args = parser.parse_args()

    paths = _expand(args.inputs)
    if not paths:
        parser.error("no images matched")

    for path in paths:
        out = process_image(path, args.out_dir, args.mm_per_px, args.label_all, args.debug)
        summary = out["manifest"]["summary"]
        scale = out["manifest"]["scale"]["mm_per_px"]
        print(
            f"{os.path.basename(path)}: {summary['brick_count']} bricks "
            f"({summary['rejected_region_count']} regions dropped), "
            f"mm/px={scale if scale else 'n/a'}, top={out['manifest']['grasp_order']} -> {os.path.basename(out['png'])}"
        )


if __name__ == "__main__":
    main()
