"""m1 submodule 3: the pile perception -- segment the pile, measure every brick, score the grasps.

**A library, not a step.** Deciding *which* brick to grasp used to live here and be handed to
submodule_1 through ``config.PILE_TARGET_PATH``; it now lives in :mod:`m1.physical.submodule_1`
alongside going there, exactly as it already did in the simulation, because the two halves of that
decision were never usefully separable -- the choice depends on two viewpoints and the position
depends on the choice. What is left here is the part both stacks share and neither duplicates:

    :func:`analyse_pile` -- one frame in, a ranked list of measured bricks out.

:mod:`m1.simulation.submodule_1` calls that function on rendered frames and
:mod:`m1.physical.submodule_1` calls it on RealSense frames, and it cannot tell the difference. That
is the point: the perception is not simulated, it is the same code.

The ``main()`` at the bottom is still here and still works, as a way to look at what the perception
makes of a pile -- or of a saved ``--from-capture`` frame -- without tying up the robot or committing
to a grasp. It writes ``config.PILE_TARGET_PATH``, which nothing in the pipeline reads any more.

The robot-side twin of ``src/pile_perception.py``. That script is RGB-only and robot-free -- it takes
a still photograph of the pile and ranks the grasps in pixels, with millimetres inferred from the 8 mm
grid lego is moulded on. This module does the same job with the wrist RealSense and the hand-eye
calibration in hand, which changes three things and only three:

  * **Scale is measured, not inferred.** Every outline is back-projected onto the table plane the arm
    touched off, so the footprints come out in metres in the robot's base frame. No stud-pitch ruler,
    and no dependence on the pile containing enough visible studs to calibrate against.
  * **Depth decides what is a brick.** The one question a single RGB frame cannot always answer is a
    tan brick against a knot in the plywood -- same colour, same size, same crisp outline. One of them
    is 10 mm off the table, so the height map settles it outright. The RGB confidence cues are kept as
    the fallback for the pixels where the RealSense returns nothing, which on brick edges is many.
  * **Burial replaces "exposure" as the top-of-pile test.** The still-photo version infers that a
    clean isolated rectangle is a brick nothing lies on top of. Here the height of the ring around
    each brick says so directly.

Everything else -- the table appearance model, the hysteresis foreground, the watershed / merge /
split instance segmentation, the clearance and scoring -- is the same pipeline, re-implemented here
so that ``m1`` depends on nothing outside ``config.py``.

Pipeline:
  1. capture_pile_view    -- move to the viewpoint, grab colour + depth + the camera pose
  2. build_scene          -- per-pixel base-frame position and height above the touched-off table
  3. fit_table_model      -- what the bare table looks like at every pixel, seeded from the height map
  4. segment_foreground   -- brick pixels vs. table pixels, inside the pile's workspace
  5. segment_instances    -- cut that foreground into one region per brick
  6. build_bricks         -- metric geometry, height and colour per region, with a confidence
  7. measure_clearance    -- room around each brick, on a top-down millimetre map of the table
  8. rank_bricks          -- order by how safely a top-down Robotiq 2F-85 grasp would work
  9. write_pile_target    -- (``main()`` only) the winner, for inspection

Steps 1-8 are :func:`analyse_pile`, which is what submodule_1 calls. Nothing here moves the arm except
to reach the viewpoint.

    python src/m1/physical/submodule_3.py
    python src/m1/physical/submodule_3.py --stay --debug-dir /tmp/pile
    python src/m1/physical/submodule_3.py --from-capture /tmp/pile/capture_1754.npz
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, NumpyIntImageType
from loguru import logger
from scipy import ndimage as ndi

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import (  # noqa: E402
    APPROX_ARM_REACH,
    CAMERA_RESOLUTIONS,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    FALLBACK_BRICK_HEIGHT,
    PILE_TARGET_MAX_AGE,
    PILE_TARGET_PATH,
    PREGRASP_HEIGHT,
    SUPPORTED_ROBOT_TYPES,
    TABLE_Z,
    TablePlane,
    connect_arm,
    ensure_control_ready,
    find_reachable_hover_orientation,
    load_camera_pose_in_tcp,
    load_table_plane,
    open_camera,
)

# The joint configuration the pile is looked at from. Same viewpoint style as submodule_1's POS1: far
# enough back that the whole pile is in frame and inside the RealSense's minimum range, tilted enough
# that the pile is not hidden behind the gripper. --stay skips the move and uses wherever the arm is.
PILE_VIEW: List[float] = [-0.08343679, -1.31992237, 0.26209098, -0.40548201, -1.20620281, -1.63604099]

# --- working resolution ---------------------------------------------------------------------------
# Every stage below runs on a copy of the frame resized to this long side, and the intrinsics are
# scaled with it, so a 1080p and a 720p capture behave identically. The pixel thresholds are all
# derived from millimetres at runtime (see PixelScale), so the resize is about cost, not tuning.
WORKING_LONG_SIDE = 1280

# --- what the depth stream is allowed to say ------------------------------------------------------
MIN_VALID_DEPTH = 0.10  # metres; below this the RealSense is reporting its own blind zone
MAX_VALID_DEPTH = 1.50
# Height band, above the touched-off table plane, that the scene is allowed to occupy. Below the floor
# is depth noise punching through the tabletop; above the ceiling is the gripper, an arm or a hand,
# none of which are the pile.
SCENE_FLOOR_M = -0.010
SCENE_CEILING_M = 0.100
TABLE_BAND_M = 0.0015  # within this of the plane, a pixel is bare table and seeds the colour model

# --- the table's appearance -----------------------------------------------------------------------
TABLE_MODEL_SIGMA_FRAC = 0.06  # radius of the model, as a fraction of the frame diagonal
TABLE_MODEL_ITERATIONS = 3  # fewer than the RGB-only version needs: the depth seed is already right
BACKGROUND_DOWNSAMPLE_SIGMA_PX = 8.0
TABLE_INLIER_SIGMA = 2.5

# --- foreground ----------------------------------------------------------------------------------
# Depth first, and the thresholds are set by the thinnest part rather than by the noise: a lego plate
# is 3.2 mm tall and is the shortest thing that can be on the table, so a "strong" step has to be
# comfortably under that or every plate in the pile is invisible. What makes 2.5 mm safe against a
# RealSense's per-pixel noise is that the height map is median-filtered over a millimetre or so first
# (see _denoise_height) -- a plate is 8 mm across at its narrowest, so it survives that filter whole
# while single-pixel depth speckle does not.
FOREGROUND_STRONG_HEIGHT_M = 0.0025
FOREGROUND_WEAK_HEIGHT_M = 0.0010
# Colour, in standard deviations of the table's own scatter, used where the depth map has holes --
# which on a brick's edges, on dark plastic and on anything glossy is most of the interesting pixels.
FOREGROUND_STRONG_SIGMA = 7.0
FOREGROUND_WEAK_SIGMA = 3.5
FOREGROUND_SEED_AREA_MM2 = 15.0
FOREGROUND_CLOSE_MM = 2.0
FOREGROUND_OPEN_MM = 1.2
FOREGROUND_MIN_AREA_MM2 = 40.0

# The bricks were all tipped out in one place, so the pile's own footprint grown by a margin is the
# workspace, and a mark on the wood at the far edge of the table is not.
PILE_BRIDGE_MM = 12.0  # blobs closer than this belong to the same pile
PILE_MARGIN_MM = 30.0  # how far outside the pile's hull a brick that skidded clear is still workspace

# --- instance splitting -----------------------------------------------------------------------------
EDGE_BLUR_MM = 0.9  # smooths stud shading and grain so only brick borders survive as ridges
# A height step this big is a full border on its own, whatever the two bricks' colours are. Half a
# plate: two bricks lying side by side on the table rarely differ by more than noise, two stacked ones
# differ by a whole part height.
HEIGHT_EDGE_SCALE_M = 0.0016
SEED_GRADIENT_FRACTION = 0.30  # a watershed seed is foreground flatter than this much of the 95th pct
SEED_MIN_AREA_MM2 = 12.0
MERGE_ROUNDS = 3
MERGE_COLOUR_TOL = 2.6  # region colour distance, in table-scatter sigmas
MERGE_EDGE_TOL = 0.42  # normalized gradient along the shared border below which it is not a border
MERGE_HEIGHT_TOL_M = 0.0025  # two fragments of one brick sit at one height; a stack does not
MERGE_ABSORB_AREA_MM2 = 25.0  # fragments smaller than this always join their closest neighbour
MERGE_MIN_BORDER_MM = 2.0  # two bricks meeting at a corner are not one brick
MERGE_MIN_FILL = 0.62  # a merge that leaves an L or a T joined two bricks, not one brick's pieces
SPLIT_FILL_TARGET = 0.68  # a region filling its own box worse than this is more than one brick
SPLIT_MIN_SIDE_MM = 5.0  # no lego part is thinner than this, so no cut may leave a sliver
SPLIT_MIN_EDGE = 0.30  # a cut has to run along a real gradient ridge, not through a flat brick top
SPLIT_MAX_DEPTH = 3

# --- what counts as a brick -----------------------------------------------------------------------
# A 1x1 plate is 7.8 mm square and 61 mm2, and it is the smallest part in the catalogue. Under
# MIN_BRICK_AREA_MM2 a region is a screw hole or a chip in the wood -- both of which are otherwise
# perfect grasps, being small and alone, which is exactly why the floor is here.
MIN_BRICK_AREA_MM2 = 45.0
MAX_BRICK_AREA_MM2 = 2500.0  # 5 x 5 cm; larger than any part in the set, so it is a merge failure
MIN_BRICK_SIDE_MM = 6.0
MAX_ASPECT_RATIO = 14.0
MIN_RECTANGULARITY = 0.42
MIN_SOLIDITY = 0.62
MIN_BRICK_HEIGHT_M = 0.0018  # under this a region is at table level, so it is the table
MAX_BRICK_HEIGHT_M = 0.060  # a stack six bricks tall; taller than that is not the pile

# How sure the detector is that a region is a brick at all. With depth the answer is nearly always
# obvious, so the cues are mostly a way of noticing when it is not:
#
#   height -- how far off the table it stands, which is the whole question
#   flatness -- the spread of heights across it. One brick's top face is flat; a region covering two
#       stacked bricks, or a brick and the table beside it, is not
#   coverage -- how much of the region the depth stream actually returned. A region with no depth is
#       being judged on colour alone, and says so
#   deviation -- how far its colour is from the table's, in the table's own sigmas
#
# Below MIN_DEPTH_COVERAGE the depth cues have nothing to say and the RGB-only confidence from
# pile_perception.py is used instead, so a brick the RealSense could not see is still a candidate --
# just a less trusted one.
# A plate is 3.2 mm and nothing else on this table is 3 mm thick, so a region standing that far off it
# is as certain as the cue ever gets; asking for a full brick's 9.6 mm would rate every plate in the
# pile as a maybe.
CONFIDENT_HEIGHT_M = 0.0032
FLATNESS_TOL_M = 0.0030
CONFIDENT_COVERAGE = 0.55
MIN_DEPTH_COVERAGE = 0.15
CONFIDENCE_DEVIATION_SIGMA = 12.0
CONFIDENCE_EDGE_SUPPORT = 0.80
CONFIDENCE_INTERIOR_TEXTURE = 0.70
DEPTH_CONFIDENCE_WEIGHTS = {"height": 0.40, "flatness": 0.20, "coverage": 0.15, "deviation": 0.25}
RGB_CONFIDENCE_WEIGHTS = {"deviation": 0.45, "edge_support": 0.30, "interior_texture": 0.25}
RING_INNER_MM = 2.0  # the ring just outside a brick, where its surroundings are measured
RING_OUTER_MM = 8.0
RING_MIN_TABLE_PX = 40  # below this much bare table beside it, the texture cue has nothing to say
MIN_BRICK_CONFIDENCE = 0.45  # under this the region is dropped outright
PRIORITY_MIN_CONFIDENCE = 0.70  # under this it is reported and outlined, but never sent the arm at

# --- the gripper ------------------------------------------------------------------------------------
# Robotiq 2F-85. The jaws close on the brick's two long sides, so the brick's short side is the width
# the gripper has to span and the long sides are where the fingertips need room. The usable width is
# the stroke less the margin submodule_2 opens to before descending -- a brick that only fits with the
# fingers already touching it cannot be approached.
GRIPPER_STROKE_MM = 85.0
GRIPPER_APPROACH_MARGIN_MM = 14.0
GRIPPER_MIN_WIDTH_MM = 4.0
GRIPPER_MAX_WIDTH_MM = GRIPPER_STROKE_MM - GRIPPER_APPROACH_MARGIN_MM
GRIPPER_IDEAL_WIDTH_MM = 16.0
GRIPPER_WIDTH_SIGMA_MM = 12.0
FINGER_HALF_THICKNESS_MM = 6.0  # room one fingertip needs beside the brick
# How far down a part's side the fingertips can get before the table stops them, and how much of that
# is enough. A 2F-85's pads are 37 mm tall, but what actually holds a part is the few millimetres of
# it they can reach past: a 9.6 mm brick offers 8 mm of side wall, a 3.2 mm plate offers 1.7 mm, and
# a plate is the part that gets shoved out from between the closing fingers rather than pinched.
# Nothing else in the score notices the difference -- a plate lying alone on bare table scores full
# marks on clearance, isolation, exposure and visibility -- so without this term the ranking sends the
# arm at the hardest grasps in the pile first.
FINGERTIP_CLEARANCE_MM = 1.5  # kept above the table, matching submodule_2's descent cap
GRIP_DEPTH_GOOD_MM = 6.0  # engagement at which the term saturates
CLEARANCE_GOOD_MM = 14.0  # fingertip room at which the clearance term saturates
ISOLATION_GOOD_MM = 26.0  # gap to the rest of the pile at which the isolation term saturates
CONFIDENT_AREA_MM2 = 260.0  # footprint of a 2x4 plate; the area term saturates there
BURIAL_SCALE_M = 0.006  # neighbours this much taller than a brick mean it is underneath them

# --- the top-down map -------------------------------------------------------------------------------
# Clearance is measured on an orthographic millimetre raster of the table rather than in image pixels,
# because a pixel near the top of the frame and one near the bottom are not the same number of
# millimetres and a perspective distance transform quietly mixes the two.
TOPDOWN_MM_PER_CELL = 1.0
TOPDOWN_MARGIN_MM = 40.0
TOPDOWN_MAX_CELLS = 1200  # per side; a guard against a stray footprint blowing the raster up
JAW_PROBE_MARGIN_MM = 2.0  # how far outside the brick the fingertip clearance is sampled
EXPOSED_FREE_MM = 2.5  # bare table this far from a perimeter point counts that point as exposed

SCORE_WEIGHTS = {
    "clearance": 0.28,  # room for the fingertips beside the brick -- the thing that fails first
    "grip_depth": 0.18,  # how far down its side the fingertips get before the table stops them
    "isolation": 0.14,  # how far the brick sits from the rest of the pile
    "top_of_pile": 0.20,  # nothing standing over it, straight off the height map
    "exposure": 0.10,  # how much of its outline borders bare table rather than another brick
    "visibility": 0.08,  # a clean rectangle is a brick nothing is lying across
    "width_fit": 0.08,  # short side comfortably inside the gripper's range
    "size": 0.06,  # a whole brick rather than the corner of one, and a bigger target for the jaws
    "confidence": 0.14,  # how sure the detector is this is a brick at all and not the table
}

PRIORITY_COUNT = 5
MAX_REACHABILITY_CHECKS = 12  # IK round trips are not free; see assign_priorities

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

# Robot-side sanity limits on where a grasp is allowed to be, mirroring submodule_2's.
MIN_BASE_DISTANCE = 0.15
WORKSPACE_MARGIN = 0.90


# =================================================================================================
# what one look at the pile consists of
# =================================================================================================


@dataclass(frozen=True)
class PileView:
    """One capture: what the camera saw, where it was, and how the arm was standing."""

    image_rgb: NumpyIntImageType
    depth_map: Optional[np.ndarray]  # metres, aligned to image_rgb
    intrinsics_matrix: CameraIntrinsicsMatrixType
    X_base_camera: HomogeneousMatrixType
    joint_configuration: Optional[np.ndarray] = None
    name: str = "pile view"

    def save(self, path: str) -> str:
        """Write the capture to an ``.npz`` so a run can be replayed without the robot."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            image_rgb=self.image_rgb,
            depth_map=self.depth_map if self.depth_map is not None else np.zeros((0, 0), np.float32),
            intrinsics_matrix=self.intrinsics_matrix,
            X_base_camera=self.X_base_camera,
            joint_configuration=(
                self.joint_configuration if self.joint_configuration is not None else np.zeros(0)
            ),
        )
        return path

    @staticmethod
    def load(path: str) -> "PileView":
        payload = np.load(path)
        depth = payload["depth_map"]
        joints = payload["joint_configuration"]
        return PileView(
            image_rgb=payload["image_rgb"],
            depth_map=None if depth.size == 0 else depth.astype(np.float32),
            intrinsics_matrix=payload["intrinsics_matrix"],
            X_base_camera=payload["X_base_camera"],
            joint_configuration=None if joints.size == 0 else joints,
            name=os.path.basename(path),
        )


@dataclass(frozen=True)
class PixelScale:
    """How many working-resolution pixels a millimetre on the table is worth.

    Straight off the pinhole model at the distance the pile actually is: ``f / Z``. It is what turns
    every threshold in this file from a number tuned against one photograph into a physical size.
    """

    px_per_mm: float
    distance_m: float

    def length(self, mm: float, minimum: int = 1) -> int:
        return max(minimum, int(round(mm * self.px_per_mm)))

    def area(self, mm2: float, minimum: int = 1) -> int:
        return max(minimum, int(round(mm2 * self.px_per_mm**2)))

    def odd(self, mm: float, minimum: int = 3) -> int:
        """A morphological kernel size: odd, so the structuring element has a centre."""
        size = max(minimum, int(round(mm * self.px_per_mm)))
        return size if size % 2 == 1 else size + 1


@dataclass
class Scene:
    """The frame turned into geometry: where every pixel is, and how far off the table it stands."""

    bgr: np.ndarray  # working resolution, BGR, because every cv2 call below assumes it
    intrinsics_matrix: CameraIntrinsicsMatrixType  # scaled to the working resolution
    X_base_camera: HomogeneousMatrixType
    plane: Tuple[float, float, float]  # z = a*x + b*y + c, the touched-off tabletop
    table_xy: np.ndarray  # (H, W, 2) base-frame x, y where each ray crosses the table plane
    height: np.ndarray  # (H, W) metres above the table plane, from depth; 0 where invalid
    depth_valid: np.ndarray  # (H, W) bool
    reach_mask: np.ndarray  # (H, W) bool, inside the arm's usable horizontal workspace
    table_mask: np.ndarray  # (H, W) bool, bare tabletop
    scale: PixelScale
    working_scale: float  # working pixels per source pixel

    @property
    def shape(self) -> Tuple[int, int]:
        return self.bgr.shape[:2]

    def table_z_at(self, x: float, y: float) -> float:
        a, b, c = self.plane
        return float(a * x + b * y + c)


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
    """One segmented region, measured in the robot's base frame."""

    index: int
    mask: np.ndarray = field(repr=False)
    contour: np.ndarray = field(repr=False)  # working-resolution pixels
    footprint_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False)  # base frame

    # where it is, in the base frame
    center_m: Tuple[float, float] = (0.0, 0.0)  # centre of the minimum-area rectangle
    grasp_pixel: Tuple[int, int] = (0, 0)  # that centre, back in the source frame's pixels
    height_m: float = 0.0  # top face above the tabletop
    height_measured: bool = True  # False when it is config.FALLBACK_BRICK_HEIGHT, for want of depth
    table_z: float = 0.0
    width_mm: float = 0.0  # short side: what the fingers close on
    length_mm: float = 0.0
    long_axis_heading: float = 0.0  # radians, base frame
    area_mm2: float = 0.0

    # how it looks
    area_px: float = 0.0
    rectangularity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 1.0
    colour_name: str = "unknown"
    colour_rgb: Tuple[int, int, int] = (0, 0, 0)

    # how sure we are it is a brick
    deviation: float = 0.0
    edge_support: float = 0.0
    interior_texture: float = 0.0
    depth_coverage: float = 0.0
    height_spread_m: float = 0.0
    neighbour_height_m: float = 0.0
    confidence: float = 0.0
    confidence_source: str = "depth"

    # how good a grasp it is
    jaw_clearance_mm: float = 0.0
    isolation_mm: float = 0.0
    exposed_ratio: float = 0.0
    score: float = 0.0
    score_terms: Dict[str, float] = field(default_factory=dict)
    priority: Optional[int] = None
    is_clump: bool = False
    graspable: bool = True
    reachable: Optional[bool] = None
    reject_reason: Optional[str] = None

    @property
    def closing_heading(self) -> float:
        """Base-frame direction the fingers must close along: square to the brick's long axis."""
        return self.long_axis_heading + math.pi / 2

    @property
    def top_face_z(self) -> float:
        return self.table_z + self.height_m

    def describe(self) -> str:
        return (
            f"#{self.index} {self.colour_name} {self.width_mm:.1f} x {self.length_mm:.1f} x "
            f"{self.height_m * 1000:.1f} mm at ({self.center_m[0]:.3f}, {self.center_m[1]:.3f}) m, "
            f"long axis {math.degrees(self.long_axis_heading):.0f} deg, "
            f"{self.jaw_clearance_mm:.1f} mm fingertip room, score {self.score:.3f}"
        )


@dataclass
class PileAnalysis:
    """Everything one look at the pile produced."""

    view: PileView
    scene: Scene
    model: TableModel
    deviation: np.ndarray
    foreground: np.ndarray
    workspace: np.ndarray
    labels: np.ndarray
    bricks: List[Brick]
    rejected: List[Brick]
    ordered: List[Brick]

    @property
    def target(self) -> Optional[Brick]:
        """The brick to grasp: the highest-ranked one that survived every check."""
        for brick in self.ordered:
            if brick.priority == 1:
                return brick
        return None


# =================================================================================================
# capture
# =================================================================================================


def capture_pile_view(
    arm: Optional[PositionManipulator],
    camera: RGBDCamera,
    X_tcp_camera: HomogeneousMatrixType,
    joint_configuration: Optional[np.ndarray] = None,
    joint_speed: float = 0.1,
    name: str = "pile view",
) -> PileView:
    """Move to the viewpoint if one is given, then grab colour, depth and the camera pose together.

    The arm is stationary by the time the frame is grabbed, so the image and the TCP pose it is paired
    with describe the same instant -- which is the whole basis of the eye-in-hand back-projection.
    Colour and depth come from one ``grab_images`` buffer, so the height map describes the pixels the
    segmentation is about to run on and not the ones from a frame earlier.
    """
    if arm is not None and joint_configuration is not None:
        logger.info(f"Moving to the pile viewpoint: {np.round(joint_configuration, 3)} rad ...")
        arm.move_to_joint_configuration(joint_configuration, joint_speed=joint_speed).wait()

    camera.grab_images()
    image = camera.retrieve_rgb_image_as_int()
    X_base_camera = arm.get_tcp_pose() @ X_tcp_camera if arm is not None else np.eye(4)

    try:
        depth = np.asarray(camera.retrieve_depth_map(), dtype=np.float32)
    except Exception as exception:  # noqa: BLE001 - a missing depth map degrades, it does not abort
        logger.warning(
            f"No depth map available ({exception}); falling back to the RGB-only cues, which cannot "
            "tell a tan brick from a knot in the plywood."
        )
        depth = None

    joints = None
    if arm is not None:
        try:
            joints = np.asarray(arm.get_joint_configuration(), dtype=float)
        except Exception as exception:  # noqa: BLE001 - only recorded for the debug capture
            logger.debug(f"Could not read the joint configuration: {exception}")

    return PileView(
        image_rgb=image,
        depth_map=depth,
        intrinsics_matrix=camera.intrinsics_matrix(),
        X_base_camera=X_base_camera,
        joint_configuration=joints,
        name=name,
    )


def resize_to_working_resolution(
    image: np.ndarray, intrinsics_matrix: CameraIntrinsicsMatrixType, depth: Optional[np.ndarray]
) -> Tuple[np.ndarray, CameraIntrinsicsMatrixType, Optional[np.ndarray], float]:
    """Shrink the frame to :data:`WORKING_LONG_SIDE` and carry the intrinsics and depth with it.

    The intrinsics have to be scaled too, or every back-projection below lands somewhere else. Pixel
    *centres* are what map linearly under a resize, hence the half-pixel shift on the principal point;
    ignoring it is a half-pixel bias, which is under a tenth of a millimetre here but costs nothing to
    get right. Depth is resampled nearest-neighbour so no pixel ends up holding the average of a
    brick's top face and the table beside it.
    """
    height, width = image.shape[:2]
    ratio = WORKING_LONG_SIDE / max(height, width)
    if ratio >= 0.999:
        return image, np.asarray(intrinsics_matrix, float), depth, 1.0

    size = (int(round(width * ratio)), int(round(height * ratio)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    scale = size[0] / width

    K = np.asarray(intrinsics_matrix, float).copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * scale - 0.5

    resized_depth = None if depth is None else cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    return resized, K, resized_depth, scale


# =================================================================================================
# geometry: pixels to the base frame
# =================================================================================================


def back_project_depth(
    depth: np.ndarray, intrinsics_matrix: CameraIntrinsicsMatrixType, X_base_camera: HomogeneousMatrixType
) -> Tuple[np.ndarray, np.ndarray]:
    """Every depth pixel as a base-frame point. Returns ``(points (H, W, 3), valid (H, W))``."""
    height, width = depth.shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

    valid = np.isfinite(depth) & (depth > MIN_VALID_DEPTH) & (depth < MAX_VALID_DEPTH)
    z = np.where(valid, depth, 0.0).astype(np.float32)
    points_camera = np.stack([(columns - cx) * z / fx, (rows - cy) * z / fy, z], axis=-1)
    points_base = points_camera @ np.asarray(X_base_camera[:3, :3], np.float32).T + np.asarray(
        X_base_camera[:3, 3], np.float32
    )
    return points_base, valid


def _fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``values`` with every invalid pixel replaced by its nearest valid neighbour.

    Needed before any spatial filter touches the height map: a hole holds 0, and 0 is not "missing"
    to a median or a Sobel, it is "exactly at table height" -- so an unfilled hole reads as a crater
    and every hole edge reads as a cliff.
    """
    if valid.all() or not valid.any():
        return values
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
    return values[iy, ix]


def _denoise_height(height: np.ndarray, valid: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Median-filter the height map over about a millimetre of table.

    A median rather than a blur, because the whole point of the map is its steps: a blur would ramp
    the 3.2 mm edge of a plate over its own kernel and take the thinnest parts in the pile with it,
    while a median leaves a step a step and simply deletes the isolated pixels a stereo depth sensor
    scatters over a matte surface. One millimetre is far below the 7.8 mm narrowest part and far above
    the speckle.
    """
    if not valid.any():
        return height
    kernel = 5 if px_per_mm >= 1.6 else 3
    smoothed = cv2.medianBlur(_fill_invalid(height, valid).astype(np.float32), kernel)
    return np.where(valid, smoothed, 0.0).astype(np.float32)


def project_pixels_onto_plane(
    pixels: np.ndarray,
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    plane: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    """Back-project ``(N, 2)`` pixels onto the plane ``z = a*x + b*y + c``, in the base frame.

    The tilted generalisation of intersecting a ray with a horizontal plane, so a tabletop that is not
    exactly square to the robot's base is followed rather than averaged away -- 1 degree of tilt is
    7 mm across a 40 cm workspace, which is wider than the brick.

    Returns ``None`` if any ray runs away from the plane instead of crossing it, which means the
    outline being projected is not on the table at all.
    """
    a, b, c = plane
    homogeneous = np.column_stack([np.asarray(pixels, float), np.ones(len(pixels))])
    directions_camera = homogeneous @ np.linalg.inv(np.asarray(intrinsics_matrix, float)).T
    directions_base = directions_camera @ np.asarray(X_base_camera[:3, :3], float).T
    origin = np.asarray(X_base_camera[:3, 3], float)

    denominator = directions_base[:, 2] - a * directions_base[:, 0] - b * directions_base[:, 1]
    if np.any(np.abs(denominator) < 1e-9):
        return None
    distances = (a * origin[0] + b * origin[1] + c - origin[2]) / denominator
    if np.any(distances <= 0):
        return None
    return origin + distances[:, None] * directions_base


def _plane_grid(
    shape: Tuple[int, int],
    intrinsics_matrix: CameraIntrinsicsMatrixType,
    X_base_camera: HomogeneousMatrixType,
    plane: Tuple[float, float, float],
) -> np.ndarray:
    """Where every pixel's ray crosses the table plane, as an ``(H, W, 2)`` map of base-frame x, y.

    Used for the things that need a position for *every* pixel -- the reach mask, and a fallback x, y
    wherever the depth map has a hole. A brick 10 mm off the table lands a fraction of a millimetre
    from its true x, y this way, which is well inside what the hand-eye calibration contributes.
    """
    height, width = shape
    a, b, c = plane
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    directions_camera = np.stack(
        [(columns - cx) / fx, (rows - cy) / fy, np.ones_like(columns)], axis=-1
    )
    directions_base = directions_camera @ np.asarray(X_base_camera[:3, :3], np.float32).T
    origin = np.asarray(X_base_camera[:3, 3], np.float32)

    denominator = directions_base[..., 2] - a * directions_base[..., 0] - b * directions_base[..., 1]
    denominator = np.where(np.abs(denominator) < 1e-9, np.nan, denominator)
    distances = (a * origin[0] + b * origin[1] + c - origin[2]) / denominator
    distances = np.where(distances > 0, distances, np.nan)
    return origin[:2] + distances[..., None] * directions_base[..., :2]


def build_scene(view: PileView, plane: Tuple[float, float, float], robot_type: str) -> Scene:
    """Turn the capture into geometry: base-frame positions, heights, and what is in reach."""
    bgr_full = cv2.cvtColor(view.image_rgb, cv2.COLOR_RGB2BGR)
    bgr, intrinsics, depth, working_scale = resize_to_working_resolution(
        bgr_full, view.intrinsics_matrix, view.depth_map
    )
    shape = bgr.shape[:2]

    table_xy = _plane_grid(shape, intrinsics, view.X_base_camera, plane)
    horizontal = np.hypot(table_xy[..., 0], table_xy[..., 1])
    reach = APPROX_ARM_REACH.get(robot_type, 0.5)
    reach_mask = np.isfinite(horizontal) & (horizontal > MIN_BASE_DISTANCE) & (
        horizontal < WORKSPACE_MARGIN * reach
    )

    if depth is None:
        height = np.zeros(shape, np.float32)
        depth_valid = np.zeros(shape, bool)
        distance = 0.40  # nothing better to go on; only sets the pixel-per-millimetre scale
    else:
        points_base, depth_valid = back_project_depth(depth, intrinsics, view.X_base_camera)
        a, b, c = plane
        surface = a * points_base[..., 0] + b * points_base[..., 1] + c
        height = np.where(depth_valid, points_base[..., 2] - surface, 0.0).astype(np.float32)
        depth_valid &= (height > SCENE_FLOOR_M) & (height < SCENE_CEILING_M)
        height = np.where(depth_valid, height, 0.0).astype(np.float32)
        distance = float(np.median(depth[depth_valid])) if depth_valid.any() else 0.40

    focal = 0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1]))
    scale = PixelScale(px_per_mm=focal / (distance * 1000.0), distance_m=distance)
    height = _denoise_height(height, depth_valid, scale.px_per_mm)
    table_mask = reach_mask & depth_valid & (height < TABLE_BAND_M)

    logger.info(
        f"Pile seen from {distance * 100:.0f} cm, so one millimetre on the table is "
        f"{scale.px_per_mm:.2f} working pixel(s); {100.0 * depth_valid.mean():.0f}% of the frame has depth."
    )
    return Scene(
        bgr=bgr,
        intrinsics_matrix=intrinsics,
        X_base_camera=view.X_base_camera,
        plane=plane,
        table_xy=table_xy,
        height=height,
        depth_valid=depth_valid,
        reach_mask=reach_mask,
        table_mask=table_mask,
        scale=scale,
        working_scale=working_scale,
    )


# =================================================================================================
# the table's appearance
# =================================================================================================


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


def fit_table_model(opponent: np.ndarray, seed: np.ndarray) -> TableModel:
    """Blur the table's own pixels into an estimate of it, re-deciding which pixels those are.

    A polynomial over the whole board cannot follow the lighting band along one edge, and a plain blur
    is dragged upward by the pile sitting in the middle of it. Weighting the blur by the current table
    mask does both: the estimate under the pile is extrapolated from the wood ringing it, and the mask
    sharpens each round as the bricks drop out of it.

    ``seed`` is where the table is *known* to be -- the depth map's answer -- which is why this needs
    half the iterations the RGB-only version does. Without depth the seed is the whole region and this
    reduces exactly to it.
    """
    sigma = TABLE_MODEL_SIGMA_FRAC * float(np.hypot(*opponent.shape[:2]))
    inliers = seed.copy()
    residual = np.zeros_like(opponent)
    cov = np.eye(3, dtype=np.float64)
    for _ in range(TABLE_MODEL_ITERATIONS):
        residual = opponent - _normalized_convolution(opponent, inliers, sigma)
        samples = residual[inliers].reshape(-1, 3)
        cov = np.cov(samples.T) + np.eye(3) * 1e-9
        d = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", residual, np.linalg.inv(cov), residual), 0.0))
        inliers = seed & (d < TABLE_INLIER_SIGMA)
        if not inliers.any():
            inliers = seed.copy()
            break
    return TableModel(residual, cov, np.sqrt(np.diag(cov)), inliers)


def build_table_model(scene: Scene) -> Tuple[TableModel, np.ndarray]:
    """Fit the table's colour, seeded by the height map where there is one."""
    opponent = opponent_image(scene.bgr)
    region = scene.reach_mask
    seed = scene.table_mask
    if seed.sum() < 0.02 * region.sum():
        logger.warning(
            f"Only {int(seed.sum())} pixel(s) of bare table were found by depth, too few to seed the "
            "table's colour model; falling back to fitting it from the colour alone, which is what "
            "pile_perception.py does without a robot."
        )
        seed = region
    return fit_table_model(opponent, seed), region


# =================================================================================================
# foreground
# =================================================================================================


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(max(count, 1), bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def pile_workspace(strong: np.ndarray, region: np.ndarray, scale: PixelScale) -> np.ndarray:
    """The part of the table the pile was tipped out onto, grown by a margin."""
    bridge = scale.odd(PILE_BRIDGE_MM)
    bridged = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((bridge, bridge), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged)
    if count < 2:
        return region
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    grown = np.zeros(strong.shape, np.uint8)
    cv2.fillPoly(grown, [hull], 1)
    margin = 2 * scale.length(PILE_MARGIN_MM) + 1
    grown = cv2.dilate(grown, np.ones((margin, margin), np.uint8))
    return grown.astype(bool) & region


def segment_foreground(scene: Scene, deviation: np.ndarray, region: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Brick pixels vs. table pixels, by height where depth reaches and by colour where it does not.

    Hysteresis, as in the RGB-only version: a brick has to be unmistakable somewhere (STRONG) to be
    admitted, and is then grown to its full silhouette (WEAK). What changed is that "unmistakable" now
    means "4 mm off the table" wherever the RealSense returned a reading, and only falls back to
    "7 sigma from the table's colour" where it did not -- which is why a knot in the plywood no longer
    has a vote and a dark brick the depth stream missed still does.

    The workspace crop is where the two differ most. ``pile_perception.py`` has to crop to the pile's
    own footprint, because colour alone cannot tell a tan brick at the edge of the board from a stain
    in the wood, and the bricks were at least all tipped out in one place. Here the crop is only
    applied to the pixels the depth map could not see: anything standing measurably off the table is
    a brick wherever on the table it is, including the one that skidded away from the pile.
    """
    scale = scene.scale
    depth_strong = scene.depth_valid & (scene.height > FOREGROUND_STRONG_HEIGHT_M)
    depth_weak = scene.depth_valid & (scene.height > FOREGROUND_WEAK_HEIGHT_M)
    colour_strong = ~scene.depth_valid & (deviation > FOREGROUND_STRONG_SIGMA)
    colour_weak = ~scene.depth_valid & (deviation > FOREGROUND_WEAK_SIGMA)

    seed = cv2.morphologyEx((region & depth_strong).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seed = _drop_small(seed, scale.area(FOREGROUND_SEED_AREA_MM2))
    if seed.any():
        # Depth found the bricks, so the workspace is the whole reachable tabletop and the pile hull
        # only decides where an unsupported *colour* detection is still allowed to be believed.
        workspace = region
        colour_region = pile_workspace(seed, region, scale)
    else:
        # No depth anywhere: this is pile_perception.py's situation exactly, so use its answer -- the
        # pile's own footprint, grown by a margin, cropping everything the colour test might invent.
        logger.warning(
            "Nothing stands measurably above the table in the depth map, so the pile is being found by "
            "colour alone and cropped to its own footprint, as the robot-free pile_perception.py does."
        )
        seed = cv2.morphologyEx((region & colour_strong).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        seed = _drop_small(seed, scale.area(FOREGROUND_SEED_AREA_MM2))
        workspace = pile_workspace(seed, region, scale)
        colour_region = workspace

    strong = (region & (depth_strong | (colour_strong & colour_region))).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strong = _drop_small(strong, scale.area(FOREGROUND_SEED_AREA_MM2)) * workspace
    weak = region & (depth_weak | (colour_weak & colour_region))

    weak = (weak & workspace).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(weak)
    seeded = np.unique(labels[strong > 0])
    fg = np.isin(labels, seeded[seeded > 0]).astype(np.uint8)

    close = scale.odd(FOREGROUND_CLOSE_MM)
    open_ = scale.odd(FOREGROUND_OPEN_MM)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((open_, open_), np.uint8))
    fg = _drop_small(fg, scale.area(FOREGROUND_MIN_AREA_MM2))
    return ndi.binary_fill_holes(fg > 0).astype(np.uint8), workspace


# =================================================================================================
# instances
# =================================================================================================


def edge_strength(model: TableModel, scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """Border evidence, from colour and from height, on one 0..1 scale.

    Dividing each colour channel by the table's own scatter is what lets one number stand for both
    "these two bricks are different colours" and "there is a shadow line between them"; in raw values
    the intensity channel would drown the chroma ones and same-brightness neighbours of different
    colours would have no border at all.

    The height gradient is then folded in with a max rather than a sum: a step from one brick down to
    the one below it is a border on its own evidence, whatever the two look like, and taking the
    larger of the two cues keeps the combined number on the same 0..1 scale the merge and split
    thresholds are written in.
    """
    normalized = np.clip((model.residual / model.sigma) * 24.0 + 128.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), max(EDGE_BLUR_MM * scene.scale.px_per_mm, 0.6))
    total = np.zeros(blurred.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(blurred[..., c], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(blurred[..., c], cv2.CV_32F, 0, 1, 3)
        total += gx * gx + gy * gy
    return np.sqrt(total), normalized


def height_edges(scene: Scene) -> np.ndarray:
    """Normalized border evidence from the height map alone; 0 wherever depth is missing.

    Both neighbours have to have depth for a step between them to mean anything, so the gradient is
    computed on a height map with the holes filled by their nearest valid neighbour and then masked
    back to where the reading was real. Without that, every hole edge reads as a metre-high cliff.
    """
    if not scene.depth_valid.any():
        return np.zeros(scene.shape, np.float32)
    filled = _fill_invalid(scene.height, scene.depth_valid)
    smooth = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), max(0.8 * scene.scale.px_per_mm, 0.6))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, 3)
    # Sobel's 3x3 kernel sums to 8 over a unit step, so dividing by 8 puts the result back in metres
    # per pixel; a step of HEIGHT_EDGE_SCALE_M across one pixel is then a full-strength border.
    magnitude = np.sqrt(gx * gx + gy * gy) / 8.0
    reliable = cv2.erode(scene.depth_valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.where(reliable, np.clip(magnitude / HEIGHT_EDGE_SCALE_M, 0.0, 1.0), 0.0).astype(np.float32)


def _nearest_label(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Flood every unassigned pixel of ``region`` with its nearest assigned label."""
    holes = (labels <= 0) & (region > 0)
    if not holes.any():
        return labels
    if not (labels > 0).any():
        return labels
    _, (iy, ix) = ndi.distance_transform_edt(labels <= 0, return_indices=True)
    out = labels.copy()
    out[holes] = labels[iy[holes], ix[holes]]
    return out


def oversegment(fg: np.ndarray, normalized: np.ndarray, gradient_norm: np.ndarray, scale: PixelScale) -> np.ndarray:
    """Watershed the foreground from its flat interiors, deliberately into too many pieces."""
    seeds = ((gradient_norm < SEED_GRADIENT_FRACTION) & (fg > 0)).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seeds = _drop_small(seeds, scale.area(SEED_MIN_AREA_MM2))

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


def _region_heights(labels: np.ndarray, scene: Scene) -> np.ndarray:
    """Mean height per label over its valid-depth pixels; NaN where a label has none."""
    n = int(labels.max()) + 1
    counted = labels * scene.depth_valid
    counts = np.bincount(counted.ravel(), minlength=n).astype(np.float64)
    sums = np.bincount(counted.ravel(), weights=(scene.height * scene.depth_valid).ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        heights = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    heights[0] = np.nan
    return heights


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


def merge_regions(labels: np.ndarray, model: TableModel, scene: Scene, gradient_norm: np.ndarray) -> np.ndarray:
    """Glue the fragments of each brick back together: same colour, same height, no border between.

    Lego is moulded in flat uniform colour, so two pieces of one brick match closely while two bricks
    rarely do -- but two neighbouring bricks of the *same* colour also match, which is why the gradient
    along the shared border has to be weak as well, and why the height has to agree: the commonest way
    for two same-coloured bricks to sit against each other with no visible seam is for one to be lying
    on the other, and that is a 3 to 10 mm step.
    """
    feature = model.residual / model.sigma
    for _ in range(MERGE_ROUNDS):
        areas, means = _region_stats(labels, feature)
        heights = _region_heights(labels, scene)
        pairs, lengths, edges = _adjacency(labels, gradient_norm)
        if not len(pairs):
            break
        colours = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
        steps = np.abs(heights[pairs[:, 0]] - heights[pairs[:, 1]])
        # A pair with no depth on one side has a NaN step and cannot be refused on height, so it
        # falls back to the colour and border tests alone -- which is the RGB-only behaviour.
        with np.errstate(invalid="ignore"):
            level = ~(steps > MERGE_HEIGHT_TOL_M)
        small = np.minimum(areas[pairs[:, 0]], areas[pairs[:, 1]]) < scene.scale.area(MERGE_ABSORB_AREA_MM2)
        joins = (
            (colours < MERGE_COLOUR_TOL)
            & (edges < MERGE_EDGE_TOL)
            & (lengths >= scene.scale.length(MERGE_MIN_BORDER_MM))
            & level
        )
        absorbs = small & level & ((colours < MERGE_COLOUR_TOL * 1.6) | (edges < MERGE_EDGE_TOL * 1.6))

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


def _best_cut(mask: np.ndarray, gradient_norm: np.ndarray, scale: PixelScale) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    min_side = scale.length(SPLIT_MIN_SIDE_MM, minimum=3)
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    u = (xs - cx) * cos + (ys - cy) * sin  # along the box's w side
    v = -(xs - cx) * sin + (ys - cy) * cos
    g = gradient_norm[ys, xs]

    best = None
    for coords, extent in ((u, w), (v, h)):
        if extent < 2 * min_side:
            continue
        lo, hi = -extent / 2 + min_side, extent / 2 - min_side
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
    part_a = _drop_small(part_a, min_side**2)
    part_b = _drop_small(part_b, min_side**2)
    if part_a.sum() == 0 or part_b.sum() == 0:
        return None
    if min(_fill_ratio(part_a), _fill_ratio(part_b)) <= _fill_ratio(mask):
        return None
    return part_a, part_b


def split_regions(labels: np.ndarray, gradient_norm: np.ndarray, scale: PixelScale) -> np.ndarray:
    out = np.zeros_like(labels)
    next_label = 1
    min_area = scale.area(MIN_BRICK_AREA_MM2)
    for value in range(1, int(labels.max()) + 1):
        mask = (labels == value).astype(np.uint8)
        if mask.sum() < min_area:
            if mask.any():
                out[mask > 0] = next_label
                next_label += 1
            continue
        stack = [(mask, 0)]
        while stack:
            part, depth = stack.pop()
            cut = (
                _best_cut(part, gradient_norm, scale)
                if depth < SPLIT_MAX_DEPTH and _fill_ratio(part) < SPLIT_FILL_TARGET
                else None
            )
            if cut is None:
                out[part > 0] = next_label
                next_label += 1
            else:
                stack.extend((piece, depth + 1) for piece in cut)
    return out


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


def segment_instances(fg: np.ndarray, model: TableModel, scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    gradient, normalized = edge_strength(model, scene)
    reference = float(np.percentile(gradient[fg > 0], 95)) if (fg > 0).any() else 1.0
    gradient_norm = np.maximum(gradient / max(reference, 1e-6), height_edges(scene))
    labels = oversegment(fg, normalized, gradient_norm, scene.scale)
    labels = merge_regions(labels, model, scene, gradient_norm)
    labels = split_regions(labels, gradient_norm, scene.scale)
    labels = _nearest_label(labels, fg)
    # A merge across a diagonal touch or a straight cut can leave one label in two pieces, and one
    # brick per label is what everything downstream assumes.
    return _split_disconnected(labels), gradient_norm


# =================================================================================================
# bricks
# =================================================================================================


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


def _shrink_for_view_tilt(
    dimensions: Tuple[float, float],
    axes: Tuple[np.ndarray, np.ndarray],
    view_direction: np.ndarray,
    height: float,
) -> Tuple[float, float]:
    """Remove the side walls a tilted view adds to a silhouette.

    A camera that is not directly overhead sees a brick's near wall as well as its top face, so the
    silhouette projected onto the top-face plane is longer than the brick by ``height * tan(tilt)``
    along the camera's azimuth. A rectangle axis only picks up the component of that along itself.
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


def _brick_confidence(brick: Brick) -> Tuple[float, str]:
    """How sure we are this region is a brick, from depth where there is depth and colour where not."""
    deviation = float(np.clip(brick.deviation / CONFIDENCE_DEVIATION_SIGMA, 0.0, 1.0))
    if brick.depth_coverage >= MIN_DEPTH_COVERAGE:
        cues = {
            "height": float(np.clip((brick.height_m - MIN_BRICK_HEIGHT_M) / (CONFIDENT_HEIGHT_M - MIN_BRICK_HEIGHT_M), 0.0, 1.0)),
            "flatness": float(np.clip(1.0 - brick.height_spread_m / FLATNESS_TOL_M, 0.0, 1.0)),
            "coverage": float(np.clip(brick.depth_coverage / CONFIDENT_COVERAGE, 0.0, 1.0)),
            "deviation": deviation,
        }
        return float(sum(DEPTH_CONFIDENCE_WEIGHTS[k] * v for k, v in cues.items())), "depth"

    # No usable depth on this region, so this is exactly pile_perception.py's RGB-only judgement:
    # how far off the table's colour it is, how much of a step rather than a fade its outline is, and
    # whether the wood's grain runs through it (a stain) or stops at its edge (a moulded surface).
    cues = {
        "deviation": deviation,
        "edge_support": float(np.clip(brick.edge_support / CONFIDENCE_EDGE_SUPPORT, 0.0, 1.0)),
        "interior_texture": float(np.clip((1.0 - brick.interior_texture) / CONFIDENCE_INTERIOR_TEXTURE, 0.0, 1.0)),
    }
    return float(sum(RGB_CONFIDENCE_WEIGHTS[k] * v for k, v in cues.items())), "colour"


def measure_footprint(
    contour: np.ndarray, scene: Scene, height_m: float
) -> Optional[Tuple[np.ndarray, Tuple[float, float], float, float, float, float]]:
    """The region's outline in metres, projected onto the plane of its own top face.

    Returns ``(polygon, centre, width_m, length_m, long_axis_heading, area_m2)`` or ``None`` when the
    outline does not cross the plane at all -- which means the region is not on the table and there is
    nothing sensible to report about it.
    """
    a, b, c = scene.plane
    projected = project_pixels_onto_plane(
        contour.reshape(-1, 2), scene.intrinsics_matrix, scene.X_base_camera, (a, b, c + height_m)
    )
    if projected is None or len(projected) < 3:
        return None

    polygon = projected[:, :2].astype(np.float64)
    # The rectangle is fitted in millimetres, not metres. cv2.minAreaRect works in float32 internally
    # and a table-frame coordinate is around 0.3 -- fitting a 8 mm side out of numbers that size is
    # asking for the answer to be quantised by the format rather than by the pixels.
    polygon_mm = (polygon * 1000.0).astype(np.float32)
    (center_x, center_y), (side_a, side_b), angle_deg = cv2.minAreaRect(polygon_mm)
    center_x, center_y = center_x / 1000.0, center_y / 1000.0
    side_a, side_b = side_a / 1000.0, side_b / 1000.0
    angle = math.radians(angle_deg)
    axis_a = np.array([math.cos(angle), math.sin(angle)])
    axis_b = np.array([-math.sin(angle), math.cos(angle)])

    centre = np.array([center_x, center_y])
    view_direction = np.array([centre[0], centre[1], a * centre[0] + b * centre[1] + c + height_m]) - scene.X_base_camera[:3, 3]
    norm = float(np.linalg.norm(view_direction))
    if norm < 1e-9:
        return None
    view_direction = view_direction / norm
    side_a, side_b = _shrink_for_view_tilt((side_a, side_b), (axis_a, axis_b), view_direction, height_m)

    if side_a >= side_b:
        length, width, long_axis = side_a, side_b, axis_a
    else:
        length, width, long_axis = side_b, side_a, axis_b

    area = float(abs(cv2.contourArea(polygon_mm))) / 1e6
    return (
        polygon,
        (float(centre[0]), float(centre[1])),
        float(width),
        float(length),
        float(math.atan2(long_axis[1], long_axis[0])),
        area,
    )


def build_bricks(
    labels: np.ndarray,
    scene: Scene,
    deviation: np.ndarray,
    gradient_norm: np.ndarray,
    foreground: np.ndarray,
) -> Tuple[List[Brick], List[Brick]]:
    """One :class:`Brick` per region, measured in metres, with the regions that cannot be bricks split off."""
    scale = scene.scale
    inner = scale.odd(RING_INNER_MM)
    outer = scale.odd(RING_OUTER_MM)
    erode = scale.odd(1.5)

    kept: List[Brick] = []
    rejected: List[Brick] = []
    for value in range(1, int(labels.max()) + 1):
        mask = (labels == value).astype(np.uint8)
        area_px = float(mask.sum())
        if area_px < 1:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        long_px, short_px = max(rect[1]), min(rect[1])
        if long_px < 1 or short_px < 1:
            continue

        # The brick's own height, from the pixels the depth stream actually returned inside it. The
        # median, not the mean: the studs stand 1.6 mm proud over a minority of the top face and a
        # mean would ride up on them, putting the projection plane above the surface being grasped.
        interior = cv2.erode(mask, np.ones((erode, erode), np.uint8))
        if interior.sum() < 9:
            interior = mask
        inside = (interior > 0) & scene.depth_valid
        coverage = float(inside.sum() / max(interior.sum(), 1))
        if coverage >= MIN_DEPTH_COVERAGE:
            samples = scene.height[inside]
            height_m = float(np.median(samples))
            spread = float(np.median(np.abs(samples - height_m)))
            measured_height = True
        else:
            # Not enough depth on this region to say how tall it is. Assuming a standard brick beats
            # assuming zero: the height sets the plane the outline is projected onto, and at zero the
            # region is measured on the tabletop -- so its own side walls are counted into its
            # footprint and it comes out a couple of millimetres too big in every direction.
            height_m, spread, measured_height = FALLBACK_BRICK_HEIGHT, 0.0, False

        measured = measure_footprint(contour, scene, max(height_m, 0.0))
        if measured is None:
            continue
        polygon, centre, width_m, length_m, heading, area_m2 = measured

        # What stands around it. A ring higher than the brick means something is lying over it, which
        # is the thing that makes a grasp fail without looking wrong from above.
        ring = cv2.dilate(mask, np.ones((outer, outer), np.uint8)) - cv2.dilate(mask, np.ones((inner, inner), np.uint8))
        ring_depth = (ring > 0) & scene.depth_valid
        neighbour_height = float(np.percentile(scene.height[ring_depth], 90)) if ring_depth.any() else 0.0
        # Measured against bare table only. A brick buried in the pile is ringed by other bricks, whose
        # tops are as flat as its own, and comparing it to those says nothing about whether it is wood.
        bare_ring = ring * (foreground == 0)

        outline = np.zeros_like(mask)
        cv2.drawContours(outline, [contour], -1, 1, 2)
        bgr_median = np.median(scene.bgr[mask > 0].reshape(-1, 3), axis=0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

        centre_pixel = np.mean(contour.reshape(-1, 2), axis=0)
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 1e-6:
            centre_pixel = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])

        brick = Brick(
            index=len(kept) + len(rejected),
            mask=mask,
            contour=contour,
            footprint_m=polygon,
            center_m=centre,
            grasp_pixel=(
                int(round((centre_pixel[0] + 0.5) / scene.working_scale - 0.5)),
                int(round((centre_pixel[1] + 0.5) / scene.working_scale - 0.5)),
            ),
            height_m=float(np.clip(height_m, 0.0, MAX_BRICK_HEIGHT_M)),
            height_measured=measured_height,
            table_z=scene.table_z_at(*centre),
            width_mm=width_m * 1000.0,
            length_mm=length_m * 1000.0,
            long_axis_heading=heading,
            area_mm2=area_m2 * 1e6,
            area_px=area_px,
            rectangularity=float(area_px / max(long_px * short_px, 1.0)),
            solidity=float(area_px / max(hull_area, 1.0)),
            aspect_ratio=float(long_px / max(short_px, 1e-6)),
            colour_rgb=(int(bgr_median[2]), int(bgr_median[1]), int(bgr_median[0])),
            deviation=float(np.median(deviation[mask > 0])),
            edge_support=float(np.median(gradient_norm[outline > 0])),
            interior_texture=(
                float(np.median(gradient_norm[interior > 0]) / max(float(np.median(gradient_norm[bare_ring > 0])), 1e-6))
                if bare_ring.sum() >= RING_MIN_TABLE_PX
                else 0.0
            ),
            depth_coverage=coverage,
            height_spread_m=spread,
            neighbour_height_m=neighbour_height,
        )
        brick.colour_name = _colour_name(brick.colour_rgb)
        brick.confidence, brick.confidence_source = _brick_confidence(brick)

        # Failing the shape tests means the region is not one brick -- but on this much height and
        # colour evidence it is certainly bricks, so it is kept as an unresolved clump rather than
        # thrown away. Outlining it says "there is something here the detector could not take apart",
        # which is the truth, where dropping it would draw bare table over several real bricks.
        brick.is_clump = brick.rectangularity < MIN_RECTANGULARITY or brick.solidity < MIN_SOLIDITY

        reason = None
        if brick.area_mm2 < MIN_BRICK_AREA_MM2:
            reason = "too_small"
        elif brick.area_mm2 > MAX_BRICK_AREA_MM2:
            reason = "too_large"
        elif brick.width_mm < MIN_BRICK_SIDE_MM:
            reason = "too_thin"
        elif brick.aspect_ratio > MAX_ASPECT_RATIO:
            reason = "not_brick_shaped"
        elif brick.depth_coverage >= MIN_DEPTH_COVERAGE and brick.height_m < MIN_BRICK_HEIGHT_M:
            reason = "flat_on_the_table"
        elif brick.height_m > MAX_BRICK_HEIGHT_M - 1e-9:
            reason = "too_tall_to_be_the_pile"
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


# =================================================================================================
# clearance, on a top-down map of the table
# =================================================================================================


@dataclass
class TopDownMap:
    """An orthographic millimetre raster of the tabletop, so distances come out in millimetres.

    Measuring clearance in image pixels and multiplying by a single millimetres-per-pixel is wrong by
    however much the perspective varies across the frame -- with the camera 40 cm up and the pile
    20 cm across, that is several percent between the near and far edge of the pile, on the one number
    (fingertip room) that decides whether the grasp collides. An orthographic raster has no such
    gradient: one cell is one millimetre everywhere.
    """

    origin: np.ndarray  # base-frame x, y of cell (0, 0)
    shape: Tuple[int, int]
    mm_per_cell: float

    def to_cells(self, points: np.ndarray) -> np.ndarray:
        cells = (np.asarray(points, float)[:, :2] - self.origin) / (self.mm_per_cell / 1000.0)
        return np.round(cells).astype(np.int32)

    def raster(self, polygon: np.ndarray) -> np.ndarray:
        canvas = np.zeros(self.shape, np.uint8)
        cells = self.to_cells(polygon)
        cv2.fillPoly(canvas, [cells.reshape(-1, 1, 2)], 1)
        return canvas


def build_topdown_map(bricks: Sequence[Brick]) -> Optional[TopDownMap]:
    points = np.vstack([b.footprint_m for b in bricks if len(b.footprint_m)]) if bricks else np.zeros((0, 2))
    if len(points) < 3:
        return None
    margin = TOPDOWN_MARGIN_MM / 1000.0
    low = points.min(axis=0) - margin
    high = points.max(axis=0) + margin
    cells = np.ceil((high - low) / (TOPDOWN_MM_PER_CELL / 1000.0)).astype(int) + 1
    if np.any(cells > TOPDOWN_MAX_CELLS) or np.any(cells < 3):
        logger.warning(
            f"The brick footprints span {(high - low) * 100} cm, which is not a pile on a table; skipping the "
            "top-down clearance map and leaving every clearance at zero."
        )
        return None
    return TopDownMap(origin=low, shape=(int(cells[1]), int(cells[0])), mm_per_cell=TOPDOWN_MM_PER_CELL)


def measure_clearance(bricks: List[Brick], table_map: Optional[TopDownMap]) -> None:
    """How much bare table each brick has around it, and beside its two grasp faces, in millimetres."""
    if table_map is None:
        return

    rasters = {brick.index: table_map.raster(brick.footprint_m) for brick in bricks}
    occupancy = np.zeros(table_map.shape, np.uint8)
    for raster in rasters.values():
        occupancy |= raster

    for brick in bricks:
        others = occupancy.copy()
        others[rasters[brick.index] > 0] = 0
        free = cv2.distanceTransform((1 - others).astype(np.uint8), cv2.DIST_L2, 3) * table_map.mm_per_cell

        # Where the fingertips go: out along the normal of the brick's long axis, just past its side.
        centre = np.array(brick.center_m)
        normal = np.array([-math.sin(brick.long_axis_heading), math.cos(brick.long_axis_heading)])
        reach = (brick.width_mm / 2.0 + JAW_PROBE_MARGIN_MM) / 1000.0
        clearances = []
        for sign in (1, -1):
            cell = table_map.to_cells((centre + sign * reach * normal).reshape(1, 2))[0]
            row = int(np.clip(cell[1], 0, table_map.shape[0] - 1))
            column = int(np.clip(cell[0], 0, table_map.shape[1] - 1))
            clearances.append(float(free[row, column]))
        brick.jaw_clearance_mm = float(min(clearances))

        interior = cv2.distanceTransform(rasters[brick.index], cv2.DIST_L2, 3) * table_map.mm_per_cell
        cell = table_map.to_cells(centre.reshape(1, 2))[0]
        row = int(np.clip(cell[1], 0, table_map.shape[0] - 1))
        column = int(np.clip(cell[0], 0, table_map.shape[1] - 1))
        brick.isolation_mm = float(max(free[row, column] - interior[row, column], 0.0))

        outline = table_map.to_cells(brick.footprint_m)
        rows = np.clip(outline[:, 1], 0, table_map.shape[0] - 1)
        columns = np.clip(outline[:, 0], 0, table_map.shape[1] - 1)
        brick.exposed_ratio = float(np.mean(free[rows, columns] > EXPOSED_FREE_MM))


# =================================================================================================
# ranking
# =================================================================================================


def rank_bricks(bricks: List[Brick]) -> List[Brick]:
    """Order the bricks by how safely a top-down parallel-jaw grasp would work on each.

    The terms are the same ones ``pile_perception.py`` scores on, with one swap: where the RGB-only
    version *infers* that a clean, isolated rectangle is a brick nothing is lying on top of, the height
    map says so outright. ``top_of_pile`` is one minus how far the ring around the brick stands above
    it -- a brick with bare table or shorter bricks beside it scores 1, a brick with a 6 mm-taller
    neighbour leaning over it scores 0.

    ``grip_depth`` is the second thing depth buys, and it is not in the RGB-only version at all
    because a single photograph cannot measure it: how much of the part's side wall the fingertips can
    reach past before the table stops them. Every other term rates a plate lying alone on bare table
    as the best grasp in the pile, and every one of them is right -- it is isolated, exposed, clean
    and the right width. It is also 3.2 mm tall, which leaves the pads under two millimetres of purchase
    and turns the close into a shove.
    """
    for brick in bricks:
        clearance = float(
            np.clip((brick.jaw_clearance_mm - FINGER_HALF_THICKNESS_MM * 0.5) / CLEARANCE_GOOD_MM, 0.0, 1.0)
        )
        isolation = float(np.clip(brick.isolation_mm / ISOLATION_GOOD_MM, 0.0, 1.0))
        grip_depth = float(
            np.clip((brick.height_m * 1000.0 - FINGERTIP_CLEARANCE_MM) / GRIP_DEPTH_GOOD_MM, 0.0, 1.0)
        )
        burial = float(np.clip((brick.neighbour_height_m - brick.height_m) / BURIAL_SCALE_M, 0.0, 1.0))
        top_of_pile = 1.0 - burial
        exposure = float(np.clip(brick.exposed_ratio, 0.0, 1.0))
        visibility = float(np.clip((brick.rectangularity - MIN_RECTANGULARITY) / (0.92 - MIN_RECTANGULARITY), 0.0, 1.0))
        size = float(np.clip(brick.area_mm2 / CONFIDENT_AREA_MM2, 0.0, 1.0))

        if GRIPPER_MIN_WIDTH_MM <= brick.width_mm <= GRIPPER_MAX_WIDTH_MM:
            width_fit = float(np.exp(-((brick.width_mm - GRIPPER_IDEAL_WIDTH_MM) ** 2) / (2 * GRIPPER_WIDTH_SIGMA_MM**2)))
        else:
            width_fit = 0.0

        brick.graspable = (
            GRIPPER_MIN_WIDTH_MM <= brick.width_mm <= GRIPPER_MAX_WIDTH_MM
            and brick.area_mm2 >= MIN_BRICK_AREA_MM2
            and brick.jaw_clearance_mm >= FINGER_HALF_THICKNESS_MM
            # An unresolved clump has no single outline to aim at, so its geometry is not a grasp.
            and not brick.is_clump
        )

        brick.score_terms = {
            "clearance": clearance,
            "grip_depth": grip_depth,
            "isolation": isolation,
            "top_of_pile": top_of_pile,
            "exposure": exposure,
            "visibility": visibility,
            "width_fit": width_fit,
            "size": size,
            "confidence": brick.confidence,
        }
        brick.score = float(sum(SCORE_WEIGHTS[k] * v for k, v in brick.score_terms.items()))
        if not brick.graspable:
            brick.score *= 0.25

    return sorted(bricks, key=lambda b: b.score, reverse=True)


def assign_priorities(
    ordered: List[Brick],
    arm: Optional[PositionManipulator],
    pregrasp_height: float,
    count: int = PRIORITY_COUNT,
) -> None:
    """Number the bricks the arm should actually be sent at, best first.

    Three ways a well-scored brick still does not get a number:

    * the detector is not confident it is a brick -- an unconfident region is most likely a mark in the
      wood, and the cost of closing the gripper on bare table is a wasted cycle plus a fingertip into
      the board;
    * its geometry is not a grasp (too wide for the jaws, too small to be a part, an unresolved clump,
      or no room for a fingertip beside it);
    * the arm cannot reach a straight-down pregrasp above it, which is a real and common outcome at
      the far corner of the table and is far better found here than after the arm has committed.

    The reachability test is eight IK calls over the network per brick, so the search gives up after
    :data:`MAX_REACHABILITY_CHECKS` candidates: if that many of the best-scoring bricks in a row are
    all out of reach, the pile is in the wrong place and no amount of further searching fixes it.
    """
    rank = 0
    checked = 0
    for brick in ordered:
        if rank >= count:
            break
        if brick.confidence < PRIORITY_MIN_CONFIDENCE or not brick.graspable:
            continue
        if arm is not None:
            if checked >= MAX_REACHABILITY_CHECKS:
                logger.warning(
                    f"Stopped checking reachability after {checked} candidate(s); the rest of the pile is "
                    "reported but not offered as a target. If nothing was reachable, the pile is outside the "
                    "arm's comfortable workspace -- move it closer to the base."
                )
                break
            checked += 1
            position = np.array([brick.center_m[0], brick.center_m[1], brick.top_face_z + pregrasp_height])
            try:
                find_reachable_hover_orientation(arm, position)
                brick.reachable = True
            except RuntimeError:
                brick.reachable = False
                logger.debug(
                    f"Brick #{brick.index} at {position.round(3)} m scored {brick.score:.3f} but has no reachable "
                    "straight-down pregrasp; skipping it."
                )
                continue
        rank += 1
        brick.priority = rank


# =================================================================================================
# the pipeline
# =================================================================================================


def analyse_pile(view: PileView, plane: Tuple[float, float, float], robot_type: str) -> PileAnalysis:
    """Run the whole perception pipeline on one capture. Nothing here touches the robot."""
    scene = build_scene(view, plane, robot_type)
    model, region = build_table_model(scene)
    deviation = model.deviation()
    foreground, workspace = segment_foreground(scene, deviation, region)
    labels, gradient_norm = segment_instances(foreground, model, scene)
    bricks, rejected = build_bricks(labels, scene, deviation, gradient_norm, foreground)
    measure_clearance(bricks, build_topdown_map(bricks))
    ordered = rank_bricks(bricks)
    return PileAnalysis(
        view=view,
        scene=scene,
        model=model,
        deviation=deviation,
        foreground=foreground,
        workspace=workspace,
        labels=labels,
        bricks=bricks,
        rejected=rejected,
        ordered=ordered,
    )


# =================================================================================================
# the handoff to submodule_1
# =================================================================================================


def _target_record(brick: Brick) -> Dict:
    return {
        "id": brick.index,
        "priority": brick.priority,
        "score": round(brick.score, 4),
        "score_terms": {k: round(v, 3) for k, v in brick.score_terms.items()},
        "confidence": round(brick.confidence, 3),
        "confidence_source": brick.confidence_source,
        "colour": {"name": brick.colour_name, "rgb": list(brick.colour_rgb)},
        # Everything submodule_1 needs to stand over this brick, and submodule_2 to close on it.
        "position": [round(brick.center_m[0], 5), round(brick.center_m[1], 5), round(brick.top_face_z, 5)],
        "pixel": list(brick.grasp_pixel),
        "table_z": round(brick.table_z, 5),
        "height": round(brick.height_m, 5),
        "height_measured": bool(brick.height_measured),
        "width": round(brick.width_mm / 1000.0, 5),
        "length": round(brick.length_mm / 1000.0, 5),
        "long_axis_heading": round(brick.long_axis_heading, 5),
        "closing_heading": round(brick.closing_heading, 5),
        "geometry": {
            "area_mm2": round(brick.area_mm2, 1),
            "aspect_ratio": round(brick.aspect_ratio, 2),
            "rectangularity": round(brick.rectangularity, 3),
            "solidity": round(brick.solidity, 3),
            "unresolved_clump": bool(brick.is_clump),
        },
        "context": {
            "jaw_clearance_mm": round(brick.jaw_clearance_mm, 1),
            "isolation_mm": round(brick.isolation_mm, 1),
            "exposed_perimeter_ratio": round(brick.exposed_ratio, 3),
            "neighbour_height_mm": round(brick.neighbour_height_m * 1000.0, 1),
            "depth_coverage": round(brick.depth_coverage, 3),
            "height_spread_mm": round(brick.height_spread_m * 1000.0, 2),
            "table_deviation_sigma": round(brick.deviation, 1),
        },
        "graspable": bool(brick.graspable),
        "reachable": brick.reachable,
    }


def build_pile_manifest(analysis: PileAnalysis, robot_type: str, pregrasp_height: float) -> Dict:
    """Everything one look at the pile concluded, with the chosen brick promoted to the top."""
    target = analysis.target
    reasons: Dict[str, int] = {}
    for brick in analysis.rejected:
        reasons[brick.reject_reason or "unknown"] = reasons.get(brick.reject_reason or "unknown", 0) + 1
    colours: Dict[str, int] = {}
    for brick in analysis.bricks:
        colours[brick.colour_name] = colours.get(brick.colour_name, 0) + 1

    candidates = [b for b in analysis.ordered if b.priority]
    return {
        "schema": "pile_target/1",
        "written_at": time.time(),
        "robot_type": robot_type,
        "pregrasp_height": pregrasp_height,
        "target": _target_record(target) if target is not None else None,
        "alternatives": [_target_record(b) for b in candidates[1:]],
        "table_plane": {"a": analysis.scene.plane[0], "b": analysis.scene.plane[1], "c": analysis.scene.plane[2]},
        "view": {
            "X_base_camera": np.asarray(analysis.scene.X_base_camera, float).round(6).tolist(),
            "intrinsics_matrix": np.asarray(analysis.scene.intrinsics_matrix, float).round(4).tolist(),
            "working_scale": round(analysis.scene.working_scale, 6),
            "distance_m": round(analysis.scene.scale.distance_m, 4),
            "px_per_mm": round(analysis.scene.scale.px_per_mm, 4),
            "joint_configuration": (
                np.asarray(analysis.view.joint_configuration, float).round(6).tolist()
                if analysis.view.joint_configuration is not None
                else None
            ),
            "depth_coverage": round(float(analysis.scene.depth_valid.mean()), 4),
        },
        "summary": {
            "brick_count": len(analysis.bricks),
            "graspable_count": int(sum(1 for b in analysis.bricks if b.graspable)),
            "unresolved_clump_count": int(sum(1 for b in analysis.bricks if b.is_clump)),
            "rejected_region_count": len(analysis.rejected),
            "rejected_reasons": reasons,
            "colour_histogram": dict(sorted(colours.items(), key=lambda kv: -kv[1])),
            "median_brick_width_mm": (
                round(float(np.median([b.width_mm for b in analysis.bricks])), 1) if analysis.bricks else None
            ),
            "median_jaw_clearance_mm": (
                round(float(np.median([b.jaw_clearance_mm for b in analysis.bricks])), 1) if analysis.bricks else None
            ),
            "highest_brick_mm": (
                round(1000.0 * max(b.height_m for b in analysis.bricks), 1) if analysis.bricks else None
            ),
        },
        "grasp_order": [b.index for b in sorted(candidates, key=lambda b: b.priority or 0)],
        "params": {
            "working_long_side": WORKING_LONG_SIDE,
            "foreground_strong_height_m": FOREGROUND_STRONG_HEIGHT_M,
            "min_brick_area_mm2": MIN_BRICK_AREA_MM2,
            "min_brick_confidence": MIN_BRICK_CONFIDENCE,
            "priority_min_confidence": PRIORITY_MIN_CONFIDENCE,
            "gripper_width_mm": [GRIPPER_MIN_WIDTH_MM, GRIPPER_MAX_WIDTH_MM],
            "score_weights": SCORE_WEIGHTS,
        },
    }


def write_pile_target(path: str, manifest: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote the pile target to {path}.")


def read_pile_target(path: str = PILE_TARGET_PATH, max_age: float = PILE_TARGET_MAX_AGE) -> Optional[Dict]:
    """Load submodule_3's chosen brick, or ``None`` with a reason logged if it should not be trusted.

    Two ways it is refused, both meaning "this describes a pile that is no longer on the table": the
    file is missing, or it is older than ``max_age``. Every pick disturbs the pile, so an old target is
    not merely stale -- the brick it names has very likely moved or been buried by the last grasp.
    """
    if not os.path.exists(path):
        logger.warning(f"No pile target at {path}; run submodule_3 first.")
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exception:
        logger.warning(f"Could not read the pile target at {path}: {exception}")
        return None

    age = time.time() - float(payload.get("written_at", 0.0))
    if age > max_age:
        logger.warning(
            f"The pile target at {path} is {age / 60:.1f} min old (limit {max_age / 60:.0f} min), so it describes "
            "an arrangement of the pile that has since been picked at. Re-run submodule_3."
        )
        return None
    if payload.get("target") is None:
        logger.warning(f"The pile target at {path} records no graspable brick.")
        return None
    return payload


# =================================================================================================
# output for the eyes
# =================================================================================================

OUTLINE_COLOUR = (80, 235, 90)
PRIORITY_COLOURS = [(60, 60, 245), (0, 150, 255), (0, 215, 255), (190, 220, 60), (230, 160, 60)]


def render_overlay(analysis: PileAnalysis) -> np.ndarray:
    """The frame with every brick outlined and the ones to grasp numbered 1..5.

    Drawn at working resolution, which is what every measurement was made at -- an overlay that has
    been resampled back to the source frame no longer shows exactly the pixels that were judged.
    """
    out = analysis.scene.bgr.copy()
    for brick in analysis.bricks:
        cv2.drawContours(out, [brick.contour], -1, OUTLINE_COLOUR, 2, cv2.LINE_AA)

    for brick in sorted((b for b in analysis.ordered if b.priority), key=lambda b: b.priority or 0):
        colour = PRIORITY_COLOURS[min((brick.priority or 1) - 1, len(PRIORITY_COLOURS) - 1)]
        cv2.drawContours(out, [brick.contour], -1, colour, 3, cv2.LINE_AA)

        rect = cv2.minAreaRect(brick.contour)
        (cx, cy), (w, h), angle = rect
        if h > w:
            angle += 90.0
        theta = math.radians(angle)
        nx, ny = -math.sin(theta), math.cos(theta)
        reach = min(w, h) / 2 + analysis.scene.scale.length(JAW_PROBE_MARGIN_MM)
        for sign in (1, -1):
            a = (int(round(cx + sign * nx * reach * 0.75)), int(round(cy + sign * ny * reach * 0.75)))
            b = (int(round(cx + sign * nx * reach * 1.6)), int(round(cy + sign * ny * reach * 1.6)))
            cv2.line(out, a, b, colour, 2, cv2.LINE_AA)

        label = str(brick.priority)
        centre = (int(round(cx)), int(round(cy)))
        cv2.circle(out, centre, 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, centre, 15, colour, 2, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.putText(
            out, label, (centre[0] - size[0] // 2, centre[1] + size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA,
        )

    target = analysis.target
    if target is not None:
        caption = (
            f"#{target.index} {target.colour_name} {target.width_mm:.1f}x{target.length_mm:.1f}x"
            f"{target.height_m * 1000:.1f} mm @ ({target.center_m[0]:.3f}, {target.center_m[1]:.3f}) m, "
            f"close along {math.degrees(target.closing_heading):.0f} deg"
        )
        cv2.putText(out, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_debug_panel(analysis: PileAnalysis) -> np.ndarray:
    """The intermediate stages stacked up: height, foreground, instances, and what was dropped."""
    scene = analysis.scene
    height = cv2.applyColorMap(
        (np.clip(scene.height / 0.03, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    height[~scene.depth_valid] = 0

    foreground = scene.bgr.copy()
    foreground[analysis.foreground == 0] = (foreground[analysis.foreground == 0] * 0.25).astype(np.uint8)
    contours, _ = cv2.findContours(analysis.workspace.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(foreground, contours, -1, (0, 220, 255), 2)

    rng = np.random.default_rng(0)
    palette = rng.integers(60, 255, (int(analysis.labels.max()) + 2, 3)).astype(np.uint8)
    segmentation = np.where((analysis.labels > 0)[..., None], palette[analysis.labels], 0).astype(np.uint8)

    drops = scene.bgr.copy()
    for brick in analysis.rejected:
        cv2.drawContours(drops, [brick.contour], -1, (60, 60, 245), 2)
        text = f"{brick.reject_reason} h{brick.height_m * 1000:.1f} c{brick.confidence:.2f}"
        anchor = (int(brick.contour[:, 0, 0].mean()) - 40, int(brick.contour[:, 0, 1].mean()))
        cv2.putText(drops, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([height, foreground, cv2.addWeighted(scene.bgr, 0.45, segmentation, 0.55, 0), drops])


def save_debug_output(debug_dir: str, analysis: PileAnalysis, manifest: Dict) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(debug_dir, f"pile_{stamp}_perception.png"), render_overlay(analysis))
    cv2.imwrite(os.path.join(debug_dir, f"pile_{stamp}_debug.png"), render_debug_panel(analysis))
    with open(os.path.join(debug_dir, f"pile_{stamp}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    analysis.view.save(os.path.join(debug_dir, f"pile_{stamp}_capture.npz"))
    logger.info(f"Wrote the overlay, the stage-by-stage panel, the manifest and the raw capture to {debug_dir}.")


def report(analysis: PileAnalysis) -> None:
    """Log what was found, and why the chosen brick was chosen."""
    logger.info(
        f"{len(analysis.bricks)} brick(s) in the pile, {sum(1 for b in analysis.bricks if b.graspable)} graspable "
        f"({len(analysis.rejected)} region(s) dropped)."
    )
    for brick in sorted((b for b in analysis.ordered if b.priority), key=lambda b: b.priority or 0):
        logger.info(f"  {brick.priority}. {brick.describe()}")

    target = analysis.target
    if target is None:
        logger.error(
            "No brick in this view is a grasp the arm can be sent at. Either the pile is out of frame, the "
            "table has not been touched off (so every height is measured from a guess), or everything left is "
            "too tightly packed for a fingertip. Try `python src/calibrate_table.py`, then move the pile "
            "closer to the base and look again."
        )
        return
    terms = ", ".join(f"{k}={v:.2f}" for k, v in sorted(target.score_terms.items(), key=lambda kv: -kv[1]))
    logger.success(f"Grasp {target.describe()}")
    logger.info(f"  score {target.score:.3f} from {terms}")
    logger.info(
        f"  submodule_1 should stand at ({target.center_m[0]:.4f}, {target.center_m[1]:.4f}) m, "
        f"top face z={target.top_face_z:.4f} m; submodule_2 closes {target.width_mm:.1f} mm along "
        f"{math.degrees(target.closing_heading):.0f} deg."
    )
    if target.confidence_source == "colour":
        logger.warning(
            "The chosen brick had almost no depth on it, so it was judged on colour alone -- the one case "
            "where a mark in the plywood can still be mistaken for a brick -- and its height is the assumed "
            f"{FALLBACK_BRICK_HEIGHT * 1000:.1f} mm rather than a measurement. Check the overlay before "
            "running submodule_1."
        )


# =================================================================================================
# CLI
# =================================================================================================


def resolve_table_plane(table_z: Optional[float]) -> Tuple[float, float, float]:
    """The tabletop as ``z = a*x + b*y + c``, best source first.

    The touched-off plane wins: the arm measured it by touching, so unlike anything the camera says it
    carries no hand-eye calibration error -- and every height in this module is measured from it, so an
    error here is an error in every brick at once. ``--table-z`` overrides it with a level plane (the
    tilt is lost), and ``config.TABLE_Z`` is the last resort.
    """
    if table_z is not None:
        logger.info(f"--table-z {table_z:+.4f} m given; using a level plane at that height.")
        return 0.0, 0.0, float(table_z)

    plane: Optional[TablePlane] = load_table_plane()
    if plane is not None:
        logger.info(f"Using the {plane.describe()}.")
        return plane.a, plane.b, plane.c

    logger.warning(
        f"The table has never been touched off, so every brick height below is measured from "
        f"config.TABLE_Z={TABLE_Z:+.4f} m, which is a guess. A 2 cm error there turns every brick into a "
        "clump and every clump into nothing. Run `python src/calibrate_table.py` first."
    )
    return 0.0, 0.0, TABLE_Z


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm the camera is mounted on; sets the reach the workspace is cropped to.",
)
@click.option(
    "--ip-address",
    default=None,
    help="Robot controller IP address. Defaults per robot type "
    f"(ur3e: {DEFAULT_IP_ADDRESSES['ur3e']}, realman: {DEFAULT_IP_ADDRESSES['realman']}).",
)
@click.option(
    "--port",
    default=DEFAULT_REALMAN_PORT,
    show_default=True,
    help="Controller port (RealMan only; ignored for the UR3e).",
)
@click.option(
    "--speed-ratio",
    type=click.IntRange(1, 100),
    default=10,
    show_default=True,
    help="1..100, fraction of the arm's max joint speed for the move to the viewpoint.",
)
@click.option(
    "--calibration-path",
    default=DEFAULT_CALIBRATION_DIR,
    show_default=True,
    help="Path to the hand-eye-calibration --calibration_dir output directory.",
)
@click.option(
    "--camera-resolution",
    type=click.Choice(list(CAMERA_RESOLUTIONS)),
    default=DEFAULT_CAMERA_RESOLUTION,
    show_default=True,
    help="RealSense colour resolution (height). A D415/D435 needs USB 3 to stream colour+depth.",
)
@click.option(
    "--stay",
    is_flag=True,
    help="Look at the pile from wherever the arm is standing instead of moving to PILE_VIEW first. "
    "Use it when the arm is already parked over the pile and you only want a second opinion.",
)
@click.option(
    "--table-z",
    type=float,
    default=None,
    help="Height of the table's surface in the base frame (metres), overriding the measured table "
    "plane. You should not normally need this: `python src/calibrate_table.py` measures it, tilt "
    "included, and passing a flat value here throws the tilt away.",
)
@click.option(
    "--pregrasp-height",
    type=click.FloatRange(0.0, 0.10, min_open=True),
    default=PREGRASP_HEIGHT,
    show_default=True,
    help="Metres above the brick's top face that submodule_1 will hover at. Only used here to check "
    "the arm can actually reach that pose before offering the brick as a target.",
)
@click.option(
    "--target-path",
    default=PILE_TARGET_PATH,
    show_default=True,
    help="Where to write the chosen brick for submodule_1 to pick up.",
)
@click.option(
    "--debug-dir",
    default=None,
    help="If set, save the outlined frame, the stage-by-stage panel, the manifest and the raw capture "
    "here. The capture can be replayed with --from-capture, which is the way to retune the thresholds "
    "without tying up the robot.",
)
@click.option(
    "--from-capture",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Re-run the perception on a capture saved by --debug-dir instead of connecting to anything. "
    "No arm, so reachability is not checked and the chosen brick is only a proposal.",
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Report the choice without writing the target file, so nothing downstream picks it up.",
)
def main(
    robot_type: str,
    ip_address: Optional[str],
    port: int,
    speed_ratio: int,
    calibration_path: str,
    camera_resolution: str,
    stay: bool,
    table_z: Optional[float],
    pregrasp_height: float,
    target_path: str,
    debug_dir: Optional[str],
    from_capture: Optional[str],
    no_write: bool,
) -> None:
    """Perceive the pile, choose the brick to grasp, and hand it to submodule_1."""
    plane = resolve_table_plane(table_z)

    if from_capture is not None:
        logger.info(f"Replaying the capture at {from_capture}; no robot and no camera.")
        view = PileView.load(from_capture)
        analysis = analyse_pile(view, plane, robot_type)
        assign_priorities(analysis.ordered, None, pregrasp_height)
        manifest = build_pile_manifest(analysis, robot_type, pregrasp_height)
        report(analysis)
        logger.warning("Replayed without an arm, so no brick was checked for reachability.")
        if debug_dir:
            save_debug_output(debug_dir, analysis, manifest)
        if no_write:
            logger.info("--no-write: the target file was left alone.")
        elif analysis.target is None:
            logger.warning(f"Nothing graspable was found, so {target_path} was left alone.")
        else:
            write_pile_target(target_path, manifest)
        return

    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]
    X_tcp_camera = load_camera_pose_in_tcp(calibration_path)

    with connect_arm(robot_type, ip_address, port) as arm, open_camera(
        CAMERA_RESOLUTIONS[camera_resolution]
    ) as camera:
        # The reachability checks below go through getInverseKinematics, which reports every pose as
        # unreachable while the UR control script is stopped -- and a previous run's long pause is
        # exactly how it gets stopped.
        ensure_control_ready(arm)

        joint_configuration = None
        if not stay:
            joint_configuration = np.asarray(PILE_VIEW, dtype=float)
            if joint_configuration.size != arm.manipulator_specs.dof:
                raise click.ClickException(
                    f"PILE_VIEW has {joint_configuration.size} joint value(s) but the {robot_type} arm has "
                    f"{arm.manipulator_specs.dof} joints. Fill it in at the top of this file, or pass --stay."
                )
        joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)

        view = capture_pile_view(arm, camera, X_tcp_camera, joint_configuration, joint_speed)
        analysis = analyse_pile(view, plane, robot_type)
        assign_priorities(analysis.ordered, arm, pregrasp_height)
        manifest = build_pile_manifest(analysis, robot_type, pregrasp_height)
        report(analysis)

        if debug_dir:
            save_debug_output(debug_dir, analysis, manifest)
        if no_write:
            logger.info("--no-write: the target file was left alone.")
        elif analysis.target is None:
            logger.warning(f"Nothing graspable was found, so {target_path} was left alone.")
        else:
            write_pile_target(target_path, manifest)

    os._exit(0)


if __name__ == "__main__":
    main()
