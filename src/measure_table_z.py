"""Cross-check the hand-eye calibration by measuring the board's plane from every sample.

SUPERSEDED for the table height: use ``calibrate_table.py``, which touches the tabletop with the arm.
This script reaches the base frame *through* the hand-eye calibration, so it carries that
calibration's error -- it put the tabletop at z = -0.0240 m where touching it found z = -0.0044 m, and
that 19.6 mm drove the fingertips into the table on every grasp. What it is still good for is
measuring that error: run both and compare.

Each calibration sample photographs the charuco board from a different arm pose, so mapping the
detected board into the base frame with ``tcp_pose @ camera_pose_in_tcp @ board_pose_in_camera``
should put its surface at the same height every time. Agreement across the samples is the thing to
read: they look from genuinely different angles, so a consistent answer means the camera pose is
consistent, while a spread of centimetres means the hand-eye calibration is inconsistent and needs
more board poses.

Note that the height it reports is the height of *whatever surface the board is lying on*. That is
only the tabletop if the board is lying on the tabletop::

    python src/measure_table_z.py
    python src/measure_table_z.py --calibration-dir /path/to/other_calibration_dir
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    detect_charuco_board,
)
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from loguru import logger

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import DEFAULT_CALIBRATION_DIR, load_camera_pose_in_tcp, load_table_plane  # noqa: E402

# Above this spread between samples, the samples are not describing one flat plane and the number
# should not be trusted -- which points at the hand-eye calibration, not at the table.
SUSPICIOUS_SPREAD = 0.005
# A board more than this far off horizontal is not lying flat on the table, so nothing below applies.
MAX_BOARD_TILT_DEG = 5.0


def _load_pose(path: str) -> HomogeneousMatrixType:
    with open(path) as f:
        pose = json.load(f)
    translation = np.array([pose["position_in_meters"][axis] for axis in ("x", "y", "z")])
    euler = np.array([pose["rotation_euler_xyz_in_radians"][angle] for angle in ("roll", "pitch", "yaw")])
    return SE3Container.from_euler_angles_and_translation(euler, translation).homogeneous_matrix


def _load_intrinsics(data_dir: str) -> np.ndarray:
    with open(os.path.join(data_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    focal = intrinsics["focal_lengths_in_pixels"]
    principal = intrinsics["principal_point_in_pixels"]
    return np.array([[focal["fx"], 0, principal["cx"]], [0, focal["fy"], principal["cy"]], [0, 0, 1]])


def measure_table_z(calibration_dir: str) -> Tuple[Optional[float], List[Tuple[int, float, float]]]:
    """Table height in the base frame, and the ``(sample, height, board tilt)`` it came from."""
    data_dir = os.path.join(calibration_dir, "data")
    image_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.png")))
    pose_paths = sorted(glob.glob(os.path.join(data_dir, "tcp_pose_*.json")))
    if not image_paths or len(image_paths) != len(pose_paths):
        raise click.ClickException(
            f"Expected matching image_*.png and tcp_pose_*.json in {data_dir}; found "
            f"{len(image_paths)} image(s) and {len(pose_paths)} pose(s)."
        )

    intrinsics_matrix = _load_intrinsics(data_dir)
    X_tcp_camera = load_camera_pose_in_tcp(calibration_dir)

    samples: List[Tuple[int, float, float]] = []
    for index, (image_path, pose_path) in enumerate(zip(image_paths, pose_paths)):
        image = cv2.imread(image_path)
        board_in_camera = detect_charuco_board(
            image, intrinsics_matrix, aruco_dict=AIRO_DEFAULT_ARUCO_DICT, charuco_board=AIRO_DEFAULT_CHARUCO_BOARD
        )
        if board_in_camera is None:
            logger.warning(f"No board detected in {os.path.basename(image_path)}; skipping it.")
            continue

        board_in_base = _load_pose(pose_path) @ X_tcp_camera @ board_in_camera
        tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(board_in_base[2, 2])))))
        samples.append((index, float(board_in_base[2, 3]), tilt_deg))

    if not samples:
        return None, []
    return float(np.median([height for _, height, _ in samples])), samples


@click.command()
@click.option(
    "--calibration-dir",
    default=DEFAULT_CALIBRATION_DIR,
    show_default=True,
    help="The hand-eye-calibration output directory, holding data/ and results_n=*/.",
)
def main(calibration_dir: str) -> None:
    """Check a hand-eye calibration's consistency against the board plane it was computed from."""
    table_z, samples = measure_table_z(calibration_dir)
    if table_z is None:
        raise click.ClickException(
            f"No charuco board could be detected in any image in {calibration_dir}/data. The table height "
            "cannot be measured from this calibration."
        )

    for index, height, tilt_deg in samples:
        logger.info(f"sample {index}: board surface at z={height:+.4f} m, {tilt_deg:.1f} deg off horizontal.")

    heights = np.array([height for _, height, _ in samples])
    spread = float(heights.max() - heights.min())
    worst_tilt = max(tilt for _, _, tilt in samples)

    if worst_tilt > MAX_BOARD_TILT_DEG:
        logger.warning(
            f"The board reads up to {worst_tilt:.1f} deg off horizontal (limit {MAX_BOARD_TILT_DEG:.0f}). It was "
            "not lying flat on the table, or the hand-eye calibration's rotation is off -- either way the height "
            "below is not the table's."
        )
    if spread > SUSPICIOUS_SPREAD:
        logger.warning(
            f"The {len(samples)} samples disagree by {spread * 1000:.1f} mm (limit {SUSPICIOUS_SPREAD * 1000:.0f}). "
            "They see one flat table from different angles, so they should agree; this much spread means the "
            "hand-eye calibration is inconsistent. Re-run it with more board poses before trusting this number."
        )
    else:
        logger.success(f"The {len(samples)} samples agree to within {spread * 1000:.1f} mm.")

    logger.info(f"The board's surface reads z={table_z:+.4f} m in the base frame through this calibration.")

    plane = load_table_plane()
    if plane is not None:
        touched = plane.z_at(0.0, 0.0) if (plane.a == 0.0 and plane.b == 0.0) else None
        logger.info(f"For comparison, the arm touched the table: {plane.describe()}.")
        if touched is not None:
            logger.info(
                f"Camera-through-calibration says {table_z:+.4f} m, touching says {touched:+.4f} m -- "
                f"a {abs(table_z - touched) * 1000:.1f} mm gap, which is this calibration's vertical error "
                "*if* the board is lying on the same surface the arm touched."
            )

    click.echo(
        "\nThis is a calibration cross-check, not the table height the pick uses.\n"
        "  - It only equals the tabletop if the board is lying flat on the tabletop itself. If the board is\n"
        "    taped to a panel, a wall or a riser, this number is that surface's height, not the table's.\n"
        "  - It reaches the base frame through the hand-eye calibration, so it carries that calibration's\n"
        "    error. That is why it is not used: it read -0.0240 m where the arm touched -0.0044 m.\n"
        "\nFor the height the pick actually descends to, run:\n\n    python src/calibrate_table.py\n\n"
        "which touches the tabletop with the arm and needs no camera at all. What agreement (or not) between\n"
        "the two tells you is how far off the hand-eye calibration is vertically.\n"
    )


if __name__ == "__main__":
    main()
