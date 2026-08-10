"""Measure the tabletop's height in the robot's base frame, from the hand-eye calibration's images.

``config.TABLE_Z`` is the one number the whole pick is anchored to: submodule_1 projects the clicked
rays onto the plane at ``TABLE_Z + brick height``, so an error in it lifts the pregrasp by that much
and, on a line of sight that is not vertical, slides the target sideways as well. It is not zero --
the base frame's z = 0 is the robot's mounting flange and the arm is bolted to the table through a
plate -- so it has to be measured.

It already has been, by the calibration: the charuco board lies flat on this same table, and every
calibration sample photographs it from a different arm pose. Mapping the detected board into the base
frame with ``tcp_pose @ camera_pose_in_tcp @ board_pose_in_camera`` puts its surface at the table's
height, once per sample. The board is a printed sheet lying directly on the tabletop, so its surface
*is* the tabletop.

Agreement across the samples is the thing to read: they look from genuinely different angles, so a
consistent answer means the camera pose is consistent too, while a spread of centimetres means the
hand-eye calibration is the problem rather than the table.

Run it after every re-calibration, and put the number it prints into ``config.TABLE_Z``::

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
from config import DEFAULT_CALIBRATION_DIR, TABLE_Z, load_camera_pose_in_tcp  # noqa: E402

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
    """Measure config.TABLE_Z from a hand-eye calibration's board images."""
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

    logger.info(f"config.TABLE_Z is currently {TABLE_Z:+.4f} m; this calibration measures {table_z:+.4f} m.")
    if abs(table_z - TABLE_Z) > 0.001:
        click.echo(f"\nUpdate src/config.py:\n\n    TABLE_Z = {table_z:.4f}\n")
    else:
        click.echo(f"\nconfig.TABLE_Z = {TABLE_Z:.4f} still matches this calibration; nothing to change.\n")


if __name__ == "__main__":
    main()
