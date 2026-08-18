"""Why the perception cannot find the bare table, measured at each viewpoint.

The symptom this exists for is ``build_table_model``'s warning::

    Only 477 pixel(s) of bare table were found by depth, too few to seed the table's colour model

with a camera feed in which the table plainly fills the frame. "Bare table" is not what the eye sees;
it is ``reach_mask & depth_valid & (height < TABLE_BAND_M)`` -- inside the arm's reach, *with a depth
reading*, and within **1.5 mm** of the touched-off plane. Two independent things can empty that mask
while the table sits there in plain sight, and they need opposite fixes:

1. **The depth stream does not reach the table.** Dark, glossy or featureless tabletops return nothing
   to a stereo sensor, so the table's pixels are simply not in ``depth_valid``. Fix the surface or the
   exposure -- no threshold will conjure the missing readings.
2. **The table has depth, but not at the height the plane says.** ``height`` is a depth reading routed
   through the hand-eye calibration minus a plane routed through the fingertip TCP: two calibrations
   with millimetres of error each, compared against a 1.5 mm band. A few millimetres of bias puts the
   entire tabletop above the band -- and above ``FOREGROUND_STRONG_HEIGHT_M`` (2.5 mm) as well, which
   is the same table coming back as dozens of "bricks".

So this script goes to each viewpoint, builds exactly the :class:`~m1.physical.submodule_3.Scene` the
perception builds, and prints what the two masks actually contain: how much of the reachable frame has
depth at all, where the height readings sit relative to the touched-off plane, and -- by fitting a
plane to the depth itself -- how far the camera's idea of the tabletop is from the arm's, in offset and
in tilt. The verdict at the end names which of the two cases this rig is in.

Nothing moves except the arm going to the two viewpoints, and nothing is written to the calibration.

Usage::

    python src/tools/diagnose_table.py
    python src/tools/diagnose_table.py --table-z 0.03      # try a level plane at that height instead
    python src/tools/diagnose_table.py --save-dir run/table_diagnosis   # + height maps as PNGs
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from loguru import logger

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.config import DEFAULT_CALIBRATION_DIR, DEFAULT_CAMERA_RESOLUTION, SUPPORTED_ROBOT_TYPES  # noqa: E402
from m1.physical import cell as C  # noqa: E402
from m1.physical import submodule_3 as perception  # noqa: E402
from m1.physical.submodule_1 import VIEWPOINTS, VIEW_TARGET  # noqa: E402

#: Height bins the text histogram is drawn over, in millimetres above the touched-off plane. Wide
#: enough to show a tabletop that came out centimetres off, fine enough to show a 1 mm bias.
HISTOGRAM_EDGES_MM = (-20, -10, -5, -2.5, -1.5, 0, 1.5, 2.5, 5, 10, 20, 40, 100)
#: A tabletop this much of the reachable frame is expected to be. Below it, the "flat and level" fit
#: below is fitting something else -- a pile that has taken over the frame, most likely.
MIN_TABLE_FRACTION = 0.2
#: Beyond this the camera's tabletop and the arm's are not the same surface, and the segmentation is
#: measuring the difference between two calibrations rather than the pile.
SUSPICIOUS_OFFSET_MM = 2.0
SUSPICIOUS_TILT_DEG = 0.5


fit_plane_to_depth = perception.fit_plane_to_depth


def tilt_degrees(plane: Sequence[float]) -> float:
    """How far the plane leans off level."""
    a, b, _ = plane
    return float(np.degrees(np.arctan(np.hypot(a, b))))


def _normal(plane: Sequence[float]) -> np.ndarray:
    normal = np.array([-plane[0], -plane[1], 1.0])
    return normal / np.linalg.norm(normal)


def angle_between(first: Sequence[float], second: Sequence[float]) -> float:
    """Angle between two planes, in degrees.

    The angle between their *normals*, not the difference of their tilts: two planes can lean by the
    same amount in opposite directions, which is a difference of zero by tilt and twice the tilt in
    fact. Getting this wrong reports two surfaces 3 degrees apart as agreeing to half a degree.
    """
    return float(np.degrees(np.arccos(np.clip(float(_normal(first) @ _normal(second)), -1.0, 1.0))))


def histogram(heights_mm: np.ndarray, width: int = 46) -> None:
    """A text histogram of the height readings, in the bins the thresholds care about."""
    total = max(heights_mm.size, 1)
    counts, _ = np.histogram(heights_mm, bins=np.asarray(HISTOGRAM_EDGES_MM, float))
    below = int((heights_mm < HISTOGRAM_EDGES_MM[0]).sum())
    above = int((heights_mm >= HISTOGRAM_EDGES_MM[-1]).sum())
    tallest = max(counts.max() if counts.size else 1, below, above, 1)

    def row(label: str, count: int, note: str = "") -> None:
        bar = "#" * int(round(width * count / tallest))
        print(f"    {label:>16s} {100.0 * count / total:5.1f}%  {bar:<{width}s} {note}")

    print("    height above the touched-off plane, over the reachable frame that has depth:")
    row(f"< {HISTOGRAM_EDGES_MM[0]:g}", below, "(dropped by SCENE_FLOOR_M)")
    for low, high, count in zip(HISTOGRAM_EDGES_MM, HISTOGRAM_EDGES_MM[1:], counts):
        note = ""
        if high <= perception.SCENE_FLOOR_M * 1000:
            note = "(dropped by SCENE_FLOOR_M)"
        elif high <= perception.TABLE_BAND_M * 1000:
            note = "<- bare table"
        elif low >= perception.FOREGROUND_STRONG_HEIGHT_M * 1000:
            note = "<- read as brick"
        row(f"{low:g} .. {high:g}", int(count), note)
    row(f">= {HISTOGRAM_EDGES_MM[-1]:g}", above, "(dropped by SCENE_CEILING_M)")


def diagnose(cell: C.Cell, name: str, eye: Sequence[float]) -> None:
    """Go to one viewpoint and report what the perception's two table masks contain."""
    configuration = C.VIEWPOINT_JOINT_CONFIGURATIONS.get(name)
    if configuration is not None:
        logger.info(f"{name}: moving to the measured viewpoint configuration ...")
        cell.move_arm_to(configuration)
    else:
        pose = C.look_at_tool_pose(cell, np.asarray(eye, float), np.asarray(VIEW_TARGET, float))
        q = C.solve_tool_ik(cell, pose)
        if q is None:
            logger.error(f"{name} at {np.round(eye, 3)} m is not reachable; skipping it.")
            return
        cell.move_arm_to(q)
    cell.advance(0.4)

    view = cell.capture(name)
    scene = perception.build_scene(view, cell.table_plane, cell.robot_type)

    frame = scene.height.size
    reachable = scene.reach_mask
    with_depth = reachable & scene.depth_valid
    table = scene.table_mask
    print(f"\n{'=' * 88}\n{name}\n{'=' * 88}")
    print(f"  camera {np.round(np.asarray(view.X_base_camera, float)[:3, 3], 4)} m, "
          f"{scene.scale.distance_m * 100:.0f} cm from what it is looking at")
    print(f"  reachable frame       {100.0 * reachable.mean():5.1f}% of {frame} working pixel(s)")
    print(f"  ... with depth        {100.0 * with_depth.sum() / max(reachable.sum(), 1):5.1f}% of that "
          f"({with_depth.sum()} px)")
    print(f"  ... bare table        {100.0 * table.sum() / max(reachable.sum(), 1):5.1f}% of that "
          f"({table.sum()} px; the colour model needs 2% to seed from depth)")
    if with_depth.any():
        print()
        histogram(scene.height[with_depth] * 1000.0)

    fitted = fit_plane_to_depth(scene)
    if fitted is None:
        print("\n  Not enough depth over the reachable frame to fit a tabletop to it at all.")
        print("  -> the depth stream, not the plane, is the problem. See the verdict below.")
        return
    plane, rms, inliers = fitted
    x, y = C.PILE_CENTER
    touched_z = cell.table_z_at(x, y)
    fitted_z = plane[2] + plane[0] * x + plane[1] * y
    offset_mm = (fitted_z - touched_z) * 1000.0
    fraction = inliers / max(reachable.sum(), 1)

    print()
    print(f"  the arm's tabletop    z = {cell.table_plane[2]:+.4f} {cell.table_plane[0]:+.5f} x "
          f"{cell.table_plane[1]:+.5f} y   ({tilt_degrees(cell.table_plane):.2f} deg tilt), "
          f"z={touched_z:+.4f} m under the pile")
    print(f"  the camera's tabletop z = {plane[2]:+.4f} {plane[0]:+.5f} x {plane[1]:+.5f} y   "
          f"({tilt_degrees(plane):.2f} deg tilt), z={fitted_z:+.4f} m under the pile")
    print(f"  fitted to {inliers} px ({100.0 * fraction:.0f}% of the reachable frame), "
          f"flat to {rms * 1000:.2f} mm RMS")
    print(f"  -> the camera puts the tabletop {offset_mm:+.1f} mm from where the fingertips found it under "
          f"PILE_CENTER, and the two surfaces are {angle_between(plane, cell.table_plane):.2f} deg apart")

    # The offset under PILE_CENTER is only meaningful if that is where the camera is pointing. Say where
    # it actually is looking, and what the two planes do there -- an extrapolated plane is not evidence.
    px_all, py_all = scene.table_xy[..., 0][with_depth], scene.table_xy[..., 1][with_depth]
    mx, my = float(np.median(px_all)), float(np.median(py_all))
    here_touched = cell.table_plane[2] + cell.table_plane[0] * mx + cell.table_plane[1] * my
    here_fitted = plane[2] + plane[0] * mx + plane[1] * my
    print(f"  the frame covers x {px_all.min():+.3f}..{px_all.max():+.3f}, y {py_all.min():+.3f}..{py_all.max():+.3f} m, "
          f"centred on ({mx:+.3f}, {my:+.3f})")
    print(f"  there: fingertips say z={here_touched:+.4f} m, the camera says z={here_fitted:+.4f} m "
          f"-> {(here_fitted - here_touched) * 1000:+.1f} mm apart")
    if np.hypot(mx - C.PILE_CENTER[0], my - C.PILE_CENTER[1]) > 0.10:
        print(f"     !! that is {np.hypot(mx - C.PILE_CENTER[0], my - C.PILE_CENTER[1]) * 100:.0f} cm from "
              f"PILE_CENTER {C.PILE_CENTER}, so the touched-off plane is being extrapolated to get here "
              "and PILE_CENTER needs measuring (src/tools/teach_pose.py)")

    # What the same frame would have yielded had the tabletop come from the depth itself. Heights are
    # measured from the touched-off plane, so swapping planes is just adding the gap between them:
    # height' = z - fitted(x, y) = height + touched(x, y) - fitted(x, y).
    px, py = scene.table_xy[..., 0][with_depth], scene.table_xy[..., 1][with_depth]
    touched_surface = cell.table_plane[2] + cell.table_plane[0] * px + cell.table_plane[1] * py
    fitted_surface = plane[2] + plane[0] * px + plane[1] * py
    refitted = scene.height[with_depth] + (touched_surface - fitted_surface)
    would_be = int((refitted < perception.TABLE_BAND_M).sum())
    print(f"  -> with the camera's own plane, bare table would be "
          f"{100.0 * would_be / max(reachable.sum(), 1):.1f}% of the reachable frame "
          f"({would_be} px) instead of {table.sum()}")

    if fraction < MIN_TABLE_FRACTION:
        print("     (careful: that fit covers little of the frame, so it may be a flat part of the pile)")


def save_maps(cell: C.Cell, name: str, directory: str) -> None:
    """Write the colour frame, the depth-valid mask and the height map, to look at afterwards."""
    view = cell.capture(name)
    scene = perception.build_scene(view, cell.table_plane, cell.robot_type)
    stem = os.path.join(directory, name.replace(" ", "_"))
    os.makedirs(directory, exist_ok=True)

    cv2.imwrite(f"{stem}_colour.png", scene.bgr)
    cv2.imwrite(f"{stem}_depth_valid.png", (scene.depth_valid * 255).astype(np.uint8))
    cv2.imwrite(f"{stem}_table_mask.png", (scene.table_mask * 255).astype(np.uint8))
    # 0 mm is black, 20 mm and up is white, so a tabletop that reads as elevated is visibly grey.
    normalised = np.clip(scene.height / 0.020, 0.0, 1.0)
    coloured = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~scene.depth_valid] = (0, 0, 0)
    cv2.imwrite(f"{stem}_height.png", coloured)
    np.savez_compressed(
        f"{stem}.npz",
        height=scene.height,
        depth_valid=scene.depth_valid,
        reach_mask=scene.reach_mask,
        table_xy=scene.table_xy,
        plane=np.asarray(scene.plane, float),
        X_base_camera=np.asarray(view.X_base_camera, float),
    )
    logger.info(f"{name}: wrote {stem}_*.png and {stem}.npz")


@click.command()
@click.option("--robot-type", type=click.Choice(SUPPORTED_ROBOT_TYPES), default="ur3e", show_default=True)
@click.option("--ip-address", default=None, help="Robot controller IP. Defaults per robot type.")
@click.option("--calibration-path", default=DEFAULT_CALIBRATION_DIR, show_default=True)
@click.option("--camera-resolution", type=click.Choice(list(C.CAMERA_RESOLUTIONS)), default=DEFAULT_CAMERA_RESOLUTION, show_default=True)
@click.option("--speed-ratio", type=click.IntRange(1, 100), default=C.DEFAULT_SPEED_RATIO, show_default=True)
@click.option("--table-z", type=float, default=None, help="Diagnose against a level plane at this height instead of the touched-off one.")
@click.option("--save-dir", default=None, help="Also write each view's colour, height map and masks here.")
def command(
    robot_type: str,
    ip_address: Optional[str],
    calibration_path: str,
    camera_resolution: str,
    speed_ratio: int,
    table_z: Optional[float],
    save_dir: Optional[str],
) -> None:
    with C.build_cell(
        robot_type=robot_type,
        ip_address=ip_address,
        calibration_path=calibration_path,
        camera_resolution=camera_resolution,
        speed_ratio=speed_ratio,
        table_z=table_z,
        with_gripper=False,
    ) as cell:
        for name, eye in VIEWPOINTS:
            diagnose(cell, name, eye)
            if save_dir:
                save_maps(cell, name, save_dir)

        print(f"\n{'=' * 88}\nhow to read this\n{'=' * 88}")
        print(
            "  Little of the reachable frame has depth\n"
            "      -> the sensor is not seeing the tabletop: dark, glossy or featureless wood, or the\n"
            f"         exposure is wrong. Nothing in submodule_3 can fix that.\n\n"
            "  Depth is there, but the heights sit above the 1.5 mm band and the camera's own plane is\n"
            f"  more than {SUSPICIOUS_OFFSET_MM:.0f} mm or {SUSPICIOUS_TILT_DEG:.1f} deg from the touched-off one\n"
            "      -> the two calibrations disagree, and the segmentation is measuring that disagreement\n"
            "         rather than the pile. Re-run calibrate_table.py with the gripper at the width the\n"
            "         picks use and more touch points spread wider; if the disagreement survives that, the\n"
            "         table level for *segmentation* has to come from the depth itself, keeping the\n"
            "         touched-off plane for the grasp height.\n\n"
            "  Both planes agree and the heights straddle the band\n"
            "      -> ordinary depth noise against a 1.5 mm band. TABLE_BAND_M is the knob."
        )


if __name__ == "__main__":
    command()
