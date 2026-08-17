"""How far the perception's idea of a brick is from where the brick actually is, measured.

The symptom this exists for is "the arm went to the brick and was way off". There is no ground truth on
a bench, so this makes one: the perception says where a brick is, then *you put the fingertips on that
same brick* and the controller says where they are. The difference between those two numbers is the
error the grasp suffers, in the frame the grasp happens in.

Measured at three or four bricks in different parts of the frame, the error's *shape* names its cause,
and this fits for both at once:

* **A constant offset** in x and y, the same everywhere -- the hand-eye calibration's translation. The
  camera is not where the calibration says it is, so every ray starts in the wrong place.
* **An offset growing with distance from the point directly under the camera**, proportional to
  ``tan(angle off vertical)`` -- a height error. Every position here is the brick's outline projected
  onto the plane of its own top face (:func:`m1.physical.submodule_3.measure_footprint`), so if that
  plane is at the wrong height the outline slides sideways along the line of sight. The fit reports the
  height error that would explain what it sees, in millimetres.
* **Neither fits, and the residual stays large** -- the hand-eye *rotation*, which tilts every ray by a
  different amount. At 37 cm, one degree is 6.5 mm.

**Two cheap things to rule out first, which this prints without needing you to touch anything.** Whether
the region it picked is the brick at all -- the overlay it saves shows what it outlined, and on a rig
where the table reads as elevated the best-scoring "brick" is a patch of bare wood, which no calibration
error explains. And whether the brick's height was *measured* or guessed: a part standing on edge shows
the camera a narrow top face, depth coverage over it can fall below
:data:`m1.physical.submodule_3.MIN_DEPTH_COVERAGE`, and the height then falls back to
``config.FALLBACK_BRICK_HEIGHT`` (9.6 mm). A part 25 mm tall projected onto a plane 9.6 mm up lands
sideways by 15 mm times the tangent of the view angle, and the fingers stop 15 mm above it as well.

Usage::

    python src/tools/verify_pick_accuracy.py                  # freedrive to each brick in turn
    python src/tools/verify_pick_accuracy.py --samples 5
    python src/tools/verify_pick_accuracy.py --view "view 2"   # check the other viewpoint the same way

The arm moves once, to the viewpoint. After that it is limp (freedrive) while you place the fingertips,
and freedrive is switched off before every reading.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from loguru import logger

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.config import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_RESOLUTION,
    FALLBACK_BRICK_HEIGHT,
    PREGRASP_HEIGHT,
    SUPPORTED_ROBOT_TYPES,
)
from m1.physical import cell as C  # noqa: E402
from m1.physical import submodule_3 as perception  # noqa: E402
from m1.physical.submodule_1 import VIEWPOINTS, VIEW_TARGET  # noqa: E402

#: A touch further than this from every brick the perception reported is not a measurement of any of
#: them -- either the wrong brick was touched, or the error is larger than the pile is spread out.
MAX_ASSOCIATION_M = 0.05
#: A part reading taller than this is not a lego part on a table; it is the table itself reading high,
#: which means the plane is wrong and every position is wrong sideways too.
IMPLAUSIBLE_HEIGHT_M = 0.030


def look(cell: C.Cell, name: str, eye: Sequence[float]) -> perception.PileAnalysis:
    """Go to the viewpoint and run the real perception on what it sees."""
    configuration = C.VIEWPOINT_JOINT_CONFIGURATIONS.get(name)
    if configuration is not None:
        logger.info(f"{name}: moving to the measured viewpoint configuration ...")
        cell.move_arm_to(configuration)
    else:
        pose = C.look_at_tool_pose(cell, np.asarray(eye, float), np.asarray(VIEW_TARGET, float))
        q = C.solve_tool_ik(cell, pose)
        if q is None:
            raise click.ClickException(f"{name} at {np.round(eye, 3)} m is not reachable.")
        cell.move_arm_to(q)
    cell.advance(0.4)

    view = cell.capture(name)
    analysis = perception.analyse_pile(view, cell.table_plane, cell.robot_type)
    perception.assign_priorities(analysis.ordered, None, PREGRASP_HEIGHT)
    return analysis


def report_bricks(analysis: perception.PileAnalysis) -> List[perception.Brick]:
    """Print what the perception found, flagging the two cheap explanations."""
    bricks = [b for b in analysis.ordered if b.graspable and b.confidence >= 0.7]
    print(f"\n{len(bricks)} graspable brick(s), best first:\n")
    print(f"{'#':>2} {'colour':10s} {'w x l mm':>13s} {'position m':>18s} {'height':>9s} {'depth':>6s} {'score':>6s}")
    for rank, brick in enumerate(bricks, start=1):
        height = f"{brick.height_m * 1000:5.1f}mm" + ("" if brick.height_measured else "*")
        print(
            f"{rank:2d} {brick.colour_name:10s} {brick.width_mm:5.1f} x {brick.length_mm:5.1f} "
            f"({brick.center_m[0]:+.4f}, {brick.center_m[1]:+.4f}) {height:>9s} "
            f"{brick.depth_coverage:6.2f} {brick.score:6.3f}"
        )
    guessed = [b for b in bricks if not b.height_measured]
    if guessed:
        print(
            f"\n  * {len(guessed)} brick(s) had too little depth on them to measure a height, so they were "
            f"projected onto a plane {FALLBACK_BRICK_HEIGHT * 1000:.1f} mm above the table whatever their real\n"
            "    height is. A part standing on edge is the usual cause, and it lands sideways by the height "
            "error times the tangent of the view angle."
        )
    tall = [b for b in bricks if b.height_m > IMPLAUSIBLE_HEIGHT_M]
    if tall:
        print(
            f"\n  !! {len(tall)} brick(s) read taller than {IMPLAUSIBLE_HEIGHT_M * 1000:.0f} mm. Lego parts are "
            "not that tall: the table plane is reading low, so every position here is displaced sideways too.\n"
            "     Run src/tools/diagnose_table.py before trusting anything below."
        )
    return bricks


def touch(arm, prompt: str) -> Optional[np.ndarray]:
    """Freedrive until the operator says the fingertips are placed, then read the TCP position."""
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None or not hasattr(rtde_control, "teachMode"):
        raise click.ClickException(
            "This driver has no freedrive (teachMode), so the fingertips cannot be placed by hand. Move the "
            "arm with the pendant and use --from-current-pose instead."
        )
    rtde_control.teachMode()
    print(f"\n  FREEDRIVE ON -- {prompt}")
    try:
        answer = input("  press enter when the fingertips are on it (or 's' to skip, 'q' to stop) ").strip().lower()
    finally:
        rtde_control.endTeachMode()
    if answer.startswith("q"):
        return None
    if answer.startswith("s"):
        return np.array([np.nan, np.nan, np.nan])
    return np.asarray(arm.get_tcp_pose(), float)[:3, 3]


def geometry(brick: perception.Brick, analysis: perception.PileAnalysis, plane: Tuple[float, float, float]):
    """Where the camera is over this brick, and how far off vertical it is looking at it."""
    camera = np.asarray(analysis.view.X_base_camera, float)[:3, 3]
    a, b, c = plane
    top_face_z = c + a * brick.center_m[0] + b * brick.center_m[1] + brick.height_m
    offset = np.array(brick.center_m) - camera[:2]      # away from the point under the camera
    distance = float(np.linalg.norm(offset))
    drop = float(camera[2] - top_face_z)
    radial = offset / distance if distance > 1e-9 else np.zeros(2)
    return radial, (distance / drop if drop > 1e-6 else 0.0), top_face_z


def decompose(samples: List[dict]) -> None:
    """Split the measured errors into a constant offset and a height error, and say what is left.

    Least squares on ``error = (dx, dy) + dz * tan(theta) * radial``: two unknowns that are the same at
    every brick and one that grows with how obliquely the brick is seen, which is what separates a
    camera in the wrong place from a projection plane at the wrong height.
    """
    rows, targets = [], []
    for sample in samples:
        radial, tangent, _ = sample["geometry"]
        rows.append([[1.0, 0.0, tangent * radial[0]], [0.0, 1.0, tangent * radial[1]]])
        targets.append(sample["error"][:2])
    design = np.vstack([np.asarray(r) for r in rows])
    observed = np.concatenate(targets)
    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ solution
    dx, dy, dz = solution

    print(f"\n{'=' * 88}\nwhat explains the error\n{'=' * 88}")
    print(f"  constant offset      ({dx * 1000:+.1f}, {dy * 1000:+.1f}) mm  -> the hand-eye translation")
    print(f"  height error         {dz * 1000:+.1f} mm            -> the plane the outlines are projected onto")
    print(f"  unexplained residual {np.sqrt(np.mean(residual ** 2)) * 1000:.1f} mm RMS  -> hand-eye rotation, "
          "or one bad touch")
    if len(samples) < 3:
        print("  (two samples fit three numbers almost exactly -- take three or four for this to mean anything)")

    magnitude = np.hypot(dx, dy)
    print()
    if magnitude > 0.005 and magnitude > abs(dz) * 0.5:
        print("  The offset is mostly constant, which points at the hand-eye calibration's translation.")
        print(f"  {DEFAULT_CALIBRATION_DIR} was solved from 3 board poses; redo it with 15 or more, and check")
        print("  the four solvers agree to a few mm before trusting the result.")
    elif abs(dz) > 0.005:
        print(f"  The offset grows with the view angle, which is a height error of about {dz * 1000:.0f} mm --")
        print("  either the touched-off table plane or the brick heights (look at the depth column above).")
    else:
        print("  Both terms are small; whatever is left is in the residual, i.e. the hand-eye rotation.")


@click.command()
@click.option("--robot-type", type=click.Choice(SUPPORTED_ROBOT_TYPES), default="ur3e", show_default=True)
@click.option("--ip-address", default=None, help="Robot controller IP. Defaults per robot type.")
@click.option("--calibration-path", default=DEFAULT_CALIBRATION_DIR, show_default=True)
@click.option("--camera-resolution", type=click.Choice(list(C.CAMERA_RESOLUTIONS)), default=DEFAULT_CAMERA_RESOLUTION, show_default=True)
@click.option("--speed-ratio", type=click.IntRange(1, 100), default=C.DEFAULT_SPEED_RATIO, show_default=True)
@click.option("--view", "view_name", default=VIEWPOINTS[0][0], show_default=True, help="Which viewpoint to check.")
@click.option("--samples", type=click.IntRange(1, 12), default=4, show_default=True, help="How many bricks to touch.")
@click.option("--save-dir", default="run/pick_accuracy", show_default=True, help="Where the overlay is written.")
def command(
    robot_type: str,
    ip_address: Optional[str],
    calibration_path: str,
    camera_resolution: str,
    speed_ratio: int,
    view_name: str,
    samples: int,
    save_dir: str,
) -> None:
    eye = dict(VIEWPOINTS).get(view_name)
    if eye is None:
        raise click.ClickException(f"Unknown viewpoint {view_name!r}; the ones defined are {[n for n, _ in VIEWPOINTS]}.")

    with C.build_cell(
        robot_type=robot_type,
        ip_address=ip_address,
        calibration_path=calibration_path,
        camera_resolution=camera_resolution,
        speed_ratio=speed_ratio,
        with_gripper=False,
    ) as cell:
        analysis = look(cell, view_name, eye)
        bricks = report_bricks(analysis)

        os.makedirs(save_dir, exist_ok=True)
        overlay_path = os.path.join(save_dir, f"{view_name.replace(' ', '_')}_overlay.png")
        cv2.imwrite(overlay_path, perception.render_overlay(analysis))
        cv2.imwrite(os.path.join(save_dir, f"{view_name.replace(' ', '_')}_stages.png"),
                    perception.render_debug_panel(analysis))
        print(f"\n  wrote {overlay_path} -- look at it before touching anything. If the numbered regions are")
        print("  wood rather than bricks, stop here: the positions are right about the wrong thing.")

        if not bricks:
            raise click.ClickException("Nothing graspable was found, so there is nothing to measure against.")

        measured: List[dict] = []
        for index in range(min(samples, len(bricks))):
            placed = touch(cell.arm, f"put the fingertips on the CENTRE OF THE TOP FACE of any brick above ({index + 1}/{min(samples, len(bricks))})")
            if placed is None:
                break
            if not np.all(np.isfinite(placed)):
                continue

            distances = [float(np.linalg.norm(np.array(b.center_m) - placed[:2])) for b in bricks]
            nearest = int(np.argmin(distances))
            brick = bricks[nearest]
            if distances[nearest] > MAX_ASSOCIATION_M:
                print(f"  that touch is {distances[nearest] * 1000:.0f} mm from every brick reported -- either it "
                      "was not one of them, or the error is bigger than the pile. Not counted.")
                continue

            geo = geometry(brick, analysis, cell.table_plane)
            error = np.array(brick.center_m + (geo[2],)) - placed
            measured.append({"brick": brick, "error": error, "geometry": geo, "touched": placed})
            print(f"  brick {nearest + 1} ({brick.colour_name}): perception "
                  f"({brick.center_m[0]:+.4f}, {brick.center_m[1]:+.4f}, {geo[2]:+.4f}), "
                  f"fingertips ({placed[0]:+.4f}, {placed[1]:+.4f}, {placed[2]:+.4f})")
            print(f"      off by ({error[0] * 1000:+.1f}, {error[1] * 1000:+.1f}) mm sideways, "
                  f"{np.linalg.norm(error[:2]) * 1000:.1f} mm total, and {error[2] * 1000:+.1f} mm in z; "
                  f"seen {np.degrees(np.arctan(geo[1])):.0f} deg off vertical")

        if not measured:
            raise click.ClickException("No usable measurements were taken.")

        print(f"\n{'=' * 88}\nmeasured error over {len(measured)} brick(s)\n{'=' * 88}")
        sideways = np.array([np.linalg.norm(s["error"][:2]) for s in measured])
        print(f"  sideways: mean {sideways.mean() * 1000:.1f} mm, worst {sideways.max() * 1000:.1f} mm")
        print(f"  in z:     mean {np.mean([s['error'][2] for s in measured]) * 1000:+.1f} mm")
        if len(measured) >= 2:
            decompose(measured)


if __name__ == "__main__":
    command()
