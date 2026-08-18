"""Measure the tabletop by touching it with the arm, and fit a plane to what it touched.

Usage::

    # park the arm over the middle of the brick area first, tool pointing down, then:
    python src/tools/calibrate_table.py                     # probe a 5-point pattern around it
    python src/tools/calibrate_table.py --points 9 --half-width 0.08
    python src/tools/calibrate_table.py --max-probe-travel 0.25   # parked well above the table
    python src/tools/calibrate_table.py --freedrive         # hand-guide each touch instead of probing
    python src/tools/calibrate_table.py --from-current-pose # record where the tip is standing right now

"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import click
import numpy as np
from airo_robots.manipulators.position_manipulator import PositionManipulator
from loguru import logger

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.config import (  # noqa: E402
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    SUPPORTED_ROBOT_TYPES,
    TABLE_PLANE_PATH,
    TablePlane,
    connect_arm,
    ensure_control_ready,
    find_reachable_hover_orientation,
    load_table_plane,
)

# Downward speed of the probe. Slow enough that the arm stops within a fraction of a millimetre of
# first contact, and that a mistake pushes the table rather than damaging anything.
PROBE_SPEED = 0.005  # m/s
PROBE_ACCELERATION = 0.25  # m/s^2
# How far above the expected table each probe starts, and retreats to afterwards.
PROBE_CLEARANCE = 0.03  # metres
# Runaway guard only: how far a probe may descend before we conclude it is never going to find the
# table. Generous on purpose -- it is measured from the start of the descent, which is anchored to
# wherever the arm was parked, so a tight value here rejects good touches taken from a safe parking
# height rather than catching bad ones. The descent runs at PROBE_SPEED under contact detection, so a
# long travel costs seconds, not safety. What actually catches a bad touch is MAX_TOUCH_SPREAD.
MAX_PROBE_TRAVEL = 0.15  # metres
# Once one point is on the tabletop, the tabletop is known. A later touch further than this from the
# ones already taken landed on a brick, a fixture or over the table's edge -- which is the check the
# travel limit was standing in for, made against something that means anything.
MAX_TOUCH_SPREAD = 0.02  # metres

# Above this worst-case residual the touched points do not lie on one plane, so the fit is meaningless.
SUSPICIOUS_RESIDUAL = 0.003  # metres
# A tabletop tilted further than this relative to the robot's base is not what this assumes.
SUSPICIOUS_TILT_DEG = 3.0
# A TCP offset shorter than this means the tool centre point is at (or near) the flange rather than
# the fingertip, so "the TCP touched the table" would be measuring the wrong thing entirely.
MIN_PLAUSIBLE_TCP_OFFSET = 0.05  # metres


def probe_pattern(center: np.ndarray, n_points: int, half_width: float) -> List[np.ndarray]:
    """``n_points`` x, y positions spread around ``center``, for the plane fit to span.

    A cross/ring rather than a line: three collinear points fix a plane's height and one slope but
    leave the other slope completely undetermined, and the fit would silently report a confident
    answer for it.
    """
    if n_points <= 1:
        return [center[:2].copy()]

    points = [center[:2].copy()]
    ring = n_points - 1
    for index in range(ring):
        angle = 2 * np.pi * index / ring
        points.append(center[:2] + half_width * np.array([np.cos(angle), np.sin(angle)]))
    return points


def probe_point(
    arm: PositionManipulator,
    xy: np.ndarray,
    start_z: float,
    joint_speed: float,
    max_travel: float = MAX_PROBE_TRAVEL,
    max_spread: float = MAX_TOUCH_SPREAD,
    reference_z: Optional[float] = None,
) -> Optional[float]:
    """Touch the table at ``xy`` and return the TCP z where contact happened.

    The arm is taken to ``start_z`` above the point first (a normal move), then descends under
    ``moveUntilContact``, which watches the UR's own force estimate and retracts to the contact point
    when it fires.

    ``None`` means the touch was not usable: no contact inside ``max_travel``, or -- once
    ``reference_z`` is known from an earlier touch -- a contact more than ``max_spread`` away from the
    tabletop the other points found. The arm is always returned to its hover pose first, so a rejected
    probe never leaves the tip resting on the table for the next move to drag it sideways.
    """
    rtde_control = getattr(arm, "rtde_control", None)
    rtde_receive = getattr(arm, "rtde_receive", None)
    if rtde_control is None or rtde_receive is None:
        raise click.ClickException(
            "Touch-off probing needs the ur-rtde driver's contact detection; this arm does not expose it. "
            "Use --freedrive to guide each touch by hand instead."
        )

    above = np.array([xy[0], xy[1], start_z])
    try:
        pose = find_reachable_hover_orientation(arm, above)
    except RuntimeError as exception:
        logger.warning(f"Cannot reach {above.round(3)} m to probe from: {exception}")
        return None
    arm.move_to_tcp_pose(pose, joint_speed=joint_speed).wait()

    logger.info(f"Probing at x={xy[0]:+.4f} y={xy[1]:+.4f} from z={start_z:+.4f} m ...")
    # Contact is only looked for downward, so the sideways nudge of getting into position, or a cable
    # brushing the arm, cannot be mistaken for the table.
    contacted = rtde_control.moveUntilContact(
        [0.0, 0.0, -PROBE_SPEED, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], PROBE_ACCELERATION
    )
    touched_z = float(rtde_receive.getActualTCPPose()[2])
    travel = start_z - touched_z

    def retreat() -> None:
        arm.move_to_tcp_pose(pose, joint_speed=joint_speed).wait()

    if not contacted:
        logger.warning(
            f"No contact detected at {xy.round(3)} within {travel * 1000:.0f} mm; the probe stopped at "
            f"z={touched_z:+.4f} m. Either there is no table under this point, or the descent ran out of "
            f"travel before reaching it -- raise --max-probe-travel if the arm started well above the table."
        )
        retreat()
        return None
    if travel > max_travel:
        logger.warning(
            f"The probe at {xy.round(3)} descended {travel * 1000:.0f} mm before contact, past the "
            f"{max_travel * 1000:.0f} mm runaway limit, so it is being discarded. If the arm was simply parked "
            f"high above the table then this touch was fine: re-run with --max-probe-travel {travel * 1.5:.2f} "
            "(or park the tip closer to the tabletop first)."
        )
        retreat()
        return None
    if reference_z is not None and abs(touched_z - reference_z) > max_spread:
        logger.warning(
            f"The probe at {xy.round(3)} touched z={touched_z:+.4f} m, {abs(touched_z - reference_z) * 1000:.0f} mm "
            f"from the z={reference_z:+.4f} m the other points found (limit {max_spread * 1000:.0f} mm). That is a "
            "brick, a fixture or the edge of the table rather than the tabletop; discarding it."
        )
        retreat()
        return None

    logger.success(f"Touched the table at z={touched_z:+.4f} m after {travel * 1000:.0f} mm of descent.")
    retreat()
    return touched_z


def freedrive_point(arm: PositionManipulator, index: int) -> Optional[Tuple[np.ndarray, float]]:
    """Let the user hand-guide the tip onto the table, then record where it is.

    The fallback for when force probing is not wanted -- and the way the first measurement of this
    table was actually taken.
    """
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is not None:
        rtde_control.teachMode()
    try:
        answer = click.prompt(
            f"  touch point {index + 1}: put the fingertip on the table, then press Enter (or 'q' to finish)",
            default="",
            show_default=False,
        )
    finally:
        if rtde_control is not None:
            rtde_control.endTeachMode()
    if answer.strip().lower() in ("q", "quit", "done"):
        return None

    pose = arm.get_tcp_pose()
    tilt_deg = float(np.degrees(np.arccos(np.clip(-pose[2, 2], -1.0, 1.0))))
    if tilt_deg > 15.0:
        logger.warning(
            f"The tool is {tilt_deg:.0f} deg off vertical at this point. The TCP is still where it is, so the "
            "reading stands, but a tilted tool usually means a fingertip corner is touching rather than the tip."
        )
    logger.success(f"Recorded x={pose[0, 3]:+.4f} y={pose[1, 3]:+.4f} z={pose[2, 3]:+.4f} m.")
    return pose[:2, 3].copy(), float(pose[2, 3])


def fit_plane(points: Sequence[Tuple[np.ndarray, float]]) -> Tuple[float, float, float, float]:
    """Least-squares ``z = a*x + b*y + c`` through the touched points, plus the worst residual.

    With fewer than three points there is not enough to determine a tilt, so the plane is taken level
    at the mean height -- honest about what was measured rather than fitting noise.
    """
    xy = np.array([point for point, _ in points], dtype=float)
    z = np.array([height for _, height in points], dtype=float)

    if len(points) < 3:
        return 0.0, 0.0, float(z.mean()), float(np.max(np.abs(z - z.mean())) if len(z) > 1 else 0.0)

    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    (a, b, c), *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = float(np.max(np.abs(design @ np.array([a, b, c]) - z)))
    return float(a), float(b), float(c), residual


def check_tcp_offset(arm: PositionManipulator) -> Optional[Tuple[float, ...]]:
    """Report the arm's configured TCP offset, and complain if it cannot be the fingertip.

    This is the assumption the whole touch-off rests on: that ``get_tcp_pose()`` describes the point
    that touches the table. If Polyscope's TCP is still at the flange, every z measured here is the
    flange's, roughly a gripper-length above the tabletop, and nothing downstream would work -- so it
    is worth one sentence of output every run rather than a silent assumption.
    """
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None or not hasattr(rtde_control, "getTCPOffset"):
        return None
    try:
        offset = tuple(float(value) for value in rtde_control.getTCPOffset())
    except Exception as exception:  # noqa: BLE001 - a diagnostic must not stop the calibration
        logger.debug(f"Could not read the TCP offset: {exception}")
        return None

    distance = float(np.linalg.norm(offset[:3]))
    logger.info(f"The robot's configured TCP offset is {np.round(offset[:3], 4)} m ({distance * 1000:.0f} mm out).")
    if distance < MIN_PLAUSIBLE_TCP_OFFSET:
        logger.warning(
            f"That is only {distance * 1000:.0f} mm from the flange, so the TCP is not at the gripper's "
            "fingertip. Everything below measures wherever that point is instead of the tabletop, and the pick "
            "will be wrong by the difference. Set the tool centre point in Polyscope's installation first."
        )
    return offset


def save_plane(path: str, plane: TablePlane) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "a": plane.a,
                "b": plane.b,
                "c": plane.c,
                "residual": plane.residual,
                "n_samples": plane.n_samples,
                "measured_at": plane.measured_at,
                "tcp_offset": list(plane.tcp_offset) if plane.tcp_offset else None,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote the table plane to {path}.")


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm to touch off with.",
)
@click.option("--ip-address", default=None, help="Robot controller IP address. Defaults per robot type.")
@click.option("--port", default=DEFAULT_REALMAN_PORT, show_default=True, help="Controller port (RealMan only).")
@click.option(
    "--speed-ratio",
    type=click.IntRange(1, 100),
    default=10,
    show_default=True,
    help="1..100, fraction of the arm's max joint speed for the moves *between* probe points. The "
    "descent onto the table always goes at the slow probe speed.",
)
@click.option(
    "--points",
    type=click.IntRange(1, 25),
    default=5,
    show_default=True,
    help="How many points to probe. Three is the minimum for a tilt; more averages down the noise.",
)
@click.option(
    "--half-width",
    type=click.FloatRange(0.01, 0.30),
    default=0.06,
    show_default=True,
    help="Radius of the probe pattern around the arm's current position, in metres. Make it span the "
    "area bricks are actually picked from -- a plane fitted over 2 cm says little about 20 cm away.",
)
@click.option(
    "--max-probe-travel",
    type=click.FloatRange(0.01, 0.40),
    default=MAX_PROBE_TRAVEL,
    show_default=True,
    help="Runaway guard: how far a probe may descend before it is given up on, in metres. Measured "
    "from the start of the descent, so it has to cover however far above the table the arm was "
    "parked -- raise it rather than discard a touch that actually found the tabletop.",
)
@click.option(
    "--max-touch-spread",
    type=click.FloatRange(0.002, 0.20),
    default=MAX_TOUCH_SPREAD,
    show_default=True,
    help="How far a touch may sit from the ones already taken before it is treated as a brick, a "
    "fixture or the table's edge instead of the tabletop.",
)
@click.option(
    "--freedrive",
    is_flag=True,
    help="Hand-guide each touch instead of probing with contact detection. Slower, but it needs no "
    "force sensing and lets you put the tip exactly where you want it.",
)
@click.option(
    "--from-current-pose",
    is_flag=True,
    help="Record only where the tip is standing right now, as a single point, and write a level plane "
    "at that height. The quickest way to get a usable table height when the tip is already touching.",
)
@click.option(
    "--output",
    default=TABLE_PLANE_PATH,
    show_default=True,
    help="Where to write the fitted plane.",
)
@click.option("--dry-run", is_flag=True, help="Measure and report, but do not write the plane file.")
def main(
    robot_type: str,
    ip_address: Optional[str],
    port: int,
    speed_ratio: int,
    points: int,
    half_width: float,
    max_probe_travel: float,
    max_touch_spread: float,
    freedrive: bool,
    from_current_pose: bool,
    output: str,
    dry_run: bool,
) -> None:
    """Touch the table with the arm and record the plane the pick descends onto."""
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]

    previous = load_table_plane(output, warn_if_old=False)
    if previous is not None:
        logger.info(f"Replacing the existing {previous.describe()}.")

    with connect_arm(robot_type, ip_address, port) as arm:
        ensure_control_ready(arm)
        tcp_offset = check_tcp_offset(arm)
        joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)

        touched: List[Tuple[np.ndarray, float]] = []
        start_pose = arm.get_tcp_pose()

        if from_current_pose:
            logger.info("Recording the current TCP position as a single touch point.")
            touched.append((start_pose[:2, 3].copy(), float(start_pose[2, 3])))
        elif freedrive:
            logger.info(f"Freedrive touch-off: guide the fingertip onto the table {points} time(s).")
            for index in range(points):
                recorded = freedrive_point(arm, index)
                if recorded is None:
                    break
                touched.append(recorded)
        else:
            start_z = float(start_pose[2, 3]) + PROBE_CLEARANCE
            pattern = probe_pattern(start_pose[:3, 3], points, half_width)
            logger.info(
                f"Probing {len(pattern)} point(s) around x={start_pose[0, 3]:+.3f} y={start_pose[1, 3]:+.3f}, "
                f"descending from z={start_z:+.4f} m at {PROBE_SPEED * 1000:.0f} mm/s, giving up after "
                f"{max_probe_travel * 1000:.0f} mm."
            )
            for xy in pattern:
                # The median of what has been touched so far is the tabletop as currently known; the
                # first probe has nothing to compare against and is judged on travel alone.
                reference_z = float(np.median([z for _, z in touched])) if touched else None
                touched_z = probe_point(
                    arm,
                    xy,
                    start_z,
                    joint_speed,
                    max_travel=max_probe_travel,
                    max_spread=max_touch_spread,
                    reference_z=reference_z,
                )
                if touched_z is not None:
                    touched.append((xy, touched_z))

    if not touched:
        raise click.ClickException(
            "No touch points were recorded, so the table height is unknown. If the probes did find contact "
            f"but were discarded for descending too far, they were fine and the {max_probe_travel * 1000:.0f} mm "
            "--max-probe-travel is simply shorter than the gap the arm was parked above the table: raise it. "
            "If no contact was found at all, park the tip closer to the tabletop, or use --freedrive."
        )

    a, b, c, residual = fit_plane(touched)
    plane = TablePlane(
        a=a, b=b, c=c, residual=residual, n_samples=len(touched), measured_at=time.time(), tcp_offset=tcp_offset
    )

    logger.info(f"Fitted {plane.describe()}.")
    for xy, height in touched:
        logger.info(
            f"  touched ({xy[0]:+.4f}, {xy[1]:+.4f}) at z={height:+.4f} m "
            f"(fit says {plane.z_at(xy[0], xy[1]):+.4f}, off by {(height - plane.z_at(*xy)) * 1000:+.2f} mm)"
        )

    if len(touched) < 3:
        logger.warning(
            f"Only {len(touched)} point(s), so the plane is assumed level. That is fine if the table is, but a "
            "tilt of even 1 degree is 7 mm across 40 cm. Probe 3 or more points to measure it."
        )
    if residual > SUSPICIOUS_RESIDUAL:
        logger.warning(
            f"The touched points miss a single plane by up to {residual * 1000:.1f} mm (limit "
            f"{SUSPICIOUS_RESIDUAL * 1000:.0f} mm). Either the table is not flat here, something was lying under "
            "the tip at one of the points, or the contact detection fired early on one of them."
        )
    if plane.tilt_deg > SUSPICIOUS_TILT_DEG:
        logger.warning(
            f"The fitted table is tilted {plane.tilt_deg:.1f} deg relative to the robot's base (limit "
            f"{SUSPICIOUS_TILT_DEG:.0f}). Check the robot is bolted down flat before trusting this."
        )

    if dry_run:
        logger.info("--dry-run: not writing the plane file.")
        return
    save_plane(output, plane)
    click.echo(
        f"\nTable plane measured from {len(touched)} touch point(s). submodule_1 and submodule_2 now read it "
        f"from {output}; no constant to edit.\n"
    )


if __name__ == "__main__":
    main()
