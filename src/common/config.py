"""
Single source of truth for robot configuration
"""

from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import socket
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from loguru import logger

if TYPE_CHECKING:
    from airo_camera_toolkit.cameras.realsense.realsense import Realsense

# --- robot connection ------------------------------------------------------------------------------
SUPPORTED_ROBOT_TYPES: Tuple[str, ...] = ("ur3e", "realman")
DEFAULT_IP_ADDRESSES: Dict[str, str] = {"ur3e": "10.43.0.162", "realman": "192.168.1.18"}
DEFAULT_REALMAN_PORT = 8080
#: How far the arm reaches horizontally, to the *flange*, as the datasheet gives it: a UR3e is 500 mm,
#: a Realman RM65 is 850 mm.
#:
#: **Not the fingertips.** A 231 mm gripper looks like 231 mm of extra reach and is nothing of the kind
#: for the grasps this project makes: pointing straight down, the tool spends its whole length going
#: *down*, adding zero horizontally -- and worse, the wrist then has to sit 32 cm above the fingertips,
#: which eats the arm's vertical budget and makes a top-down pose reach *less* far the higher it is.
#: That is why freedriving to a brick proves nothing about whether it can be grasped: freedrive puts the
#: tool at whatever angle it likes, where the same 231 mm does buy horizontal reach.
#:
#: This was 0.66 -- the UR3e's 500 mm with the gripper added on -- which let the pile perception offer
#: bricks 59 cm out (0.9 x 0.66) that no straight-down pose can get near, and the failure then surfaced
#: as an IK error in the middle of a pick instead of a rejected region at perception time. The 0.9 factor
#: the two callers apply to this number is what turns the flange envelope into a usable top-down radius.
APPROX_ARM_REACH: Dict[str, float] = {"ur3e": 0.50, "realman": 0.85}

# --- camera ------------------------------------------------------------------------------------------
CAMERA_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "1080": (1920, 1080),
    "720": (1280, 720),
    "540": (960, 540),
    "480": (848, 480),
}
DEFAULT_CAMERA_RESOLUTION = "720"

# --- calibration ---------------------------------------------------------------------------------------
DEFAULT_CALIBRATION_DIR = "/home/joon/int2026/calibration_dir"

# --- Table height relative to base -----------------------------------------------------------------------------------------
TABLE_Z = -0.0044

TABLE_PLANE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run", "table_plane.json")
)
TABLE_PLANE_MAX_AGE = 7 * 24 * 3600.0

# --- the brick -----------------------------------------------------------------------------------------
FALLBACK_BRICK_HEIGHT = 0.0096
FALLBACK_BRICK_WIDTH = 0.0078
FALLBACK_BRICK_LENGTH = 0.0238

BRICK_HANDOFF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run", "brick_handoff.json")
)
BRICK_HANDOFF_MAX_AGE = 1800.0
FOOTPRINT_MATCH_TOLERANCE = 0.0025

# --- the pile ------------------------------------------------------------------------------------------
# Where submodule_3 leaves the brick it picked out of the pile for submodule_1 to go and stand over.
# Short-lived on purpose: it describes one arrangement of a pile that every pick disturbs, so a target
# older than a couple of minutes is describing a pile that no longer exists.
PILE_TARGET_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run", "pile_target.json")
)
PILE_TARGET_MAX_AGE = 180.0

PREGRASP_HEIGHT = 0.03

# --- top-down tool orientation ---------------------------------------------------------------------------
HOVER_ORIENTATION_EULER = np.array([0, np.pi, 0.0001])
HOVER_YAW_CANDIDATES = np.linspace(0, 2 * np.pi, 8, endpoint=False)


# =================================================================================================
# the table, as measured by touching it
# =================================================================================================


@dataclass(frozen=True)
class TablePlane:
    """The tabletop as ``z = a*x + b*y + c`` in the robot's base frame, from calibrate_table.py.

    A plane rather than a single height because the table is not exactly perpendicular to the robot's
    z axis. A tilt of only 1 degree is 7 mm across a 40 cm workspace -- comfortably more than the
    1.5 mm of fingertip clearance a plate leaves, so a single number that is right in the middle of
    the table is wrong at its edges.
    """

    a: float
    b: float
    c: float
    residual: float  # metres, worst |touched z - fitted z| over the samples
    n_samples: int
    measured_at: float  # unix time
    tcp_offset: Optional[Tuple[float, ...]] = None  # the arm's TCP offset when it was measured

    def z_at(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arctan(np.hypot(self.a, self.b))))

    def describe(self) -> str:
        age_hours = (time.time() - self.measured_at) / 3600.0
        return (
            f"table plane from {self.n_samples} touch point(s), {age_hours:.1f} h old: "
            f"z = {self.c:+.4f} m at the base axis, tilted {self.tilt_deg:.2f} deg, "
            f"worst residual {self.residual * 1000:.1f} mm"
        )


def load_table_plane(path: str = TABLE_PLANE_PATH, warn_if_old: bool = True) -> Optional[TablePlane]:
    """The measured table plane, or ``None`` if the table has never been touched off."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        plane = TablePlane(
            a=float(payload["a"]),
            b=float(payload["b"]),
            c=float(payload["c"]),
            residual=float(payload["residual"]),
            n_samples=int(payload["n_samples"]),
            measured_at=float(payload["measured_at"]),
            tcp_offset=tuple(payload["tcp_offset"]) if payload.get("tcp_offset") else None,
        )
    except (OSError, ValueError, KeyError) as exception:
        logger.warning(f"Could not read the table plane at {path}: {exception}")
        return None

    if warn_if_old and time.time() - plane.measured_at > TABLE_PLANE_MAX_AGE:
        logger.warning(
            f"The table plane at {path} is {(time.time() - plane.measured_at) / 86400:.1f} days old. If the table "
            "or the gripper has been touched since, re-run `python src/tools/calibrate_table.py`."
        )
    return plane


def table_z_at(x: float, y: float, plane: Optional[TablePlane] = None) -> float:
    """Height of the tabletop under ``(x, y)``, from the touch-off if there is one.

    Falls back to :data:`TABLE_Z` with a warning, because that constant came from the camera and the
    whole point of the touch-off is that the camera cannot measure this well enough to descend on.
    """
    if plane is None:
        plane = load_table_plane()
    if plane is None:
        logger.warning(
            f"No touched-off table plane at {TABLE_PLANE_PATH}; falling back to TABLE_Z={TABLE_Z:+.4f} m, which "
            "is a guess. Run `python src/tools/calibrate_table.py` to measure it."
        )
        return TABLE_Z
    return plane.z_at(x, y)


# =================================================================================================
# the arm
# =================================================================================================


def create_arm(robot_type: str, ip_address: str, port: int) -> PositionManipulator:
    if robot_type == "realman":
        from airo_robots.manipulators.hardware.realman import RealmanControl

        return RealmanControl(ip_address, port)
    if robot_type == "ur3e":
        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        arm = URrtde(ip_address)
        detected_model = arm.model.value.lower()
        if detected_model != "ur3e":
            logger.warning(
                f"--robot_type ur3e was requested, but the arm at {ip_address} reports model "
                f"'{arm.model.value}'; continuing with the connected robot."
            )
        return arm
    raise ValueError(f"Unsupported robot type {robot_type!r}; expected one of {SUPPORTED_ROBOT_TYPES}.")


def _disconnect_arm(arm: PositionManipulator) -> None:
    try:
        close = getattr(arm, "close", None)
        if callable(close):
            close()
            return
        for interface_name in ("rtde_control", "rtde_receive"):
            interface = getattr(arm, interface_name, None)
            if interface is not None and hasattr(interface, "disconnect"):
                interface.disconnect()
    except Exception as exception:  # noqa: BLE001 - teardown must never raise
        logger.warning(f"Ignoring error while disconnecting the arm: {exception}")


#: RTDE's port on a UR controller. Probed before the driver is built, never spoken to directly.
UR_RTDE_PORT = 30004
#: How long that probe waits. A controller on the same subnet answers a TCP connect in milliseconds, so
#: this is long enough to be sure and short enough that a mistyped address is a two-second mistake.
CONNECT_PROBE_TIMEOUT = 3.0


def _unreachable(ip_address: str, port: int, timeout: float = CONNECT_PROBE_TIMEOUT) -> Optional[str]:
    """Why ``ip_address:port`` cannot be opened, or ``None`` if it can.

    This exists because ur_rtde's ``RTDEControlInterface`` constructor **retries forever** rather than
    raising: point it at an address with nothing behind it and the whole program stops on the "Connecting
    to ..." line with no error, no traceback and no timeout. The careful message below is unreachable
    without a check like this one, and an infinite hang is the least debuggable failure there is.
    """
    try:
        with socket.create_connection((ip_address, port), timeout=timeout):
            return None
    except OSError as exception:
        return str(exception) or type(exception).__name__


@contextlib.contextmanager
def connect_arm(robot_type: str, ip_address: str, port: int) -> Iterator[PositionManipulator]:
    logger.info(f"Connecting to {robot_type} arm at {ip_address}...")
    probe_port = port if robot_type == "realman" else UR_RTDE_PORT
    failure = _unreachable(ip_address, probe_port)
    if failure is not None:
        raise RuntimeError(
            f"Nothing answered at {ip_address}:{probe_port} within {CONNECT_PROBE_TIMEOUT:.0f} s ({failure}), so "
            f"the {robot_type} arm is not reachable from this machine. In order of how often it is each one:\n"
            f"  * this host has no address on the robot's subnet. `ip -brief addr` should show one; if the\n"
            f"    interface the robot is cabled to has no IPv4, that is the fault. A NetworkManager profile\n"
            f"    bound to an interface name that no longer exists (NIC names shift when hardware moves) is\n"
            f"    the usual reason -- `nmcli con show <profile> | grep interface-name` against\n"
            f"    `ip -brief addr` will show it.\n"
            f"  * the controller is powered off, or still booting.\n"
            f"  * the cable is out, or in the wrong port.\n"
            f"  * the address is wrong: this one came from --ip-address or config.DEFAULT_IP_ADDRESSES.\n"
            f"Check with `ping {ip_address}` before running this again."
        )
    try:
        arm = create_arm(robot_type, ip_address, port)
    except Exception as exception:
        endpoint = f"{ip_address}:{port}" if robot_type == "realman" else ip_address
        raise RuntimeError(
            f"Could not connect to the {robot_type} arm at {endpoint}. Check the IP/port, that the "
            f"robot is powered on and in remote control, and the network connection. "
            f"Original error: {exception}"
        ) from exception
    logger.info(f"Connected to {robot_type} arm.")
    try:
        yield arm
    finally:
        _disconnect_arm(arm)


def ensure_control_ready(arm: PositionManipulator) -> None:
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None:
        return
    try:
        if not rtde_control.isProgramRunning():
            logger.warning("The UR control script had stopped; reuploading it before moving.")
            rtde_control.reuploadScript()
    except Exception as exception:  # noqa: BLE001 - never let a recovery attempt crash the run
        logger.warning(f"Could not verify/restart the UR control script: {exception}")


def find_reachable_hover_orientation(
    arm: PositionManipulator,
    position: np.ndarray,
    joint_configuration_near: Optional[np.ndarray] = None,
) -> HomogeneousMatrixType:
    for yaw in HOVER_YAW_CANDIDATES:
        orientation_euler = np.array([0, np.pi, yaw])
        pose = SE3Container.from_euler_angles_and_translation(orientation_euler, position).homogeneous_matrix
        if arm.inverse_kinematics(pose, joint_configuration_near) is not None:
            return pose

    raise RuntimeError(
        f"No reachable straight-down orientation at {position.round(3)} m (base frame) across "
        f"{len(HOVER_YAW_CANDIDATES)} yaw angles; the position itself is likely out of reach."
    )


# =================================================================================================
# the camera
# =================================================================================================


def _realsense_diagnostics() -> str:
    try:
        import pyrealsense2 as rs

        descriptions = []
        for device in rs.context().query_devices():
            name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "unknown"
            usb = (
                device.get_info(rs.camera_info.usb_type_descriptor)
                if device.supports(rs.camera_info.usb_type_descriptor)
                else "?"
            )
            descriptions.append(f"{name} on a USB {usb} link")
        return "; ".join(descriptions) if descriptions else "no RealSense devices detected"
    except Exception as exception:  # noqa: BLE001 - diagnostics must never mask the real error
        return f"(could not enumerate RealSense devices: {exception})"


@contextlib.contextmanager
def open_camera(resolution: Tuple[int, int]) -> Iterator["Realsense"]:
    from airo_camera_toolkit.cameras.realsense.realsense import Realsense
    try:
        camera = Realsense(resolution=resolution, fps=30, enable_hole_filling=False)
    except RuntimeError as exception:
        raise RuntimeError(
            f"Could not start the RealSense at {resolution[0]}x{resolution[1]} (color) + depth: {exception}. "
            f"Detected {_realsense_diagnostics()}. A D415/D435 needs a USB-3 connection (a blue USB-3 port, "
            f"the camera's USB-3 cable, and no USB-2 hub in between) to stream color+depth; on a USB-2 link "
            f"librealsense fails with 'Couldn't resolve requests'. Switch to USB 3, or try a lower "
            f"--camera-resolution."
        ) from exception
    try:
        yield camera
    finally:
        camera.pipeline.stop()


#: Which hand-eye solver's answer to use, in order of preference; the first one with a pose file and a
#: finite residual wins. Pinned rather than picked by lowest residual: on a small sample the residuals
#: separate the methods by less than their spread (Daniilidis 0.012632 vs Tsai 0.012775 here, a
#: difference in the fourth decimal), so "lowest residual" is choosing on noise while the methods
#: themselves disagree by centimetres. ``None`` restores the old lowest-residual behaviour.
HAND_EYE_METHOD_PREFERENCE: Optional[Sequence[str]] = ("Tsai", "Horaud", "Daniilidis", "Andreff", "Park")


def select_hand_eye_method(residual_errors: Dict[str, float], available: Sequence[str]) -> str:
    """Pick the solver to trust: the first preferred one that actually solved, else lowest residual.

    A method is only a candidate if it produced a pose file *and* a finite residual -- an ``Infinity``
    means the solve failed outright, which is what a degenerate set of calibration poses does to Park.
    """
    usable = [m for m in available if math.isfinite(residual_errors.get(m, math.inf))]
    if not usable:
        raise RuntimeError(
            f"No hand-eye method in {sorted(available)} produced a finite residual, so none of them solved. "
            "The calibration poses are degenerate; re-run it with more, and more varied, orientations."
        )
    for method in HAND_EYE_METHOD_PREFERENCE or ():
        if method in usable:
            return method
    return min(usable, key=lambda method: residual_errors[method])


def load_camera_pose_in_tcp(calibration_dir: str) -> HomogeneousMatrixType:
    results_dirs = glob.glob(os.path.join(calibration_dir, "results_n=*"))
    if not results_dirs:
        raise RuntimeError(f"No results_n=* directory found in {calibration_dir}.")
    results_dir = max(results_dirs, key=lambda path: int(path.rsplit("=", 1)[-1]))

    with open(os.path.join(results_dir, "residual_errors.json")) as f:
        residual_errors = json.load(f)
    available = [
        os.path.basename(path)[len("camera_pose_") : -len(".json")]
        for path in glob.glob(os.path.join(results_dir, "camera_pose_*.json"))
    ]
    best_method = select_hand_eye_method(residual_errors, available)

    pose_path = os.path.join(results_dir, f"camera_pose_{best_method}.json")
    logger.info(f"Using {best_method} hand-eye calibration (residual {residual_errors[best_method]:.4f}): {pose_path}")
    spread = [residual_errors[m] for m in available if math.isfinite(residual_errors.get(m, math.inf))]
    if len(spread) > 1 and max(spread) - min(spread) < 1e-3:
        logger.warning(
            f"The {len(spread)} solvers' residuals span only {(max(spread) - min(spread)):.2e}, so they are "
            "indistinguishable on this data and the choice between them is arbitrary. That is a sign of too "
            "few calibration poses, not of agreement -- compare the pose files before trusting any of them."
        )
    with open(pose_path) as f:
        calibration = json.load(f)

    position = np.array([calibration["position_in_meters"][axis] for axis in ("x", "y", "z")])
    orientation_euler = np.array(
        [calibration["rotation_euler_xyz_in_radians"][angle] for angle in ("roll", "pitch", "yaw")]
    )
    return SE3Container.from_euler_angles_and_translation(orientation_euler, position).homogeneous_matrix
