"""
Single source of truth for robot configuration
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple

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
APPROX_ARM_REACH: Dict[str, float] = {"ur3e": 0.66, "realman": 0.85}

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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "table_plane.json")
)
TABLE_PLANE_MAX_AGE = 7 * 24 * 3600.0

# --- the brick -----------------------------------------------------------------------------------------
FALLBACK_BRICK_HEIGHT = 0.0096
FALLBACK_BRICK_WIDTH = 0.0078
FALLBACK_BRICK_LENGTH = 0.0238

BRICK_HANDOFF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "brick_handoff.json")
)
BRICK_HANDOFF_MAX_AGE = 1800.0
FOOTPRINT_MATCH_TOLERANCE = 0.0025

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
            "or the gripper has been touched since, re-run `python src/calibrate_table.py`."
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
            "is a guess. Run `python src/calibrate_table.py` to measure it."
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


@contextlib.contextmanager
def connect_arm(robot_type: str, ip_address: str, port: int) -> Iterator[PositionManipulator]:
    logger.info(f"Connecting to {robot_type} arm at {ip_address}...")
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


def load_camera_pose_in_tcp(calibration_dir: str) -> HomogeneousMatrixType:
    results_dirs = glob.glob(os.path.join(calibration_dir, "results_n=*"))
    if not results_dirs:
        raise RuntimeError(f"No results_n=* directory found in {calibration_dir}.")
    results_dir = max(results_dirs, key=lambda path: int(path.rsplit("=", 1)[-1]))

    with open(os.path.join(results_dir, "residual_errors.json")) as f:
        residual_errors = json.load(f)
    best_method = min(residual_errors, key=lambda method: residual_errors[method])

    pose_path = os.path.join(results_dir, f"camera_pose_{best_method}.json")
    logger.info(f"Using {best_method} hand-eye calibration (residual {residual_errors[best_method]:.4f}): {pose_path}")
    with open(pose_path) as f:
        calibration = json.load(f)

    position = np.array([calibration["position_in_meters"][axis] for axis in ("x", "y", "z")])
    orientation_euler = np.array(
        [calibration["rotation_euler_xyz_in_radians"][angle] for angle in ("roll", "pitch", "yaw")]
    )
    return SE3Container.from_euler_angles_and_translation(orientation_euler, position).homogeneous_matrix
