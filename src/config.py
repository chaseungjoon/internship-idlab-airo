from typing import Dict, Tuple

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
