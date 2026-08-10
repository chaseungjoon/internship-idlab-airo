import contextlib
import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

import click
import cv2
import numpy as np
from airo_robots.manipulators.position_manipulator import PositionManipulator
from loguru import logger

# capture.py lives at src/capture.py, next to config.py. submodule_0 carries the connection/
# error-handling logic this reuses and lives one level down, at src/m1/physical/submodule_0.py.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SRC_DIR, "m1", "physical"))
sys.path.insert(0, _SRC_DIR)
from config import (  # noqa: E402
    CAMERA_RESOLUTIONS,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    SUPPORTED_ROBOT_TYPES,
)
from submodule_0 import connect_arm, open_camera  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lego_pic"))
FILENAME_PREFIX = "lego_pile"

# cv2.imwrite params per --format. jpg matches the repo's existing lego_pile_1.jpeg and keeps a long
# session's worth of photos small; png is available for when a session needs to be lossless.
IMAGE_FORMATS: Dict[str, Tuple[str, list]] = {
    "jpg": (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    "png": (".png", [cv2.IMWRITE_PNG_COMPRESSION, 3]),
}
DEPTH_SUFFIX = "_depth"  # companion file for --save-depth: <same stem>_depth.png, 16-bit mm, lossless

# Seconds to let the arm and the camera's auto-exposure settle after freedrive stops, before the
# frame that gets saved is grabbed. See the module docstring for why.
SETTLE_TIME = 0.4

WINDOW_TITLE = "Lego pile capture -- SPACE: capture, Q: quit"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_FREEDRIVE = (255, 200, 0)  # BGR, cyan-ish
COLOR_CAPTURING = (0, 255, 255)  # yellow
COLOR_SAVED = (0, 220, 0)  # green
MESSAGE_DURATION = 1.5  # seconds the "SAVED ..." status stays up before reverting to the freedrive one

_QUIT_KEYS = (ord("q"), 27)  # q or Esc


def ensure_control_ready(arm: PositionManipulator) -> None:
    """Re-arm the UR control script if it has stopped, e.g. after a protective stop mid-freedrive.

    A long hands-on session is exactly when a protective stop (bumping the arm into something while
    dragging it) is likely, and once the control script has stopped, ``teachMode()``/``endTeachMode()``
    silently do nothing -- the next SPACE would look like it worked but leave the arm stuck. Only
    ``URrtde`` exposes ``rtde_control``; ``RealmanControl`` doesn't need this and is left untouched.
    """
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None:
        return
    try:
        if not rtde_control.isProgramRunning():
            logger.warning("The UR control script had stopped (protective stop?); reuploading it.")
            rtde_control.reuploadScript()
    except Exception as exception:  # noqa: BLE001 - never let a recovery attempt crash the run
        logger.warning(f"Could not verify/restart the UR control script: {exception}")


def timestamped_stem(prefix: str = FILENAME_PREFIX) -> str:
    """``lego_pile_<YYYYMMDD>_<HHMMSS>``, i.e. the requested ``lego_pile_[date]_[time]`` pattern."""
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"


def unique_path(directory: str, filename: str) -> str:
    """``directory/filename``, or the same with a ``_2``, ``_3``, ... suffix if that already exists.

    Two captures inside the same second would otherwise get the same stem and the second would
    silently overwrite the first -- the worst possible failure mode when the whole point of the
    session is not to lose photos.
    """
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    suffix = 1
    while os.path.exists(candidate):
        suffix += 1
        candidate = os.path.join(directory, f"{stem}_{suffix}{ext}")
    return candidate


def save_rgb(image_rgb: np.ndarray, output_dir: str, image_format: str) -> str:
    """Save one RGB frame as ``lego_pile_<date>_<time>.<ext>`` and return the path written."""
    ext, imwrite_params = IMAGE_FORMATS[image_format]
    path = unique_path(output_dir, timestamped_stem() + ext)
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), imwrite_params)
    return path


def save_depth_map(depth_m: np.ndarray, output_dir: str, rgb_path: str) -> str:
    """Save a depth map as a 16-bit PNG in millimetres, named after the RGB photo it goes with.

    16-bit PNG in millimetres is the common RGBD-dataset convention (lossless, and millimetre
    resolution comfortably covers a RealSense's actual precision). Sharing the RGB photo's stem
    keeps the pair obviously matched on disk without needing a separate index.
    """
    depth_mm = np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65535).astype(np.uint16)
    stem, _ = os.path.splitext(os.path.basename(rgb_path))
    path = os.path.join(output_dir, f"{stem}{DEPTH_SUFFIX}.png")
    cv2.imwrite(path, depth_mm)
    return path


def draw_hud(
    image_rgb: np.ndarray, output_dir: str, capture_count: int, status: str, status_color: Tuple[int, int, int]
) -> np.ndarray:
    """Overlay the status line, capture count and key legend onto a BGR copy of ``image_rgb``.

    Drawn on a copy for the preview window only -- the frame that gets saved is grabbed fresh and
    never passes through this function, so nothing here ever ends up burned into the pile photos.
    """
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 64), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 36), (width, height), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    cv2.putText(canvas, status, (12, 28), FONT, 0.7, status_color, 2)
    cv2.putText(canvas, f"Captured: {capture_count}  ->  {output_dir}", (12, 52), FONT, 0.5, (210, 210, 210), 1)
    cv2.putText(canvas, "[SPACE] capture     [Q] quit", (12, height - 12), FONT, 0.55, (255, 255, 255), 1)
    return canvas


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm to freedrive.",
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
    "--camera-resolution",
    type=click.Choice(list(CAMERA_RESOLUTIONS)),
    default=DEFAULT_CAMERA_RESOLUTION,
    show_default=True,
    help="RealSense colour resolution (height). A D415/D435 needs USB 3 to stream colour+depth.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory photos are saved to.",
)
@click.option(
    "--format",
    "image_format",
    type=click.Choice(sorted(IMAGE_FORMATS)),
    default="jpg",
    show_default=True,
    help="Image format for the saved photo.",
)
@click.option(
    "--save-depth",
    is_flag=True,
    help="Also save each frame's depth map as a 16-bit PNG (millimetres), named "
    f"<same stem>{DEPTH_SUFFIX}.png next to the photo.",
)
def main(
    robot_type: str,
    ip_address: str,
    port: int,
    camera_resolution: str,
    output_dir: str,
    image_format: str,
    save_depth: bool,
) -> None:
    """Freedrive the arm, press SPACE to photograph the pile, Q to stop."""
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]
    os.makedirs(output_dir, exist_ok=True)

    with connect_arm(robot_type, ip_address, port) as arm, open_camera(CAMERA_RESOLUTIONS[camera_resolution]) as camera:
        ensure_control_ready(arm)
        ext = IMAGE_FORMATS[image_format][0].lstrip(".")
        logger.info(f"Saving photos to {output_dir} as {FILENAME_PREFIX}_<date>_<time>.{ext}")
        logger.info("Freedrive is on -- move the arm into position, SPACE to capture, Q (or Esc) to quit.")
        arm.start_freedrive()
        in_freedrive = True
        capture_count = 0
        status, status_color, status_until = "FREEDRIVE -- move the arm, SPACE to capture", COLOR_FREEDRIVE, float("inf")

        try:
            while True:
                camera.grab_images()
                image_rgb = camera.retrieve_rgb_image_as_int()

                if time.monotonic() > status_until:
                    status, status_color = "FREEDRIVE -- move the arm, SPACE to capture", COLOR_FREEDRIVE
                cv2.imshow(WINDOW_TITLE, draw_hud(image_rgb, output_dir, capture_count, status, status_color))
                key = cv2.waitKey(30) & 0xFF

                if key == ord(" "):
                    # Immediate feedback before the (briefly) blocking capture sequence below, so the
                    # keypress doesn't feel like it was dropped.
                    cv2.imshow(WINDOW_TITLE, draw_hud(image_rgb, output_dir, capture_count, "CAPTURING...", COLOR_CAPTURING))
                    cv2.waitKey(1)

                    ensure_control_ready(arm)
                    arm.stop_freedrive()
                    in_freedrive = False
                    time.sleep(SETTLE_TIME)

                    camera.grab_images()
                    rgb_path = save_rgb(camera.retrieve_rgb_image_as_int(), output_dir, image_format)
                    capture_count += 1
                    logger.success(f"[{capture_count}] Saved {rgb_path}")
                    if save_depth:
                        depth_path = save_depth_map(camera.retrieve_depth_map(), output_dir, rgb_path)
                        logger.info(f"    + depth: {depth_path}")

                    ensure_control_ready(arm)
                    arm.start_freedrive()
                    in_freedrive = True
                    status, status_color = f"SAVED {os.path.basename(rgb_path)}", COLOR_SAVED
                    status_until = time.monotonic() + MESSAGE_DURATION

                elif key in _QUIT_KEYS or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Quit requested.")
                    break
        except KeyboardInterrupt:
            logger.warning("Interrupted.")
        finally:
            if in_freedrive:
                with contextlib.suppress(Exception):
                    arm.stop_freedrive()
            with contextlib.suppress(Exception):
                cv2.destroyWindow(WINDOW_TITLE)
                cv2.waitKey(1)

        logger.success(f"Session ended: {capture_count} picture(s) saved to {output_dir}.")

    os._exit(0)


if __name__ == "__main__":
    main()
