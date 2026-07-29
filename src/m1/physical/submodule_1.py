"""M1 submodule 1 (physical): triangulate a hand-clicked point from two views, then pregrasp above it.

Workflow:
  1. Move the arm to two predefined joint configurations (``POS1``, ``POS2``), grabbing an RGB frame
     at each.
  2. In a popup of each frame, click the same physical point and press Enter to confirm.
  3. Each (pixel, eye-in-hand camera pose) pair back-projects to a ray in the robot base frame; the
     camera pose is forward kinematics (``arm.get_tcp_pose()``) composed with the hand-eye
     calibration (camera-in-TCP). The two rays are triangulated to a single 3D point.
  4. The arm moves to a straight-down pregrasp pose ``PREGRASP_HEIGHT`` above that point.

Connection, camera, calibration loading and reachable-orientation handling are reused from
``submodule_0``; pose maths uses airo-mono's ``SE3Container`` and imaging uses airo-camera-toolkit.
"""

import os
import sys
from typing import List, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_robots.exceptions import RobotConfigurationException
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, NumpyIntImageType
from loguru import logger

# submodule_1 lives next to submodule_0; make sure it is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submodule_0 import (  # noqa: E402
    CAMERA_RESOLUTIONS,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    SUPPORTED_ROBOT_TYPES,
    connect_arm,
    find_reachable_hover_orientation,
    load_camera_pose_in_tcp,
    open_camera,
)

# Joint configurations (radians) for the two viewpoints. Fill these in. They must be far enough
# apart that the two lines of sight to the target have real parallax (see triangulate_rays()).
POS1: List[float] = [-0.08343679, -1.31992237,  0.26209098, -0.40548201, -1.20620281, -1.63604099]
POS2: List[float] = [0.81975543, -1.24165185,  0.23308164, -0.76548697, -1.72945053, -0.95741016]

PREGRASP_HEIGHT = 0.03  # metres above the triangulated point for the pregrasp pose.
PARALLEL_RAY_EPS = 1e-6  # |1 - (da·db)^2| below this -> the two views are effectively collinear.
LARGE_TRIANGULATION_GAP = 0.02  # metres between the two rays' closest points above which we warn.

# Rough maximum TCP reach from the base (metres), for an out-of-workspace warning/diagnosis.
APPROX_ARM_REACH = {"ur3e": 0.50, "realman": 0.85}

_CONFIRM_KEYS = (13, 10, 32)  # Enter (either code) or Space.
_ABORT_KEYS = (27, ord("q"))  # Esc or q.


def click_pixel(image_rgb: NumpyIntImageType, window_title: str) -> Tuple[int, int]:
    """Show ``image_rgb`` and return the ``(u, v)`` pixel the user clicks and confirms.

    Left-click to place the marker (re-click to move it), Enter/Space to confirm, Esc/q to abort.

    Raises:
        RuntimeError: if the user aborts or closes the window before confirming a pixel.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    selection: dict = {}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["uv"] = (int(x), int(y))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, on_mouse)
    logger.info(f"[{window_title}] click the target point, then press Enter to confirm (Esc to abort).")
    try:
        while True:
            canvas = image_bgr.copy()
            if "uv" in selection:
                cv2.drawMarker(canvas, selection["uv"], (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in _CONFIRM_KEYS and "uv" in selection:
                return selection["uv"]
            if key in _ABORT_KEYS:
                raise RuntimeError("Pixel selection aborted by the user.")
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Pixel-selection window was closed before a pixel was confirmed.")
    finally:
        cv2.destroyWindow(window_title)
        cv2.waitKey(1)  # let the window actually close on all backends.


def pixel_to_base_ray(
    u: float, v: float, intrinsics_matrix: CameraIntrinsicsMatrixType, X_base_camera: HomogeneousMatrixType
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project pixel ``(u, v)`` to a ray ``(origin, unit direction)`` in the robot base frame.

    Pinhole model: the camera-frame ray is ``K^-1 [u, v, 1]`` (optical convention, +z forward, the
    same one the RealSense point cloud and submodule_0 use), rotated into the base frame by the
    eye-in-hand camera pose. The ray origin is the camera centre.
    """
    direction_camera = np.linalg.inv(intrinsics_matrix) @ np.array([u, v, 1.0])
    camera_in_base = SE3Container.from_homogeneous_matrix(X_base_camera)
    direction_base = camera_in_base.rotation_matrix @ direction_camera
    direction_base = direction_base / np.linalg.norm(direction_base)
    return camera_in_base.translation, direction_base


def triangulate_rays(
    ray_a: Tuple[np.ndarray, np.ndarray], ray_b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Midpoint triangulation of two rays, each an ``(origin, unit direction)`` tuple.

    Returns the midpoint of the mutually-closest points on the two (generally skew) rays, and the
    gap between those closest points as a quality metric.

    Raises:
        RuntimeError: if the rays are (near-)parallel, i.e. the viewpoints give too little parallax.
    """
    origin_a, direction_a = ray_a
    origin_b, direction_b = ray_b

    b = float(direction_a @ direction_b)
    denominator = 1.0 - b * b  # == (da·da)(db·db) - (da·db)^2, with unit directions.
    if abs(denominator) < PARALLEL_RAY_EPS:
        raise RuntimeError(
            "The two lines of sight are almost parallel; the viewpoints have too little parallax to "
            "triangulate. Choose two joint configurations that view the target from more different angles."
        )

    between = origin_a - origin_b
    d = float(direction_a @ between)
    e = float(direction_b @ between)
    s = (b * e - d) / denominator  # parameter along ray_a
    t = (e - b * d) / denominator  # parameter along ray_b

    closest_a = origin_a + s * direction_a
    closest_b = origin_b + t * direction_b
    point = 0.5 * (closest_a + closest_b)
    gap = float(np.linalg.norm(closest_a - closest_b))
    return point, gap


def capture_view_ray(
    arm: PositionManipulator,
    camera: RGBDCamera,
    X_tcp_camera: HomogeneousMatrixType,
    joint_configuration: np.ndarray,
    joint_speed: float,
    view_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Move to ``joint_configuration``, grab a frame, and return the clicked base-frame ray.

    The arm is stationary once the move completes, so the frame and the TCP pose it is paired with
    are taken together, giving a consistent eye-in-hand camera pose for the back-projection.
    """
    logger.info(f"Moving to {view_name}: {np.round(joint_configuration, 3)} rad ...")
    arm.move_to_joint_configuration(joint_configuration, joint_speed=joint_speed).wait()

    camera.grab_images()
    image = camera.retrieve_rgb_image_as_int()
    X_base_camera = arm.get_tcp_pose() @ X_tcp_camera  # eye-in-hand: FK composed with hand-eye calibration.

    u, v = click_pixel(image, f"{view_name} - click the target point")
    logger.info(f"{view_name}: clicked pixel (u={u}, v={v}).")
    return pixel_to_base_ray(u, v, camera.intrinsics_matrix(), X_base_camera)


def out_of_reach_message(robot_type: str, position: np.ndarray, exception: Exception) -> str:
    """Human-readable explanation for an unreachable target, with its distance from the base."""
    horizontal = float(np.hypot(position[0], position[1]))
    straight = float(np.linalg.norm(position[:3]))
    limit = APPROX_ARM_REACH.get(robot_type, 0.5)
    return (
        f"The pregrasp pose is not reachable (the controller found no IK solution). The target is "
        f"{horizontal * 100:.0f} cm horizontally / {straight * 100:.0f} cm straight-line from the robot "
        f"base, and a {robot_type} reaches only ~{limit * 100:.0f} cm. Move the target closer to the base "
        f"(aim for < ~{0.8 * limit * 100:.0f} cm horizontally for a comfortable top-down grasp), then "
        f"re-run. [{type(exception).__name__}]"
    )


def ensure_control_ready(arm: PositionManipulator) -> None:
    """Re-arm the UR control script if it has stopped (e.g. during a long pixel-selection pause).

    ur-rtde's reachability check (``isPoseWithinSafetyLimits``, used by ``move_to_tcp_pose``) and the
    moves themselves need the control script running. If it has stopped, ur-rtde prints "RTDE control
    script is not running!" and then reports *every* pose as unreachable — which looks exactly like a
    kinematic/safety rejection. This restarts the script so the reachability decision is trustworthy.
    RealMan and other drivers don't expose ``rtde_control`` and are left untouched.
    """
    rtde_control = getattr(arm, "rtde_control", None)
    if rtde_control is None:
        return
    try:
        if not rtde_control.isProgramRunning():
            logger.warning("The UR control script had stopped; reuploading it before moving.")
            rtde_control.reuploadScript()
    except Exception as exception:  # noqa: BLE001 - never let a recovery attempt crash the run
        logger.warning(f"Could not verify/restart the UR control script: {exception}")


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm to control.",
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
    help="1..100, fraction of the arm's max joint speed.",
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
    "--pregrasp-height",
    type=click.FloatRange(min=0.0, min_open=True),
    default=PREGRASP_HEIGHT,
    show_default=True,
    help="Metres above the triangulated point for the pregrasp pose.",
)
def main(
    robot_type: str,
    ip_address: str,
    port: int,
    speed_ratio: int,
    calibration_path: str,
    camera_resolution: str,
    pregrasp_height: float,
) -> None:
    """Triangulate a hand-clicked 3D point from two views and move to a pregrasp above it."""
    view_configurations = [np.asarray(POS1, dtype=float), np.asarray(POS2, dtype=float)]
    if any(configuration.size == 0 for configuration in view_configurations):
        raise click.ClickException(
            "Fill in POS1 and POS2 (the two viewpoint joint configurations, in radians) at the top of "
            "this file first."
        )

    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]

    X_tcp_camera = load_camera_pose_in_tcp(calibration_path)

    with connect_arm(robot_type, ip_address, port) as arm, open_camera(
        CAMERA_RESOLUTIONS[camera_resolution]
    ) as camera:
        dof = arm.manipulator_specs.dof
        for name, configuration in zip(("POS1", "POS2"), view_configurations):
            if configuration.size != dof:
                raise click.ClickException(
                    f"{name} has {configuration.size} joint value(s) but the {robot_type} arm has {dof} joints."
                )

        joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)

        ray_1 = capture_view_ray(arm, camera, X_tcp_camera, view_configurations[0], joint_speed, "view 1")
        ray_2 = capture_view_ray(arm, camera, X_tcp_camera, view_configurations[1], joint_speed, "view 2")

        point, gap = triangulate_rays(ray_1, ray_2)
        logger.info(f"Triangulated point: {point.round(4)} m (base frame); rays miss by {gap * 1000:.1f} mm.")
        if gap > LARGE_TRIANGULATION_GAP:
            logger.warning(
                f"The two rays miss each other by {gap * 1000:.1f} mm (> {LARGE_TRIANGULATION_GAP * 1000:.0f} mm); "
                "the two clicks may not be on the same point, or the hand-eye calibration is off. "
                "Continuing with their midpoint."
            )

        pregrasp_position = point + np.array([0.0, 0.0, pregrasp_height])
        # The long pixel-selection pauses can leave the UR control script stopped, which makes the
        # reachability checks below (getInverseKinematics / isPoseWithinSafetyLimits) unreliable and
        # every pose look unreachable; re-arm it first.
        ensure_control_ready(arm)
        # No explicit IK seed: ur-rtde already seeds from the arm's current joint configuration when
        # none is given, and passing a numpy seed trips a bug in the ur-rtde wrapper
        # (`joint_configuration_guess or np.array([])` -> "truth value of an array is ambiguous").
        try:
            X_base_pregrasp = find_reachable_hover_orientation(arm, pregrasp_position)
        except RuntimeError as exception:
            logger.error(out_of_reach_message(robot_type, pregrasp_position, exception))
            os._exit(1)

        horizontal_reach = float(np.hypot(X_base_pregrasp[0, 3], X_base_pregrasp[1, 3]))
        logger.info(
            f"Pregrasp target: {X_base_pregrasp[:3, 3].round(4)} m (base frame), {pregrasp_height * 100:.0f} cm "
            f"above the clicked point; {horizontal_reach * 100:.0f} cm from the base horizontally."
        )
        reach_limit = APPROX_ARM_REACH.get(robot_type)
        if reach_limit is not None and horizontal_reach > 0.9 * reach_limit:
            logger.warning(
                f"That is near the {robot_type} reach limit (~{reach_limit * 100:.0f} cm); the move may fail as "
                "unreachable. If it does, move the target closer to the base and re-run."
            )

        input(f"about to move to the pregrasp at {X_base_pregrasp[:3, 3].round(3)} m (base frame), press enter to continue")
        try:
            arm.move_to_tcp_pose(X_base_pregrasp, joint_speed=joint_speed).wait()
        except RobotConfigurationException as exception:
            logger.error(out_of_reach_message(robot_type, X_base_pregrasp[:3, 3], exception))
            os._exit(1)
        logger.info(f"reached:\n{arm.get_tcp_pose()}")

    os._exit(0)


if __name__ == "__main__":
    main()
