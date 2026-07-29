import contextlib
import glob
import json
import os
import time
from typing import Iterator, Optional, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, NumpyIntImageType
from loguru import logger

# The two supported arms share the airo-robots PositionManipulator interface, so everything below
# the connection layer is robot-agnostic. Their drivers are imported lazily in create_arm() so a
# machine set up for only one arm needn't have the other's (optional) driver installed.
SUPPORTED_ROBOT_TYPES = ("ur3e", "realman")
DEFAULT_IP_ADDRESSES = {"ur3e": "10.43.0.162", "realman": "192.168.1.18"}
DEFAULT_REALMAN_PORT = 8080

# Named RealSense colour resolutions (the wrapper picks the depth resolution itself). Lower is
# lighter on USB bandwidth; a D415/D435 still needs a USB-3 link to stream colour+depth at all.
CAMERA_RESOLUTIONS = {
    "1080": Realsense.RESOLUTION_1080,
    "720": Realsense.RESOLUTION_720,
    "540": Realsense.RESOLUTION_540,
    "480": Realsense.RESOLUTION_480,
}
DEFAULT_CAMERA_RESOLUTION = "720"

DEFAULT_CALIBRATION_DIR = "/home/joon/int2026/calibration_dir"

HOVER_ORIENTATION_EULER = np.array([0, np.pi, 0.0001]) 
HOVER_YAW_CANDIDATES = np.linspace(0, 2 * np.pi, 8, endpoint=False)

MIN_VALID_DEPTH = 0.10 
COLOR_ANOMALY_THRESHOLD = 40
MIN_BRICK_CONTOUR_AREA = 200
MAX_HEIGHT_ABOVE_TABLE = 0.05

SERVO_INTERVAL = 0.5  # seconds between re-detections while the arm is moving towards the brick.

REPLAN_POSITION_THRESHOLD = 0.01
MAX_VISUAL_SERVO_DURATION = 20.0
CONFIRMATION_SAMPLES = 3  # detections that must all agree before a re-plan is trusted.

# A RealSense can't measure depth closer than its Min-Z (~0.3-0.45 m for the D415 at these
# resolutions, less for a D435). This eye-in-hand camera looks straight down while approaching a
# hover pose only centimetres above the table, so once it descends past this distance the brick
# falls inside the sensor's blind zone and depth-based detection returns garbage. Below it we stop
# visually correcting and let the current move finish open-loop toward the last confident target
# (the brick's x, y is already known; only the descent remains). Tune to your camera's Min-Z.
MIN_RELIABLE_DEPTH_DISTANCE = 0.30



def create_arm(robot_type: str, ip_address: str, port: int) -> PositionManipulator:
    """Construct and connect the manipulator driver for ``robot_type``.

    Args:
        robot_type: one of :data:`SUPPORTED_ROBOT_TYPES` (``"ur3e"`` or ``"realman"``).
        ip_address: robot controller IP address.
        port: controller port. Used by the RealMan driver only; the ur-rtde driver reaches the
            UR3e on its fixed RTDE ports, so ``port`` is ignored for ``"ur3e"``.

    Raises:
        ValueError: if ``robot_type`` is not supported.
    """
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
    """Release the robot connection, whatever teardown the driver exposes.

    ``RealmanControl`` provides ``close()``; ``URrtde`` has no teardown of its own, so its
    underlying RTDE interfaces are disconnected directly. Cleanup errors are logged and
    swallowed so they can never mask an exception propagating out of the ``with`` block.
    """
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
    """Yield a connected arm and guarantee it is released afterwards.

    Presents both drivers behind one ``with`` block despite their differing lifecycles
    (``RealmanControl`` is itself a context manager, ``URrtde`` is not), and turns a failed
    connection into an actionable error rather than a raw driver traceback.
    """
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


def _realsense_diagnostics() -> str:
    """Best-effort one-line description of the connected RealSense(s) for error messages."""
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
def open_camera(resolution: Tuple[int, int]) -> Iterator[Realsense]:
    """Open the RealSense at ``resolution``, turning a failed start into an actionable error.

    librealsense reports an unsatisfiable stream request as the opaque "Couldn't resolve
    requests"; the usual cause is a D415/D435 on a USB-2 link, which can't stream the
    colour+depth the pipeline needs. This adds the detected device/USB type and a concrete fix
    to that message, and guarantees the pipeline is stopped on exit.
    """
    # Hole-filling is disabled: on a poorly-reflecting table (poor IR depth return), it was
    # smearing unrelated far-away depth across the holes instead of leaving them invalid, which is
    # what fill_invalid_depth_with_table_estimate needs to do instead (fill with the *table's* depth).
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
    """Load the eye-in-hand camera pose (in the TCP frame) from a hand-eye-calibration output dir.

    Picks the ``results_n=*`` subdirectory with the most samples, then the
    ``camera_pose_<method>.json`` with the lowest residual error in that directory
    (mirrors the selection logic in ``do_camera_robot_calibration``).

    Args:
        calibration_dir: the ``--calibration_dir`` passed to `airo-camera-toolkit
            hand-eye-calibration`.
    """
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
    position = np.array(list(calibration["position_in_meters"].values()))
    orientation_euler = np.array(list(calibration["rotation_euler_xyz_in_radians"].values()))
    return SE3Container.from_euler_angles_and_translation(orientation_euler, position).homogeneous_matrix


MIN_GENUINE_DEPTH_POINTS = 10  # below this, we can't judge a region's height; trust the colour detection instead


def fill_invalid_depth_with_table_estimate(
    points_camera: np.ndarray,
    intrinsics_matrix: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill points with invalid/missing depth using a representative table depth.

    Black, absorptive, or small/thin surfaces (a black table, a small lego brick) often return
    no valid depth from the RealSense's IR-based sensing. Rather than discard those pixels, this
    assumes they lie approximately at the table's own depth (true for a thin brick resting on
    it) and backprojects them with that substitute depth using the pinhole model, so a
    colour-detected region stays usable for localization even where its own depth failed.

    Args:
        points_camera: point cloud in the camera frame, flattened in image raster order,
            shape (H*W, 3).
        intrinsics_matrix: 3x3 camera intrinsics matrix.
        image_shape: (height, width) of the image the point cloud is aligned to.

    Returns:
        A tuple of (filled point cloud, boolean mask of originally-valid points), both
        matching ``points_camera`` in shape/order.

    Raises:
        RuntimeError: if there's not enough valid depth anywhere in the frame to even estimate
            a table depth.
    """
    valid = points_camera[:, 2] > MIN_VALID_DEPTH
    if valid.sum() < MIN_GENUINE_DEPTH_POINTS:
        raise RuntimeError("Too few points with valid depth in the whole frame to estimate the table depth.")
    table_depth_camera = float(np.median(points_camera[valid, 2]))

    height, width = image_shape
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    row, col = np.mgrid[0:height, 0:width]
    depth = np.where(valid.reshape(height, width), points_camera[:, 2].reshape(height, width), table_depth_camera)

    x = (col - cx) * depth / fx
    y = (row - cy) * depth / fy
    points_filled = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    return points_filled, valid


def detect_brick_mask(
    image: NumpyIntImageType,
    points_camera: np.ndarray,
    genuinely_valid_depth: np.ndarray,
    X_base_camera: HomogeneousMatrixType,
) -> np.ndarray:
    """Segment the brick as whatever stands out from the table's own majority colour.

    MVP assumption: a single brick on an otherwise uniformly-coloured table, camera pointed
    straight at it. Rather than assuming the table is any specific (e.g. black) colour, this
    takes the frame's median colour as "the table" and flags anything far enough from it (see
    :data:`COLOR_ANOMALY_THRESHOLD`) as a brick candidate — adapts automatically to whatever the
    table's colour and the lighting actually are, instead of breaking when either isn't what was
    assumed.

    A candidate with too little genuinely-measured depth (see
    :func:`fill_invalid_depth_with_table_estimate`) is trusted as the brick outright — a real
    depth *reading* failure on a colour blob is itself evidence it's the small brick, not
    something larger like the arm/gripper. Candidates with plenty of real depth data are only
    rejected if they're clearly too *tall* for a standalone brick (above
    ``MAX_HEIGHT_ABOVE_TABLE``) — which is what filters out the arm's own wrist/gripper/camera
    mount when it's visible (and larger than the brick) in an eye-in-hand frame. There's no
    lower bound: the RealSense can't reliably resolve a lego brick's few-millimetre height, so
    any small or borderline depth reading near the table is still trusted as the brick rather
    than rejected as noise.

    Args:
        image: RGB image (H, W, 3), uint8.
        points_camera: point cloud in the camera frame (invalid depth already filled in, see
            :func:`fill_invalid_depth_with_table_estimate`), flattened to match ``image`` in
            raster order, shape (H*W, 3).
        genuinely_valid_depth: boolean mask (H*W,), True where the camera actually measured
            depth (before filling).
        X_base_camera: camera pose in the robot base frame.

    Returns:
        Boolean mask (H, W), True on the brick.

    Raises:
        RuntimeError: if no colour anomaly is found, or none of them have a brick-like height.
    """
    majority_color = np.median(image.reshape(-1, 3), axis=0)
    color_distance = np.linalg.norm(image.astype(np.float32) - majority_color, axis=2)
    anomaly = (color_distance > COLOR_ANOMALY_THRESHOLD).astype(np.uint8) * 255
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(anomaly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_BRICK_CONTOUR_AREA]
    if not contours:
        raise RuntimeError("Could not find any region that stands out from the table's colour; no brick detected.")
    contours.sort(key=cv2.contourArea, reverse=True)
    logger.debug(f"{len(contours)} colour-anomaly candidate(s) this frame, areas={[int(cv2.contourArea(c)) for c in contours]}")

    points_base = points_camera @ X_base_camera[:3, :3].T + X_base_camera[:3, 3]
    # Rough table height from the whole scene, only to tell candidates apart by height; the brick
    # itself is refined afterwards in compute_hover_pose using the region this function returns.
    rough_table_z = float(np.percentile(points_base[genuinely_valid_depth, 2], 10))

    for contour in contours:
        mask_image = np.zeros(anomaly.shape, dtype=np.uint8)
        cv2.drawContours(mask_image, [contour], -1, 255, thickness=cv2.FILLED)
        candidate_mask = mask_image.reshape(-1) > 0
        candidate_genuine = candidate_mask & genuinely_valid_depth

        if candidate_genuine.sum() < MIN_GENUINE_DEPTH_POINTS:
            logger.debug(
                f"Only {int(candidate_genuine.sum())} genuinely-measured depth points on this "
                "region; trusting the colour detection (likely the brick, too small/dark for "
                "reliable depth) rather than its height."
            )
            return mask_image > 0

        height_above_table = float(np.median(points_base[candidate_genuine, 2])) - rough_table_z
        if height_above_table <= MAX_HEIGHT_ABOVE_TABLE:
            return mask_image > 0

        logger.debug(
            f"Rejected a colour anomaly {height_above_table:.3f} m above the table "
            f"(> {MAX_HEIGHT_ABOVE_TABLE} m) — likely the arm/gripper in frame, not the brick."
        )

    raise RuntimeError(
        "Found colour anomaly region(s), but all of them are too tall to be a single standalone "
        f"brick (> {MAX_HEIGHT_ABOVE_TABLE} m above the table); check whether the arm or gripper "
        "is visible in the camera's field of view."
    )


def compute_hover_pose(
    points_camera: np.ndarray,
    brick_mask: np.ndarray,
    X_base_camera: HomogeneousMatrixType,
    hover_height: float,
) -> Tuple[HomogeneousMatrixType, np.ndarray]:
    """Compute a TCP pose hovering above the table plane under the detected brick.

    Args:
        points_camera: point cloud in the camera frame (invalid depth already filled in, see
            :func:`fill_invalid_depth_with_table_estimate`), flattened to match ``brick_mask``
            in raster order, shape (H*W, 3).
        brick_mask: boolean mask (H, W), True on the brick (see :func:`detect_brick_mask`).
        X_base_camera: camera pose in the robot base frame.
        hover_height: metres above the table plane to hover at.

    Returns:
        A tuple of the hover TCP pose (base frame) and the brick's (x, y, table z) position
        used to compute it, for logging/visualization.

    Raises:
        RuntimeError: if the brick mask leaves too few (or no) pixels to measure the table
            plane from — this must fail loudly rather than let a NaN/degenerate table height
            silently flow into the returned pose.
    """
    brick_mask_flat = brick_mask.reshape(-1)
    table_mask_flat = ~brick_mask_flat
    if brick_mask_flat.sum() < 1 or table_mask_flat.sum() < MIN_GENUINE_DEPTH_POINTS:
        raise RuntimeError(
            f"Brick mask covers {brick_mask_flat.sum()}/{brick_mask_flat.size} pixels, leaving too "
            "few to measure the table plane from; rejecting this detection instead of returning a "
            "degenerate (NaN) target."
        )

    points_base = points_camera @ X_base_camera[:3, :3].T + X_base_camera[:3, 3]

    brick_xy = points_base[brick_mask_flat, :2].mean(axis=0)
    table_z = float(np.median(points_base[table_mask_flat, 2]))

    hover_position = np.array([brick_xy[0], brick_xy[1], table_z + hover_height])
    X_base_hover = SE3Container.from_euler_angles_and_translation(
        HOVER_ORIENTATION_EULER, hover_position
    ).homogeneous_matrix
    brick_pose_for_logging = np.array([brick_xy[0], brick_xy[1], table_z])
    return X_base_hover, brick_pose_for_logging


def find_reachable_hover_orientation(
    arm: PositionManipulator,
    position: np.ndarray,
    joint_configuration_near: Optional[np.ndarray] = None,
) -> HomogeneousMatrixType:
    """Find a straight-down TCP pose at ``position`` that the arm can actually reach.

    Tries each of ``HOVER_YAW_CANDIDATES`` (rotations about the vertical tool axis) and returns
    the first one the arm's :meth:`~airo_robots.manipulators.position_manipulator.PositionManipulator.inverse_kinematics`
    reports as solvable. A fixed yaw can
    be kinematically unreachable at some (x, y) — hitting a joint limit or wrist singularity —
    even though the position itself, and a straight-down orientation in general, clearly aren't
    the problem; the yaw is a free choice here, so there's no reason to fail instead of trying
    another one.

    Args:
        arm: the robot, used only to query IK (no motion is commanded).
        position: target TCP position (base frame), metres.
        joint_configuration_near: IK seed; defaults to the arm's current configuration.

    Raises:
        RuntimeError: if none of the candidate yaws are reachable.
    """
    for yaw in HOVER_YAW_CANDIDATES:
        orientation_euler = np.array([0, np.pi, yaw])
        pose = SE3Container.from_euler_angles_and_translation(orientation_euler, position).homogeneous_matrix
        if arm.inverse_kinematics(pose, joint_configuration_near) is not None:
            return pose

    raise RuntimeError(
        f"No reachable straight-down orientation at {position.round(3)} m (base frame) across "
        f"{len(HOVER_YAW_CANDIDATES)} yaw angles; the position itself is likely out of reach."
    )


def save_debug_frame(debug_dir: str, image: NumpyIntImageType, brick_mask: np.ndarray) -> str:
    """Save the frame with the detected brick mask outlined, for after-the-fact inspection.

    There's no live display during a run, so this is the only way to see what
    :func:`detect_brick_mask` actually picked (or mis-picked) on the real camera feed.
    """
    os.makedirs(debug_dir, exist_ok=True)
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    contours, _ = cv2.findContours(brick_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    path = os.path.join(debug_dir, f"frame_{time.time():.3f}.png")
    cv2.imwrite(path, overlay)
    return path


def estimate_target_pose(
    arm: PositionManipulator,
    camera: RGBDCamera,
    X_tcp_camera: HomogeneousMatrixType,
    hover_height: float,
    debug_dir: Optional[str] = None,
) -> Tuple[HomogeneousMatrixType, np.ndarray]:
    """One-shot brick detection: grab a frame from wherever the arm currently is and estimate
    the hover pose above the brick.

    Since the camera is eye-in-hand, this re-derives the camera's pose in the base frame from
    the arm's *current* TCP pose every time it's called, so it stays correct while the arm is
    moving (see :func:`move_to_brick_with_visual_correction`), not just from a single pose
    taken before any motion started. The image is grabbed *before* reading the TCP pose so the
    two are as close in time as possible — while the arm is moving, reading the pose first would
    pair the frame with a TCP pose from slightly before/after it was actually taken.

    Args:
        debug_dir: if given, save this frame with the detected brick outlined (see
            :func:`save_debug_frame`) — for inspecting what's actually being detected.
    """
    camera.grab_images()
    X_base_tcp = arm.get_tcp_pose()
    X_base_camera = X_base_tcp @ X_tcp_camera

    image = camera.retrieve_rgb_image_as_int()
    point_cloud = camera.retrieve_colored_point_cloud()
    points_camera, genuinely_valid_depth = fill_invalid_depth_with_table_estimate(
        point_cloud.points, camera.intrinsics_matrix(), image.shape[:2]
    )

    brick_mask = detect_brick_mask(image, points_camera, genuinely_valid_depth, X_base_camera)
    if debug_dir is not None:
        path = save_debug_frame(debug_dir, image, brick_mask)
        logger.debug(f"Saved debug frame to {path}")
    X_base_hover, brick_pose = compute_hover_pose(points_camera, brick_mask, X_base_camera, hover_height)
    X_base_hover = find_reachable_hover_orientation(arm, X_base_hover[:3, 3])
    return X_base_hover, brick_pose


def move_to_brick_with_visual_correction(
    arm: PositionManipulator,
    camera: RGBDCamera,
    X_tcp_camera: HomogeneousMatrixType,
    joint_speed: float,
    hover_height: float,
    debug_dir: Optional[str] = None,
) -> Tuple[HomogeneousMatrixType, np.ndarray]:
    """Move to the hover pose above the brick, re-detecting and correcting the target while moving.

    A single detection made before moving bakes in whatever hand-eye calibration and RGBD
    detection error existed at that instant. Instead, this re-estimates the brick's pose every
    ``SERVO_INTERVAL`` seconds *while the arm is still moving there*, and re-issues the move if
    the updated estimate has drifted from the target currently being executed by more than
    ``REPLAN_POSITION_THRESHOLD`` — giving the arm several chances to correct onto the brick,
    rather than one blind shot.

    A drifted estimate is only trusted once it's reproduced across ``CONFIRMATION_SAMPLES``
    consecutive, mutually-agreeing detections: a single noisy re-detection (e.g. a reflection or
    the gripper momentarily mis-detected as the brick) would otherwise immediately override a
    perfectly good target with a bad one, and even a *pair* of frames agreeing by chance on the
    same wrong candidate can still slip through — requiring a larger consensus makes that far
    less likely.

    Correction stops once the camera descends within ``MIN_RELIABLE_DEPTH_DISTANCE`` of the table:
    inside the depth sensor's Min-Z blind zone the brick can't be measured, so the move simply
    finishes open-loop toward the last confident target rather than reacting to garbage depth.

    Assumes the controller accepts a new ``move_to_tcp_pose`` while a previous one is still
    in flight and replaces it (rather than queueing or rejecting it) — verify this on hardware;
    if it doesn't, this will need an explicit stop/cancel before re-issuing the move.

    Returns:
        The final target pose that was moved to, and the brick pose (x, y, table z) it was
        computed from.
    """
    X_base_target, brick_pose = estimate_target_pose(arm, camera, X_tcp_camera, hover_height, debug_dir)
    logger.info(f"initial target: {X_base_target[:3, 3].round(3)} m (base frame)")
    action = arm.move_to_tcp_pose(X_base_target, joint_speed=joint_speed)

    start_time = time.monotonic()
    while time.monotonic() - start_time < MAX_VISUAL_SERVO_DURATION:
        time.sleep(SERVO_INTERVAL)

        if action.is_action_done():
            break

        # Once the camera descends inside the depth sensor's blind zone (see
        # MIN_RELIABLE_DEPTH_DISTANCE), depth-based detection only produces garbage, which would
        # otherwise get the real brick rejected as "too tall". Stop correcting and let the move
        # finish toward the last confident target instead of chasing that noise.
        X_base_camera_now = arm.get_tcp_pose() @ X_tcp_camera
        camera_height_above_table = float(X_base_camera_now[2, 3] - brick_pose[2])
        if camera_height_above_table < MIN_RELIABLE_DEPTH_DISTANCE:
            logger.info(
                f"Camera is {camera_height_above_table * 100:.0f} cm above the table, inside the depth "
                f"sensor's blind zone (< {MIN_RELIABLE_DEPTH_DISTANCE * 100:.0f} cm Min-Z); stopping "
                "visual correction and finishing the move to the last confident target."
            )
            break

        try:
            X_base_target_updated, updated_brick_pose = estimate_target_pose(
                arm, camera, X_tcp_camera, hover_height, debug_dir
            )
        except RuntimeError as exception:
            logger.debug(f"Skipping a visual correction step (detection failed this frame): {exception}")
            continue

        drift = float(np.linalg.norm(X_base_target_updated[:3, 3] - X_base_target[:3, 3]))
        if drift <= REPLAN_POSITION_THRESHOLD:
            continue

        # Gather more independent samples and only re-plan if they all agree with each other -
        # not just with the one that triggered this check.
        samples = [X_base_target_updated]
        sample_brick_poses = [updated_brick_pose]
        for _ in range(CONFIRMATION_SAMPLES - 1):
            try:
                sample, sample_brick_pose = estimate_target_pose(arm, camera, X_tcp_camera, hover_height, debug_dir)
            except RuntimeError as exception:
                logger.debug(f"Skipping a visual correction step (confirmation frame failed): {exception}")
                break
            samples.append(sample)
            sample_brick_poses.append(sample_brick_pose)
        else:
            positions = np.stack([sample[:3, 3] for sample in samples])
            consensus_position = positions.mean(axis=0)
            spread = float(np.max(np.linalg.norm(positions - consensus_position, axis=1)))

            if spread > REPLAN_POSITION_THRESHOLD:
                logger.debug(
                    f"Re-detected brick {drift * 100:.1f} cm from the current target, but "
                    f"{len(samples)} confirmation frames disagree by up to {spread * 100:.1f} cm; "
                    "treating this as noise, not re-planning."
                )
                continue

            logger.info(
                f"Re-detected brick {drift * 100:.1f} cm from the current target, confirmed by "
                f"{len(samples)} consistent frames (spread {spread * 100:.1f} cm); re-planning."
            )
            X_base_target = find_reachable_hover_orientation(arm, consensus_position)
            brick_pose = sample_brick_poses[-1]
            action = arm.move_to_tcp_pose(X_base_target, joint_speed=joint_speed)
    else:
        logger.warning(f"Visual correction budget ({MAX_VISUAL_SERVO_DURATION:.0f}s) exceeded; letting it finish.")

    action.wait()
    return X_base_target, brick_pose


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
    "--hover-height",
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.07,
    show_default=True,
    help="Metres above the table plane to hover at.",
)
@click.option(
    "--camera-resolution",
    type=click.Choice(list(CAMERA_RESOLUTIONS)),
    default=DEFAULT_CAMERA_RESOLUTION,
    show_default=True,
    help="RealSense colour resolution (height). A D415/D435 needs USB 3 to stream colour+depth.",
)
@click.option(
    "--debug-dir",
    default=None,
    help="If set, save every detection frame here with the detected brick outlined (see save_debug_frame) "
    "for after-the-fact inspection -- there's no live display during a run.",
)
def main(
    robot_type: str,
    ip_address: Optional[str],
    port: int,
    speed_ratio: int,
    calibration_path: str,
    hover_height: float,
    camera_resolution: str,
    debug_dir: Optional[str],
) -> None:
    """Detect a lego brick on a table and move the TCP to hover above it (UR3e or RealMan)."""
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]

    X_tcp_camera = load_camera_pose_in_tcp(calibration_path)

    with connect_arm(robot_type, ip_address, port) as arm, open_camera(
        CAMERA_RESOLUTIONS[camera_resolution]
    ) as camera:
        joint_speed = speed_ratio / 100 * min(arm.manipulator_specs.max_joint_speeds)

        X_base_preview, brick_pose_preview = estimate_target_pose(
            arm, camera, X_tcp_camera, hover_height, debug_dir
        )
        logger.info(
            f"brick at (x={brick_pose_preview[0]:.3f}, y={brick_pose_preview[1]:.3f}) m, "
            f"table z={brick_pose_preview[2]:.3f} m (base frame); hovering {hover_height:.2f} m above the table."
        )
        input(f"about to hover at {X_base_preview[:3, 3].round(3)} m (base frame), press enter to continue")

        X_base_hover, brick_pose = move_to_brick_with_visual_correction(
            arm, camera, X_tcp_camera, joint_speed, hover_height, debug_dir
        )
        logger.info(
            f"final target: (x={brick_pose[0]:.3f}, y={brick_pose[1]:.3f}) m, table z={brick_pose[2]:.3f} m "
            f"(base frame)"
        )
        logger.info(f"reached:\n{arm.get_tcp_pose()}")

    os._exit(0)


if __name__ == "__main__":
    main()
