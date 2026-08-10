"""
m1 submodule 1: locate a hand-clicked brick from two views, then pregrasp above it.
"""
import os
import sys
from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_robots.exceptions import RobotConfigurationException
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, NumpyIntImageType
from loguru import logger

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from config import (
    APPROX_ARM_REACH,
    BRICK_HANDOFF_PATH,
    CAMERA_RESOLUTIONS,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    FALLBACK_BRICK_HEIGHT,
    FOOTPRINT_MATCH_TOLERANCE,
    PREGRASP_HEIGHT,
    SUPPORTED_ROBOT_TYPES,
    TABLE_Z,
    connect_arm,
    ensure_control_ready,
    find_reachable_hover_orientation,
    load_camera_pose_in_tcp,
    load_table_plane,
    open_camera,
)
from lego_catalog import load_catalog, parts_in_set
from m1.physical.brick_measure import ViewObservation, measure_brick, write_handoff

POS1: List[float] = [-0.08343679, -1.31992237,  0.26209098, -0.40548201, -1.20620281, -1.63604099]
POS2: List[float] = [0.81975543, -1.24165185,  0.23308164, -0.76548697, -1.72945053, -0.95741016]

MAX_PREGRASP_HEIGHT = 0.05
PARALLEL_RAY_EPS = 1e-6
LARGE_TRIANGULATION_GAP = 0.02

MIN_VALID_DEPTH = 0.10
MIN_TABLE_DEPTH_POINTS = 500
SUSPICIOUS_TABLE_DISAGREEMENT = 0.02
LARGE_VIEW_DISAGREEMENT = 0.01
TABLE_HEIGHT_BIN_SIZE = 0.002
TABLE_LOWER_SURFACE_PERCENTILE = 35.0
MAX_RUNTIME_FLOOR_TIGHTENING = 0.004
MAX_RUNTIME_FLOOR_VIEW_SPREAD = 0.004

_CONFIRM_KEYS = (13, 10, 32)
_ABORT_KEYS = (27, ord("q"))

MAX_CLICK_WINDOW_WIDTH = 1280
MAX_CLICK_WINDOW_HEIGHT = 800


def click_pixel(image_rgb: NumpyIntImageType, window_title: str) -> Tuple[int, int]:
    """Show ``image_rgb`` and return the ``(u, v)`` pixel the user clicks and confirms.

    Left-click to place the marker (re-click to move it), Enter/Space to confirm, Esc/q to abort.

    The image is downscaled *here*, by a factor this function knows, and shown in a fixed-size
    (``WINDOW_AUTOSIZE``) window, rather than handed at full size to a resizable ``WINDOW_NORMAL``
    window. In a resizable window the picture is stretched to fit whatever size the window manager
    gives it, and whether the mouse coordinates that come back are image pixels or window pixels is
    up to the highgui backend -- a silent, backend-dependent scale factor on the one input the whole
    module is built on. Scaling it ourselves makes the mapping ours: the click is divided by a known
    ``scale`` and nothing else can rescale it.

    Raises:
        RuntimeError: if the user aborts or closes the window before confirming a pixel.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    scale = min(1.0, MAX_CLICK_WINDOW_WIDTH / width, MAX_CLICK_WINDOW_HEIGHT / height)
    display = image_bgr if scale == 1.0 else cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    selection: dict = {}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["xy"] = (int(x), int(y))

    def to_image_pixel(x: int, y: int) -> Tuple[int, int]:
        """Displayed-window pixel -> full-resolution image pixel, clamped to the image."""
        return (
            int(min(max(round(x / scale), 0), width - 1)),
            int(min(max(round(y / scale), 0), height - 1)),
        )

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    logger.info(
        f"[{window_title}] click the target point, then press Enter to confirm (Esc to abort). "
        f"Showing the {width}x{height} frame at {scale:.2f}x, so one click pixel is {1 / scale:.1f} image pixel(s)."
    )
    try:
        while True:
            canvas = display.copy()
            if "xy" in selection:
                cv2.drawMarker(canvas, selection["xy"], (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in _CONFIRM_KEYS and "xy" in selection:
                return to_image_pixel(*selection["xy"])
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


def project_ray_onto_height(ray: Tuple[np.ndarray, np.ndarray], z: float) -> np.ndarray:
    """Intersect a ``(origin, unit direction)`` base-frame ray with the horizontal plane at ``z``.

    This is what replaced triangulating the two rays against each other. Both rays start from the
    camera centre, which the hand-eye calibration gets wrong by centimetres, so their intersection is
    wrong by centimetres in every axis. Intersecting with a plane whose height is *known* -- the table
    plus a brick -- fixes the answer's z outright and leaves only x and y carrying the calibration
    error, which is the trade worth making: a few millimetres sideways still grasps a 7.8 mm brick,
    9 cm too high grasps nothing at all.

    Raises:
        RuntimeError: if the ray is (near-)parallel to the plane, or the plane is behind the camera.
    """
    return project_ray_onto_plane(ray, 0.0, 0.0, z)


def project_ray_onto_plane(ray: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float) -> np.ndarray:
    """Intersect a base-frame ray with the plane ``z = a*x + b*y + c``.

    The tilted generalisation of :func:`project_ray_onto_height`, so the clicked rays can be projected
    onto the table plane ``calibrate_table.py`` actually touched rather than onto an assumed level one.
    A tabletop tilted 1 degree relative to the robot's base is 7 mm out across a 40 cm workspace --
    more than the fingertip clearance a plate leaves -- so the tilt is worth carrying through exactly
    rather than approximating it away.

    Substituting ``p = origin + t * direction`` into the plane equation gives ``t`` in closed form; no
    iteration, and it reduces to the horizontal case when ``a = b = 0``.

    Raises:
        RuntimeError: if the ray is (near-)parallel to the plane, or the plane is behind the camera.
    """
    origin, direction = ray
    if abs(direction[2]) < 0.2:  # more than ~78 degrees off vertical
        raise RuntimeError(
            "The line of sight is almost horizontal, so it barely crosses the table plane and the "
            "projected position would be meaningless. Use viewpoints that look down at the pile."
        )

    denominator = direction[2] - a * direction[0] - b * direction[1]
    if abs(denominator) < 1e-9:
        raise RuntimeError("The line of sight runs parallel to the table plane; it never crosses it.")
    distance = (a * origin[0] + b * origin[1] + c - origin[2]) / denominator
    if distance <= 0:
        raise RuntimeError(
            "The table plane lies behind the camera. Check the table calibration and the hand-eye calibration."
        )
    return origin + distance * direction


def measure_table_height(camera: RGBDCamera, X_base_camera: HomogeneousMatrixType) -> Optional[float]:
    """Height of the table in the base frame as the depth stream sees it, or ``None`` if it cannot.

    The table is the lowest broad surface in view. A plain "largest histogram bin over all heights"
    turned out too easy to hijack with a close wrist, a dense pile or another object occupying more
    pixels than the bare tabletop, which can manufacture a fake table *above* the real one. So the
    histogram is deliberately restricted to the lower third of the observed heights and the densest
    2 mm band there is taken as the table.

    Used only as a cross-check on ``--table-z``. It goes through the same (mis-)calibrated camera pose
    as everything else, so it is not an independent measurement of the table -- but that is exactly
    what makes it useful: whatever it disagrees with ``--table-z`` by is an estimate of the hand-eye
    calibration's error along the camera's view direction.
    """
    try:
        depth = np.asarray(camera.retrieve_depth_map(), dtype=np.float32)
    except Exception as exception:  # noqa: BLE001 - a cross-check must never break the run
        logger.debug(f"No depth map available for the table cross-check: {exception}")
        return None

    valid = np.isfinite(depth) & (depth > MIN_VALID_DEPTH)
    if valid.sum() < MIN_TABLE_DEPTH_POINTS:
        logger.debug(f"Only {int(valid.sum())} valid depth pixel(s); skipping the table cross-check.")
        return None

    height, width = depth.shape
    intrinsics = camera.intrinsics_matrix()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    row, col = np.mgrid[0:height, 0:width]
    z = depth[valid]
    points_camera = np.stack([(col[valid] - cx) * z / fx, (row[valid] - cy) * z / fy, z], axis=-1)
    heights = points_camera @ X_base_camera[:3, :3].T[:, 2] + X_base_camera[2, 3]

    low = float(np.percentile(heights, 1.0))
    lower_surface_high = float(np.percentile(heights, TABLE_LOWER_SURFACE_PERCENTILE))
    lower_surface = heights[(heights >= low) & (heights <= lower_surface_high)]
    if lower_surface.size < MIN_TABLE_DEPTH_POINTS // 5:
        lower_surface = heights
        lower_surface_high = float(np.percentile(heights, 99.0))
    if lower_surface_high - low < 1e-3:
        return float(np.median(lower_surface))
    counts, edges = np.histogram(
        lower_surface,
        bins=max(4, int(round((lower_surface_high - low) / TABLE_HEIGHT_BIN_SIZE))),
        range=(low, lower_surface_high),
    )
    peak = int(np.argmax(counts))
    band = lower_surface[(lower_surface >= edges[peak]) & (lower_surface <= edges[peak + 1])]
    return float(np.median(band if band.size else lower_surface))


def resolve_runtime_table_floor(table_z: float, measured: List[float]) -> float:
    """Conservative per-run table floor, but only from depth evidence strong enough to trust.

    A higher floor is safer only when it is still plausibly the *table*. In practice that means:

    * it must not disagree wildly with ``--table-z``;
    * if two views contribute, they must agree with each other to within a few millimetres;
    * if only one view contributes, it may only tighten the floor by a few millimetres.

    Anything looser than that is logged as a warning but not promoted into a hard stop.
    """
    trusted = [z for z in measured if abs(z - table_z) <= SUSPICIOUS_TABLE_DISAGREEMENT]
    if not trusted:
        return float(table_z)

    tightening = max(trusted) - table_z
    if len(trusted) >= 2:
        spread = max(trusted) - min(trusted)
        if spread > MAX_RUNTIME_FLOOR_VIEW_SPREAD:
            logger.warning(
                f"The depth table estimates disagree by {spread * 1000:.1f} mm, so they are not trusted as a hard "
                "runtime floor. Keeping --table-z as the floor for this run."
            )
            return float(table_z)
    if tightening > MAX_RUNTIME_FLOOR_TIGHTENING:
        logger.warning(
            f"The depth-based table floor would tighten --table-z by {tightening * 1000:.1f} mm, which is too "
            "large to trust as the tabletop from these views. Keeping --table-z as the floor for this run."
        )
        return float(table_z)
    return float(max([table_z, *trusted]))


def capture_view_ray(
    arm: PositionManipulator,
    camera: RGBDCamera,
    X_tcp_camera: HomogeneousMatrixType,
    joint_configuration: np.ndarray,
    joint_speed: float,
    view_name: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[float], ViewObservation]:
    """Move to ``joint_configuration``, grab a frame, and return the clicked ray and the table height.

    The arm is stationary once the move completes, so the frame and the TCP pose it is paired with are
    taken together, giving a consistent eye-in-hand camera pose for the back-projection. Colour and
    depth come from the same ``grab_images`` buffer, so the table cross-check describes the same
    instant as the click.

    The frame, the camera pose and the clicked pixel are also returned as a :class:`ViewObservation`,
    which is everything :func:`brick_measure.measure_brick` needs to work out *which brick* this is --
    measured from the same viewpoint, at the same instant, as the click that chose it.
    """
    logger.info(f"Moving to {view_name}: {np.round(joint_configuration, 3)} rad ...")
    arm.move_to_joint_configuration(joint_configuration, joint_speed=joint_speed).wait()

    camera.grab_images()
    image = camera.retrieve_rgb_image_as_int()
    X_base_camera = arm.get_tcp_pose() @ X_tcp_camera  # eye-in-hand: FK composed with hand-eye calibration.
    table_z = measure_table_height(camera, X_base_camera)
    try:
        depth_map = np.asarray(camera.retrieve_depth_map(), dtype=np.float32)
    except Exception as exception:  # noqa: BLE001 - depth only breaks ties; its absence is survivable
        logger.debug(f"No depth map in {view_name}: {exception}")
        depth_map = None

    u, v = click_pixel(image, f"{view_name} - click the target point")
    logger.info(f"{view_name}: clicked pixel (u={u}, v={v}).")
    observation = ViewObservation(
        image_rgb=image,
        click_uv=(u, v),
        intrinsics_matrix=camera.intrinsics_matrix(),
        X_base_camera=X_base_camera,
        depth_map=depth_map,
        name=view_name,
    )
    return pixel_to_base_ray(u, v, camera.intrinsics_matrix(), X_base_camera), table_z, observation


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
    type=click.FloatRange(0.0, MAX_PREGRASP_HEIGHT, min_open=True),
    default=PREGRASP_HEIGHT,
    show_default=True,
    help=f"Metres above the brick's top face for the pregrasp pose. Capped at {MAX_PREGRASP_HEIGHT * 100:.0f} cm: "
    "submodule_2 descends from here open-loop, so a higher hover is a longer blind move.",
)
@click.option(
    "--table-z",
    type=float,
    default=None,
    help="Height of the table's surface in the robot's base frame (metres), overriding the measured "
    "table plane. You should not normally need this: run `python src/calibrate_table.py` once and the "
    "arm touches the table to measure it, tilt included. Passing a flat value here throws that tilt "
    "away. Without either, it falls back to config.TABLE_Z, which is a guess.",
)
@click.option(
    "--brick-height",
    type=click.FloatRange(0.0, 0.10, min_open=True),
    default=None,
    help="Height of the brick above the table (metres), if you want to state it rather than have it "
    "measured. Giving it turns the measurement off entirely and pins every part to this height "
    "(0.0096 for a standard brick, 0.0032 for a plate). By default the brick is measured and looked "
    "up in the part catalog instead.",
)
@click.option(
    "--measure-brick/--no-measure-brick",
    "measure_brick_flag",  # named apart from the measure_brick() this calls
    default=True,
    show_default=True,
    help="Measure the clicked brick's footprint from both views and identify it in the part catalog "
    "(lego_3d/), instead of assuming every brick is a 1x3. Its real width, length and height are "
    "written to the handoff file for submodule_2 to grasp with.",
)
@click.option(
    "--debug-dir",
    default=None,
    help="If set, save each view's segmentation here with the chosen brick outlined -- the only way to "
    "check that the measurement outlined one brick rather than two touching ones.",
)
@click.option(
    "--any-part",
    is_flag=True,
    help="Match the measured footprint against every part with a mesh, not only those listed in "
    "lego_list.csv. Use it when the brick on the table is not from this set.",
)
def main(
    robot_type: str,
    ip_address: str,
    port: int,
    speed_ratio: int,
    calibration_path: str,
    camera_resolution: str,
    pregrasp_height: float,
    table_z: float,
    brick_height: Optional[float],
    measure_brick_flag: bool,
    debug_dir: Optional[str],
    any_part: bool,
) -> None:
    """Locate a hand-clicked brick from two views and move to a pregrasp above it."""
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

        ray_1, table_z_1, view_1 = capture_view_ray(
            arm, camera, X_tcp_camera, view_configurations[0], joint_speed, "view 1"
        )
        ray_2, table_z_2, view_2 = capture_view_ray(
            arm, camera, X_tcp_camera, view_configurations[1], joint_speed, "view 2"
        )

        # Where the table is. The touched-off plane wins: the arm measured it by touching, so unlike
        # anything the camera says it carries no hand-eye calibration error. --table-z overrides it
        # (flat, so the tilt is lost), and config.TABLE_Z is the last resort.
        table_plane = None if table_z is not None else load_table_plane()
        if table_plane is not None:
            plane_a, plane_b, plane_c = table_plane.a, table_plane.b, table_plane.c
            logger.info(f"Using the {table_plane.describe()}.")
        else:
            plane_a = plane_b = 0.0
            plane_c = table_z if table_z is not None else TABLE_Z
            if table_z is not None:
                logger.info(f"--table-z {table_z:+.4f} m given; using a level plane at that height.")
            else:
                logger.warning(
                    f"The table has never been touched off, so this falls back to config.TABLE_Z={TABLE_Z:+.4f} m, "
                    "which is a guess and was 19.6 mm out last time it was checked against the real table. Run "
                    "`python src/calibrate_table.py` before grasping."
                )

        # The measurement below needs a scalar height to start from; take the plane under the middle
        # of where the two cameras are looking, then let the exact tilted projection refine it.
        table_z = float(plane_c + plane_a * view_1.X_base_camera[0, 3] + plane_b * view_1.X_base_camera[1, 3])

        # Which brick is this? Both views saw it, from 40 cm and in focus, which is the only place in
        # the whole pick where it *can* be measured -- by the pregrasp the camera is inside its own
        # blind zone. So the footprint is measured here and the part looked up here, and submodule_2
        # is told the answer rather than assuming a 1x3.
        brick = None
        if brick_height is not None:
            logger.info(
                f"--brick-height {brick_height * 1000:.1f} mm given, so the brick is not measured; every part "
                "is treated as that tall."
            )
        elif not measure_brick_flag:
            logger.warning(
                "--no-measure-brick: falling back to the default 1x3 dimensions, which are wrong for every "
                "other part in the pile."
            )
        else:
            brick = measure_brick(
                [view_1, view_2],
                table_z=table_z,
                catalog=load_catalog(),
                tolerance=FOOTPRINT_MATCH_TOLERANCE,
                initial_height=FALLBACK_BRICK_HEIGHT,
                restrict_to=None if any_part else parts_in_set(),
                debug_dir=debug_dir,
            )

        if brick is not None and brick.part is not None:
            brick_height = brick.height
            logger.success(f"Brick identified: {brick.describe()}.")
            if brick.part.obstruction > 0:
                logger.warning(
                    f"This part carries {brick.part.obstruction * 1000:.1f} mm of structure above the face being "
                    "grasped; a top-down grasp aimed at that face may foul it."
                )
        else:
            if brick is not None:
                logger.warning("The brick was measured but matched no catalog part; using the default dimensions.")
            brick_height = brick_height if brick_height is not None else FALLBACK_BRICK_HEIGHT

        # The brick's top face: one brick height above the table. This is the plane both clicked rays
        # are projected onto (see the module docstring).
        brick_top_z = table_z + brick_height

        measured = [z for z in (table_z_1, table_z_2) if z is not None]
        trusted_measured = [z for z in measured if abs(z - table_z) <= SUSPICIOUS_TABLE_DISAGREEMENT]
        runtime_safe_table_z = resolve_runtime_table_floor(table_z, measured)

        # Cross-check first, because if it fires it explains everything that follows.
        if measured:
            measured_table_z = float(np.mean(measured))
            disagreement = measured_table_z - table_z
            logger.info(
                f"Depth sees the table at z={measured_table_z:.4f} m; --table-z says {table_z:.4f} m "
                f"(difference {disagreement * 100:+.1f} cm)."
            )
            if len(trusted_measured) != len(measured):
                ignored = [z for z in measured if abs(z - table_z) > SUSPICIOUS_TABLE_DISAGREEMENT]
                logger.warning(
                    f"Ignoring suspicious depth table estimate(s) {', '.join(f'{z:.4f}' for z in ignored)} m for "
                    f"the runtime floor because they disagree with --table-z={table_z:.4f} m by more than "
                    f"{SUSPICIOUS_TABLE_DISAGREEMENT * 100:.0f} cm and are likely not the tabletop."
                )
            if abs(disagreement) > SUSPICIOUS_TABLE_DISAGREEMENT:
                logger.warning(
                    f"That is {abs(disagreement) * 100:.1f} cm apart, and nothing else in the chain can shift the "
                    "table that far. Either --table-z is wrong for this table (it is measured from the "
                    f"calibration board in {calibration_path}; re-measure it if the robot has been re-mounted), "
                    "or the hand-eye calibration is off by about that much along the camera's view direction -- "
                    "which is also roughly how far off x and y will be. Re-running the hand-eye calibration with "
                    "10-20 board poses instead of 3 is the fix for the second case. The pregrasp *height* below "
                    "is unaffected either way: it is anchored to --table-z, not to the camera."
                )
            logger.info(
                f"submodule_2 will enforce a runtime TCP floor from z={runtime_safe_table_z:.4f} m upward: the "
                "highest depth estimate from this run that remained consistent enough to trust as the tabletop."
            )
        else:
            logger.warning(
                "No usable depth in either view, so the table height could not be cross-checked; trusting "
                f"--table-z={table_z:.4f} m outright."
            )
            logger.warning(
                "Without a depth-based table measurement this run cannot add a tighter runtime floor; "
                "submodule_2 will fall back to the configured table height only."
            )

        # Each view's own answer for where the brick is. Their disagreement replaces the triangulation
        # gap as the quality metric, and measures the same thing more usefully: how far apart the two
        # views put the *brick*, in the plane the grasp happens in.
        # Projected onto the *tilted* table plane raised by one brick height, so a table that is not
        # exactly square to the robot's base is followed rather than averaged away.
        try:
            point_1 = project_ray_onto_plane(ray_1, plane_a, plane_b, plane_c + brick_height)
            point_2 = project_ray_onto_plane(ray_2, plane_a, plane_b, plane_c + brick_height)
        except RuntimeError as exception:
            logger.error(str(exception))
            os._exit(1)

        point = 0.5 * (point_1 + point_2)
        # Now that the brick's x, y is known, the table height *under it* is the one that matters --
        # which is what submodule_2's floor and the handoff below should carry, not the value from
        # wherever the plane happened to be sampled earlier.
        table_z = float(plane_c + plane_a * point[0] + plane_b * point[1])
        brick_top_z = table_z + brick_height
        # Re-derive the depth-based floor against that refined height, so the number handed to
        # submodule_2 describes the table under the brick rather than under the camera.
        runtime_safe_table_z = resolve_runtime_table_floor(table_z, measured)
        view_disagreement = float(np.linalg.norm(point_1 - point_2))
        logger.info(
            f"View 1 puts the brick at {point_1.round(4)} m, view 2 at {point_2.round(4)} m; "
            f"using their midpoint {point.round(4)} m (views disagree by {view_disagreement * 1000:.1f} mm)."
        )
        if view_disagreement > LARGE_VIEW_DISAGREEMENT:
            logger.warning(
                f"The two views disagree by {view_disagreement * 1000:.1f} mm (> "
                f"{LARGE_VIEW_DISAGREEMENT * 1000:.0f} mm), which is wider than the 7.8 mm brick. Either the two "
                "clicks were not on the same point, or the hand-eye calibration's lateral error is large. "
                "Continuing with the midpoint, but expect the grasp to be off sideways."
            )

        # Triangulation is still computed, purely so the size of the disagreement is on the record: it
        # is the number that used to drive the pregrasp, and it is why the pregrasp used to be wrong.
        try:
            triangulated, gap = triangulate_rays(ray_1, ray_2)
            logger.info(
                f"For reference, triangulating the two rays instead gives {triangulated.round(4)} m "
                f"({(triangulated[2] - point[2]) * 100:+.1f} cm in z, "
                f"{np.linalg.norm(triangulated[:2] - point[:2]) * 100:.1f} cm sideways), with the rays missing "
                f"each other by {gap * 1000:.1f} mm. Not used."
            )
            if gap > LARGE_TRIANGULATION_GAP:
                logger.info(
                    f"Those rays miss by more than {LARGE_TRIANGULATION_GAP * 1000:.0f} mm, which is itself a "
                    "sign the clicks or the calibration are off."
                )
        except RuntimeError as exception:
            logger.debug(f"Reference triangulation unavailable: {exception}")

        pregrasp_position = point + np.array([0.0, 0.0, pregrasp_height])
        logger.info(
            f"Brick top face at z={brick_top_z:.4f} m (table {table_z:.4f} + brick {brick_height * 1000:.1f} mm); "
            f"pregrasp {pregrasp_height * 100:.1f} cm above it."
        )
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

        # Written before the move rather than after: if the move fails or is interrupted, the file
        # still describes the brick at a position that no longer matches where the arm is standing,
        # and submodule_2's position check will refuse it -- which is the correct outcome. The
        # runtime table floor is carried even when the part could not be identified, because the
        # descent limit still matters in that case.
        write_handoff(
            BRICK_HANDOFF_PATH,
            brick_position=point[:2],
            brick=brick,
            height=brick_height,
            configured_table_z=table_z,
            measured_table_zs=measured,
            safe_table_z=runtime_safe_table_z,
        )

        input(f"about to move to the pregrasp at {X_base_pregrasp[:3, 3].round(3)} m (base frame), press enter to continue")
        try:
            arm.move_to_tcp_pose(X_base_pregrasp, joint_speed=joint_speed).wait()
        except RobotConfigurationException as exception:
            logger.error(out_of_reach_message(robot_type, X_base_pregrasp[:3, 3], exception))
            os._exit(1)
        logger.info(f"reached:\n{arm.get_tcp_pose()}")
        if brick is not None and brick.part is not None:
            logger.info(
                f"submodule_2 will grasp a {brick.width * 1000:.1f} mm wide part with the fingers closing along "
                f"{np.degrees(brick.closing_heading):.0f} deg -- no --yaw-deg needed."
            )

    os._exit(0)


if __name__ == "__main__":
    main()
