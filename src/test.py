import argparse
import sys
import numpy as np
from typing import Tuple
from dotenv import load_dotenv
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_dataset_tools.data_parsers.pose import Pose
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.manipulators.hardware import realman
from airo_spatial_algebra import SE3Container
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyIntImageType,
    Vector2DType,
    Vector3DType,
    JointConfigurationType,
)
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_camera_toolkit.interfaces import RGBDCamera

UR_MODELS = {"ur3", "ur3e", "ur5e"}
DEFAULT_REALMAN_IP = "192.168.1.18"
DEFAULT_UR3E_IP = "10.43.0.162"
DEFAULT_REALMAN_PORT = 8080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot",
        default="ur3e",
        choices=sorted(UR_MODELS | {"realman"}),
        help="Robot type to connect to.",
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_UR3E_IP,
        help=f"IP address of the robot controller (default for realman: {DEFAULT_REALMAN_IP}; required for UR robots).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_REALMAN_PORT,
        help="Controller port, only used with --robot realman (default: %(default)s).",
    )
    return parser.parse_args()


def connect(args: argparse.Namespace):
    if args.robot in UR_MODELS:
        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        if args.ip is None:
            print("error: --ip is required for UR robots.", file=sys.stderr)
            sys.exit(1)

        arm = URrtde(args.ip)

        detected_model = arm.model.value.lower()
        if detected_model != args.robot:
            print(
                f"warning: requested --robot {args.robot}, but detected {detected_model} at {args.ip}.",
                file=sys.stderr,
            )
        return arm

    from airo_robots.manipulators.hardware.realman import RealmanControl

    ip_address = args.ip or DEFAULT_REALMAN_IP
    return RealmanControl(ip_address, args.port)


def main() -> None:
    args = parse_args()

    try:
        arm = connect(args)
    except (RuntimeError, ConnectionError) as e:
        print(f"error: could not connect to {args.robot} robot: {e}", file=sys.stderr)
        sys.exit(1)

    arm.rtde_control.teachMode()
    input("press any key")
    arm.rtde_control.endTeachMode()
    print(f"Joint configuration: {arm.get_joint_configuration()}")
    print(f"TCP pose:\n{arm.get_tcp_pose()}")

    gripper = Robotiq2F85(args.ip)
    gripper.open()
    gripper.close()


if __name__ == "__main__":
    main()
