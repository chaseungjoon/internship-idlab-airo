"""
    python print_joint_tcp.py --robot ur3e --ip 10.43.0.162
    python print_joint_tcp.py --robot realman
"""

import argparse
import sys

UR_MODELS = {"ur3", "ur3e", "ur5e"}
DEFAULT_REALMAN_IP = "192.168.1.18"
DEFAULT_REALMAN_PORT = 8080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot",
        default="realman",
        choices=sorted(UR_MODELS | {"realman"}),
        help="Robot type to connect to.",
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_REALMAN_IP,
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

    print(f"Joint configuration: {arm.get_joint_configuration()}")
    print(f"TCP pose:\n{arm.get_tcp_pose()}")


if __name__ == "__main__":
    main()
