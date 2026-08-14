"""Hand-guide the arm somewhere and print where it ended up.

Freedrive is on while this waits for a keypress: push the arm where you want it, press enter, and it
prints the joint configuration and the TCP pose it settled at. That is how the constants elsewhere in
this repo were measured, and it is the way to re-measure them for your own bench:

* ``cell.VIEWPOINT_JOINT_CONFIGURATIONS`` -- the two places the camera looks at the pile from. Drive
  the arm until the wrist camera sees the whole pile from a comfortable angle, and paste the joint
  configuration in.
* ``cell.HOME_CONFIGURATION`` -- the parking pose every cross-table leg goes via.
* ``cell.PILE_CENTER`` and ``submodule_2.DROP_POSITION`` -- put the tip on the middle of the pile, and
  then on the corner picked bricks should go to, and read the TCP translation off.

::

    python src/tools/teach_pose.py
    python src/tools/teach_pose.py --open-gripper     # let go of whatever is held first

Was ``src/test.py``, which was neither a test nor findable; it also shadowed the standard library's
``test`` package for anything that put ``src/`` on the path.
"""

from __future__ import annotations

import os
import sys

import click
import numpy as np
from loguru import logger

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from common.config import (  # noqa: E402
    DEFAULT_IP_ADDRESSES,
    DEFAULT_REALMAN_PORT,
    SUPPORTED_ROBOT_TYPES,
    connect_arm,
    ensure_control_ready,
)


@click.command()
@click.option(
    "--robot-type",
    "robot_type",
    type=click.Choice(SUPPORTED_ROBOT_TYPES),
    default="ur3e",
    show_default=True,
    help="Which arm to hand-guide.",
)
@click.option(
    "--ip-address",
    default=None,
    help="Robot controller IP address. Defaults per robot type "
    f"(ur3e: {DEFAULT_IP_ADDRESSES['ur3e']}, realman: {DEFAULT_IP_ADDRESSES['realman']}).",
)
@click.option("--port", default=DEFAULT_REALMAN_PORT, show_default=True, help="Controller port (RealMan only).")
@click.option(
    "--open-gripper",
    is_flag=True,
    help="Open the Robotiq before freedriving, so a brick left in the jaws is dropped rather than "
    "carried around while you push the arm about. UR3e only.",
)
def main(robot_type: str, ip_address: str, port: int, open_gripper: bool) -> None:
    """Freedrive the arm, then print the joint configuration and TCP pose it was left in."""
    if ip_address is None:
        ip_address = DEFAULT_IP_ADDRESSES[robot_type]

    with connect_arm(robot_type, ip_address, port) as arm:
        ensure_control_ready(arm)

        if open_gripper:
            if robot_type != "ur3e":
                raise click.ClickException("--open-gripper is UR3e only; nothing else here carries the Robotiq.")
            from m1.physical.cell import connect_gripper  # noqa: PLC0415 - only needed for this flag

            with connect_gripper(ip_address) as gripper:
                gripper.open().wait()

        rtde_control = getattr(arm, "rtde_control", None)
        if rtde_control is None or not hasattr(rtde_control, "teachMode"):
            raise click.ClickException(
                f"The {robot_type} driver has no freedrive (teachMode) in airo-mono, so this can only report "
                "where the arm already is. Move it with the pendant and re-run without freedrive."
            )

        rtde_control.teachMode()
        logger.info("Freedrive is ON -- push the arm where you want it, then press enter here.")
        try:
            input("press enter when the arm is where you want it ")
        finally:
            # In a finally block on purpose: leaving the arm in teachMode after a Ctrl-C means it stays
            # limp, which is a surprise nobody wants next to a table full of bricks.
            rtde_control.endTeachMode()
        logger.info("Freedrive is OFF.")

        joints = np.asarray(arm.get_joint_configuration(), dtype=float)
        pose = arm.get_tcp_pose()
        print()
        print("joint configuration (rad), paste-ready:")
        print(f"    np.array([{', '.join(f'{v:.8f}' for v in joints)}])")
        print()
        print("TCP position (m), paste-ready:")
        print(f"    np.array([{', '.join(f'{v:.4f}' for v in pose[:3, 3])}])")
        print()
        print("full TCP pose:")
        print(np.round(pose, 4))


if __name__ == "__main__":
    main()
