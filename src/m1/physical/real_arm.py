"""M1 physical stack: the real RealMan RM75-6F arm over RM_API2 (Ethernet, port 8080).

Tech stack: Robotic_Arm.rm_robot_interface (RealMan's official RM_API2 Python binding) wrapped
behind airo_robots.manipulators.PositionManipulator - the same role rtde_control/rtde_receive
play for airo_robots' URrtde implementation. Install on the robot PC:

    pip install Robotic_Arm

RM_API2 reference: https://develop.realman-robotics.com/en/robot/apipython/
The dict keys returned by rm_get_current_arm_state() and the exact algo function names are not
fully documented publicly; print the raw state once after connecting and check it against your
installed RM_API2 version before trusting the values used here.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

RMPoseType = np.ndarray
"""RM_API2 pose: [x, y, z, rx, ry, rz], translation in meters, XYZ-Euler orientation in radians."""

# https://www.realman-robotics.com/en/products/rm75.html - verify against your unit's datasheet.
# J1-J2: 180 deg/s max. J3-J7: 225 deg/s max. Joint ranges: +-178,130,178,135,178,128,360 deg.
RM75_6F_SPECS = ManipulatorSpecs(
    max_joint_speeds=[np.radians(180), np.radians(180)] + [np.radians(225)] * 5,
    max_linear_speed=1.0,
)
RM75_6F_JOINT_LIMITS_DEG = np.array([178, 130, 178, 135, 178, 128, 360])


class RMArm(PositionManipulator):
    """PositionManipulator implementation for the RealMan RM75-6F over RM_API2.

    No (self-)collision checking or obstacle avoidance is performed. `connect=0` is used on every
    move (execute immediately, no trajectory blending) and `block=0` (non-blocking), matching the
    async command + AwaitableAction pattern used throughout airo_robots.
    """

    def __init__(
        self,
        ip_address: str,
        port: int = 8080,
        manipulator_specs: Optional[ManipulatorSpecs] = None,
        gripper: Optional[ParallelPositionGripper] = None,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip_address, port)
        if self.handle.id == -1:
            raise RuntimeError(
                f"Could not connect to the RM75-6F at {ip_address}:{port}. "
                "Is the arm powered on, reachable on the network, and not already claimed by another client?"
            )
        logger.info(f"Connected to RM75-6F at {ip_address}:{port}, handle id {self.handle.id}")

        super().__init__(manipulator_specs or RM75_6F_SPECS, gripper)

        self.default_speed_ratio = 20  # 1..100, percentage of max joint/linear velocity
        self.default_blend_radius = 0  # 0 = come to a full stop at every waypoint

        self._pose_reached_L2_threshold = 0.01
        self._joint_config_reached_L2_threshold = 0.01

    # -- state -------------------------------------------------------------------------
    def _state(self) -> dict:
        code, state = self.arm.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with error code {code}")
        return state

    def get_joint_configuration(self) -> JointConfigurationType:
        return np.radians(self._state()["joint"])

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        return self._convert_rm_pose_to_homogeneous_pose(np.asarray(self._state()["pose"]))

    # -- moves -------------------------------------------------------------------------
    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_joint_configuration_is_valid(joint_configuration)
        speed_ratio = self._speed_to_ratio(joint_speed, min(self.manipulator_specs.max_joint_speeds))

        joint_degrees = list(np.degrees(joint_configuration))
        code = self.arm.rm_movej(joint_degrees, speed_ratio, self.default_blend_radius, 0, 0)
        if code != 0:
            raise RuntimeError(f"rm_movej failed with error code {code}")
        return AwaitableAction(
            lambda: bool(
                np.linalg.norm(self.get_joint_configuration() - joint_configuration)
                < self._joint_config_reached_L2_threshold
            )
        )

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_pose_is_valid(tcp_pose)
        speed_ratio = self._speed_to_ratio(joint_speed, min(self.manipulator_specs.max_joint_speeds))
        rm_pose = self._convert_homogeneous_pose_to_rm_pose(tcp_pose)
        code = self.arm.rm_movej_p(list(rm_pose), speed_ratio, self.default_blend_radius, 0, 0)
        if code != 0:
            raise RuntimeError(f"rm_movej_p failed with error code {code}")
        return AwaitableAction(
            lambda: bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_L2_threshold)
        )

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_pose_is_valid(tcp_pose)
        linear_speed = linear_speed or self.default_linear_speed
        self._assert_linear_speed_is_valid(linear_speed)
        speed_ratio = self._speed_to_ratio(linear_speed, self.manipulator_specs.max_linear_speed)
        rm_pose = self._convert_homogeneous_pose_to_rm_pose(tcp_pose)
        code = self.arm.rm_movel(list(rm_pose), speed_ratio, self.default_blend_radius, 0, 0)
        if code != 0:
            raise RuntimeError(f"rm_movel failed with error code {code}")
        return AwaitableAction(
            lambda: bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_L2_threshold)
        )

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        joint_degrees = list(np.degrees(joint_configuration))
        code = self.arm.rm_movej_canfd(joint_degrees, follow=True)
        if code != 0:
            raise RuntimeError(f"rm_movej_canfd failed with error code {code}")
        return self._timed_awaitable(duration)

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        rm_pose = self._convert_homogeneous_pose_to_rm_pose(tcp_pose)
        code = self.arm.rm_movep_canfd(list(rm_pose), follow=True)
        if code != 0:
            raise RuntimeError(f"rm_movep_canfd failed with error code {code}")
        return self._timed_awaitable(duration)

    # -- kinematics ----------------------------------------------------------------
    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_near: Optional[JointConfigurationType] = None
    ) -> Optional[JointConfigurationType]:
        q_near = joint_configuration_near if joint_configuration_near is not None else self.get_joint_configuration()
        rm_pose = self._convert_homogeneous_pose_to_rm_pose(tcp_pose)
        code, joint_degrees = self.arm.rm_algo_inverse_kinematics(
            {"q_in": list(np.degrees(q_near)), "q_pose": list(rm_pose), "flag": 1}
        )
        if code != 0:
            return None
        return np.radians(joint_degrees)

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        rm_pose = self.arm.rm_algo_forward_kinematics(list(np.degrees(joint_configuration)), 1)
        return self._convert_rm_pose_to_homogeneous_pose(np.asarray(rm_pose))

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        limit = np.radians(RM75_6F_JOINT_LIMITS_DEG)
        return bool(np.all(np.abs(joint_configuration) <= limit))

    # -- lifecycle --------------------------------------------------------------------
    def close(self) -> None:
        self.arm.rm_delete_robot_arm()

    # -- conversions --------------------------------------------------------------
    @staticmethod
    def _convert_rm_pose_to_homogeneous_pose(rm_pose: RMPoseType) -> HomogeneousMatrixType:
        return SE3Container.from_euler_angles_and_translation(rm_pose[3:], rm_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homogeneous_pose_to_rm_pose(homogeneous_pose: HomogeneousMatrixType) -> RMPoseType:
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        return np.concatenate([se3.translation, se3.orientation_as_euler_angles])

    def _speed_to_ratio(self, speed: Optional[float], max_speed: float) -> int:
        speed = speed or self.default_joint_speed
        return int(np.clip(round(100 * speed / max_speed), 1, 100))

    def _timed_awaitable(self, duration: float) -> AwaitableAction:
        action_sent_time = time.time_ns()
        return AwaitableAction(
            lambda: time.time_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )


if __name__ == "__main__":
    """test script for the RM75-6F.
    e.g. python src/m1/physical/real_arm.py --ip_address 192.168.1.18
    """
    import click
    from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot

    @click.command()
    @click.option("--ip_address", help="IP address of the RM75-6F")
    @click.option("--port", default=8080, help="port of the RM75-6F Ethernet interface")
    def test_rm_arm(ip_address: str, port: int) -> None:
        arm = RMArm(ip_address, port)
        try:
            manual_test_robot(arm)
        finally:
            arm.close()

    test_rm_arm()
