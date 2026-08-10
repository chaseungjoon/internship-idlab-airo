from typing import Tuple

import cv2
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
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

load_dotenv()
ROBOT_IP = os.getenv("ROBOT_IP")

gripper = Robotiq2F85(ROBOT_IP)
gripper.open()
gripper.close()