import numpy as np

from airo_robots.manipulators.hardware.realman import RealmanControl

IP_ADDRESS = "192.168.1.18"
PORT = 8080
SPEED_RATIO = 10 
arm = RealmanControl(IP_ADDRESS, PORT)

print(arm.get_joint_configuration())
print(arm.get_tcp_pose())