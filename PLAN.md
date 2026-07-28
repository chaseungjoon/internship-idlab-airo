# Objective

From a pile of lego bricks, grab, identify and sort each brick (by shape / shape + color) in an efficient and fast way.

# Execution plan

## Background

```
Pile of unsorted lego bricks
Realman arm
BrainCo Bionic Dexterous Hand (BC-Revo-2)
Realsense RGBD camera
```

## Outcome

Lego bricks sorted by shape or shape and color

## Modules

### Module 0

> Objective: Practice - command the BrainCo Revo2 hand directly, in sim and on hardware

- Description: Used to learn the joint space, find and store the grasp poses that M1 replays, and measure where the sim disagrees with the hardware.

- Interface
    - Input: 6 normalized finger commands in [0, 1] (thumb, thumb_aux, index, middle, ring, pinky)
    - Output: hand pose in sim / on hardware, measured positions and currents

- Implementation
    - Simulation ([src/m0/simulation](src/m0/simulation)): pydrake + Meshcat, kinematic. Playground for the joint space, aperture calibration, grasp poses sized to brick dimensions -> `poses.json`
    - Physical ([src/m0/physical](src/m0/physical)): bc_stark_sdk over RS-485. Replays `poses.json`, detects contact from motor current, records what the hand reached -> `poses_measured.json`
    - The two stacks share only [src/m0/hand_model.py](src/m0/hand_model.py): the finger table and the normalized pose vector

### Module 1

> Objective: Grasp a single lego brick from the pile

- Description: From an unorganized pile of lego bricks (from [lego_list.csv](lego_list.csv])), the robot hand will grasp a singular lego brick. The chosen lego brick will not be random nor pre-set, but the most optimal grab in the pile.

- Interface
    - Input: Lego brick pile RGBD camera frame
    - Output: TCP Pose, gripper action

- Implementation
    - Submodule 0: Grasp single standalone block
        - eye_in_hand: need a proper camera mount
        - eye_to_hand: not tried yet
    - Submodule 1: Grasp a specific block from the pile
    - Submodule 2: Identify which block to grasp from the pile

### Module 2

> Objective: Identify grasped lego brick

- Description: From the camera frames of the grasped lego brick, Module 2 updates the confidence score of the brick identification in real time. When reached a certain confidence score threshold, the brick will be identified wiith brick_id and orientation(optional)

- Interface
    - Input: Camera frame(s), grasp pose
    - Output: brick_id, (orientation)

- Implementation
    - Submodule 0: Identify brick orientation
    - Submodule 1: Identify brick_id from database

### Module 3

> Objective: Sort identified lego brick into a category

- Description: Categorize each lego brick in [lego_list.csv](lego_list.csv) into a handful of categories regarding its shape and color. This module is not a runtime active module, but will need to run before the execution of the pipeline

### Module 4

> Objective: Wire Module 1~3 and add termination checking


# Task Scope

# Actions

# Data

# Compute

# Performance metrics

# Success criteria & failure modes

# Usage of software architecture

# Use-cases & behaviors
