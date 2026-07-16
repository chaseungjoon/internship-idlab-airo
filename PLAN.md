# Objective

From a pile of lego bricks, grab, identify and sort each brick (by shape / shape + color) in an efficient and fast way.

# Execution plan

## Background

- Pile of unsorted lego bricks (manifest: lego_list.csv)
- Realman arm
- BrainCo Bionic Dexterous Hand (BC-Revo-2)
- Realsense RGBD camera

## Outcome

Lego bricks sorted by shape or shape and color

## Modules

### Module 1

- Objective: Grasp a single lego brick from the pile

- Interface
    - Input: Lego brick pile RGBD camera frame
    - Output: TCP Pose, gripper action

- Implementation
    - Submodule 0: Grasp single standalone block 
    - Submodule 1: Grasp a specific block from the pile
    - Submodule 2: Identify which block to grasp from the pile

### Module 2

- Objective: Identify grasped lego brick

- Interface
    - Input: Camera frame(s), grasp pose
    - Output: brick_id, orientation

- Implementation
    - Submodule 0: Identify brick orientation
    - Submodule 1: Identify brick_id from database

### Module 3

- Objective: Sort identified lego brick into a category

### Module 4

- Objective: Wire Module 1~3 and add termination checking


# Task Scope

# Actions

# Data

# Compute

# Performance metrics

# Success criteria & failure modes

# Usage of software architecture

# Use-cases & behaviors
