# info

* Bachelor's Internship @ IDLab-AIRO, UGent
* Timeline: 13JUL26-26AUG26
* Robot ip: 10.42.0.162
* Robot type: universal robots UR3
* Materials: int2026/Materials
* Lego list: int2026/lego_list.csv
* Lego 3d models: int2026/lego_3d

# Objective

* Learn robot manipulation and imitation learning utilizing UR3 and lego bricks

# Plan

## MVP

1. Use `.urdf` in Drake to simulate gripper and grab simulation
2. Rotate wrist a few times and render, and pretrain
3. Using UR3, test frame capture

### Setup (MVP 1 & 2)

Environment + code for MVP items 1 & 2 lives in `/src`: Drake + Meshcat +
`airo-mono`, mirroring `materials/environment.yaml` (the course setup).

* `src/environment.yaml` - Linux conda env, exact copy of the professor's
  `materials/environment.yaml`. Supports all 3 MVP items, including the real
  UR3 and its RealSense camera.
* `src/environment-macos.yaml` - macOS variant (same packages, python bumped
  3.10 -> 3.11 since `drake==1.32.0` has no macOS `cp310` wheel). Supports
  MVP items 1 & 2 fully. **Does not support MVP item 3's camera capture** -
  `pyrealsense2` has no macOS distribution at all. Real UR3 *arm control*
  (`airo-robots`/`ur_rtde`) does build and import on macOS, so it'll work
  from a Mac too if the Mac is on the robot's network - only the RealSense
  camera is macOS-blocked. Verified by actually building both the Drake
  scene and the full `airo-mono` stack on macOS 26 (arm64) before writing
  this.
* `src/scene.py`, `src/grab_demo.py`, `src/wrist_render.py` - MVP items 1 & 2
  implementation (UR3e + Robotiq 2F-85 grab simulation using a `lego_3d`
  brick; wrist rotation + rendered/saved frames for later pretraining).
* `src/test.ipynb` - run this after creating the env to confirm everything
  works, including an on-site-only check for the real UR3 (and its camera,
  Linux-only).

```
conda env create -f src/environment.yaml        # Linux
conda env create -f src/environment-macos.yaml  # macOS
conda activate irm
python -m ipykernel install --user --name irm --display-name "Python (irm)"
jupyter lab src/test.ipynb   # select the "Python (irm)" kernel
```

## Task Scope

## Actions

## Data

## Compute

## Users

## Performance metrics

## Success criteria & failure modes

## Usage of software architecture

## Use-cases & behaviors
