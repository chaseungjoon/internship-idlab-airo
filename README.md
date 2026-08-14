# IDLab-AIRO Summer Internship

## Info

* **Bachelor's Internship** @ ***IDLab-AIRO, UGent***
* **Timeline**: 1JUL26-26AUG26
* **Robot type**: RM75, UR3e
* **TCP type**: BrainCo Revo2, Robotiq 2F-85

## Objective

* Learn robot manipulation and imitation learning utilizing robot hand and lego bricks

* [PLAN.md](PLAN.md)

## Layout

```text
src/
  common/       shared libraries — imported, never run
    config.py         single source of truth for robot/camera/calibration constants and connections
    lego_catalog.py   the lego part catalog: meshes, footprints, part matching
    scene.py          Drake scene building (arm, gripper, meshcat) for the simulations
  tools/        command-line utilities, run by hand
    calibrate_table.py  touch the tabletop with the arm and fit a plane to it — run this first
    capture.py          grab frames from the wrist camera
    measure_table_z.py  cross-check the hand-eye calibration against the board (superseded for table height)
    teach_pose.py       freedrive the arm and print where it ended up, for measuring bench constants
  prototypes/   earlier standalone versions, kept because they still run without a robot
    pile_perception.py  RGB-only pile perception on a still photograph
    wrist_render.py     render a wrist-camera sweep to a frame dataset stub
  m0/           module 0 — command the BrainCo Revo2 hand directly
  m1/           module 1 — grasp one brick from the pile (simulation/ and physical/)
```

Everything puts `src/` on `sys.path` and imports as `common.config`, `m1.physical.cell`, and so on;
scripts are run by path from the repo root (`python src/tools/calibrate_table.py`).

## Quickstart

> Prerequisites: python3.10.*, conda

- Install

```bash
git clone https://github.com/chaseungjoon/internship-idlab-airo
cd internship-idlab-airo
```

- Setup

```bash
conda env create -f src/environment-latest.yaml   
```
```bash
conda activate airo-mono
```

- Run simulation

M1 in Drake + Meshcat: a UR3e with a Robotiq 2F-85 and a wrist RealSense picking real lego parts out
of a flat pile, one survey at a time. Open the notebook and run the cells in order; it prints a
Meshcat URL to watch in.

```bash
jupyter notebook src/m1/simulation/main.ipynb
```

- Run physical

Calibrate the camera, then touch off the table once (the arm measures it, tilt included):

```bash
airo-camera-toolkit hand-eye-calibration --mode eye_in_hand --robot_ip=[ROBOT_IP]
python3 src/tools/calibrate_table.py
```

Then the notebook, which runs both M1 submodules in one process — the same two, in the same order, as
the simulation:

```bash
jupyter notebook src/m1/physical/main.ipynb
```

Or the two-terminal workflow, when you want to stop between the halves and look at what was chosen:

```bash
python3 src/m1/physical/submodule_1.py    # perceive the pile, choose a brick, stand over it
python3 src/m1/physical/submodule_2.py    # grasp it and lift it
```

The perception is shared, not mirrored: both stacks call `analyse_pile` in
[`src/m1/physical/submodule_3.py`](src/m1/physical/submodule_3.py), which can also be run on its own
to look at what it makes of a pile without committing to a grasp.

## Revo2 Hand setup

```bash
pip install bc-stark-sdk 
```
