# IDLab-AIRO Summer Internship

## Info

* **Bachelor's Internship** @ ***IDLab-AIRO, UGent***
* **Timeline**: 1JUL26-26AUG26
* **Robot type**: RM75, UR3e
* **TCP type**: BrainCo Revo2, Robotiq 2F-85

## Objective

* Learn robot manipulation and imitation learning utilizing robot hand and lego bricks

* [PLAN.md](PLAN.md)

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
python3 src/calibrate_table.py
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
