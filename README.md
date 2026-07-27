# Info

* Bachelor's Internship @ IDLab-AIRO, UGent
* Timeline: 1JUL26-26AUG26
* Robot type: Realman
* Hand type: BrainCo Revo2

# Objective

* Learn robot manipulation and imitation learning utilizing robot hand and lego bricks

[PLAN.md](PLAN.md)

# Quickstart

- Prerequisites: python3.10.* & conda

- Install

```bash
git clone https://github.com/chaseungjoon/internship-idlab-airo
cd internship-idlab-airo
```

- Setup

```bash
conda env create -f src/environment.yaml    #linux
conda env create -f src/environment-macos.yaml    #macos
```
```bash
conda activate int2026
```

- Run simulation

```bash
python3 src/M1/simulation/submodule_0.py
```

- Run physical

```bash
jupyter notebook src/M1/physical/submodule_0.ipynb
```

# Hand setup

```bash
pip install bc-stark-sdk    # real hand only, on the robot PC
```

# M0 - hand playground

Practice module for the BrainCo Revo2 hand. The two stacks are kept apart and share only
`src/m0/hand_model.py` (the normalized 6-finger pose) and the pose files it reads/writes.

```
src/m0/
  hand_model.py                    stack-neutral: finger table, pose vector, pose library
  poses.json                       designed in simulation, replayed on hardware
  poses_measured.json              what the hardware actually reached
  simulation/   sim_hand.py        pydrake + Meshcat + ipywidgets
                hand_playground.ipynb    sliders, joint space, fingertip geometry
                grasp_poses.ipynb        aperture calibration, size a pinch to a brick
  physical/     real_hand.py       bc_stark_sdk over RS-485 + ipywidgets
                hand_playground.ipynb    connect, arm, drive the real hand
                grasp_poses.ipynb        replay sim poses, find contact by motor current
```

```bash
jupyter notebook src/m0/simulation/hand_playground.ipynb    # anywhere
jupyter notebook src/m0/physical/hand_playground.ipynb      # on the robot PC
```