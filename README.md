# Info

* Bachelor's Internship @ IDLab-AIRO, UGent
* Timeline: 1JUL26-26AUG26
* Robot type: Realman, UR3e
* Hand type: BrainCo Revo2, 

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
conda env create -f src/environment-latest.yaml   
```
```bash
conda activate airo-mono
```

- Run simulation

M1 in Drake + Meshcat: a UR3e with a Robotiq 2F-85 and a wrist RealSense picking real lego parts out
of a flat pile. Open the notebook and run the cells in order; it prints a Meshcat URL to watch in.

```bash
jupyter notebook src/m1/simulation/main.ipynb
```

The perception is not simulated — the rendered colour and depth go into the same
`m1/physical/submodule_3.py` that runs against the real camera.

- Run physical

Touch off the table once (the arm measures it, tilt included), then run the M1 submodules in order:

```bash
python3 src/calibrate_table.py
python3 src/m1/physical/submodule_3.py    # perceive the pile, choose a brick
python3 src/m1/physical/submodule_1.py    # go above it, pre-grasp
python3 src/m1/physical/submodule_2.py    # grasp and lift
```

# Revo2 Hand setup

```bash
pip install bc-stark-sdk 
```