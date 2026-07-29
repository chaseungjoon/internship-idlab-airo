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

```bash
python3 src/m1/simulation/submodule_0.py
```

- Run physical

```bash
jupyter notebook src/m1/physical/submodule_0.py
```

# Revo2 Hand setup

```bash
pip install bc-stark-sdk 
```