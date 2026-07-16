# Info

* Bachelor's Internship @ IDLab-AIRO, UGent
* Timeline: 1JUL26-26AUG26
* Robot type: Realman
* Hand type: BrainCo Revo2

# Objective

* Learn robot manipulation and imitation learning utilizing robot hand and lego bricks

# Quickstart

- Prerequisites: python3.10.*, conda, UR3e

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