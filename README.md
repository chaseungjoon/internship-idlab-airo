# Info

* Bachelor's Internship @ IDLab-AIRO, UGent
* Timeline: 1JUL26-26AUG26
* Robot ip: 10.42.0.162
* Robot type: Universal Robots UR3e

# Objective

* Learn robot manipulation and imitation learning utilizing UR3 and lego bricks

# Quickstart

- Prerequisites
    python3.10.*, conda, UR3e

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

- Run simulation

```bash
conda activate irm

python3 src/M1/submodule_0.py
```