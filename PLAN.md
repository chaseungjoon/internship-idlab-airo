# Plan

## Objective

From a pile of unsorted lego bricks, grasp, identify and sort each brick — by shape, or by shape and
colour — quickly and without human intervention.

## Setup

| | |
|---|---|
| Arms | Universal Robots UR3e, Realman RM75 |
| End effectors | Robotiq 2F-85 adaptive gripper, BrainCo Revo2 dexterous hand |
| Camera | Intel RealSense RGB-D, wrist mounted |
| Parts | the 104 lines of [lego_list.csv](lego_list.csv), tipped out in one pile |

## Modules

| | | status |
|---|---|---|
| M0 | command the Revo2 hand directly | done |
| M1 | grasp a single brick out of the pile | done, running on the bench |
| M2 | identify the brick once it is held | not started |
| M3 | assign every part in the catalogue to a sorting category | not started |
| M4 | wire M1–M3 together, add termination checking | not started |

---

### M0 — the hand · [src/m0](src/m0)

Practice, and the source of the grasp poses M1 replays. Learn the joint space, size the poses to
brick dimensions, and measure where the simulator disagrees with the hardware.

- **In** — 6 normalised finger commands in [0, 1]: thumb, thumb_aux, index, middle, ring, pinky
- **Out** — hand pose in sim or on hardware, with measured positions and motor currents

[`simulation/`](src/m0/simulation) is pydrake + Meshcat, kinematic: a playground for the joint space,
aperture calibration and grasp poses, written to `poses.json`. [`physical/`](src/m0/physical) is
`bc_stark_sdk` over RS-485: it replays `poses.json`, detects contact from motor current, and records
what the hand actually reached to `poses_measured.json`. The two share only
[`hand_model.py`](src/m0/hand_model.py) — the finger table and the normalised pose vector.

### M1 — grasp one brick · [src/m1](src/m1)

The brick chosen is neither random nor pre-set: it is the best grasp available in the pile as it
currently lies. See the perception section of the [README](README.md#perception--finding-the-bricks-and-choosing-which-to-pick).

- **In** — one RGB-D frame of the pile
- **Out** — TCP pose and gripper action

| | |
|---|---|
| perception | segment the pile, measure every brick, score every grasp, rank them |
| submodule 1 | look from two viewpoints, locate the chosen brick, stand over it |
| submodule 2 | descend, close, verify, lift, verify again |
| `main.ipynb` | run `submodule 1 → submodule 2` in a loop until the pile is empty |

Two viewpoints cost most of the cycle time, so one survey locates *every* graspable brick and the
picks are served from that map. The pile is looked at again only when the map runs low — which is
also when the bricks occluded at the start have become the ones on top.

### M2 — identify the held brick

Update a confidence score over brick identity from the camera frames of the brick in the gripper, and
commit to a `brick_id` once it passes a threshold.

- **In** — camera frame(s), grasp pose
- **Out** — brick id, orientation

Two parts: identify orientation, then match against the catalogue. Doing this *after* the pick rather
than in the pile is deliberate — one frame of a pile does not carry enough of any one brick to
identify it, and once it is held the camera can look at it from wherever it likes.

### M3 — sorting categories

Assign every part in [lego_list.csv](lego_list.csv) to a category by shape and colour. Offline: it
runs once, before the pipeline, not during it.

### M4 — the loop

Wire M1–M3 together and decide when to stop.

## Open questions

- How many frames M2 needs before its confidence is worth acting on, and what to do with a brick it
  cannot identify.
- Whether the RGB-D perception goes back into the bench pipeline, which depends on getting the
  hand-eye calibration error below the plate thickness it has to resolve.
- Whether shape-only or shape-and-colour categories, which sets how many bins M3 needs.
