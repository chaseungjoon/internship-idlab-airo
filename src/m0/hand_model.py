"""M0 - the part of the Revo2 hand that both stacks agree on.

This module is deliberately dependency-free (numpy only): no pydrake, no bc_stark_sdk.
It defines the *interface* between the two M0 stacks:

    src/M0/simulation  -> Drake + Meshcat          (sim_hand.py)
    src/M0/physical    -> bc_stark_sdk over RS-485 (real_hand.py)

Both speak the same "normalized pose": a length-6 float vector in [0, 1], 0 = open,
1 = closed, ordered the way bc_stark_sdk orders its finger arrays. A pose found in
simulation can therefore be saved to poses.json and replayed on the hardware unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

M0_DIR = Path(__file__).resolve().parent
REPO_ROOT = M0_DIR.parent.parent

#: Poses designed in simulation. Written by src/M0/simulation, read by both stacks.
POSE_LIBRARY_PATH = M0_DIR / "poses.json"
#: Poses after hardware verification (what the hand actually reached). Written by src/M0/physical.
MEASURED_LIBRARY_PATH = M0_DIR / "poses_measured.json"


@dataclass(frozen=True)
class Finger:
    """One actuated degree of freedom of the Revo2.

    The hand has 6 actuated joints and 11 DOF: every finger's distal joint is
    mechanically coupled to its proximal joint, so you command 6 numbers and the
    linkage resolves the rest. `coupled_joint` is how the simulation reproduces that
    (the real hand does it in hardware).
    """

    name: str
    sdk_index: int                      # position in the bc_stark_sdk 6-vector
    joint: str                          # URDF joint driven directly
    joint_limit: float                  # rad reached at command == 1.0
    coupled_joint: str | None = None
    coupled_multiplier: float = 1.0


FINGERS: list[Finger] = [
    Finger("thumb", 0, "right_thumb_proximal_joint", 1.03, "right_thumb_distal_joint", 1.0),
    Finger("thumb_aux", 1, "right_thumb_metacarpal_joint", 1.57),
    Finger("index", 2, "right_index_proximal_joint", 1.41, "right_index_distal_joint", 1.155),
    Finger("middle", 3, "right_middle_proximal_joint", 1.41, "right_middle_distal_joint", 1.155),
    Finger("ring", 4, "right_ring_proximal_joint", 1.41, "right_ring_distal_joint", 1.155),
    Finger("pinky", 5, "right_pinky_proximal_joint", 1.41, "right_pinky_distal_joint", 1.155),
]
FINGER_NAMES = [f.name for f in FINGERS]
N_FINGERS = len(FINGERS)

#: Per-finger clamp applied on the way to the *hardware* only. Lower an entry if a
#: finger is fouling something; BrainCo's own examples cap the thumb around 0.4-0.5
#: in several demos because the thumb drives two joints at once.
POSITION_CAP = np.ones(N_FINGERS)

#: Fingertip body names in revo2_right_hand.urdf, keyed by finger name.
TIP_BODY = {f.name: f"right_{f.name.replace('_aux', '')}_tip_link" for f in FINGERS}

#: Starting points, not gospel - tune them in simulation and save your own over the top.
PRESETS: dict[str, list[float]] = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "point": [0.6, 0.8, 0.0, 1.0, 1.0, 1.0],
    "pinch": [0.50, 0.80, 0.50, 0.0, 0.0, 0.0],
    "tripod": [0.50, 0.80, 0.50, 0.50, 0.0, 0.0],
    "lego_pinch": [0.54, 0.80, 0.54, 0.0, 0.0, 0.0],
    "power": [0.8, 0.6, 0.9, 0.9, 0.9, 0.9],
}


def as_pose(value) -> np.ndarray:
    """Coerce a scalar / sequence / {finger: value} dict into a clipped length-6 pose.

    >>> as_pose(0.5).round(2)
    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    >>> as_pose({"index": 0.8})[2]
    0.8
    """
    if isinstance(value, dict):
        pose = np.zeros(N_FINGERS)
        for name, v in value.items():
            if name not in FINGER_NAMES:
                raise KeyError(f"unknown finger {name!r}, expected one of {FINGER_NAMES}")
            pose[FINGER_NAMES.index(name)] = v
    elif np.isscalar(value):
        pose = np.full(N_FINGERS, float(value))
    else:
        pose = np.asarray(value, dtype=float).reshape(-1)
        if pose.shape != (N_FINGERS,):
            raise ValueError(f"pose must have {N_FINGERS} entries, got {pose.shape}")
    return np.clip(pose, 0.0, 1.0)


def pose_to_sdk(pose) -> list[int]:
    """Normalized pose -> bc_stark_sdk integer positions (0..1000), safety-capped."""
    return [int(round(v)) for v in np.clip(as_pose(pose), 0.0, POSITION_CAP) * 1000]


def sdk_to_pose(positions) -> np.ndarray:
    """bc_stark_sdk integer positions (0..1000) -> normalized pose."""
    return as_pose(np.asarray(positions, dtype=float) / 1000.0)


def pose_to_joint_angles(pose) -> dict[str, float]:
    """Normalized pose -> {urdf joint name: angle in rad}, coupled distal joints included."""
    pose = as_pose(pose)
    angles: dict[str, float] = {}
    for finger in FINGERS:
        angle = pose[finger.sdk_index] * finger.joint_limit
        angles[finger.joint] = angle
        if finger.coupled_joint is not None:
            angles[finger.coupled_joint] = angle * finger.coupled_multiplier
    return angles


def describe(pose) -> str:
    """One-line human readable pose, command value and resulting joint angle per finger."""
    pose = as_pose(pose)
    return "  ".join(
        f"{f.name}={pose[f.sdk_index]:.2f}({np.rad2deg(pose[f.sdk_index] * f.joint_limit):3.0f}deg)" for f in FINGERS
    )


# ---------------------------------------------------------------------------------------
# Pose library - the handover from simulation to hardware
# ---------------------------------------------------------------------------------------
def load_poses(path: Path = POSE_LIBRARY_PATH, include_presets: bool = True) -> dict[str, np.ndarray]:
    """Named poses from `path`, layered on top of PRESETS (saved poses win)."""
    poses = {name: as_pose(values) for name, values in PRESETS.items()} if include_presets else {}
    if path.exists():
        poses.update({name: as_pose(values) for name, values in json.loads(path.read_text()).items()})
    return poses


def save_pose(name: str, pose, path: Path = POSE_LIBRARY_PATH) -> dict[str, np.ndarray]:
    """Write one named pose into `path` (created if needed) and return the whole library."""
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored[name] = [round(float(v), 3) for v in as_pose(pose)]
    path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    return load_poses(path)


def delete_pose(name: str, path: Path = POSE_LIBRARY_PATH) -> dict[str, np.ndarray]:
    """Remove a saved pose (presets cannot be deleted, only shadowed)."""
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored.pop(name, None)
    path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    return load_poses(path)
