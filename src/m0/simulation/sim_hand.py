"""M0 simulation stack: Drake + Meshcat model of the BrainCo Revo2 right hand.

Tech stack, in full: pydrake (MultibodyPlant / RobotDiagram) for kinematics, Meshcat for
the 3D view, ipywidgets for the sliders. Nothing here talks to hardware - the physical
stack lives in src/M0/physical/real_hand.py and shares only the normalized pose vector
defined in src/M0/hand_model.py.

This is a *kinematic* model: joint angles are set directly on the context and published,
there is no physics stepping. That is what you want for pose-finding (instant response to
a slider); contact simulation for grasping comes in M1.
"""

from __future__ import annotations

import html
import sys
import traceback
from pathlib import Path

import numpy as np
from airo_drake import finish_build
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.planning import RobotDiagramBuilder

M0_DIR = Path(__file__).resolve().parent.parent
if str(M0_DIR) not in sys.path:
    sys.path.insert(0, str(M0_DIR))
if str(M0_DIR.parent) not in sys.path:
    sys.path.insert(0, str(M0_DIR.parent))  # src/, for scene.py

from hand_model import (  # noqa: E402
    FINGERS,
    N_FINGERS,
    POSE_LIBRARY_PATH,
    REPO_ROOT,
    TIP_BODY,
    as_pose,
    load_poses,
    pose_to_joint_angles,
    save_pose,
)
from scene import GRIPPER_URDF  # noqa: E402

#: Where the hand is welded in the world. Fingers point along +z, the palm faces +x.
DEFAULT_MOUNT = RigidTransform(RotationMatrix.Identity(), [0.0, 0.0, 0.0])

#: A 1x1 brick (8 x 8 x 9.6 mm). Any file from lego_3d/urdf/ works.
DEFAULT_BRICK = REPO_ROOT / "lego_3d" / "urdf" / "3005__light_bluish_gray.urdf"

_MESHCAT: Meshcat | None = None


def launch_meshcat() -> Meshcat:
    """The one Meshcat server for this kernel.

    Re-using it keeps the port stable across cell re-runs - a fresh Meshcat() would grab
    the next free port (7001, 7002, ...) and leave you watching a stale browser tab.
    """
    global _MESHCAT
    if _MESHCAT is None:
        _MESHCAT = Meshcat()
        print(f"Meshcat started at {_MESHCAT.web_url()}")
    return _MESHCAT


class SimHand:
    """Kinematic Drake/Meshcat model of the Revo2 right hand."""

    def __init__(
        self,
        meshcat: Meshcat | None = None,
        mount: RigidTransform = DEFAULT_MOUNT,
        brick_urdf: Path | None = None,
    ):
        self.meshcat = meshcat if meshcat is not None else launch_meshcat()
        self.meshcat.Delete()
        self.meshcat.DeleteAddedControls()

        builder = RobotDiagramBuilder()
        self.plant = builder.plant()
        parser = builder.parser()
        parser.SetAutoRenaming(True)

        self.hand_index = parser.AddModels(str(GRIPPER_URDF))[0]
        self.plant.WeldFrames(
            self.plant.world_frame(), self.plant.GetFrameByName("right_base_link", self.hand_index), mount
        )
        self.brick_index = parser.AddModels(str(brick_urdf))[0] if brick_urdf is not None else None

        self.diagram, self.context = finish_build(builder, self.meshcat)
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.pose = np.zeros(N_FINGERS)
        self.set_pose(self.pose)

    # -- commanding --------------------------------------------------------------------
    def set_pose(self, pose) -> np.ndarray:
        """Apply a normalized pose (coupled distal joints included) and publish it."""
        self.pose = as_pose(pose)
        for joint_name, angle in pose_to_joint_angles(self.pose).items():
            self.plant.GetJointByName(joint_name, self.hand_index).set_angle(self.plant_context, angle)
        self.publish()
        return self.pose

    def publish(self) -> None:
        self.diagram.ForcedPublish(self.context)

    def refresh(self) -> None:
        """Re-send the whole scene to Meshcat.

        Use this if the browser shows an empty or stale scene - for instance when the tab
        was opened before the model was built, or after a Meshcat.Delete().
        """
        self.diagram.ExecuteInitializationEvents(self.context)
        self.publish()

    # -- reading -----------------------------------------------------------------------
    def fingertip_position(self, finger: str) -> np.ndarray:
        """World position of a fingertip link origin, in metres."""
        body = self.plant.GetBodyByName(TIP_BODY[finger], self.hand_index)
        return self.plant.EvalBodyPoseInWorld(self.plant_context, body).translation()

    def gap_mm(self, a: str = "thumb", b: str = "index") -> float:
        """Distance between two fingertips, in mm.

        Measured between tip-link *origins*, not the pad surfaces, so the fingers already
        touch at roughly 15 mm - treat it as a relative measure and confirm in Meshcat.
        """
        return float(np.linalg.norm(self.fingertip_position(a) - self.fingertip_position(b)) * 1000.0)

    def joint_angles(self, degrees: bool = True) -> dict[str, float]:
        angles = pose_to_joint_angles(self.pose)
        return {k: (np.rad2deg(v) if degrees else v) for k, v in angles.items()}

    # -- the brick ---------------------------------------------------------------------
    def place_brick_between(self, a: str = "thumb", b: str = "index") -> np.ndarray | None:
        """Park the brick halfway between two fingertips so you can eyeball a pinch.

        Nothing is simulated here: the brick stays where you put it instead of being
        gripped, because there is no physics stepping.
        """
        if self.brick_index is None:
            return None
        body = self.plant.get_body(self.plant.GetBodyIndices(self.brick_index)[0])
        midpoint = 0.5 * (self.fingertip_position(a) + self.fingertip_position(b))
        self.plant.SetFreeBodyPose(self.plant_context, body, RigidTransform(RotationMatrix.Identity(), midpoint))
        self.publish()
        return midpoint


def aperture_curve(sim: SimHand, thumb_aux: float = 0.8, samples: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thumb+index closed together, returning (closure commands, fingertip gaps mm).

    The curve is not monotonic: past the minimum the fingertips swing past each other.
    """
    closures = np.linspace(0.0, 1.0, samples)
    gaps = np.empty_like(closures)
    restore = sim.pose.copy()
    for i, c in enumerate(closures):
        sim.set_pose({"thumb": c, "thumb_aux": thumb_aux, "index": c})
        gaps[i] = sim.gap_mm()
    sim.set_pose(restore)
    return closures, gaps


def closure_for_gap(sim: SimHand, gap_mm: float, thumb_aux: float = 0.8, samples: int = 201) -> float | None:
    """Smallest thumb/index closure whose fingertip gap reaches `gap_mm`.

    Returns None if the hand cannot close that far at this thumb abduction.
    """
    closures, gaps = aperture_curve(sim, thumb_aux=thumb_aux, samples=samples)
    closing = gaps[: int(np.argmin(gaps)) + 1]   # monotonically decreasing branch
    reached = np.nonzero(closing <= gap_mm)[0]
    return float(closures[reached[0]]) if reached.size else None


# ---------------------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------------------
class SimHandUI:
    """Sliders driving a SimHand, plus the pose library.

    Every callback is wrapped so that exceptions show up in the panel instead of being
    swallowed by ipywidgets, and an update counter proves the callbacks are firing at all.
    Build this in the *same cell* as the SimHand it drives: re-running a cell that rebuilds
    the sim while an older UI still holds the previous one is the classic way to end up
    with sliders that move nothing.
    """

    def __init__(self, sim: SimHand, library_path: Path = POSE_LIBRARY_PATH, place_brick: bool = False):
        import ipywidgets as widgets

        self.w = widgets
        self.sim = sim
        self.library_path = library_path
        self.place_brick = place_brick
        self.poses = load_poses(library_path)
        self.updates = 0
        self._suspend = False
        self.listeners: list = []   # callables invoked with the new pose

        self.sliders = [
            widgets.FloatSlider(
                value=float(sim.pose[f.sdk_index]), min=0.0, max=1.0, step=0.01, description=f.name,
                continuous_update=True, readout_format=".2f",
                style={"description_width": "80px"}, layout=widgets.Layout(width="360px"),
            )
            for f in FINGERS
        ]
        for slider in self.sliders:
            slider.observe(self._guard(self._on_slider), names="value")

        self.status = widgets.HTML()
        self.error = widgets.HTML()
        self.log = widgets.HTML()
        self.preset = widgets.Dropdown(
            options=sorted(self.poses), value="open" if "open" in self.poses else None, description="pose",
            style={"description_width": "80px"}, layout=widgets.Layout(width="250px"),
        )
        self.name_box = widgets.Text(
            value="my_pose", description="save as",
            style={"description_width": "80px"}, layout=widgets.Layout(width="250px"),
        )
        link = widgets.HTML(
            f'<a href="{sim.meshcat.web_url()}" target="_blank">open Meshcat &rarr; {sim.meshcat.web_url()}</a>'
            "<br><small>if the hand does not move, make sure you are watching <i>this</i> URL</small>"
        )

        buttons = widgets.HBox([
            self._button("Open", lambda: self.set_pose(0.0), button_style="success"),
            self._button("Fist", lambda: self.set_pose(1.0)),
            self._button("Apply", self._apply_preset),
            self._button("Save", self._save),
            self._button("Brick to pinch", self._brick),
            self._button("Refresh view", self.sim.refresh),
        ])

        self.box = widgets.VBox([
            widgets.HTML("<h4>Revo2 right hand &mdash; simulation</h4>"),
            link,
            widgets.HBox([
                widgets.VBox(self.sliders),
                widgets.VBox([self.status], layout=widgets.Layout(padding="0 0 0 24px")),
            ]),
            widgets.HBox([self.preset, self.name_box]),
            buttons,
            self.log,
            self.error,
        ])
        self._refresh_status()

    # -- plumbing ----------------------------------------------------------------------
    def _guard(self, fn):
        """Run `fn`, showing any traceback in the panel rather than losing it."""

        def wrapped(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                self.error.value = ""
                return result
            except Exception:
                self.error.value = f"<pre style='color:#c00'>{html.escape(traceback.format_exc())}</pre>"

        return wrapped

    def _button(self, description, handler, **kwargs):
        button = self.w.Button(description=description, layout=self.w.Layout(width="130px"), **kwargs)
        button.on_click(self._guard(lambda _: handler()))
        return button

    def _on_slider(self, _change=None) -> None:
        if self._suspend:
            return
        pose = np.array([s.value for s in self.sliders])
        self.sim.set_pose(pose)
        if self.place_brick:
            self.sim.place_brick_between()
        self.updates += 1
        for listener in self.listeners:
            listener(pose)
        self._refresh_status()

    def _refresh_status(self) -> None:
        rows = [
            f"<b>pose</b> {np.array2string(self.sim.pose, precision=2)}",
            f"<b>thumb&ndash;index gap</b> {self.sim.gap_mm():.1f} mm",
            f"<b>thumb&ndash;middle gap</b> {self.sim.gap_mm('thumb', 'middle'):.1f} mm",
            f"<small>slider updates: {self.updates}</small>",
        ]
        self.status.value = "<br>".join(rows)

    # -- actions -----------------------------------------------------------------------
    def set_pose(self, pose) -> np.ndarray:
        """Move the sliders (and therefore the hand) to `pose`."""
        pose = as_pose(pose)
        self._suspend = True
        try:
            for slider, value in zip(self.sliders, pose):
                slider.value = float(value)
        finally:
            self._suspend = False
        self._on_slider()
        return pose

    def _apply_preset(self) -> None:
        if self.preset.value:
            self.set_pose(self.poses[self.preset.value])
            self.log.value = f"applied <b>{self.preset.value}</b>"

    def _save(self) -> None:
        name = self.name_box.value.strip()
        if not name:
            self.log.value = "<span style='color:#c00'>give the pose a name first</span>"
            return
        self.poses = save_pose(name, self.pose, self.library_path)
        self.preset.options = sorted(self.poses)
        self.preset.value = name
        self.log.value = f"saved <b>{name}</b> to {self.library_path.name}"

    def _brick(self) -> None:
        if self.sim.place_brick_between() is None:
            self.log.value = "no brick in the scene - build SimHand with brick_urdf=DEFAULT_BRICK"
        else:
            self.log.value = "brick moved to the thumb&ndash;index midpoint"

    @property
    def pose(self) -> np.ndarray:
        return self.sim.pose

    def _ipython_display_(self):
        from IPython.display import display

        display(self.box)
