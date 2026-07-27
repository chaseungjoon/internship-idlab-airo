"""M0 physical stack: the real BrainCo Revo2 right hand over RS-485 (Modbus-RTU).

Tech stack, in full: bc_stark_sdk (BrainCo's async Rust/Python SDK) on a serial port,
asyncio on a private background thread, ipywidgets for the sliders. Nothing here imports
pydrake - the simulation stack lives in src/M0/simulation/sim_hand.py and shares only the
normalized pose vector defined in src/M0/hand_model.py.

Install on the robot PC:

    pip install bc-stark-sdk
    ls /dev/ttyUSB*                   # find the RS-485 adapter
    sudo usermod -aG dialout $USER    # log out/in afterwards

Why the background thread: bc_stark_sdk is async, Jupyter already owns the main event
loop, and ipywidgets callbacks are synchronous. RealHand therefore runs its own event loop
on a daemon thread and exposes plain blocking methods that widget callbacks can call.

API reference: https://github.com/BrainCoTech/brainco-hand-sdk (python/revo2/)
"""

from __future__ import annotations

import asyncio
import html
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

M0_DIR = Path(__file__).resolve().parent.parent
if str(M0_DIR) not in sys.path:
    sys.path.insert(0, str(M0_DIR))

from hand_model import (  # noqa: E402
    FINGERS,
    MEASURED_LIBRARY_PATH,
    N_FINGERS,
    POSE_LIBRARY_PATH,
    as_pose,
    load_poses,
    pose_to_sdk,
    save_pose,
    sdk_to_pose,
)

try:
    from bc_stark_sdk import main_mod as libstark
except ImportError:  # sim-only machine
    libstark = None

#: BrainCo defaults: 0x7F (127) right hand, 0x7E (126) left hand.
RIGHT_HAND_ID = 0x7F
LEFT_HAND_ID = 0x7E

_BAUDRATE_NAMES = {
    19200: "Baud19200", 57600: "Baud57600", 115200: "Baud115200", 460800: "Baud460800",
    1000000: "Baud1Mbps", 2000000: "Baud2Mbps", 3000000: "Baud3Mbps",
    4000000: "Baud4Mbps", 5000000: "Baud5Mbps", 6000000: "Baud6Mbps",
}


def require_sdk() -> None:
    if libstark is None:
        raise RuntimeError("bc_stark_sdk is not installed on this machine; pip install bc-stark-sdk")


def baudrate_enum(value):
    """int bps -> libstark.Baudrate enum (passed through unchanged if already an enum)."""
    if not isinstance(value, int):
        return value
    if value not in _BAUDRATE_NAMES:
        raise ValueError(f"unsupported baudrate {value}, expected one of {sorted(_BAUDRATE_NAMES)}")
    return getattr(libstark.Baudrate, _BAUDRATE_NAMES[value])


class RealHand:
    """Blocking wrapper around the async bc_stark_sdk Modbus client.

    Commanding is done by a *streaming* thread that re-sends the current target at
    `stream_hz`, which is the pattern BrainCo's own examples use: the last value of a
    slider drag always lands, and the hand keeps holding the pose afterwards.

    `armed` is the master switch. While it is False the streaming thread sends nothing,
    so you can move the sliders around without the hardware following.
    """

    def __init__(
        self,
        port: str | None = None,
        slave_id: int = RIGHT_HAND_ID,
        baudrate: int = 460800,
        speed: int = 400,
        stream_hz: float = 50.0,
        status_every: int = 10,
    ):
        require_sdk()
        self.port, self.slave_id, self.baudrate = port, slave_id, baudrate
        self.speed = speed
        self.stream_hz = stream_hz
        self.status_every = status_every

        self.client = None
        self.info = None
        self.armed = False
        self.target = np.zeros(N_FINGERS)
        self.last_status: dict | None = None
        self.last_error: str | None = None
        self.sent_frames = 0

        self._io_lock = threading.Lock()      # one Modbus transaction at a time
        self._target_lock = threading.Lock()
        self._stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="revo2-sdk-loop", daemon=True)
        self._loop_thread.start()

    # -- event loop plumbing -----------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _await(self, coro, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # -- lifecycle ---------------------------------------------------------------------
    def connect(self) -> "RealHand":
        self._await(self._connect(), timeout=60.0)
        self.start_streaming()
        return self

    async def _connect(self) -> None:
        try:
            libstark.init_logging()
        except Exception:
            pass

        if self.port is None:
            protocol, port, baudrate, slave_id = await libstark.auto_detect_modbus_revo2(None, True)
            assert protocol == libstark.StarkProtocolType.Modbus, f"expected Modbus, detected {protocol}"
            self.port, self.baudrate, self.slave_id = port, baudrate, slave_id
            print(f"auto-detected {port} @ {baudrate}, slave id 0x{slave_id:02X}")
        else:
            self.baudrate = baudrate_enum(self.baudrate)

        self.client = await libstark.modbus_open(self.port, self.baudrate)
        for label, call in [
            ("hardware type", lambda: self.client.set_hardware_type(self.slave_id, libstark.StarkHardwareType.Revo2Basic)),
            ("unit mode", lambda: self.client.set_finger_unit_mode(self.slave_id, libstark.FingerUnitMode.Normalized)),
        ]:
            try:
                await call()
            except Exception as exc:  # older firmware / SDK builds
                print(f"could not set {label}: {exc}")

        self.info = await self.client.get_device_info(self.slave_id)
        print(f"connected: {self.info.description}")

    def close(self, open_hand: bool = True) -> None:
        """Open the hand, stop streaming and release the serial port."""
        if open_hand and self.armed and self.client is not None:
            try:
                self.move_to(0.0, duration=1.0)
            except Exception as exc:
                print(f"could not open the hand before closing: {exc}")
        self.armed = False
        self.stop_streaming()
        if self.client is not None:
            try:
                self._await(libstark.modbus_close(self.client))
            except Exception as exc:
                print(f"modbus_close failed: {exc}")
            finally:
                self.client = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)

    @property
    def connected(self) -> bool:
        return self.client is not None

    # -- raw I/O (blocking, serialized) ------------------------------------------------
    def send(self, pose, speed: int | None = None) -> list[int]:
        """Write one position command. Ignores `armed` - this is the raw call."""
        positions = pose_to_sdk(pose)
        speeds = [int(self.speed if speed is None else speed)] * N_FINGERS
        with self._io_lock:
            self._await(self.client.set_finger_positions_and_speeds(self.slave_id, positions, speeds))
        return positions

    def status(self) -> dict:
        """Measured positions (0..1000), speeds and currents (mA)."""
        with self._io_lock:
            s = self._await(self.client.get_motor_status(self.slave_id))
        return {
            "positions": list(s.positions),
            "speeds": list(getattr(s, "speeds", [])),
            "currents": list(getattr(s, "currents", [])),
        }

    def read_pose(self) -> np.ndarray:
        return sdk_to_pose(self.status()["positions"])

    def currents(self) -> np.ndarray:
        return np.abs(np.asarray(self.status()["currents"], dtype=float))

    # -- streaming ---------------------------------------------------------------------
    def start_streaming(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stop.clear()
        self._stream_thread = threading.Thread(target=self._stream, name="revo2-stream", daemon=True)
        self._stream_thread.start()

    def stop_streaming(self) -> None:
        self._stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

    def _stream(self) -> None:
        period = 1.0 / self.stream_hz
        frame = 0
        while not self._stop.is_set():
            if self.armed and self.client is not None:
                with self._target_lock:
                    target = self.target.copy()
                try:
                    self.send(target)
                    self.sent_frames += 1
                    if frame % self.status_every == 0:
                        self.last_status = self.status()
                    self.last_error = None
                except Exception:
                    self.last_error = traceback.format_exc(limit=2)
            frame += 1
            time.sleep(period)

    # -- commanding --------------------------------------------------------------------
    def set_target(self, pose) -> np.ndarray:
        """Set the pose the streaming thread holds. No effect while `armed` is False."""
        pose = as_pose(pose)
        with self._target_lock:
            self.target = pose
        return pose

    def move_to(self, pose, duration: float = 1.5, steps: int = 40) -> np.ndarray:
        """Ramp the target from where it is to `pose` over `duration` seconds."""
        start = self.target.copy()
        end = as_pose(pose)
        steps = max(int(steps), 2)
        for alpha in np.linspace(0.0, 1.0, steps):
            self.set_target(start + alpha * (end - start))
            time.sleep(duration / steps)
        return self.target

    def play(self, sequence, hold: float = 0.6, duration: float = 1.2, verbose: bool = True) -> None:
        """Run a list of poses, or of (name, pose) pairs, back to back."""
        for item in sequence:
            name, pose = item if isinstance(item, tuple) else (None, item)
            if name and verbose:
                print(f"-> {name}")
            self.move_to(pose, duration=duration)
            time.sleep(hold)

    def open_hand(self, duration: float = 1.0) -> np.ndarray:
        return self.move_to(0.0, duration=duration)

    # -- measurement -------------------------------------------------------------------
    def hold_and_measure(self, pose, duration: float = 1.2, settle: float = 0.6) -> dict:
        """Move to `pose`, let it settle, and report commanded vs measured.

        The per-finger error is the interesting number: a finger that stops short of its
        command while drawing current has hit something (or is fighting the linkage).
        """
        commanded = self.move_to(pose, duration=duration)
        time.sleep(settle)
        status = self.status()
        measured = sdk_to_pose(status["positions"])
        return {
            "commanded": commanded,
            "measured": measured,
            "error": measured - commanded,
            "currents_mA": np.abs(np.asarray(status["currents"], dtype=float)),
            "positions_raw": status["positions"],
        }

    def close_until_contact(
        self,
        fingers: tuple[str, ...] = ("thumb", "index"),
        base_pose=None,
        current_mA: float = 250.0,
        step: float = 0.02,
        max_closure: float = 0.85,
        settle: float = 0.12,
    ) -> dict:
        """Close `fingers` in small steps until any of them draws more than `current_mA`.

        This is the hardware answer to the simulated aperture sweep: it tells you the
        closure command at which the hand is actually gripping the object in front of it,
        which is what M1 needs. Stops at `max_closure` regardless, and leaves the hand
        holding whatever it found - call `open_hand()` afterwards.
        """
        if not self.armed:
            raise RuntimeError("hand is not armed; set hand.armed = True first")
        indices = [FINGERS[[f.name for f in FINGERS].index(name)].sdk_index for name in fingers]
        pose = as_pose(0.0 if base_pose is None else base_pose)

        trace = []
        closure = float(max(pose[i] for i in indices))
        while closure <= max_closure:
            for i in indices:
                pose[i] = closure
            self.set_target(pose)
            time.sleep(settle)
            currents = self.currents()
            peak = float(np.max(currents[indices])) if len(currents) else 0.0
            trace.append({"closure": closure, "peak_mA": peak, "currents_mA": currents})
            if peak >= current_mA:
                return {"contact": True, "closure": closure, "peak_mA": peak, "trace": trace,
                        "pose": self.target.copy()}
            closure = round(closure + step, 4)

        return {"contact": False, "closure": closure, "peak_mA": trace[-1]["peak_mA"] if trace else 0.0,
                "trace": trace, "pose": self.target.copy()}


# ---------------------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------------------
class RealHandUI:
    """Sliders driving a RealHand, with a live readout of what the hand actually did.

    Every callback is wrapped so exceptions show up in the panel instead of being swallowed
    by ipywidgets, and the counters (slider updates / frames sent) tell you immediately
    whether the problem is the UI, the arming switch, or the serial link.
    """

    def __init__(
        self,
        hand: RealHand,
        library_path: Path = POSE_LIBRARY_PATH,
        measured_path: Path = MEASURED_LIBRARY_PATH,
        monitor_hz: float = 2.5,
    ):
        import ipywidgets as widgets

        self.w = widgets
        self.hand = hand
        self.library_path = library_path
        self.measured_path = measured_path
        self.poses = load_poses(library_path)
        self.updates = 0
        self._suspend = False

        self.sliders = [
            widgets.FloatSlider(
                value=float(hand.target[f.sdk_index]), min=0.0, max=1.0, step=0.01, description=f.name,
                continuous_update=True, readout_format=".2f",
                style={"description_width": "80px"}, layout=widgets.Layout(width="360px"),
            )
            for f in FINGERS
        ]
        for slider in self.sliders:
            slider.observe(self._guard(self._on_slider), names="value")

        self.speed = widgets.IntSlider(
            value=hand.speed, min=50, max=1000, step=50, description="speed",
            style={"description_width": "80px"}, layout=widgets.Layout(width="360px"),
        )
        self.speed.observe(self._guard(lambda ch: setattr(hand, "speed", int(ch["new"]))), names="value")

        self.armed = widgets.ToggleButton(
            value=False, description="ARMED - hand will move", icon="bolt",
            button_style="danger", layout=widgets.Layout(width="230px"),
        )
        self.armed.observe(self._guard(self._on_arm), names="value")

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

        buttons = widgets.HBox([
            self._button("OPEN (stop)", self._panic, button_style="success"),
            self._button("Fist", lambda: self.set_pose(1.0)),
            self._button("Apply", self._apply_preset),
            self._button("Read hand", self._read),
            self._button("Save measured", self._save_measured),
        ])

        self.box = widgets.VBox([
            widgets.HTML("<h4>Revo2 right hand &mdash; hardware</h4>"),
            widgets.HBox([
                widgets.VBox(self.sliders + [self.speed]),
                widgets.VBox([self.status], layout=widgets.Layout(padding="0 0 0 24px")),
            ]),
            widgets.HBox([self.preset, self.name_box, self.armed]),
            buttons,
            self.log,
            self.error,
        ])

        self._monitor_period = 1.0 / monitor_hz
        threading.Thread(target=self._monitor, name="revo2-readout", daemon=True).start()
        self._refresh_status()

    # -- plumbing ----------------------------------------------------------------------
    def _guard(self, fn):
        def wrapped(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                self.error.value = ""
                return result
            except Exception:
                self.error.value = f"<pre style='color:#c00'>{html.escape(traceback.format_exc())}</pre>"

        return wrapped

    def _button(self, description, handler, **kwargs):
        button = self.w.Button(description=description, layout=self.w.Layout(width="150px"), **kwargs)
        button.on_click(self._guard(lambda _: handler()))
        return button

    def _on_slider(self, _change=None) -> None:
        if self._suspend:
            return
        self.hand.set_target([s.value for s in self.sliders])
        self.updates += 1
        self._refresh_status()

    def _on_arm(self, change) -> None:
        armed = bool(change["new"])
        if armed and not self.hand.connected:
            self.armed.value = False
            self.log.value = "<span style='color:#c00'>not connected to the hand</span>"
            return
        self.hand.armed = armed
        self.armed.description = "ARMED - hand will move" if armed else "disarmed (safe)"
        self.armed.button_style = "danger" if armed else ""
        self.log.value = "<b>streaming to hardware</b>" if armed else "disarmed; the hand holds its last pose"

    def _monitor(self) -> None:
        """Cheap UI refresh loop. Reads only cached values - no serial I/O from here."""
        while True:
            try:
                self._refresh_status()
            except Exception:
                pass
            time.sleep(self._monitor_period)

    def _refresh_status(self) -> None:
        hand = self.hand
        rows = [f"<b>target</b> {np.array2string(hand.target, precision=2)}"]
        status = hand.last_status
        if status:
            measured = np.asarray(status["positions"], dtype=float) / 1000.0
            rows.append(f"<b>measured</b> {np.array2string(measured, precision=2)}")
            if status["currents"]:
                currents = np.abs(np.asarray(status["currents"], dtype=float))
                hot = " &larr; loaded" if currents.max() > 250 else ""
                rows.append(f"<b>current mA</b> {np.array2string(currents, precision=0)}{hot}")
        rows.append(
            f"<small>{'connected' if hand.connected else 'NOT connected'} &middot; "
            f"{'ARMED' if hand.armed else 'disarmed'} &middot; "
            f"slider updates: {self.updates} &middot; frames sent: {hand.sent_frames}</small>"
        )
        if hand.last_error:
            rows.append(f"<pre style='color:#c00'>{html.escape(hand.last_error)}</pre>")
        self.status.value = "<br>".join(rows)

    # -- actions -----------------------------------------------------------------------
    def set_pose(self, pose) -> np.ndarray:
        pose = as_pose(pose)
        self._suspend = True
        try:
            for slider, value in zip(self.sliders, pose):
                slider.value = float(value)
        finally:
            self._suspend = False
        self._on_slider()
        return pose

    def _panic(self) -> None:
        self.set_pose(0.0)
        self.log.value = "opening"

    def _apply_preset(self) -> None:
        if self.preset.value:
            self.set_pose(self.poses[self.preset.value])
            self.log.value = f"applied <b>{self.preset.value}</b>"

    def _read(self) -> None:
        if not self.hand.connected:
            self.log.value = "<span style='color:#c00'>not connected to the hand</span>"
            return
        self.set_pose(self.hand.read_pose())
        self.log.value = "sliders synced to the measured positions"

    def _save_measured(self) -> None:
        """Save what the hand actually reached, not what we asked for."""
        name = self.name_box.value.strip()
        if not name:
            self.log.value = "<span style='color:#c00'>give the pose a name first</span>"
            return
        pose = self.hand.read_pose() if self.hand.connected else self.hand.target
        save_pose(name, pose, self.measured_path)
        self.log.value = f"saved measured <b>{name}</b> to {self.measured_path.name}"

    def _ipython_display_(self):
        from IPython.display import display

        display(self.box)
