from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_MESH_DIR = os.path.join(_REPO_ROOT, "lego_3d", "meshes", "visual")
DEFAULT_URDF_DIR = os.path.join(_REPO_ROOT, "lego_3d", "urdf")
DEFAULT_CACHE_PATH = os.path.join(_REPO_ROOT, "lego_3d", "catalog.json")

STUD_PITCH = 0.008
LEGO_PLAY = 0.0002
GRID_TOLERANCE = 1e-5

STUD_HEIGHT = 0.0016
OBSTRUCTION_WARNING_HEIGHT = 0.002

_URDF_NAME_PATTERN = re.compile(r"^(?P<number>.+?)__(?P<color>.+)\.urdf$")


@dataclass(frozen=True)
class LegoPart:
    """One part number's physical dimensions, in metres."""

    number: str
    width: float  # shorter footprint side -- what the gripper's fingers close on
    length: float  # longer footprint side
    height: float  # top face above the table, resting studs-up
    obstruction: float  # structure above the top face beyond a stud, 0 for a plain brick/plate
    colors: Tuple[str, ...] = ()

    @property
    def footprint(self) -> Tuple[float, float]:
        return (self.width, self.length)

    @property
    def studs(self) -> Tuple[float, float]:
        """Footprint in stud counts, e.g. ``(1.0, 3.0)`` for a 1x3 brick. Useful for logging."""
        return (round(self.width / STUD_PITCH, 1), round(self.length / STUD_PITCH, 1))

    def describe(self) -> str:
        studs = self.studs
        shape = f"{studs[0]:g}x{studs[1]:g}" if max(abs(s - round(s)) for s in studs) < 0.15 else "irregular"
        return (
            f"part {self.number} ({shape}): {self.width * 1000:.1f} x {self.length * 1000:.1f} mm footprint, "
            f"{self.height * 1000:.1f} mm tall"
        )


def _bounding_box(obj_path: str) -> Optional[Tuple[List[float], List[float]]]:
    """Axis-aligned bounds of an OBJ's vertices, as ``(min, max)`` triples in metres."""
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    with open(obj_path) as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            for axis in range(3):
                value = float(parts[axis + 1])
                low[axis] = min(low[axis], value)
                high[axis] = max(high[axis], value)
    if low[0] == float("inf"):
        return None
    return low, high


def _physical_extent(nominal: float) -> float:
    """A bounding-box extent turned into the size the part physically measures (see LEGO_PLAY)."""
    studs = nominal / STUD_PITCH
    if abs(studs - round(studs)) < GRID_TOLERANCE and round(studs) >= 1:
        return nominal - LEGO_PLAY
    return nominal


def _colors_by_part(urdf_dir: str) -> Dict[str, Tuple[str, ...]]:
    """``{part number: (colour, ...)}`` from the ``<number>__<colour>.urdf`` filenames."""
    colors: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(urdf_dir, "*.urdf"))):
        match = _URDF_NAME_PATTERN.match(os.path.basename(path))
        if match:
            colors.setdefault(match.group("number"), []).append(match.group("color"))
    return {number: tuple(sorted(set(found))) for number, found in colors.items()}


def build_catalog(mesh_dir: str = DEFAULT_MESH_DIR, urdf_dir: str = DEFAULT_URDF_DIR) -> Dict[str, LegoPart]:
    """Measure every mesh in ``mesh_dir`` into a ``{part number: LegoPart}`` catalog."""
    colors = _colors_by_part(urdf_dir) if os.path.isdir(urdf_dir) else {}
    catalog: Dict[str, LegoPart] = {}

    for path in sorted(glob.glob(os.path.join(mesh_dir, "*.obj"))):
        number = os.path.splitext(os.path.basename(path))[0]
        bounds = _bounding_box(path)
        if bounds is None:
            logger.debug(f"Skipping {path}: no vertices.")
            continue
        low, high = bounds

        footprint = sorted(_physical_extent(high[axis] - low[axis]) for axis in (0, 1))
        # LDraw puts z = 0 at the top of a plate/brick body, with the studs above it. A part whose
        # mesh sits entirely at or above z = 0 is modelled resting on that plane instead, so its top
        # face is simply its full height.
        top_face_z = 0.0 if low[2] < 0.0 else high[2]
        height = top_face_z - low[2]
        obstruction = max(0.0, high[2] - top_face_z - STUD_HEIGHT)

        if height <= 0.0 or footprint[0] <= 0.0:
            logger.debug(f"Skipping {number}: degenerate bounding box {low} .. {high}.")
            continue

        catalog[number] = LegoPart(
            number=number,
            width=footprint[0],
            length=footprint[1],
            height=height,
            obstruction=obstruction,
            colors=colors.get(number, ()),
        )

    logger.debug(f"Measured {len(catalog)} part(s) from {mesh_dir}.")
    return catalog


def load_catalog(
    mesh_dir: str = DEFAULT_MESH_DIR,
    urdf_dir: str = DEFAULT_URDF_DIR,
    cache_path: Optional[str] = DEFAULT_CACHE_PATH,
) -> Dict[str, LegoPart]:
    """The catalog, from ``cache_path`` if it is newer than the meshes, otherwise measured afresh.

    Parsing 64 meshes takes a fraction of a second, so the cache is a convenience rather than a
    necessity -- but it also makes the catalog inspectable as plain JSON, which is worth having when
    a match goes wrong and the question is what the pipeline *thinks* a part measures.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            newest_mesh = max(os.path.getmtime(p) for p in glob.glob(os.path.join(mesh_dir, "*.obj")))
            if os.path.getmtime(cache_path) >= newest_mesh:
                with open(cache_path) as f:
                    cached = json.load(f)
                return {
                    number: LegoPart(**{**entry, "colors": tuple(entry.get("colors", ()))})
                    for number, entry in cached.items()
                }
        except Exception as exception:  # noqa: BLE001 - a stale/corrupt cache is never fatal
            logger.debug(f"Ignoring the part catalog cache at {cache_path}: {exception}")

    catalog = build_catalog(mesh_dir, urdf_dir)
    if cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump({number: asdict(part) for number, part in catalog.items()}, f, indent=2, sort_keys=True)
        except OSError as exception:
            logger.debug(f"Could not write the part catalog cache to {cache_path}: {exception}")
    return catalog


@dataclass(frozen=True)
class FootprintMatch:
    """A catalog part consistent with a measured footprint, and how well it fits."""

    part: LegoPart
    width_error: float  # metres, measured - catalog
    length_error: float

    @property
    def residual(self) -> float:
        return max(abs(self.width_error), abs(self.length_error))


def match_footprint(
    catalog: Dict[str, LegoPart],
    width: float,
    length: float,
    tolerance: float,
    restrict_to: Optional[Sequence[str]] = None,
) -> List[FootprintMatch]:
    """Every catalog part whose footprint is within ``tolerance`` of ``width`` x ``length``.

    Sorted by how *short* the candidate is, then by residual. Height is the tie-break, and shortest
    wins on purpose: the grasp descends to ``table + height - grasp_depth``, so overestimating the
    height aims the fingers above the brick and misses, while underestimating grips lower down a
    brick that is really there -- and the descent is separately capped so it can never reach the
    table. Between two parts with the same footprint, missing low is the recoverable mistake.

    Args:
        catalog: from :func:`load_catalog`.
        width: measured shorter footprint side, metres.
        length: measured longer footprint side, metres.
        tolerance: largest per-dimension error accepted, metres.
        restrict_to: only consider these part numbers (e.g. the ones in ``lego_list.csv``).
    """
    matches = []
    for number, part in catalog.items():
        if restrict_to is not None and number not in restrict_to:
            continue
        width_error = width - part.width
        length_error = length - part.length
        if abs(width_error) <= tolerance and abs(length_error) <= tolerance:
            matches.append(FootprintMatch(part=part, width_error=width_error, length_error=length_error))
    return sorted(matches, key=lambda match: (match.part.height, match.residual))


def parts_in_set(csv_path: str = os.path.join(_REPO_ROOT, "lego_list.csv")) -> Tuple[str, ...]:
    """Part numbers listed in ``lego_list.csv``, to restrict matching to what is actually on the table."""
    numbers = []
    try:
        with open(csv_path) as f:
            next(f, None)  # header
            for line in f:
                number = line.split(",")[0].strip()
                if number:
                    numbers.append(number)
    except OSError as exception:
        logger.debug(f"Could not read {csv_path}: {exception}")
        return ()
    return tuple(sorted(set(numbers)))


if __name__ == "__main__":
    catalog = load_catalog()
    for part in sorted(catalog.values(), key=lambda p: (p.width, p.length, p.height)):
        flag = f"  [+{part.obstruction * 1000:.1f} mm above the top face]" if part.obstruction else ""
        print(f"{part.describe()}{flag}")
    print(f"\n{len(catalog)} parts; lego_list.csv names {len(parts_in_set())} distinct numbers.")
