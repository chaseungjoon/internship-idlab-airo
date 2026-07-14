# lego_3d

## Source data (`dat/`)

LDraw-format CAD files (`.dat`) for every part in `lego_list.csv`, mirrored from
the LDraw parts library. See `dat/README.txt` for provenance and the 2 parts
that couldn't be sourced (`3277`, `5846`).

LDraw is a hierarchical CAD format (parts reference subparts and primitives,
in LDU units, Y-down axis) — not directly loadable by PyBullet/Gazebo/ROS,
which expect flat meshes (STL/OBJ) wrapped in URDF with mass/inertia/collision
geometry. `scripts/ldraw2urdf.py` does that conversion.

## Generated assets

Run once (or after editing `lego_list.csv` / re-running the LDraw export):

```
.venv/bin/python scripts/ldraw2urdf.py
```

Produces:
- `meshes/visual/<part>.obj` — flattened, full-detail mesh, one per unique
  LDraw geometry file (64 total). Units: meters. Axes: Z-up, right-handed
  (LDraw's Y-down is rotated -90° about X on export).
- `meshes/collision/<part>_hull.obj` — convex hull of the visual mesh, for
  physics collision.
- `urdf/<number>__<color>.urdf` — one single-link URDF per (part number,
  color) pair from `MANIFEST.csv` (101 total), each with:
  - `<visual>` → full-detail mesh + an approximate LEGO color RGBA
  - `<collision>` → convex-hull mesh
  - `<inertial>` → mass + inertia tensor computed from mesh volume ×
    1050 kg/m³ (ABS density), about the mesh's actual center of mass

Load in PyBullet: `p.loadURDF("urdf/3005__light_bluish_gray.urdf")`.

## Known approximations

- **Mass is convex-hull-based for most parts.** Real LEGO bricks are hollow
  shells (open bottom, internal tubes); the LDraw meshes for these are
  correspondingly non-watertight, so mass/inertia falls back to the convex
  hull (a solid-filled proxy). This typically **overestimates mass by
  ~1.5–3x** for tall/hollow parts (bricks, arches) and is close to accurate
  for flat parts (plates, tiles). If per-part mass accuracy matters later,
  either hand-correct high-value parts or replace the density fallback with
  a shell-thickness estimate.
- **Collision = convex hull**, not the true concave shape. Fine for
  pick-and-place manipulation; insufficient if you need bricks to physically
  interlock via studs/tubes in sim. If you need that, run a proper convex
  decomposition (V-HACD) on `meshes/visual/*.obj` instead of using the hull
  — pybullet's built-in `p.vhacd()` is the usual route, but it failed to
  build from source on this machine (macOS/clang rejects Bullet's legacy
  K&R-style C in its bundled zlib); it builds fine in a Linux/ROS
  environment or via `conda install -c conda-forge pybullet`.
- **Uniform density** (1050 kg/m³, ABS) for all parts, including the
  `Trans-Black` part (actually polycarbonate, ~1200 kg/m³) — negligible for
  manipulation purposes.
- Colors are approximate RGB swatches for the 11 colors used in
  `lego_list.csv`, not exact LDraw/BrickLink palette values.
- `973pb2066c01` and `3626pb1227` (printed minifig parts) use the plain
  unprinted geometry, per `dat/README.txt` — shape is correct, print
  graphic is not modeled.
