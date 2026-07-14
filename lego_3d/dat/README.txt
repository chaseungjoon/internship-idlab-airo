LEGO 3D Parts (LDraw format) — generated for lego_list.csv
============================================================

Source: LDraw parts library (community-maintained, CAD-precision geometry
for LEGO parts), mirrored from https://github.com/pybricks/ldraw

Contents:
- parts/        : the requested part geometry files (.dat), one per part number
- parts/s/      : subpart files referenced by the parts above
- p/, p/48/     : primitive geometry (studs, cylinders, edges, etc.) referenced
                  by the parts and subparts above
- MANIFEST.csv  : your original list (number, quantity, color) with the matching
                  LDraw filename and any notes

Notes on what's included:
- 64 of your 66 unique part numbers were found directly.
- 2 printed/decorated variants were NOT in the LDraw library under their exact
  BrickLink numbers, so the PLAIN (unprinted) geometry was substituted:
    * 973pb2066c01 (minifig torso, specific print) -> 973.dat (plain torso)
    * 3626pb1227 (minifig head, specific print)     -> 3626.dat (plain head)
  Shape/dimensions are identical; only the printed graphic is missing, which
  doesn't matter for grasp/shape identification purposes.
- 2 part numbers were not found in this LDraw snapshot at all and are NOT
  included:
    * 3277  - Minifigure Hair, Short Wavy Parted on Right
    * 5846  - Brick 2x2x1 with Curved Corner Top (Curved Slope)
  These may exist in a newer LDraw release than this mirror, or may need to
  be pulled from the LDraw Parts Tracker (unofficial parts) directly at
  https://www.ldraw.org/ - I didn't have network access to that domain from
  here, so I couldn't fetch them automatically.

How to use:
- Any LDraw-compatible viewer/renderer (LDView, LeoCAD, Blender + ImportLDraw
  addon, Studio) can open these directly if you point it at this folder
  structure (it mimics the standard LDraw directory layout).
- For point-cloud / mesh work, tools like `stl2dat`-adjacent converters or
  Blender's ImportLDraw can export .dat -> .obj/.stl for each part.
- Color is not baked into geometry (LDraw uses a separate color code per
  part instance) — MANIFEST.csv has the color per element if you need to
  tag/recolor the meshes afterward.
