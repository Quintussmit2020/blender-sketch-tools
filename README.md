# Sketch Tools

A Blender add-on for the "sketch, then extrude or sweep" workflow: draw flat wire shapes or 3D paths, then turn them into live, editable meshes.

> [!WARNING]
> **This add-on is 100% vibe coded.** Every line was written by AI coding agents, directed and tested by a human.
> It is built for my own personal use, not as a supported product.
> **Use it at your own risk.** Save your work and keep backups. There is no warranty and no promise of support or fixes (see [LICENSE](LICENSE)).

## Features

**Sketching**
- **New Sketch** on the View, XY, XZ or YZ plane, or on a clicked face. **Set Plane From Face** moves an existing sketch onto a face.
- **Sketch tool** in the Edit Mode toolbar. It draws polylines, rectangles, circles, 3-point arcs and centre arcs.
- Snapping follows Blender's snap magnet (Shift+Tab), and holding Ctrl flips it. It snaps to the sketch's own vertices, then to vertices and edges of other objects, then to the grid.
- Shift locks the angle. You can type lengths in the scene unit, and Tab switches to typing an angle.
- **3D paths** are drawn straight onto the surfaces of other objects, snapping to vertices, edge midpoints and points on edges.

**Editing**
- **Fillet**: rounds selected corners. Drag or type the radius, or set a radius per vertex in the panel.
- **Trim**: click an edge to remove the part between the nearest crossings.
- **Extend**: click a loose end to extend it to the next edge in the way.
- **Subdivide**: splits selected edges into equal parts.

**Output** (rebuilds live as you edit the sketch)
- **Extrude**: closed loops become solid meshes, and nested loops become holes. Height, offset, flip and caps are adjustable.
- **Sweep**: sweeps a profile along the sketch with true mitred corners. Profiles are Square, Cylinder, Half Round, Tube, or Custom (another sketch or a curve). Pivot, offsets, rotation, mirror, mitre limit, smoothing, end caps and union are adjustable.
- **Apply**: turns the output into a plain mesh and detaches it from the sketch.

Everything lives in the **Sketch Tools** tab of the 3D Viewport's N-panel.

## Requirements

Blender 4.2 or newer. It is developed and tested on Blender 5.1.

## Install

1. Download or clone this repository.
2. Build the extension package from the `sketch_tools` folder:
   ```bash
   blender --command extension build --source-dir sketch_tools
   ```
   You can also zip the `sketch_tools` folder yourself.
3. In Blender, open **Edit > Preferences > Get Extensions**, open the drop-down menu at the top right, choose **Install from Disk**, and pick the zip.

## Quick start

1. In the N-panel, open **Sketch Tools > Create > New Sketch** and pick a plane.
2. In Edit Mode, choose the **Sketch** tool from the toolbar and click to draw. RMB or Enter ends a line, and C closes it.
3. Under **Output**, choose **Extrude** or **Sweep** and adjust the settings.
4. Keep editing the sketch, and the mesh follows. Click **Apply** when you are done.

## Tests

The headless test suite runs inside Blender:

```bash
blender --background --factory-startup --python sketch_tools/tests/run.py
```

## Project docs

- [SKETCH_TOOLS_PLAN.md](SKETCH_TOOLS_PLAN.md): the original design plan.
- [DECISIONS.md](DECISIONS.md): corrections and decisions made during the build. Where this file and the plan disagree, this file wins.

## License

[GPL-3.0-or-later](LICENSE), like Blender itself.
