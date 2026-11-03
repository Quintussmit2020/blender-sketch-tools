# Sketch Tools - Blender add-on implementation plan

Handoff document for the coding agent. Discuss anything unclear before building; do not silently change the data model or the mitre approach.

## 1. Goal

Bring the 3ds Max "sketch, then extrude or sweep" workflow to Blender:

- Sketch closed and open polylines on a working plane with click-to-place points, typed lengths, angle lock, rectangles and circles.
- Nested closed loops become holes when extruded (wall with a window opening).
- Sweep a profile (library or custom) along the sketch with true mitred corners, including correctly closed loops where the sketch meets itself again.
- Fillet corners parametrically (radius stored per vertex, resolved at build time).
- Trim and extend segments against each other.
- Everything parametric until the user hits Apply.

Non-goals (for now): bezier handles, Geometry Nodes, self-intersection union, non-planar sketches.

Target: Blender 4.2 LTS or newer, shipped as an Extension (`blender_manifest.toml`). Python only, no node groups. Development on Windows 11.

## 2. Core concepts

- **Sketch object**: a Mesh object containing only vertices and edges (a wire mesh). Its `matrix_world` defines the sketch plane: local XY is the plane, local +Z is the plane normal. All vertices live at local z = 0.
- **Chain**: a run of connected edges. Closed chain = loop. Vertices with more than two edges are junctions and split chains there.
- **Loop nesting**: even-odd rule. Depth 0 loops are solid, depth 1 loops inside them are holes, depth 2 loops inside holes are solid again (islands).
- **Fillet**: a float attribute `st_fillet_radius` on each vertex. 0 = sharp. Resolved into arc points at build time.
- **Output object**: a separate Mesh object, parented to the sketch object, rebuilt from the sketch whenever the sketch or its parameters change. Mode is Extrude or Sweep (or None).
- **Apply**: turns the output object into an ordinary mesh and detaches it from the sketch.

Why a wire mesh instead of a Curve object: custom attributes work, Blender's native snapping / Ctrl+RMB vertex extrude / vertex bevel work in edit mode, and trim/extend are plain bmesh edge splits.

## 3. Data model

### 3.1 Object-level PropertyGroup `obj.sketch_tools`

```
is_sketch:          Bool
output_mode:        Enum  NONE | EXTRUDE | SWEEP
output_object:      PointerProperty(Object)
fillet_segments:    Int   default 8, min 1, max 64   (segments per fillet arc, fixed count)

# Extrude
ex_height:          Float default 2.0 (scene units)
ex_offset:          Float default 0.0  (shift along plane normal before extruding)
ex_flip:            Bool
ex_caps:            Bool  default True

# Sweep
sw_profile:         Enum  SQUARE | CYLINDER | HALF_ROUND | TUBE | CUSTOM
sw_custom_object:   PointerProperty(Object)   (another sketch object or a Curve object)
sw_width:           Float  (SQUARE width / HALF_ROUND diameter)
sw_height:          Float  (SQUARE height)
sw_radius:          Float  (CYLINDER, TUBE outer)
sw_thickness:       Float  (TUBE wall thickness)
sw_segments:        Int    default 16 (round profiles)
sw_pivot:           Enum   9-point grid: TL TC TR ML MC MR BL BC BR  default BC
sw_offset_u:        Float  (in-plane, perpendicular to path)
sw_offset_v:        Float  (along plane normal)
sw_rotation:        Float  degrees, rotates profile about the path tangent
sw_mirror_u:        Bool
sw_mitre_limit:     Float  default 4.0 (multiple of each profile point's |u|, see DECISIONS C12)
sw_smooth:          Bool   default False
sw_cap_ends:        Bool   default True (open chains only)
```

Every property's `update` callback calls `rebuild(obj)`.

### 3.2 Per-vertex attribute

`st_fillet_radius`: FLOAT, domain POINT. Created on sketch creation. Missing attribute is treated as all zeros.

### 3.3 Output object conventions

- Name: `<sketch name>.out`
- Parented to the sketch with identity parent-inverse, so it lives in sketch local space.
- `hide_select = True` so viewport clicks reach the sketch object.
- Sketch object display: `display_type = 'WIRE'`, `show_in_front = True`.

## 4. Resolver (`core/resolve.py`)

Input: the sketch object. Output: `ResolvedSketch`.

```
@dataclass
class Chain:
    points: list[Vector]   # 2D (x, y) in sketch local space, fillets already resolved
    closed: bool
    depth: int             # nesting depth, closed chains only, -1 for open
    parent: int | None     # index of enclosing chain
    children: list[int]

@dataclass
class ResolvedSketch:
    chains: list[Chain]
    warnings: list[str]
```

Steps:

1. Read the mesh with bmesh (`bmesh.new(); bm.from_mesh(...)`; in edit mode use `bmesh.from_edit_mesh`). Ignore faces if any exist.
2. Project vertices to z = 0. If any vertex has |z| > 1e-4, add a warning (sketch shown in panel).
3. Walk edges into chains. Start chains at junction vertices (valence > 2) and endpoints (valence 1); remaining untouched edges are closed loops, start anywhere. A chain is closed when it returns to its first vertex without passing a junction.
4. Resolve fillets (section 4.1) on interior vertices of chains (every vertex of a closed chain, all but the endpoints of an open chain).
5. Remove consecutive duplicate points and zero-length segments.
6. Normalise winding: compute signed area, store it; nesting detection below does not depend on winding, but builders expect depth-even loops CCW and depth-odd loops CW. Reverse point lists to satisfy that.
7. Nesting: for each closed chain, test its first point against every other closed chain with even-odd point-in-polygon (`core/geometry2d.point_in_polygon`). Depth = number of containing loops. Parent = containing loop with the highest depth. Open chains get depth -1.

### 4.1 Fillet resolution

At vertex P with radius r > 0, incoming segment direction a (from previous point) and outgoing direction b:

- theta = angle between -a and b (the interior angle). Skip if theta is within 0.5 degrees of 180 (straight) or below 1 degree (degenerate).
- Tangent distance t = r / tan(theta / 2).
- Clamp t to 0.5 x the shorter adjacent segment length (each segment can be filleted from both ends). If clamped, recompute r = t x tan(theta / 2) and add a warning.
- Tangent points T1 = P - a_hat x t, T2 = P + b_hat x t. Centre C = P + bisector_hat x (r / sin(theta / 2)).
- Emit `fillet_segments` + 1 points on the arc from T1 to T2 around C, shortest direction. Replace P with those points.

Fillet arcs are just more corners with shallow angles; the sweep mitre math handles them without special cases.

## 5. Extrude builder (`builders/extrude.py`)

Input: `ResolvedSketch`, extrude params. Output: a bmesh in sketch local space.

For each closed chain with even depth (solid), together with its odd-depth children (holes):

1. Base height z0 = `ex_offset`, top z1 = z0 + `ex_height` (negated when `ex_flip`).
2. Side walls: for every chain in the group (outer + holes), create vertices at z0 and z1 for each point and a quad per segment.
3. Caps (if `ex_caps`): use `mathutils.geometry.tessellate_polygon([outer_pts, hole1_pts, ...])` which handles holes and returns triangle indices into the concatenated point list. Create faces for z0 and z1. Then run `bmesh.ops.join_triangles` on cap faces with `angle_face_threshold` and `angle_shape_threshold` near zero so coplanar triangles merge into ngons. A cap with holes will keep at least one bridging edge because Blender ngons cannot contain holes; that is acceptable.
4. `bmesh.ops.remove_doubles` with a small threshold to weld cap and wall vertices, then `bmesh.ops.recalc_face_normals` on all faces.

Islands (depth 2) are their own group and are processed by the same loop, since depth 2 is even.

Open chains are ignored by this builder (warning in panel: "N open chains ignored by Extrude").

## 6. Sweep builder (`builders/sweep.py`)

### 6.1 Profiles (`core/profiles.py`)

A profile is a list of closed 2D rings in profile space (u, v): u is in the sketch plane, perpendicular to the path (positive to the left of travel), v is along the sketch plane normal. Each ring is a list of (u, v) points, CCW.

Library:

- SQUARE: rectangle `sw_width` x `sw_height`. (Max "Bar".)
- CYLINDER: circle radius `sw_radius`, `sw_segments` points.
- HALF_ROUND: semicircle of diameter `sw_width`, flat side at v = 0, arc above, `sw_segments` points on the arc.
- TUBE: two rings, outer radius `sw_radius`, inner radius `sw_radius - sw_thickness`, inner ring reversed (CW) so normals face inward. Clamp thickness to below radius.
- CUSTOM: take `sw_custom_object`. If it is a sketch object, run the resolver on it and use its closed chains (local x -> u, local y -> v). If it is a Curve object, sample its splines (poly points directly; bezier via `spline.bezier_points` evaluated with `mathutils.geometry.interpolate_bezier`, `resolution_u` segments). Open chains in a custom profile are allowed and produce an open strip (no cap faces).

Transform before sweeping, in this order: mirror_u, rotation (about origin), pivot alignment (translate so the chosen bounding-box point sits at the origin), then offset_u / offset_v.

### 6.2 Mitre algorithm

This is the core of the add-on and the reason existing add-ons fail. Build the sweep per segment, not per point.

For a chain with points P_0 .. P_n-1 (after fillet resolution):

- Segment i goes P_i -> P_i+1 (wrapping when closed). Direction d_i (unit), in-plane left normal n_i = (-d_i.y, d_i.x).
- A **station** is a ring of vertices, one ring per corner point P_i (for closed chains: every point; for open chains: every point including both endpoints).
- For a corner P_i between segment i-1 and segment i, and a profile point (u, v):
  - Offset line A: P_i + n_(i-1) x u + s x d_(i-1)
  - Offset line B: P_i + n_i x u + s x d_i
  - Station vertex (2D) = intersection of A and B (`core/geometry2d.line_line_intersection`). If |cross(d_(i-1), d_i)| < 1e-6 (collinear or reversed), use P_i + n_i x u.
  - Mitre limit: if |vertex - P_i| > `sw_mitre_limit` x |u| and |u| > 0, clamp the vertex to P_i + bisector_hat x (`sw_mitre_limit` x |u|). Keeps acute corners bounded.
  - Lift to 3D: (x, y, v). The v component is never scaled; only the in-plane offset is mitred.
- Open chain endpoints: single segment, so station vertex = P + n x u.
- Faces: for consecutive stations i and i+1 (wrapping when the chain is closed), and each ring, one quad per ring edge between the two stations. Closed rings wrap their quad strip; open rings (custom profile) do not.
- Caps: open chains with `sw_cap_ends` get an ngon per closed ring at each end, built from the ring's vertices. For TUBE use `tessellate_polygon([outer, inner])` for the end caps.
- Normals: `bmesh.ops.recalc_face_normals` at the end.
- Shading: if `sw_smooth`, set all faces smooth and mark edges running along the path sharp where the profile's own corner angle exceeds 30 degrees (so a SQUARE keeps hard edges while CYLINDER is smooth). Do not use a Smooth by Angle modifier; keep the output mesh self-contained.

Because stations are computed for every corner including index 0 on a closed chain, the loop closes exactly with no seam vertex and no twist.

Multiple chains are swept independently and merged into one bmesh. Overlapping geometry where chains cross is left as-is (union is out of scope).

## 7. Build pipeline (`handlers.py`, `core/build.py`)

`rebuild(sketch_obj)`:

1. Return early if `output_mode == 'NONE'` (remove the output object if present).
2. Resolve. Store warnings on the PropertyGroup (string property) for the panel.
3. Run the builder for the mode.
4. Get or create the output object. `bm.to_mesh(output.data)`, `output.data.update()`. Preserve the output mesh's material slots and the first material assignment if present.
5. Ensure parenting and display conventions from 3.3.

Triggers:

- Every PropertyGroup `update` callback.
- `bpy.app.handlers.depsgraph_update_post`: for each update where `update.id` is a sketch object and `update.is_updated_geometry`, and the object is not in edit mode, call rebuild. Also rebuild on the transition out of edit mode (detect via a module-level `_last_mode` check inside the handler).
- Recursion guard: module-level `_rebuilding` flag; the handler returns immediately when set, and ignores updates whose id is an output object.

Live preview while in edit mode is Tier 5 (timer-based, read via `bmesh.from_edit_mesh`), not needed initially.

Apply (`ops/output.py: sketch.apply`):

- Rebuild once, clear parenting while keeping world transform, set `hide_select = False`, rename to the sketch name without `.out`, clear `sketch_tools.output_object`.
- Option `delete_sketch` (default False). If False, hide the sketch object in viewport.

## 8. Operators

All operators live under the `sketch.` prefix.

### 8.1 `sketch.new`

Creates a sketch object. Plane option: VIEW (aligned to the viewport), XY, XZ, YZ, FACE (raycast under the mouse with `scene.ray_cast`; origin at the hit point, +Z = face normal, +X = projection of the view right vector onto the face). Origin for the axis planes = 3D cursor. Adds the `st_fillet_radius` attribute, sets display conventions, enters edit mode, activates the Sketch draw tool.

### 8.2 `sketch.set_plane_from_face`

Re-orient an existing sketch's transform to a picked face, keeping vertices in place in world space (recompute local coordinates and reproject to z = 0).

### 8.3 `sketch.draw` - WorkSpaceTool + modal operator (`ui/tool.py`, `ops/draw_tool.py`)

Registered with `bpy.utils.register_tool` in Mesh Edit Mode. Modal operator run from the tool's keymap on LMB.

Point placement: `bpy_extras.view3d_utils.region_2d_to_origin_3d` / `region_2d_to_vector_3d`, intersect with the sketch plane (`mathutils.geometry.intersect_line_plane`), convert to sketch local 2D.

Modes and keys:

- Polyline (default): LMB adds a point and an edge from the previous point. RMB or Enter ends the chain. Esc cancels the current chain (removes vertices added this session). C closes the chain to its first vertex and ends it. N ends the current chain and starts a new one with the next click.
- Shift held: angle lock to 0 / 45 / 90 degrees relative to the previous segment (first segment: relative to sketch X).
- Ctrl held: snap to existing sketch vertices within 12 px, else to grid increment. Superseded by DECISIONS C23: snapping follows the snap magnet and Ctrl flips it while held.
- Typed numeric input while rubber-banding: digits, `.`, `-`, Backspace set the segment length; Tab switches to typing the angle (degrees from previous segment); Enter commits the point. Follow the pattern of Blender's transform numeric input but keep it simple (no units parsing beyond a plain float in scene units).
- R: rectangle mode. Two clicks (opposite corners) or click-drag. Typed W then Tab then H sets dimensions after the first click. Produces a closed 4-vertex chain.
- O: circle mode. Click centre, move for radius (typed radius accepted), produces a closed chain with `circle_segments` points (preference, default 32).

Overlay (`ui/overlay.py`): gpu shader lines for the rubber band, the rectangle/circle preview and the angle-lock guide; `blf` text showing current length and angle near the cursor. Snap increment from tool settings: use `context.scene.unit_settings.scale_length` and the add-on preference `grid_increment` (default 0.1).

Edits are done through `bmesh.from_edit_mesh` + `bmesh.update_edit_mesh`.

### 8.4 `sketch.fillet`

Modal. Operates on selected vertices. Mouse X drag changes radius (sensitivity scaled by view distance), typed value accepted, Enter commits, Esc restores previous radii. Writes `st_fillet_radius`. Overlay draws the resolved arc preview for the affected vertices using the resolver's fillet function directly. Also exposed as a float field in the panel that writes to all selected vertices.

### 8.5 `sketch.trim`

Modal. On LMB: find the sketch edge nearest the cursor (project edge endpoints to screen space, point-segment distance, threshold 12 px). Compute all intersections of that edge with every other edge in 2D (`core/geometry2d.segment_intersection`, excluding shared endpoints). Also treat the edge's own endpoints as boundaries. Sort intersection parameters; find the interval containing the clicked parameter; split the edge at the interval's interior bounds (`bmesh.utils.edge_split` at the fraction) and at the matching points on the crossing edges; delete the sub-edge; `bmesh.ops.remove_doubles` with 1e-5 to weld the split points. Repeat until RMB/Esc.

### 8.6 `sketch.extend`

Modal. On LMB: pick the nearest valence-1 vertex (endpoint). Cast a 2D ray from the endpoint along its edge direction; find the nearest intersection with any other edge (excluding its own). Split that edge there and add an edge from the endpoint to the new vertex. If nothing is hit, do nothing and report.

### 8.7 `sketch.set_output_mode`, `sketch.rebuild`, `sketch.apply`

Thin wrappers around the pipeline.

### 8.8 `sketch.subdivide`

Added at the owner's request on 2026-10-22. Edit mode only, on the selected edges of the active sketch (selected vertices imply their edges, as in Blender).

- Options: `mode` COUNT or LENGTH. `cuts` Int default 1, min 1, max 100 (COUNT: that many new vertices per edge). `max_length` Float in scene units (LENGTH: each edge is split into ceil(length / max_length) equal parts).
- New vertices are evenly spaced along the edge at local z = 0 and are always sharp: `st_fillet_radius` 0 and `st_fillet_id` 0. Blender's own Edge > Subdivide interpolates both attributes onto the new points (tested on 5.1.1: a corner of radius 0.5 gave the three new points 0.125, 0.25 and 0.375 with distinct non-zero ids), so a new point later dragged into a corner would carry an unexpected fillet. This operator is the supported way to subdivide a sketch.
- Existing vertices, their radii and ids are untouched. Edges stay in chain order, so closed loops stay closed and junctions stay junctions. Zero-length edges are skipped.
- After the operation only the new vertices are selected, so they can be moved straight away.
- `bl_options = {'REGISTER', 'UNDO'}`; the redo panel shows mode, cuts and max length.
- Subdividing an edge next to a filleted corner shortens that segment, so the fillet may clamp at build time with the usual warning. That is expected.

## 9. UI (`ui/panel.py`)

N-panel, tab "Sketch Tools", visible in Object and Edit mode.

- **Create**: New Sketch with plane enum; Set Plane From Face.
- **Sketch** (when the active object is a sketch): fillet radius field (selected vertices), Fillet (modal) button, Trim, Extend, Subdivide (edit mode, with cuts), fillet segments, warnings box.
- **Output**: mode enum; Extrude params or Sweep params depending on mode; Rebuild; Apply with delete-sketch checkbox.

Toolbar: one tool "Sketch" in Mesh Edit Mode with a simple icon (use a built-in icon like `GREASEPENCIL` or `CURVE_PATH` initially).

Edit-mode context menu: when the active object is a sketch, `sketch.subdivide` is also added to the Edge menu and the right-click edge context menu.

## 10. Preferences (`prefs.py`)

- `grid_increment` Float default 0.1
- `circle_segments` Int default 32
- `default_fillet_segments` Int default 8
- `snap_pixel_threshold` Int default 12

## 11. Module layout

```
sketch_tools/
  blender_manifest.toml
  __init__.py                 register/unregister, imports
  props.py                    PropertyGroup, attribute helpers
  prefs.py
  handlers.py                 depsgraph handler, recursion guard, rebuild entry point
  core/
    geometry2d.py             point_in_polygon, segment_intersection, line_line_intersection,
                              arc_points, signed_area, nearest_edge_screen
    resolve.py                chain walking, fillets, nesting -> ResolvedSketch
    profiles.py               library profiles, custom profile sampling, transforms
    build.py                  rebuild(), output object management, apply
  builders/
    extrude.py
    sweep.py
  ops/
    create.py                 sketch.new, sketch.set_plane_from_face
    draw_tool.py              sketch.draw modal
    fillet.py
    trim.py                   sketch.trim, sketch.extend
    output.py                 set_output_mode, rebuild, apply
  ui/
    panel.py
    tool.py                   WorkSpaceTool registration + keymap
    overlay.py                gpu/blf drawing
  tests/
    run.py                    entry: blender --background --python tests/run.py
    test_geometry2d.py
    test_resolve.py           fixtures: square, square with square hole, nested islands,
                              open polyline, junction, filleted square
    test_extrude.py           face counts, hole presence, manifold check
    test_sweep.py             closed square path with SQUARE profile: 16 side quads, no
                              seam, all corner widths equal; open path caps; TUBE ring count
```

`core/` and `builders/` must not import `bpy.ops` or touch context; they take plain data and bmesh so they are testable headless.

## 12. Phases and acceptance criteria

**Phase 1 - Resolver + Extrude** (sketch with existing mesh tools: Ctrl+RMB, Ctrl+Shift+B)
- New Sketch creates a correctly configured wire object on each plane option.
- Square inside square extrudes to a wall with a hole, manifold, normals outward.
- Three-level nesting produces an island.
- Changing `ex_height` live-updates; leaving edit mode after moving a vertex live-updates.
- Fillet radius set via the panel field rounds the corner with the configured segment count.
- Apply produces a plain mesh and the sketch is hidden.

**Phase 2 - Sweep**
- Closed square path, SQUARE profile: every corner has the same outer width, no seam at vertex 0, 16 side quads, no extra ring.
- Same with CYLINDER, HALF_ROUND, TUBE; TUBE has an inner surface and ring end caps on open paths.
- Filleted corners on the path sweep smoothly.
- Acute 20 degree corner is bounded by the mitre limit.
- Custom profile from a second sketch object works, including pivot alignment and rotation.

**Phase 3 - Sketch tool**
- Polyline, close, new chain, rectangle, circle, angle lock, typed length and angle, vertex snap, grid snap, overlay text.
- Tool appears in the toolbar in edit mode and is active after New Sketch.

**Phase 4 - Fillet modal, Trim, Extend**
- Fillet drag with arc preview; Esc restores.
- Trim removes exactly the clicked interval between intersections, including the common "two overlapping rectangles -> trim the inside bits" case, leaving a single clean closed loop that the resolver reads as one chain.
- Extend hits the first edge in its direction and splits it.

**Phase 5 - Later**
- Live preview while in edit mode (timer).
- Scene snapping to non-sketch geometry in the draw tool (BVH over visible objects).
- Arc primitive (three-point).
- Self-intersection union (boolean) as a sweep option.
- JSON profile library folder in preferences.

**Phase 6 - Subdivide sketch edges** (added 2026-10-22, see 8.8)
- Select edges, run Subdivide with N cuts: each selected edge gets N evenly spaced new vertices, all sharp (radius 0, id 0); unselected edges are unchanged.
- LENGTH mode: every resulting segment is no longer than `max_length`, and segments of one edge are equal.
- The resolver still sees the same chains: a closed loop stays one closed loop, a junction stays a junction, chain count unchanged.
- Fillet radii on existing corners are unchanged; the output rebuilds on leaving edit mode and matches the unsubdivided result where the new points are collinear.
- Only the new vertices are selected afterwards; Undo restores the previous sketch exactly.
- Works on an Extrude hole loop and on a Sweep path, closed and open.
- Available from the Sketch panel, the Edge menu and the edge context menu when the active object is a sketch, and not offered for other meshes.

**Phase 7 - Snap to existing objects and sweep along a 3D path** (requested by the owner on 2026-10-23; decisions settled in DECISIONS C19 on 2026-10-24)
- Goal: free-draw a path by snapping to points on existing objects (vertices, edge midpoints, points on edges, points on faces), possibly across several faces that do not share a plane, then sweep a profile along that path.
- Conflict with the current design: section 2 defines a sketch as planar (all vertices at local z = 0) and section 1 lists non-planar sketches as a non-goal. A path across several faces is not planar, so this phase needs a new path type rather than a change to existing sketches.
- Proposed shape (to confirm before building):
  - A new "3D path" object type next to the planar sketch (`sketch_tools.is_path_3d`), a wire mesh whose vertices keep full 3D positions. Planar sketches stay exactly as they are.
  - Draw tool option "Snap to scene" using a BVH over visible non-sketch objects (this absorbs the Phase 5 item "Scene snapping to non-sketch geometry"). Snap targets: vertex, edge midpoint, nearest point on edge, point on face. Ctrl keeps its meaning; Shift angle lock applies within the current face's plane.
  - Sweep along a 3D path: frames from rotation-minimising transport along the path, mitre at each corner by the bisecting plane of the two segments, the same mitre-limit rule as C2/C12. Profile "up" reference to decide: the snapped face normal at each point, world up, or the transported frame.
  - Extrude is not offered for 3D paths (it needs a plane). Fillets on a 3D path would be resolved in the plane of the two adjacent segments.
- Open decisions: (1) the profile orientation reference; (2) whether a 3D path stays linked to the snapped object (follows it when it moves) or is a one-off copy of the points; (3) whether points between snapped points may be free (projected onto the surface under the cursor) or only snapped; (4) whether Trim, Extend, Fillet and Subdivide must work on 3D paths in this phase.

**Phase 8 - Shape options: rectangle, circle and arcs** (requested by the owner on 2026-10-23; decisions settled in DECISIONS C19 on 2026-10-24)
- Today the draw tool already has rectangle (R) and circle (O); the three-point arc is listed under Phase 5.
- Proposed: make rectangle, circle and arc explicit tool options (tool header buttons and keys), add arcs (three-point, and centre-start-end), and make all shapes work on 3D paths from Phase 7 as well as on planar sketches.
- Open decisions: (1) whether arcs are baked into straight segments at draw time (fixed count, like circles today) or stay parametric like fillets (radius or bulge stored as attributes and resolved at build time); (2) whether rectangles get options such as centre-drawn, rounded corners or rotation; (3) which shapes are needed in 3D mode.

## 13. Edge cases to handle explicitly

- Duplicate vertices and zero-length edges in the sketch (weld in the resolver, do not crash).
- Vertex not on the plane (project + warn).
- Chain with fewer than 3 points marked closed (treat as open).
- Fillet radius larger than the segments allow (clamp + warn).
- Collinear consecutive segments (skip fillet, mitre falls back to plain normal offset).
- Reversed segments (180 degree turn) in a sweep: fall back to normal offset, do not produce NaN.
- Output object deleted by the user: recreate on next rebuild.
- Sketch object duplicated (Shift+D): the copy points at the original's output object; on rebuild, detect that `output_object.parent is not self` and create a fresh output.
- Undo: rebuilds triggered by depsgraph updates must not push undo steps (do not call operators from the handler, write mesh data directly).
- File load: run a rebuild pass over all sketch objects in `load_post`.

## 14. Development notes (Windows 11)

- Install for development by symlinking the `sketch_tools` folder into `%APPDATA%\Blender Foundation\Blender\4.x\extensions\user_default\sketch_tools`, then enable it under Preferences > Get Extensions > Installed. Reload with `bpy.ops.script.reload()` or by toggling the extension.
- Run tests with `"C:\Program Files\Blender Foundation\Blender 4.x\blender.exe" --background --python tests\run.py`.
- Keep `core/` and `builders/` free of `bpy.context` usage so tests run headless.
- No em dashes anywhere in code comments, UI strings or docs; use hyphens.
