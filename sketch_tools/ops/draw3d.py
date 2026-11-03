"""sketch.draw_3d: the modal that draws a 3D path on the surfaces of other objects (C19, C20).

Same layering as the planar draw tool:
    core/snap3d.PathSession     the decisions (points, closing, ending, cancelling)
    PathEditor                  applies the session's commands to the path's edit bmesh
    PathController              Blender events plus a view projector into session events
    SKETCH_OT_draw_3d           the operator lifecycle: draw handler, header text, cleanup

The controller takes the projector factory, the scene source and the header and redraw callbacks
as arguments and reads only plain event attributes, so tests drive it with fake events and a fake
view. The planar tool's overlay is reused: the session's preview holds 3D points and the
projector's to_screen takes them."""

import traceback

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from ..core import attributes, build, draw_logic, snap3d
from ..core.resolve3d import PATH_NORMAL_ATTR
from ..prefs import get_prefs
from ..ui import overlay
from . import modal_common
from .modal_common import MODIFIER_KEYS, NAVIGATION, POINTER_MOVES, PRESSES, SYSTEM_PREFIXES, TOP_ROW_CHARS

# Point snapping works on the world space vertices and edges of the visible meshes, projected on
# every cursor move in Python. Past these totals (counted over the evaluated meshes, in view
# layer order) an object is still hit by the cursor ray and still gives normals, but its
# vertices and edges are not snap targets, and the header says so.
MAX_SNAP_VERTICES = 20000
MAX_SNAP_EDGES = 60000

NO_SURFACE_HINT = "No surface under the cursor, no point placed"
NO_MESH_HINT = "No visible mesh object to draw on"
NO_PLANE_HINT = "Cursor misses the plane of the shape"

_KEYS = {'RET': 'ENTER', 'NUMPAD_ENTER': 'ENTER', 'ESC': 'ESC', 'C': 'C',
         'R': 'R', 'O': 'O', 'P': 'P', 'A': 'A'}

# Controllers of running modals, so unregistering the add-on can wind them down.
_live = modal_common.live_set()


# ---------------------------------------------------------------------------
# The scene to snap to
# ---------------------------------------------------------------------------

def _is_output(obj):
    parent = obj.parent
    return parent is not None and build.is_sketch(parent) and parent.sketch_tools.output_object == obj


def _is_snap_target(obj, path_obj, view_layer):
    return (obj.type == 'MESH' and obj != path_obj and not build.is_sketch(obj)
            and not _is_output(obj) and obj.visible_get(view_layer=view_layer))


def _adjacent_normals(mesh, matrix, world):
    """snap3d.adjacent_normals of an evaluated mesh in world space (world: the mesh's vertices
    there): the normals (and probes) of the faces around each vertex and edge, read from the mesh
    itself so snapping never has to search. The probes come from the mesh's loop triangles."""
    flat = [0.0] * (3 * len(mesh.polygons))
    mesh.polygon_normals.foreach_get("vector", flat)
    starts = [0] * len(mesh.polygons)
    totals = [0] * len(mesh.polygons)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    loop_verts = [0] * len(mesh.loops)
    loop_edges = [0] * len(mesh.loops)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    mesh.loops.foreach_get("edge_index", loop_edges)
    triangles = mesh.loop_triangles
    tri_polygon = [0] * len(triangles)
    tri_loops = [0] * (3 * len(triangles))
    triangles.foreach_get("polygon_index", tri_polygon)
    triangles.foreach_get("loops", tri_loops)
    corners_of = [[] for _ in starts]
    for t, polygon in enumerate(tri_polygon):
        start = starts[polygon]
        corners_of[polygon].append(tuple(loop - start for loop in tri_loops[3 * t:3 * t + 3]))
    # Normals move with the inverse transpose; a mirrored object keeps its outward normals.
    normal_matrix = matrix.inverted_safe().to_3x3().transposed()

    def faces():
        for i, (start, total) in enumerate(zip(starts, totals)):
            normal = normal_matrix @ Vector(flat[3 * i:3 * i + 3])
            normal.normalize()
            verts = loop_verts[start:start + total]
            corners = [world[v] for v in verts]
            probes = snap3d.face_probes(corners, corners_of[i] or snap3d.triangulate(corners))
            yield verts, loop_edges[start:start + total], normal, *probes

    return snap3d.adjacent_normals(len(mesh.vertices), len(mesh.edges), faces())


def build_scene_source(context, path_obj, normals=True):
    """A SceneSource over every visible mesh object that is not a sketch, a sketch output or the
    path itself, from the evaluated meshes (modifiers applied) in world space. Meshes without
    faces are skipped: they give no surface to hit and no normal. normals=False leaves out the
    face normals and probes around each snap point, which only the 3D tool reads (the planar tool
    snaps to points and tests their visibility, nothing more)."""
    depsgraph = context.evaluated_depsgraph_get()
    view_layer = context.view_layer
    source = snap3d.SceneSource()
    unsnapped = 0
    for obj in view_layer.objects:
        if not _is_snap_target(obj, path_obj, view_layer):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh.polygons:
                continue
            matrix = obj.matrix_world
            reach = max((matrix @ Vector(c)).length for c in evaluated.bound_box)
            source.add_tree(BVHTree.FromObject(obj, depsgraph), matrix, reach)
            if (len(source.vertices) + len(mesh.vertices) > MAX_SNAP_VERTICES
                    or len(source.edges) + len(mesh.edges) > MAX_SNAP_EDGES):
                unsnapped += 1
                continue
            flat = [0.0] * (3 * len(mesh.vertices))
            mesh.vertices.foreach_get("co", flat)
            pairs = [0] * (2 * len(mesh.edges))
            mesh.edges.foreach_get("vertices", pairs)
            world = [matrix @ Vector(flat[i:i + 3]) for i in range(0, len(flat), 3)]
            adjacency = _adjacent_normals(mesh, matrix, world) if normals else ()
            source.add_points(world, list(zip(pairs[0::2], pairs[1::2])), *adjacency)
        finally:
            evaluated.to_mesh_clear()
    if unsnapped:
        # Short: the notice shares a header line that Blender clips at both ends (C23).
        # The planar tool (normals=False) gives such an object no snaps at all, while the 3D tool
        # still places on its surface.
        what = "surface only" if normals else "not snapped"
        source.notice = f"Snap limit: {unsnapped} object(s) {what}"
    return source


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

class PathProjector(modal_common.ViewProjector):
    """The region, as the view snap3d.place wants it: world points to pixels, the cursor ray, the
    sight line to a point and the right vector (the world direction of screen +x, which orients
    shape planes). The base class supplies the region bounds test (contains)."""

    def __init__(self, region, rv3d, overlays=(), reach=0.0):
        super().__init__(region, rv3d, Matrix.Identity(4), overlays)
        self._matrix = rv3d.perspective_matrix.copy()
        self._view_inverse = rv3d.view_matrix.inverted()
        self.right = (self._view_inverse.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
        self._half = (region.width * 0.5, region.height * 0.5)
        self._reach = reach
        self.key = (tuple(c for row in self._matrix for c in row), region.width, region.height)

    def project(self, co):
        clip = self._matrix @ Vector((co[0], co[1], co[2], 1.0))
        # w is the depth for a perspective view and 1 for an orthographic one.
        if clip.w <= 1e-6:
            return None
        return (self._half[0] * (1.0 + clip.x / clip.w), self._half[1] * (1.0 + clip.y / clip.w), clip.w)

    def to_screen(self, co):
        point = self.project(co)
        return None if point is None else (point[0], point[1])

    def ray(self, mouse):
        return (view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, mouse),
                view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, mouse))

    def sight(self, co):
        co = Vector(co)
        if self.rv3d.is_perspective:
            eye = self._view_inverse.translation
            offset = co - eye
            distance = offset.length
            return eye, (offset / distance if distance > 0.0 else Vector((0.0, 0.0, -1.0))), distance
        forward = (self._view_inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        # Starts outside everything: every scene point is within reach of the origin.
        distance = self._reach + co.length + 1.0
        return co - forward * distance, forward, distance


def projector_factory(obj, source):
    def factory():
        context = bpy.context
        region, rv3d, area = context.region, context.region_data, context.area
        if region is None or rv3d is None or area is None or obj.name not in bpy.data.objects:
            return None
        rects = modal_common.overlay_rects(area)
        gizmo = modal_common.navigation_gizmo_rect(region, area, area.spaces.active, context.preferences)
        if gizmo is not None:
            rects.append(gizmo)
        return PathProjector(region, rv3d, rects, source.reach)
    return factory


# ---------------------------------------------------------------------------
# Writing the path
# ---------------------------------------------------------------------------

class PathEditor:
    """Applies PathSession commands to the edit bmesh of a 3D path.

    Session positions are world space; vertices are stored in the path's local space and the
    normal as the local unit vector. Everything it creates is tracked so CANCEL removes it again
    and restores the selection."""

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.matrix = obj.matrix_world.copy()
        self.inverse = self.matrix.inverted_safe()
        # Local coordinates are as large as the offset of the object: snap3d scales its float32
        # tolerance for reusing a vertex by it.
        self.offset = self.matrix.translation.length
        # A normal moves to local space with the transpose of the local-to-world matrix.
        self._normal_matrix = self.matrix.to_3x3().transposed()
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.rad, self.ids = attributes.ensure_bm_layers(self.bm)
        self.normals = self.bm.verts.layers.float_vector.get(PATH_NORMAL_ATTR)
        if self.normals is None:
            self.normals = self.bm.verts.layers.float_vector.new(PATH_NORMAL_ATTR)
        self._refresh_indices()
        # Existing vertices by index. Only appended to during a session, so refs stay valid
        # even while the bmesh lookup table is stale after adding vertices.
        self._old = tuple(self.bm.verts)
        self._new = {}
        self._ref_of = {}
        self._edges = []
        self._saved_selection = None

    def _refresh_indices(self):
        self.bm.verts.index_update()
        self.bm.verts.ensure_lookup_table()

    def alive(self):
        try:
            return self.bm.is_valid and self.mesh.is_editmode
        except ReferenceError:
            return False

    def _vert(self, ref):
        return self._new[ref] if ref < 0 else self._old[ref]

    def path_points(self):
        """[(ref, world co)] of every visible vertex, session vertices included: the path's own
        snap targets."""
        return [(self._ref_of.get(v, v.index), self.matrix @ v.co) for v in self.bm.verts if not v.hide]

    def _deselect_all(self):
        for elem in (*self.bm.verts, *self.bm.edges, *self.bm.faces):
            elem.select_set(False)

    def _begin(self):
        if self._saved_selection is not None:
            return
        self._saved_selection = [v.index for v in self.bm.verts if v.select]
        self._deselect_all()

    def apply(self, commands):
        """Run commands in order; one edit-mesh update at the end."""
        changed = False
        for cmd in commands:
            kind = cmd[0]
            if kind == "ADD_VERT":
                self._add_vert(cmd[1], cmd[2], cmd[3])
                changed = True
            elif kind == "ADD_EDGE":
                changed |= self._add_edge(cmd[1], cmd[2])
            elif kind == "SET_NORMAL":
                self._set_normal(cmd[1], cmd[2])
                changed = True
            elif kind == "CANCEL":
                changed |= self._rollback()
            elif kind == "END":
                self._new.clear()
                self._ref_of.clear()
                self._edges.clear()
                self._saved_selection = None
        if changed:
            self._refresh_indices()
            bmesh.update_edit_mesh(self.mesh)

    def _local_normal(self, normal):
        local = self._normal_matrix @ normal
        local.normalize()
        return local

    def _set_normal(self, ref, normal):
        """Rewrites the normal of a vertex of this session; they go with it on CANCEL."""
        self._new[ref][self.normals] = self._local_normal(normal)

    def _add_vert(self, ref, co, normal):
        self._begin()
        v = self.bm.verts.new(self.inverse @ co)
        v[self.normals] = self._local_normal(normal)
        v[self.rad] = 0.0
        v[self.ids] = 0
        v.select_set(True)
        self._new[ref] = v
        self._ref_of[v] = ref

    def _add_edge(self, ref_a, ref_b):
        self._begin()
        a, b = self._vert(ref_a), self._vert(ref_b)
        if a is b or self.bm.edges.get((a, b)) is not None:
            return False
        edge = self.bm.edges.new((a, b))
        edge.select_set(True)
        self._edges.append(edge)
        return True

    def _rollback(self):
        if self._saved_selection is None:
            return False
        for edge in self._edges:
            if edge.is_valid:
                self.bm.edges.remove(edge)
        for v in self._new.values():
            if v.is_valid:
                self.bm.verts.remove(v)
        self.bm.verts.index_update()
        # New edges between old vertices left those vertices selected.
        self._deselect_all()
        for index in self._saved_selection:
            self._old[index].select_set(True)
        self.bm.select_flush_mode()
        self._new.clear()
        self._ref_of.clear()
        self._edges.clear()
        self._saved_selection = None
        return True


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class PathController:
    """Blender events in, edits and an operator result out.

    projector_factory() returns the view snap3d.place uses plus contains(mouse, window) and
    to_screen(co), or None when there is no usable 3D view. header(text_or_None) and redraw()
    are callbacks. region_pointer is the as_pointer() of the region the modal runs in; the
    overlay draws only there."""

    def __init__(self, editor, session, projector_factory, source, snap_pixels, header, redraw,
                 region_pointer=None):
        self.editor = editor
        self.session = session
        self.source = source
        self._projector_factory = projector_factory
        self._snap_pixels = snap_pixels
        self._header = header
        self._redraw = redraw
        self.region_pointer = region_pointer
        self._cleanups = [source.clear]
        # Buttons whose last press went on to Blender; their release and click events follow it.
        self._passed = set()
        self._hint = None
        # Shift or Ctrl as of the latest event: either turns snapping off while held (the header
        # shows the state).
        self._snap_off = False
        self.mouse = None
        self.closed = False
        self._update_header()
        _live.add(self)

    def add_cleanup(self, fn):
        self._cleanups.append(fn)

    def close(self):
        """Idempotent. Every exit path ends here: finish, cancel, Esc, error, unregister."""
        if self.closed:
            return
        self.closed = True
        _live.discard(self)
        for fn in reversed(self._cleanups):
            try:
                fn()
            except Exception:
                traceback.print_exc()
        self._cleanups.clear()
        try:
            self._header(None)
        except Exception:
            traceback.print_exc()

    def abort(self):
        """Drop whatever the session added and close. For errors, a failed start and add-on
        unregister. The editor knows whether it holds uncommitted work (END clears it)."""
        try:
            if self.editor.alive():
                self.editor.apply([("CANCEL",)])
        except Exception:
            traceback.print_exc()
        self.close()

    def snapshot(self):
        """(preview, mouse, to_screen) for the overlay, None when there is nothing to draw."""
        if self.closed:
            return None
        projector = self._projector_factory()
        if projector is None:
            return None
        return self.session.preview(), self.mouse, projector.to_screen

    # -- events ------------------------------------------------------------

    def start(self, event):
        """The press that launched the operator. 'MISS' when it is over no surface: the run does
        not start."""
        # The press that starts a run may carry Shift or Ctrl, and the first header is built from it.
        self._snap_off = bool(event.shift or event.ctrl)
        placement = self._pointer(event)
        if placement is None:
            return 'MISS'
        return self._apply(self.session.handle(("CLICK", placement, self._right())))

    def handle(self, event):
        """Returns 'RUNNING_MODAL', 'PASS_THROUGH', 'FINISHED' or 'CANCELLED'."""
        if not self.editor.alive() or self._projector_factory() is None:
            # Edit mode was left, or the area is no longer a 3D view. Either way the run ends
            # and whatever is uncommitted goes.
            self.abort()
            return 'CANCELLED'
        # Before any pass-through: a modifier pressed over the header or the toolbar still flips the
        # state the header shows (C24).
        off = bool(event.shift or event.ctrl)
        if off != self._snap_off:
            self._snap_off = off
            self._update_header()
        etype = event.type
        if etype in NAVIGATION or etype.startswith('NDOF_') or etype.startswith(SYSTEM_PREFIXES):
            return 'PASS_THROUGH'
        if etype in POINTER_MOVES or etype in MODIFIER_KEYS:
            if not self._inside(event):
                self._leave()
                return 'PASS_THROUGH'
            self._move(event)
            return 'RUNNING_MODAL'
        if etype == 'LEFTMOUSE':
            if event.alt:
                # Alt+LMB navigates. Recorded like any passed press so its release still passes
                # through when Alt is let go first.
                if event.value in PRESSES:
                    self._passed.add(etype)
                return 'PASS_THROUGH'
            if event.value in PRESSES:
                if not self._inside(event):
                    # Header, toolbar, sidebar and gizmos keep working during a run.
                    self._passed.add(etype)
                    return 'PASS_THROUGH'
                self._passed.discard(etype)
                return self._click(event)
            # Release, CLICK and CLICK_DRAG of a press that went on to Blender pass through; the
            # rest of a press that placed a point is swallowed.
            return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'
        if etype == 'RIGHTMOUSE':
            if event.value in PRESSES:
                if not self._inside(event):
                    self._passed.add(etype)
                    return 'PASS_THROUGH'
                self._passed.discard(etype)
                return self._session_event(("KEY", "RMB")) if event.value == 'PRESS' else 'RUNNING_MODAL'
            return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'
        if etype.startswith('NUMPAD_') and etype != 'NUMPAD_ENTER':
            # Nothing can be typed in 3D, so the numpad always navigates.
            return 'PASS_THROUGH'
        if event.value != 'PRESS':
            return 'RUNNING_MODAL'
        if etype in _KEYS and not (event.ctrl or event.alt or getattr(event, "oskey", False)):
            return self._session_event(("KEY", _KEYS[etype]))
        # Everything else (digits included) is swallowed so no shortcut fires mid-run.
        return 'RUNNING_MODAL'

    # -- translation -------------------------------------------------------

    def _inside(self, event):
        return modal_common.over_region(self._projector_factory(), event)

    def _leave(self):
        """The pointer left the region: stop drawing the candidate there."""
        if self.mouse is not None:
            self.mouse = None
            self.session.handle(("MOVE", None))
            self._redraw()

    def _pointer(self, event):
        """Place the cursor with the event's position and modifiers and tell the session. The
        placement is also left in session.cursor; None when over no surface."""
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        projector = self._projector_factory()
        placement = None
        plane = self.session.plane()
        if projector is not None and plane is not None:
            # The rest of a rectangle, circle or centre arc lies in the plane of its first point.
            placement = snap3d.place_on_plane(
                self.source, projector, self.mouse, self._snap_pixels, plane, self.editor.path_points(),
                snapping=not (event.shift or event.ctrl), reuse_scale=self.editor.offset)
        elif projector is not None:
            anchor, previous = self.session.anchor()
            placement = snap3d.place(
                self.source, projector, self.mouse, self._snap_pixels, self.editor.path_points(),
                shift=bool(event.shift), anchor=anchor, previous_dir=previous, ctrl=bool(event.ctrl),
                previous_normal=self.session.last_normal(),
                previous_point=self.session.last_point(),
                earlier_points=self.session.earlier_points(), reuse_scale=self.editor.offset)
        if placement is not None:
            self._hint = None
        elif plane is not None:
            self._hint = NO_PLANE_HINT
        else:
            self._hint = NO_MESH_HINT if self.source.empty else NO_SURFACE_HINT
        self.session.handle(("MOVE", placement))
        return placement

    def _move(self, event):
        self._pointer(event)
        self._after()

    def _click(self, event):
        placement = self._pointer(event)
        if placement is None:
            self._after()
            return 'RUNNING_MODAL'
        return self._apply(self.session.handle(("CLICK", placement, self._right())))

    def _right(self):
        """The view's right vector for the session, which fixes the x axis of a shape plane."""
        return getattr(self._projector_factory(), "right", None)

    def _session_event(self, session_event):
        return self._apply(self.session.handle(session_event))

    def _apply(self, commands):
        if commands:
            self.editor.apply(commands)
        if self.session.done:
            result = 'CANCELLED' if self.session.cancelled else 'FINISHED'
            self.close()
            self._redraw()
            return result
        self._after()
        return 'RUNNING_MODAL'

    def _after(self):
        self._update_header()
        self._redraw()

    def _update_header(self):
        # State first, then the hint and the scene notice, then the instructions (C24).
        parts = [f"Snapping {'off' if self._snap_off else 'on'}"]
        if self._hint:
            parts.append(self._hint)
        if self.source.notice:
            parts.append(self.source.notice)
        parts.append(self.session.header_text())
        self._header(" | ".join(parts))


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

class SKETCH_OT_draw_3d(bpy.types.Operator):
    bl_idname = "sketch.draw_3d"
    bl_label = "Draw 3D Path"
    bl_description = "Draw a path on the surfaces of other objects, snapping to their vertices and edges"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        area = context.area
        return (context.mode == 'EDIT_MESH' and area is not None and area.type == 'VIEW_3D'
                and build.is_path_3d(context.active_object))

    def invoke(self, context, event):
        if modal_common.any_sketch_modal_live():
            # Same as sketch.draw: a click in another viewport or a transparent part of the UI
            # reaches the tool keymap during a run, and must reach what is under the pointer.
            return {'PASS_THROUGH'}
        obj = context.active_object
        if event.type == 'LEFTMOUSE' and not modal_common.over_region(
                projector_factory(obj, snap3d.SceneSource())(), event):
            # The tool keymap also sees presses in the transparent part of the sidebar, on the HUD
            # and on the navigation gizmo. They belong to the UI, and no scene is built for them.
            return {'PASS_THROUGH'}
        try:
            source = build_scene_source(context, obj)
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, f"3D path tool failed to start: {exc}")
            return {'CANCELLED'}
        factory = projector_factory(obj, source)
        prefs = get_prefs(context)
        header, redraw = modal_common.area_callbacks(context.area)
        region = context.region
        controller = None
        shape = getattr(context.window_manager, "sketch_tools_shape", snap3d.POLYLINE)
        session = snap3d.PathSession(
            shape if shape in snap3d.SHAPES else snap3d.POLYLINE,
            prefs.circle_segments if prefs else draw_logic.DEFAULT_CIRCLE_SEGMENTS)
        try:
            controller = PathController(
                PathEditor(obj), session, factory, source,
                prefs.snap_pixel_threshold if prefs else draw_logic.DEFAULT_SNAP_PIXELS,
                header=header, redraw=redraw,
                region_pointer=region.as_pointer() if region is not None else None)
            self._controller = controller
            result = controller.start(event)
            if result == 'MISS':
                controller.abort()
                self.report({'WARNING'}, NO_SURFACE_HINT)
                return {'CANCELLED'}
            if result != 'RUNNING_MODAL':
                return {result}
            draw = overlay.DrawOverlay(controller.snapshot, controller.region_pointer)
            draw.add()
            controller.add_cleanup(draw.remove)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            traceback.print_exc()
            if controller is not None:
                controller.abort()
            else:
                source.clear()
            self.report({'ERROR'}, f"3D path tool failed to start: {exc}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        controller = self._controller
        try:
            return {controller.handle(event)}
        except Exception as exc:
            traceback.print_exc()
            controller.abort()
            self.report({'ERROR'}, f"3D path tool failed: {exc}")
            return {'CANCELLED'}

    def cancel(self, context):
        self._controller.abort()


def close_all():
    for controller in list(_live):
        controller.abort()


classes = (SKETCH_OT_draw_3d,)
