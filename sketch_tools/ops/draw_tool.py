"""sketch.draw: the modal behind the Sketch tool (plan 8.3).

Three thin layers around core/draw_logic.DrawSession, which holds every decision:
    SketchEditor    applies the session's commands to the sketch's edit bmesh
    DrawController  turns Blender events plus a view projector into session events
    SKETCH_OT_draw  owns the operator lifecycle: draw handler, header text, cleanup

The controller takes the projector, preferences and header/redraw callbacks as arguments and
reads only plain event attributes, so tests drive it with fake events and a fake view.

Snapping looks at the sketch's own vertices first, then (C22) at the visible vertices, edge
midpoints and edge points of the other mesh objects, through core/snap3d, then falls back to the grid.
It is active when Blender's snap toggle (the magnet) and Ctrl differ (C23)."""

import math
import traceback

import bmesh
import bpy
from bpy.props import EnumProperty

from ..core import attributes, build, draw_logic, snap3d
from ..core.draw_logic import POLYLINE, DrawSession, Mods, Snap
from ..prefs import get_prefs
from ..props import FROM_HEADER, SHAPE_ITEMS
from ..ui import overlay
from . import draw3d, modal_common
from .modal_common import (
    MODIFIER_KEYS, NAVIGATION, NUMPAD_CHARS, POINTER_MOVES, PRESSES, SYSTEM_PREFIXES, TOP_ROW_CHARS,
)

# Pointer travel (pixels) between press and release that turns a click into a drag.
DRAG_PIXELS = 6.0

EDGE_ON_HINT = "View is edge-on, rotate it"
BEHIND_HINT = "Sketch is behind you, turn the view"

_KEYS = {
    'RET': 'ENTER', 'NUMPAD_ENTER': 'ENTER', 'ESC': 'ESC', 'TAB': 'TAB', 'BACK_SPACE': 'BACKSPACE',
    'C': 'C', 'N': 'N', 'R': 'R', 'O': 'O', 'P': 'P', 'A': 'A',
}

MODES = (
    *((item[0], item[1], item[2]) for item in SHAPE_ITEMS),
    (FROM_HEADER, "Header Shape", "The shape chosen in the Sketch tool's header"),
)

# Controllers of running modals, so unregistering the add-on can wind them down.
_live = modal_common.live_set()


def effective_snap(use_snap, ctrl):
    """Snapping as Blender does it: the toggle turns it on, and Ctrl flips whatever the toggle says."""
    return bool(use_snap) != bool(ctrl)


def magnet_on(context):
    """Blender's snap toggle (the magnet) read from a context, off when there is no scene or tool
    settings to read, so a missing context never aborts a run."""
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "tool_settings", None)
    return bool(getattr(settings, "use_snap", False))


class SketchEditor:
    """Applies DrawSession commands to the edit bmesh of a sketch.

    Everything it creates is tracked so CANCEL removes it again and restores the selection."""

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.rad, self.ids = attributes.ensure_bm_layers(self.bm)
        self._refresh_indices()
        # Existing vertices by index. Only appended to during a session, so refs stay valid
        # even while the bmesh lookup table is stale after adding vertices.
        self._old = tuple(self.bm.verts)
        self._new = {}
        self._ref_of = {}
        self._edges = []
        self._saved_selection = None

    def _refresh_indices(self):
        # Refs of existing vertices are bmesh indices, so they must be current.
        self.bm.verts.index_update()
        self.bm.verts.ensure_lookup_table()

    def alive(self):
        try:
            return self.bm.is_valid and self.mesh.is_editmode
        except ReferenceError:
            return False

    def _vert(self, ref):
        return self._new[ref] if ref < 0 else self._old[ref]

    def ref_of(self, vert):
        return self._ref_of.get(vert, vert.index)

    def snap_candidates(self, to_screen):
        """(ref, local_xy, screen_xy) for every visible vertex, session vertices included."""
        for v in self.bm.verts:
            if v.hide:
                continue
            xy = (v.co.x, v.co.y)
            yield self.ref_of(v), xy, to_screen(xy)

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
                self._add_vert(cmd[1], cmd[2])
                changed = True
            elif kind == "ADD_EDGE":
                changed |= self._add_edge(cmd[1], cmd[2])
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

    def _add_vert(self, ref, xy):
        self._begin()
        v = self.bm.verts.new((xy[0], xy[1], 0.0))
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



class DrawController:
    """Blender events in, edits and an operator result out.

    projector_factory() returns an object with contains(mouse, window), ray_to_local(mouse), miss (why
    the last ray_to_local found nothing) and to_screen(xy), or None when there is no usable view.
    header(text_or_None) and redraw() are callbacks. region_pointer is the as_pointer() of the
    region the modal runs in; the overlay draws only there.

    scene (a snap3d.SceneSource) and scene_view_factory (returns the view find_snap wants: world
    point to pixels, sight lines, a cache key; None without a usable view) switch scene snapping
    on. The controller owns the scene: closing the run clears it.

    snap_toggle() returns Blender's snap toggle (scene.tool_settings.use_snap); None reads as off.
    It is read on every event so Shift+Tab during a run takes effect at once."""

    def __init__(self, editor, session, projector_factory, snap_pixels, header, redraw,
                 region_pointer=None, scene=None, scene_view_factory=None, snap_toggle=None):
        self.editor = editor
        self.session = session
        self._projector_factory = projector_factory
        self._scene = scene
        self._scene_view_factory = scene_view_factory
        self._snap_toggle = snap_toggle
        self._snap_pixels = snap_pixels
        self._header = header
        self._redraw = redraw
        self.region_pointer = region_pointer
        self._cleanups = [scene.clear] if scene is not None else []
        self._press_px = None
        # Buttons whose last press went on to Blender; their release and click events follow it.
        self._passed = set()
        self._hint = None
        # Ctrl as of the latest event, for the snapping state in the header.
        self._ctrl = False
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
        unregister. The editor knows whether it holds uncommitted work (END clears it), so this
        does not trust session.done: an edit that failed half way leaves it set."""
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
        """The press that launched the operator."""
        self._press_px = self._mouse_of(event)
        # The press that starts a run may carry Ctrl, and the first header is built from it.
        self._ctrl = bool(event.ctrl)
        return self._click(event)

    def handle(self, event):
        """Returns 'RUNNING_MODAL', 'PASS_THROUGH', 'FINISHED' or 'CANCELLED'."""
        if not self.editor.alive() or self._projector_factory() is None:
            # Edit mode was left, or the area is no longer a 3D view. Either way the run ends
            # and whatever is uncommitted goes.
            self.abort()
            return 'CANCELLED'
        # Before any pass-through: Ctrl pressed over the header or the toolbar still flips the state
        # the header shows.
        ctrl = bool(event.ctrl)
        if ctrl != self._ctrl:
            self._ctrl = ctrl
            self._update_header()
        etype = event.type
        if etype == 'TAB' and event.shift:
            # Shift+Tab is Blender's snap toggle; plain Tab stays the next-field key.
            return 'PASS_THROUGH'
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
                # through when Alt is let go first, instead of being read against a stale _press_px.
                if event.value in PRESSES:
                    self._passed.add(etype)
                return 'PASS_THROUGH'
            if event.value in PRESSES:
                if not self._inside(event):
                    # Header, toolbar, sidebar and gizmos keep working during a run.
                    self._passed.add(etype)
                    return 'PASS_THROUGH'
                self._passed.discard(etype)
                self._press_px = self._mouse_of(event)
                return self._click(event)
            if etype in self._passed:
                # Release, CLICK and CLICK_DRAG of a press that went on to Blender. Kept until the
                # next press because CLICK may arrive before or after the release.
                return 'PASS_THROUGH'
            if event.value == 'RELEASE':
                if not self._inside(event):
                    return 'RUNNING_MODAL'
                return self._release(event)
            return 'RUNNING_MODAL'
        if etype == 'RIGHTMOUSE':
            if event.value in PRESSES:
                if not self._inside(event):
                    self._passed.add(etype)
                    return 'PASS_THROUGH'
                self._passed.discard(etype)
                return self._session_event(("KEY", "RMB")) if event.value == 'PRESS' else 'RUNNING_MODAL'
            return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'
        if etype.startswith('NUMPAD_'):
            # Numpad digits navigate the view until a number is being typed.
            if etype in NUMPAD_CHARS and self.session.numeric.is_typing() and event.value == 'PRESS':
                return self._session_event(("TYPE", NUMPAD_CHARS[etype]))
            if etype != 'NUMPAD_ENTER':
                return 'PASS_THROUGH'
        if event.value != 'PRESS':
            return 'RUNNING_MODAL'
        if etype in TOP_ROW_CHARS:
            return self._session_event(("TYPE", TOP_ROW_CHARS[etype]))
        if etype in _KEYS and not (event.ctrl or event.alt or getattr(event, "oskey", False)):
            return self._session_event(("KEY", _KEYS[etype]))
        return 'RUNNING_MODAL'

    # -- translation -------------------------------------------------------

    @staticmethod
    def _mouse_of(event):
        return (event.mouse_region_x, event.mouse_region_y)

    def _inside(self, event):
        """True when the pointer is over the modal's region. handle() has already ended the run
        when there is no usable view."""
        return modal_common.over_region(self._projector_factory(), event)

    def _leave(self):
        """The pointer left the region: stop drawing the readout there. The header still follows Ctrl."""
        self._update_header()
        if self.mouse is not None:
            self.mouse = None
            self._redraw()

    def _pointer(self, event):
        """(local_xy, mods) under the mouse; local_xy is None when the view cannot place a point."""
        self.mouse = self._mouse_of(event)
        projector = self._projector_factory()
        if projector is None:
            return None, Mods()
        xy = projector.ray_to_local(self.mouse)
        snap = None
        # Ctrl is a keyboard modifier elsewhere, but here it means "flip Blender's snap toggle", and
        # everything the session does under Ctrl (snaps, grid points and rounded sizes) follows it.
        snapping = self._snapping(event.ctrl)
        if snapping and xy is not None:
            snap = draw_logic.nearest_snap(
                self.mouse, self.editor.snap_candidates(projector.to_screen), self._snap_pixels)
            # Shift turns scene snapping off, as it does in 3D, so Shift+Ctrl keeps the locked
            # direction with the length rounded to the grid unless a sketch vertex is in reach.
            if snap is None and not event.shift:
                snap = self._scene_snap()
        if xy is None:
            self._hint = BEHIND_HINT if projector.miss == draw_logic.MISS_BEHIND else EDGE_ON_HINT
        else:
            self._hint = None
        return xy, Mods(bool(event.shift), snapping, snap)

    def _snapping(self, ctrl):
        return effective_snap(self._snap_toggle() if self._snap_toggle is not None else False, ctrl)

    def _scene_snap(self):
        """The scene point under the mouse projected onto the sketch plane, as a Snap that is not
        an existing vertex (ref None), or None. Local z is dropped: an orthogonal projection."""
        if self._scene is None or self._scene.empty or self._scene_view_factory is None:
            return None
        view = self._scene_view_factory()
        if view is None:
            return None
        found = snap3d.find_snap(self._scene, view, self.mouse, self._snap_pixels)
        if found is None:
            return None
        local = self.editor.obj.matrix_world.inverted_safe() @ found.co
        return Snap(None, (local.x, local.y), found.kind)

    def _move(self, event):
        xy, mods = self._pointer(event)
        if xy is not None:
            self.session.handle(("MOVE", xy, mods))
        self._after()

    def _click(self, event):
        xy, mods = self._pointer(event)
        if xy is None:
            self._after()
            return 'RUNNING_MODAL'
        return self._apply(self.session.handle(("CLICK", xy, mods)))

    def _release(self, event):
        start = self._press_px
        xy, mods = self._pointer(event)
        if xy is None or start is None:
            self._after()
            return 'RUNNING_MODAL'
        dragged = math.hypot(self.mouse[0] - start[0], self.mouse[1] - start[1]) > DRAG_PIXELS
        return self._apply(self.session.handle(("RELEASE", xy, mods, dragged)))

    def _session_event(self, session_event):
        return self._apply(self.session.handle(session_event))

    def _apply(self, commands):
        result = 'RUNNING_MODAL'
        if commands:
            self.editor.apply(commands)
        if self.session.done:
            result = 'CANCELLED' if self.session.cancelled else 'FINISHED'
            self.close()
            self._redraw()
            return result
        self._after()
        return result

    def _after(self):
        self._update_header()
        self._redraw()

    def _update_header(self):
        # State first, then the short hint and notice, then the instructions. Blender clips an
        # overflowing header at both ends, so the whole line is kept short (C23).
        parts = [f"Snapping {'on' if self._snapping(self._ctrl) else 'off'}"]
        if self._hint:
            parts.append(self._hint)
        if self._scene is not None and self._scene.notice:
            parts.append(self._scene.notice)
        parts.append(self.session.header_text())
        self._header(" | ".join(parts))



def _build_scene(obj, report):
    """The other objects' snap points (C22), or None when they cannot be read: drawing still works,
    only snapping to other objects is off. bpy.context stands in for the operator's context
    because the projector factories read it too. No face normals: only the 3D tool uses them."""
    try:
        return draw3d.build_scene_source(bpy.context, obj, normals=False)
    except Exception as exc:
        traceback.print_exc()
        report({'WARNING'}, f"Snapping to other objects is unavailable: {exc}")
        return None


class SKETCH_OT_draw(bpy.types.Operator):
    bl_idname = "sketch.draw"
    bl_label = "Sketch"
    bl_description = "Draw polylines, rectangles, circles and arcs on the sketch plane"
    bl_options = {'UNDO'}

    mode: EnumProperty(name="Mode", items=MODES, default=POLYLINE)

    @classmethod
    def poll(cls, context):
        area = context.area
        return (context.mode == 'EDIT_MESH' and area is not None and area.type == 'VIEW_3D'
                and build.is_planar_sketch(context.active_object))

    def invoke(self, context, event):
        if modal_common.any_sketch_modal_live():
            # A click in another viewport or a transparent part of the UI reaches the tool keymap
            # while a run is active. A second run would edit the same bmesh, so none starts and
            # the event still reaches whatever is under the pointer. No report: Blender discards
            # the reports of an invoke that passes the event through, and a click on the UI
            # should stay silent anyway.
            return {'PASS_THROUGH'}
        obj = context.active_object
        projector_factory = modal_common.projector_factory(obj)
        if event.type == 'LEFTMOUSE' and not modal_common.over_region(projector_factory(), event):
            # The tool keymap also sees presses in the transparent part of the sidebar, on the HUD
            # and on the navigation gizmo. They belong to the UI, not to a new run.
            return {'PASS_THROUGH'}
        prefs = get_prefs(context)
        mode = context.window_manager.sketch_tools_shape if self.mode == FROM_HEADER else self.mode
        units = context.scene.unit_settings
        # The grid increment is a DISTANCE property, which Blender already stores in local units.
        # Typed lengths and the readout are in the scene base unit, so those convert.
        session = DrawSession(
            mode,
            grid=prefs.grid_increment if prefs else draw_logic.DEFAULT_GRID_INCREMENT,
            circle_segments=prefs.circle_segments if prefs else draw_logic.DEFAULT_CIRCLE_SEGMENTS,
            unit_scale=draw_logic.effective_unit_scale(units.system, units.scale_length))
        header, redraw = modal_common.area_callbacks(context.area)
        region = context.region
        scene = _build_scene(obj, self.report)
        try:
            controller = DrawController(
                SketchEditor(obj), session, projector_factory,
                prefs.snap_pixel_threshold if prefs else draw_logic.DEFAULT_SNAP_PIXELS,
                header=header, redraw=redraw,
                region_pointer=region.as_pointer() if region is not None else None,
                scene=scene, scene_view_factory=draw3d.projector_factory(obj, scene) if scene is not None else None,
                snap_toggle=lambda: magnet_on(bpy.context))
        except Exception:
            if scene is not None:
                scene.clear()
            raise
        self._controller = controller
        try:
            # A key start (R, O) has no press to place a point from; it waits for the first click.
            result = controller.start(event) if event.type == 'LEFTMOUSE' else 'RUNNING_MODAL'
            if result != 'RUNNING_MODAL':
                return {result}
            draw = overlay.DrawOverlay(controller.snapshot, controller.region_pointer)
            draw.add()
            controller.add_cleanup(draw.remove)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            traceback.print_exc()
            controller.abort()
            self.report({'ERROR'}, f"Sketch tool failed to start: {exc}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        controller = self._controller
        try:
            return {controller.handle(event)}
        except Exception as exc:
            traceback.print_exc()
            controller.abort()
            self.report({'ERROR'}, f"Sketch tool failed: {exc}")
            return {'CANCELLED'}

    def cancel(self, context):
        self._controller.abort()


def close_all():
    for controller in list(_live):
        controller.abort()


classes = (SKETCH_OT_draw,)
