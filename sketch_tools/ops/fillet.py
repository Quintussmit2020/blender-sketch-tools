"""sketch.fillet: the modal that rounds the selected sketch corners (plan 8.4).

Layers, as in the draw tool:
    FilletEditor      reads the selected corners from the edit bmesh, writes the committed radius
    FilletController  turns Blender events plus a view into core/fillet_logic.FilletSession events
    SKETCH_OT_fillet  owns the operator lifecycle: draw handler, header text, cleanup

The operator is usually started from a panel button, so the pointer starts over the sidebar and
the modal's context region is not the viewport. Everything therefore works in window coordinates
against the 3D view's WINDOW region, which is looked up once at invoke.

Nothing is written while the modal runs: the arcs are only previewed. Cancelling therefore has
nothing to restore, and committing is one edit-mesh update and so one undo step."""

import traceback

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

from ..core import attributes, build, draw_logic
from ..core.fillet_logic import FilletSession, find_corners
from ..prefs import get_prefs
from ..ui import fillet_overlay
from . import modal_common
from .modal_common import (
    MODIFIER_KEYS, NAVIGATION, NUMPAD_CHARS, POINTER_MOVES, PRESSES, SYSTEM_PREFIXES, TOP_ROW_CHARS,
    ViewProjector, navigation_gizmo_rect, overlay_rects,
)

_KEYS = {'RET': 'ENTER', 'NUMPAD_ENTER': 'ENTER', 'ESC': 'ESC', 'BACK_SPACE': 'BACKSPACE'}

# Pixels between the two points that measure the on-screen size of a pixel.
PROBE_PIXELS = 100.0

NO_CORNERS = "No filletable vertex selected (chain endpoints, junctions and straight corners are skipped)"

STALE_CORNERS = "Fillet cancelled: a selected vertex was deleted during the run"

# Controllers of running modals, so unregistering the add-on can wind them down.
_live = modal_common.live_set()


class FilletEditor:
    """The sketch's edit bmesh, seen as a list of corners to round."""

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.rad, self.ids = attributes.ensure_bm_layers(self.bm)
        self._verts = ()

    def alive(self):
        try:
            return self.bm.is_valid and self.mesh.is_editmode
        except ReferenceError:
            return False

    def collect(self):
        """(corners, skipped) for the selected vertices (find_corners)."""
        bm = self.bm
        bm.verts.index_update()
        self._verts = tuple(bm.verts)
        return find_corners(
            [(v.co.x, v.co.y) for v in self._verts],
            [v[self.rad] for v in self._verts],
            [(e.verts[0].index, e.verts[1].index) for e in bm.edges],
            [v.index for v in self._verts if v.select])

    def commit(self, corners, radius):
        """Write radius to the given corners with a fresh id each, or id 0 for a radius of 0 (C10),
        then rebuild the output the way the panel field does. Returns False, having written
        nothing, when a panel operator deleted a corner's vertex during the run."""
        if not all(self._verts[corner.index].is_valid for corner in corners):
            return False
        for corner in corners:
            v = self._verts[corner.index]
            v[self.rad] = radius
            v[self.ids] = attributes.new_fillet_id() if radius > 0.0 else 0
        bmesh.update_edit_mesh(self.mesh, loop_triangles=False, destructive=False)
        build.safe_rebuild(self.obj)
        build.rebuild_dependents(self.obj)
        return True


class FilletView(ViewProjector):
    """The draw tool's projector, plus window coordinates and the on-screen size of a pixel."""

    def to_region(self, window):
        return (window[0] - self.region.x, window[1] - self.region.y)

    def units_per_pixel(self, xy):
        """Sketch local units one pixel spans at the depth of the sketch point xy, so a drag moves
        the radius as far as the pointer moves over the sketch at any zoom. None behind the camera."""
        world = self.world @ Vector((xy[0], xy[1], 0.0))
        origin = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, world)
        if origin is None:
            return None
        a = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, origin, world)
        b = view3d_utils.region_2d_to_location_3d(
            self.region, self.rv3d, origin + Vector((PROBE_PIXELS, 0.0)), world)
        # Vectors are float32: a one pixel step far from the world origin would drown in rounding.
        step = (self.inverse.to_3x3() @ (b - a)).length / PROBE_PIXELS
        return step if step > 0.0 else None


def _safely(fn, *args):
    """Cleanup and redraw calls on an area that may be gone already."""
    try:
        fn(*args)
    except ReferenceError:
        pass
    except Exception:
        traceback.print_exc()


class FilletController:
    """Blender events in, a committed radius and an operator result out.

    view_factory() returns an object with contains(mouse, window), to_region(window),
    to_screen(xy) and units_per_pixel(xy), or None when the view is gone. header(text_or_None),
    redraw() and report(level, text) are callbacks. region_pointer is the as_pointer() of the
    region the overlay draws in."""

    def __init__(self, editor, session, view_factory, header, redraw, report, region_pointer=None):
        self.editor = editor
        self._report = report
        self.session = session
        self._view_factory = view_factory
        self._header = header
        self._redraw = redraw
        self.region_pointer = region_pointer
        self._cleanups = []
        # Buttons whose last press went on to Blender; their release and click events follow it.
        self._passed = set()
        n = len(session.corners)
        self._anchor = (sum(c.p.x for c in session.corners) / n, sum(c.p.y for c in session.corners) / n)
        self._upp = None
        self.mouse = None
        self.closed = False
        self._update_header()
        _live.add(self)

    def add_cleanup(self, fn):
        self._cleanups.append(fn)

    def close(self):
        """Idempotent. Every exit path ends here: commit, cancel, Esc, error, unregister."""
        if self.closed:
            return
        self.closed = True
        _live.discard(self)
        for fn in reversed(self._cleanups):
            _safely(fn)
        self._cleanups.clear()
        _safely(self._header, None)

    def abort(self):
        """Close without writing. For errors, a failed start and add-on unregister."""
        self.close()
        _safely(self._redraw)

    def snapshot(self):
        """(preview, mouse, to_screen) for the overlay, None when there is nothing to draw."""
        if self.closed:
            return None
        view = self._view_factory()
        if view is None:
            return None
        return self.session.preview(), self.mouse, view.to_screen

    # -- events ------------------------------------------------------------

    def start(self, event):
        """Seed the drag from the invoking event when it is already over the viewport (a keymap
        start); from a panel button it is not, and the first move over the viewport does it."""
        view = self._view_factory()
        if view is not None and self._inside(view, event):
            self._move(view, event)

    def handle(self, event):
        """Returns 'RUNNING_MODAL', 'PASS_THROUGH', 'FINISHED' or 'CANCELLED'."""
        view = self._view_factory() if self.editor.alive() else None
        if view is None:
            self.abort()
            return 'CANCELLED'
        etype = event.type
        if etype in NAVIGATION or etype.startswith('NDOF_') or etype.startswith(SYSTEM_PREFIXES):
            return 'PASS_THROUGH'
        if etype in POINTER_MOVES or etype in MODIFIER_KEYS:
            if not self._inside(view, event):
                self._leave()
                return 'PASS_THROUGH'
            self._move(view, event)
            return 'RUNNING_MODAL'
        if etype in ('LEFTMOUSE', 'RIGHTMOUSE'):
            return self._button(view, event)
        if etype == 'ESC' and event.value == 'PRESS':
            return self._session_event(("KEY", "ESC"))
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

    def _button(self, view, event):
        etype = event.type
        if etype == 'LEFTMOUSE' and event.alt:
            # Alt+LMB navigates. Recorded like any passed press so its release still passes
            # through when Alt is let go first.
            if event.value in PRESSES:
                self._passed.add(etype)
            return 'PASS_THROUGH'
        if event.value in PRESSES:
            if not self._inside(view, event):
                # Header, toolbar, sidebar and gizmos keep working during the run.
                self._passed.add(etype)
                return 'PASS_THROUGH'
            self._passed.discard(etype)
            if event.value == 'DOUBLE_CLICK':
                return 'RUNNING_MODAL'
            return self._session_event(("KEY", "LMB" if etype == 'LEFTMOUSE' else "RMB"))
        # Release, CLICK and CLICK_DRAG of a press that went on to Blender pass through too.
        return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'

    # -- translation -------------------------------------------------------

    @staticmethod
    def _window(event):
        return (event.mouse_x, event.mouse_y)

    def _inside(self, view, event):
        window = self._window(event)
        return view.contains(view.to_region(window), window)

    def _leave(self):
        """The pointer left the viewport: stop drawing the readout there."""
        self.session.handle(("LEAVE",))
        if self.mouse is not None:
            self.mouse = None
            self._redraw()

    def _move(self, view, event):
        self.mouse = view.to_region(self._window(event))
        upp = view.units_per_pixel(self._anchor)
        if upp:
            self._upp = upp
        self.session.handle(("MOVE", self.mouse[0], self._upp, bool(event.ctrl)))
        self._after()

    def _session_event(self, session_event):
        self.session.handle(session_event)
        if self.session.done:
            return self._finish()
        self._after()
        return 'RUNNING_MODAL'

    def _finish(self):
        session = self.session
        result = 'CANCELLED' if session.cancelled else 'FINISHED'
        if not session.cancelled and not self.editor.commit(session.corners, session.radius):
            result = 'CANCELLED'
            self._report('WARNING', STALE_CORNERS)
        self.close()
        _safely(self._redraw)
        return result

    def _after(self):
        self._update_header()
        self._redraw()

    def _update_header(self):
        self._header(self.session.header_text())


def _window_region(context):
    """(area, region) of the 3D view the modal works in, or an error text. The invoking region may
    be the sidebar, so the area's WINDOW region is taken."""
    area = context.area
    if area is None or area.type != 'VIEW_3D':
        return None, "Fillet needs a 3D viewport"
    if len(area.spaces.active.region_quadviews):
        return None, "Fillet does not work in quad view, switch to a single viewport"
    for region in area.regions:
        if region.type == 'WINDOW' and region.data is not None:
            return (area, region), None
    return None, "Fillet needs a 3D viewport"


def _view_factory(obj, area_pointer):
    """The area is looked up on every call: a captured Area dangles once it is closed."""
    def factory():
        try:
            if obj.name not in bpy.data.objects:
                return None
            area = modal_common.find_area(area_pointer)
            if area is None or area.type != 'VIEW_3D':
                return None
            region = modal_common.window_region(area)
            space = area.spaces.active
            if region is None or region.data is None or space is None:
                return None
            rects = overlay_rects(area)
            gizmo = navigation_gizmo_rect(region, area, space, bpy.context.preferences)
            if gizmo is not None:
                rects.append(gizmo)
            return FilletView(region, region.data, obj.matrix_world, rects)
        except ReferenceError:
            return None
    return factory


class SKETCH_OT_fillet(bpy.types.Operator):
    bl_idname = "sketch.fillet"
    bl_label = "Fillet"
    bl_description = ("Round the selected sketch corners: drag or type the radius, Ctrl snaps to the "
                      "grid, Enter or click confirms, Esc cancels")
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        area = context.area
        return (context.mode == 'EDIT_MESH' and area is not None and area.type == 'VIEW_3D'
                and build.is_planar_sketch(context.active_object))

    def invoke(self, context, event):
        if modal_common.any_sketch_modal_live():
            # Fillet starts from the panel or the search only, no keymap re-enters it, and
            # Blender shows the reports of CANCELLED but not of PASS_THROUGH.
            self.report({'WARNING'}, modal_common.BUSY_MESSAGE)
            return {'CANCELLED'}
        found, error = _window_region(context)
        if found is None:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        area, region = found
        obj = context.active_object
        editor = FilletEditor(obj)
        corners, skipped = editor.collect()
        if not corners:
            self.report({'WARNING'}, NO_CORNERS)
            return {'CANCELLED'}

        prefs = get_prefs(context)
        units = context.scene.unit_settings
        session = FilletSession(
            corners, skipped, obj.sketch_tools.fillet_segments,
            grid=prefs.grid_increment if prefs else draw_logic.DEFAULT_GRID_INCREMENT,
            unit_scale=draw_logic.effective_unit_scale(units.system, units.scale_length))
        header, redraw = modal_common.area_callbacks(area)
        controller = FilletController(
            editor, session, _view_factory(obj, area.as_pointer()),
            header=header, redraw=redraw,
            report=lambda level, text: self.report({level}, text),
            region_pointer=region.as_pointer())
        self._controller = controller
        try:
            controller.start(event)
            draw = fillet_overlay.FilletOverlay(controller.snapshot, controller.region_pointer)
            draw.add()
            controller.add_cleanup(draw.remove)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            traceback.print_exc()
            controller.abort()
            self.report({'ERROR'}, f"Fillet failed to start: {exc}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        controller = self._controller
        try:
            return {controller.handle(event)}
        except Exception as exc:
            traceback.print_exc()
            controller.abort()
            self.report({'ERROR'}, f"Fillet failed: {exc}")
            return {'CANCELLED'}

    def cancel(self, context):
        self._controller.abort()


def close_all():
    for controller in list(_live):
        controller.abort()


classes = (SKETCH_OT_fillet,)
