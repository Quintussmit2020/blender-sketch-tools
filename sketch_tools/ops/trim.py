"""sketch.trim and sketch.extend: modal edge tools (plan 8.5, 8.6, DECISIONS C16).

Three thin layers around core/trim_logic, which holds every decision:
    TrimEditor      applies a plan to the sketch's edit bmesh and can undo the whole session
    TrimController  turns Blender events plus a view projector into editor calls
    SKETCH_OT_trim / SKETCH_OT_extend    own the operator lifecycle: draw handler, header, cleanup

Like the draw tool, one run is one undo step: RMB or Enter keeps everything and finishes, Esc
puts the mesh back as it was. The controller takes the projector, threshold and header, redraw
and report callbacks as arguments, so tests drive it with fake events and a fake view."""

import math
import re
import traceback

import bmesh
import bpy

from ..core import attributes, build, draw_logic, trim_logic
from ..prefs import get_prefs
from ..ui import overlay
from . import modal_common
from .modal_common import MODIFIER_KEYS, NAVIGATION, POINTER_MOVES, PRESSES, SYSTEM_PREFIXES

TRIM = 'TRIM'
EXTEND = 'EXTEND'

_HELP = {
    TRIM: "Trim: LMB removes the highlighted segment, RMB or Enter done, Esc cancels all",
    EXTEND: "Extend: LMB on a loose end extends it to the next edge, RMB or Enter done, Esc cancels all",
}

_IDNAME = {TRIM: "sketch.trim", EXTEND: "sketch.extend"}
_LABEL = {TRIM: "Trim", EXTEND: "Extend"}

# Controllers of running modals, so unregistering the add-on can wind them down.
_live = modal_common.live_set()

SNAPSHOT_NAME = "sketch_trim_backup"
# The name Blender gives a second mesh of the same name carries a numeric suffix.
_SNAPSHOT_NAME = re.compile(re.escape(SNAPSHOT_NAME) + r"(\.\d+)?")
# Custom property on the copy. Unlike the name, it marks the mesh as ours: it survives memfile
# undo, saving and quit.blend, and a user's own mesh with the same name does not have it.
SNAPSHOT_TAG = "sketch_tools_snapshot"


def _remove_snapshot(mesh):
    # The copy owns a copy of the shape keys, which removing the mesh leaves behind.
    bpy.data.batch_remove([mesh] + ([mesh.shape_keys] if mesh.shape_keys else []))


def pin_snapshots(pinned):
    """Fake user on or off for the snapshots of the live Trim and Extend runs."""
    for controller in list(_live):
        controller.editor.pin_snapshot(pinned)


def remove_orphan_snapshots():
    """Remove snapshot meshes (named and tagged as ours) that only their fake user keeps alive,
    left in a .blend by an autosave, quit.blend or an undo step taken during a run. A live run
    owns its snapshot, so nothing is touched while any sketch modal runs."""
    if modal_common.any_sketch_modal_live():
        return
    for mesh in list(bpy.data.meshes):
        if (_SNAPSHOT_NAME.fullmatch(mesh.name) and mesh.get(SNAPSHOT_TAG) and mesh.library is None
                and mesh.users == (1 if mesh.use_fake_user else 0)):
            _remove_snapshot(mesh)


class TrimEditor:
    """Applies trim and extend plans to the edit bmesh of a sketch.

    The first edit copies the whole edit mesh into a temporary Mesh datablock, so restore() can
    undo the whole session exactly (every attribute, flag, vertex group and shape key travels
    with the copy): Blender cannot undo a modal operator's edits piecemeal, and one run is one
    undo step. release() drops the copy; every exit path of the controller ends there."""

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.rad, self.ids = attributes.ensure_bm_layers(self.bm)
        self._backup = None

    def alive(self):
        try:
            return self.bm.is_valid and self.mesh.is_editmode
        except ReferenceError:
            return False

    @property
    def dirty(self):
        return self._backup is not None

    def segments(self):
        """(edge, a, b) for every visible edge, a and b the local (x, y) of edge.verts."""
        out = []
        for edge in self.bm.edges:
            if edge.hide:
                continue
            v0, v1 = edge.verts
            out.append((edge, (v0.co.x, v0.co.y), (v1.co.x, v1.co.y)))
        return out

    def endpoints(self):
        """(vert, xy, neighbor_xy, edge) for every visible vertex that ends exactly one edge."""
        out = []
        for vert in self.bm.verts:
            if vert.hide or len(vert.link_edges) != 1:
                continue
            edge = vert.link_edges[0]
            if edge.hide:
                continue
            other = edge.other_vert(vert)
            out.append((vert, (vert.co.x, vert.co.y), (other.co.x, other.co.y), edge))
        return out

    # -- session ------------------------------------------------------------

    def _begin(self):
        if self._backup is not None:
            return
        # A copy of the Mesh, not bm.to_mesh() into a blank one: only a mesh that already has
        # shape keys keeps the bmesh's shape key layers.
        self.obj.update_from_editmode()
        self._backup = self.mesh.copy()
        self._backup.name = SNAPSHOT_NAME
        self._backup[SNAPSHOT_TAG] = True
        # The copy has no user, so an Outliner purge during the run would delete it and Esc
        # would have nothing to restore from. release() removes it explicitly.
        self._backup.use_fake_user = True

    def pin_snapshot(self, pinned):
        """Set or clear the snapshot's fake user. A file save clears it for the duration of the
        write (handlers.py), so a zero-user mesh is not written into the .blend."""
        if self._backup is None:
            return
        try:
            self._backup.use_fake_user = pinned
        except ReferenceError:
            pass

    def release(self):
        """Drop the snapshot, keeping the mesh as it is. Idempotent."""
        backup, self._backup = self._backup, None
        if backup is None:
            return
        try:
            _remove_snapshot(backup)
        except ReferenceError:
            # Already gone, for instance purged by a file load.
            pass

    def restore(self):
        """Put the mesh back as the first edit found it, element order included, which the
        fillet id cleanup (DECISIONS C10) depends on."""
        if self._backup is None:
            return
        bm = self.bm
        # clear() zeroes the edit bmesh's select mode and from_mesh() does not set it again;
        # without it Select > Invert and selection flushing misbehave until the mode changes.
        mode = bm.select_mode
        bm.clear()
        bm.from_mesh(self._backup, use_shape_key=True, shape_key_index=max(self.obj.active_shape_key_index, 0))
        bm.select_mode = mode
        # clear() drops the layers, from_mesh() makes them again: the old handles are stale.
        self.rad, self.ids = attributes.ensure_bm_layers(bm)
        self.release()
        self._update()

    def _update(self):
        self.bm.verts.index_update()
        self.bm.verts.ensure_lookup_table()
        bmesh.update_edit_mesh(self.mesh)

    # -- edits --------------------------------------------------------------

    def _split(self, edge, vert, fac):
        """Split edge fac of the way from vert. The new vertex is sharp (DECISIONS C10: Blender
        interpolates the end vertices' attributes onto it). Returns (piece, new_vert), piece
        being the part from vert to new_vert; edge keeps the other part."""
        selected = edge.select
        piece, new_vert = bmesh.utils.edge_split(edge, vert, fac)
        new_vert[self.rad] = 0.0
        new_vert[self.ids] = 0
        new_vert.select_set(selected)
        return piece, new_vert

    def trim(self, target):
        """Remove the planned interval of target.key and weld the split points (DECISIONS C16)."""
        self._begin()
        bm = self.bm
        edge = target.key
        plan = target.plan
        va, vb = edge.verts

        pieces = []
        groups = []
        start, start_t = va, 0.0
        for bound in plan.bounds:
            point = va.co.lerp(vb.co, bound.t)
            piece, new_vert = self._split(edge, start, (bound.t - start_t) / (1.0 - start_t))
            pieces.append(piece)
            start, start_t = new_vert, bound.t
            existing, created = [], [new_vert]
            for crossing, u in bound.hits:
                if not crossing.is_valid:
                    continue
                if u == 0.0 or u == 1.0:
                    existing.append(crossing.verts[int(u)])
                else:
                    created.append(self._split(crossing, crossing.verts[0], u)[1])
            groups.append((point, existing, created))
        pieces.append(edge)

        # Pieces run [0, first bound], [first, second], ... [last bound, 1]; the removed
        # interval is the one after the lower bound, or the first when there is none.
        bm.edges.remove(pieces[1] if plan.t_lo > 0.0 else pieces[0])
        for vert in (va, vb):
            if vert.is_valid and not vert.link_edges:
                bm.verts.remove(vert)
        for point, existing, created in groups:
            self._weld(point, existing, created)
        self._update()

    def _weld(self, point, existing, created):
        """Merge the vertices at one bound into one. An existing vertex is the target, so it
        keeps its position, fillet radius and id; remove_doubles would pick by coordinate order.
        Existing vertices that sit on the point without being a crossing (the end of an edge
        collinear with the trimmed one) join the group too."""
        existing = [v for v in existing if v.is_valid]
        created = [v for v in created if v.is_valid]
        taken = set(existing) | set(created)
        near = [v for v in self.bm.verts
                if v not in taken and math.hypot(v.co.x - point.x, v.co.y - point.y) <= trim_logic.WELD_DISTANCE]
        # Existing vertices first, the first of them is the target; with none, the first created.
        members = list(dict.fromkeys(existing + near + created))
        if len(members) > 1:
            bmesh.ops.weld_verts(self.bm, targetmap={v: members[0] for v in members[1:]})

    def extend(self, target):
        """Join the loose end to the planned hit, splitting the hit edge when it is not at a vertex."""
        self._begin()
        hit = target.hit
        crossing = hit.key
        if hit.u == 0.0 or hit.u == 1.0:
            end = crossing.verts[int(hit.u)]
        else:
            end = self._split(crossing, crossing.verts[0], hit.u)[1]
        if self.bm.edges.get((target.key, end)) is None:
            self.bm.edges.new((target.key, end))
        self._update()


class TrimController:
    """Blender events in, edits and an operator result out.

    projector_factory() returns an object with contains(mouse, window), ray_to_local(mouse) and
    to_screen(xy), or None when there is no usable view. header(text_or_None), redraw() and
    report(level, text) are callbacks. region_pointer is the as_pointer() of the region the
    modal runs in; the overlay draws only there."""

    def __init__(self, kind, editor, projector_factory, threshold, header, redraw, report,
                 region_pointer=None):
        self.kind = kind
        self.editor = editor
        self._projector_factory = projector_factory
        self._threshold = threshold
        self._header = header
        self._redraw = redraw
        self._report = report
        self.region_pointer = region_pointer
        self._cleanups = []
        self._passed = set()
        self.target = None
        self.mouse = None
        self.count = 0
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
        # Last, so a failing cleanup above cannot leave the temporary mesh behind.
        for fn in (self.editor.release, lambda: self._header(None)):
            try:
                fn()
            except Exception:
                traceback.print_exc()

    def _safely_redraw(self):
        """The area may be gone already (closed or retyped during the run)."""
        try:
            self._redraw()
        except ReferenceError:
            pass
        except Exception:
            traceback.print_exc()

    def abort(self):
        """Put the mesh back and close. For Esc, errors, a failed start and add-on unregister.
        When edit mode was left the edits so far stay (C16): the bmesh is gone, so only the
        snapshot is dropped."""
        try:
            if self.editor.alive():
                self.editor.restore()
        except Exception:
            traceback.print_exc()
        self.close()
        self._safely_redraw()

    def finish(self):
        self.close()
        self._safely_redraw()
        # Nothing changed: no undo step.
        return 'FINISHED' if self.count else 'CANCELLED'

    def snapshot(self):
        """(preview, mouse, to_screen) for the overlay, None when there is nothing to draw."""
        if self.closed or self.target is None or self.mouse is None:
            return None
        projector = self._projector_factory()
        if projector is None:
            return None
        if self.kind == TRIM:
            preview = trim_logic.trim_preview(self.target)
        else:
            preview = trim_logic.extend_preview(self.target)
        return preview, self.mouse, projector.to_screen

    # -- events ------------------------------------------------------------

    def handle(self, event):
        """Returns 'RUNNING_MODAL', 'PASS_THROUGH', 'FINISHED' or 'CANCELLED'."""
        if not self.editor.alive():
            self.abort()
            return 'CANCELLED'
        if self._projector_factory() is None:
            # The area was retyped or closed: nothing to point at any more. The edits made so
            # far stay, as when edit mode is left (C16).
            return self.finish()
        etype = event.type
        if (etype in NAVIGATION or etype.startswith('NDOF_')
                or etype.startswith(SYSTEM_PREFIXES)):
            return 'PASS_THROUGH'
        if etype in POINTER_MOVES or etype in MODIFIER_KEYS:
            if not self._inside(event):
                self._leave()
                return 'PASS_THROUGH'
            self._hover(event)
            return 'RUNNING_MODAL'
        if etype == 'LEFTMOUSE':
            if event.alt:
                # Alt+LMB navigates; recorded like any passed press so its release passes too.
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
            # Release, CLICK and CLICK_DRAG of a press that went on to Blender, kept until the
            # next press because CLICK may arrive before or after the release.
            return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'
        if etype == 'RIGHTMOUSE':
            if event.value in PRESSES:
                if not self._inside(event):
                    self._passed.add(etype)
                    return 'PASS_THROUGH'
                self._passed.discard(etype)
                return self.finish() if event.value == 'PRESS' else 'RUNNING_MODAL'
            return 'PASS_THROUGH' if etype in self._passed else 'RUNNING_MODAL'
        if etype.startswith('NUMPAD_') and etype != 'NUMPAD_ENTER':
            return 'PASS_THROUGH'
        if event.value != 'PRESS':
            return 'RUNNING_MODAL'
        if etype == 'ESC':
            self.abort()
            return 'CANCELLED'
        if etype in {'RET', 'NUMPAD_ENTER'}:
            return self.finish()
        return 'RUNNING_MODAL'

    # -- translation -------------------------------------------------------

    def _inside(self, event):
        return modal_common.over_region(self._projector_factory(), event)

    def _leave(self):
        """The pointer left the region: stop drawing the highlight there."""
        if self.mouse is not None or self.target is not None:
            self.mouse = None
            self.target = None
            self._redraw()

    def _find(self, mouse):
        projector = self._projector_factory()
        if projector is None:
            return None
        segments = self.editor.segments()
        if self.kind == TRIM:
            return trim_logic.find_trim(mouse, segments, projector, self._threshold)
        return trim_logic.find_extend(mouse, self.editor.endpoints(), segments, projector,
                                      self._threshold)

    def _hover(self, event):
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        self.target = self._find(self.mouse)
        self._after()

    def _click(self, event):
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        target = self._find(self.mouse)
        if target is not None:
            if self.kind == TRIM:
                self.editor.trim(target)
                self.count += 1
            elif target.hit is None:
                self._report('WARNING', "Extend: no edge in that direction")
            else:
                self.editor.extend(target)
                self.count += 1
        # The edit may have removed what the highlight pointed at.
        self.target = self._find(self.mouse) if target is not None else None
        self._after()
        return 'RUNNING_MODAL'

    def _after(self):
        self._update_header()
        self._redraw()

    def _update_header(self):
        text = _HELP[self.kind]
        if self.count:
            text = f"{text} ({self.count} done)"
        self._header(text)


def _relaunch(op, context, kind):
    """Started from a panel button, the modal would be bound to the sidebar region and see
    pointer coordinates relative to it. Run it again with the 3D view's main region instead."""
    area = context.area
    region = modal_common.window_region(area) if area is not None and area.type == 'VIEW_3D' else None
    if context.window is None or region is None:
        op.report({'ERROR'}, "This tool needs a 3D viewport")
        return {'CANCELLED'}
    group, name = _IDNAME[kind].split(".")
    with context.temp_override(window=context.window, area=area, region=region):
        result = getattr(getattr(bpy.ops, group), name)('INVOKE_DEFAULT')
    if 'RUNNING_MODAL' not in result:
        op.report({'WARNING'}, "The tool could not start")
    # The modal was started by the nested call; this one has nothing to keep or undo.
    return {'CANCELLED'}


def _invoke(op, context, kind):
    if modal_common.any_sketch_modal_live():
        op.report({'WARNING'}, modal_common.BUSY_MESSAGE)
        return {'CANCELLED'}
    region = context.region
    if region is None or region.type != 'WINDOW':
        return _relaunch(op, context, kind)
    obj = context.active_object
    prefs = get_prefs(context)
    header, redraw = modal_common.area_callbacks(context.area)
    controller = TrimController(
        kind, TrimEditor(obj), modal_common.projector_factory(obj),
        prefs.snap_pixel_threshold if prefs else draw_logic.DEFAULT_SNAP_PIXELS,
        header=header, redraw=redraw,
        report=lambda level, text: op.report({level}, text),
        region_pointer=region.as_pointer())
    op._controller = controller
    try:
        draw = overlay.DrawOverlay(controller.snapshot, controller.region_pointer)
        draw.add()
        controller.add_cleanup(draw.remove)
        context.window_manager.modal_handler_add(op)
    except Exception as exc:
        traceback.print_exc()
        controller.abort()
        op.report({'ERROR'}, f"{_LABEL[kind]} failed to start: {exc}")
        return {'CANCELLED'}
    return {'RUNNING_MODAL'}


def _modal(op, event):
    controller = op._controller
    try:
        return {controller.handle(event)}
    except Exception as exc:
        traceback.print_exc()
        controller.abort()
        op.report({'ERROR'}, f"{_LABEL[controller.kind]} failed: {exc}")
        return {'CANCELLED'}


def _poll(context):
    area = context.area
    return (context.mode == 'EDIT_MESH' and area is not None and area.type == 'VIEW_3D'
            and build.is_planar_sketch(context.active_object))


class SKETCH_OT_trim(bpy.types.Operator):
    bl_idname = "sketch.trim"
    bl_label = "Trim"
    bl_description = ("Click a sketch edge to remove the part between the crossings next to the click. "
                      "RMB or Enter finishes, Esc cancels everything done in this run")
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll(context)

    def invoke(self, context, event):
        return _invoke(self, context, TRIM)

    def modal(self, context, event):
        return _modal(self, event)

    def cancel(self, context):
        self._controller.abort()


class SKETCH_OT_extend(bpy.types.Operator):
    bl_idname = "sketch.extend"
    bl_label = "Extend"
    bl_description = ("Click a loose end to extend it along its edge to the first edge in the way. "
                      "RMB or Enter finishes, Esc cancels everything done in this run")
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll(context)

    def invoke(self, context, event):
        return _invoke(self, context, EXTEND)

    def modal(self, context, event):
        return _modal(self, event)

    def cancel(self, context):
        self._controller.abort()


def close_all():
    for controller in list(_live):
        controller.abort()


classes = (SKETCH_OT_trim, SKETCH_OT_extend)
