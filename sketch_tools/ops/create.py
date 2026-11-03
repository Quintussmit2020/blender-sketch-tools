import bmesh
import bpy
from bpy.props import EnumProperty
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from ..core import attributes, build
from ..prefs import get_prefs
from ..ui.tool import TOOL_ID

PLANES = (
    ('VIEW', "View", "Aligned to the viewport, origin at the 3D cursor"),
    ('XY', "XY", "Top plane, normal +Z, origin at the 3D cursor"),
    ('XZ', "XZ", "Front plane, normal -Y, origin at the 3D cursor"),
    ('YZ', "YZ", "Side plane, normal +X, origin at the 3D cursor"),
    ('FACE', "Face", "Click a face in the viewport"),
)

# Local (X, Y, Z) axes in world space for the axis planes; Z = X cross Y.
_AXIS_PLANES = {
    'XY': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'XZ': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    'YZ': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
}

# Events that keep working while waiting for the face click, so the user can navigate.
_NAV_EVENTS = {
    'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'MIDDLEMOUSE',
    'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'MOUSESMARTZOOM',
}
# Regions that can overlap the viewport's WINDOW region (region overlap).
_OVERLAY_REGIONS = {'UI', 'TOOLS', 'HEADER', 'TOOL_HEADER', 'ASSET_SHELF', 'ASSET_SHELF_HEADER'}

DRAW_TOOL = TOOL_ID
_tool_error_shown = False


def frame_matrix(origin, x, y, z):
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = origin
    return m


def axis_plane_matrix(plane, origin):
    x, y, z = _AXIS_PLANES[plane]
    return frame_matrix(Vector(origin), Vector(x), Vector(y), Vector(z))


def face_matrix(location, normal, view_rotation):
    """+Z = face normal, +X = view right projected onto the face."""
    z = Vector(normal).normalized()
    candidates = (
        view_rotation @ Vector((1.0, 0.0, 0.0)),
        view_rotation @ Vector((0.0, 1.0, 0.0)),
        Vector((1.0, 0.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
    )
    for candidate in candidates:
        x = candidate - z * candidate.dot(z)
        if x.length > 1e-6:
            break
    x.normalize()
    return frame_matrix(Vector(location), x, z.cross(x), z)


def _tool_target(context):
    """(window, area, region) of the 3D view the operator was invoked from, else None."""
    area = context.area
    if context.window is None or area is None or area.type != 'VIEW_3D':
        return None
    for region in area.regions:
        if region.type == 'WINDOW':
            return context.window, area, region
    return None


def _set_tool(window, area, region):
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.wm.tool_set_by_id(name=DRAW_TOOL)
    # An unknown tool id cancels with a warning instead of raising.
    if result != {'FINISHED'}:
        raise RuntimeError(f"tool_set_by_id returned {set(result)}")


def activate_draw_tool(context):
    """Make the Sketch tool the active tool of the invoking 3D view. Never fails sketch creation."""
    global _tool_error_shown
    try:
        target = _tool_target(context)
        if target is None:
            return
        _set_tool(*target)
    except Exception as exc:
        if not _tool_error_shown:
            _tool_error_shown = True
            print(f"Sketch Tools: could not activate {DRAW_TOOL}: {exc}")


def _new_sketch_object(context, name, matrix, ensure_attrs):
    """A wire object with the sketch display conventions, linked and active, still in object mode."""
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    mesh = bpy.data.meshes.new(name)
    ensure_attrs(mesh)
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    obj.matrix_world = matrix
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    return obj


def _make_active_and_edit(context, obj):
    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    context.tool_settings.mesh_select_mode = (True, False, False)


def create_sketch(context, matrix):
    obj = _new_sketch_object(context, "Sketch", matrix, attributes.ensure_attributes)
    st = obj.sketch_tools
    st.is_sketch = True
    prefs = get_prefs(context)
    if prefs is not None:
        st.fillet_segments = prefs.default_fillet_segments
    _make_active_and_edit(context, obj)
    return obj


def create_path_3d(context):
    """A 3D path at the world origin with an identity matrix (C20), swept with the default
    square profile. The output is created while still in object mode."""
    obj = _new_sketch_object(context, "Path3D", Matrix.Identity(4), attributes.ensure_path_attributes)
    st = obj.sketch_tools
    # is_path_3d first: the rebuild that is_sketch triggers must already see a 3D path.
    st.is_path_3d = True
    st.is_sketch = True
    st.output_mode = 'SWEEP'
    _make_active_and_edit(context, obj)
    return obj


def reorient_sketch(obj, new_world):
    """Give the sketch a new plane, keeping every vertex in place in world space,
    then flatten onto the new plane (local z = 0)."""
    to_new_local = new_world.inverted_safe() @ obj.matrix_world
    mesh = obj.data
    if mesh.is_editmode:
        bm = bmesh.from_edit_mesh(mesh)
        for v in bm.verts:
            co = to_new_local @ v.co
            co.z = 0.0
            v.co = co
        bmesh.update_edit_mesh(mesh)
    else:
        for v in mesh.vertices:
            co = to_new_local @ v.co
            co.z = 0.0
            v.co = co
        mesh.update()
    obj.matrix_world = new_world
    build.rebuild(obj)


class _FacePickMixin:
    """Short modal started from a panel button: the mouse is not over the viewport yet,
    so wait for a click in any 3D viewport and ray cast from there."""

    def start_pick(self, context):
        if context.window is None or context.area is None:
            self.report({'ERROR'}, "Face picking needs an interactive viewport")
            return {'CANCELLED'}
        self._area_ptr = context.area.as_pointer()
        context.area.header_text_set("Click a face")
        context.window.cursor_modal_set('EYEDROPPER')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def end_pick(self, context):
        for area in context.window.screen.areas:
            if area.as_pointer() == self._area_ptr:
                area.header_text_set(None)
                break
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as exc:
            self.end_pick(context)
            self.report({'ERROR'}, f"Face pick failed: {exc}")
            return {'CANCELLED'}

    def _modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
            self.end_pick(context)
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._ray_cast(context, event)
            if hit is None:
                return {'RUNNING_MODAL'}
            location, normal, view_rotation = hit
            if location is None:
                self.report({'WARNING'}, "No face under the cursor")
                return {'RUNNING_MODAL'}
            self.end_pick(context)
            return self.use_face(context, face_matrix(location, normal, view_rotation))

        if (event.type in _NAV_EVENTS
                or event.type.startswith('NUMPAD_')
                or event.type.startswith('NDOF_')):
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}

    @staticmethod
    def _ray_cast(context, event):
        """None when the click is not over a viewport; (None, None, rot) on a miss."""
        mx, my = event.mouse_x, event.mouse_y

        def inside(region):
            # Hidden regions can still report a size.
            if region.width <= 1 or region.height <= 1:
                return False
            return (region.x <= mx < region.x + region.width
                    and region.y <= my < region.y + region.height)

        for area in context.window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            if any(r.type in _OVERLAY_REGIONS and inside(r) for r in area.regions):
                continue
            for region in area.regions:
                if region.type != 'WINDOW' or region.data is None or not inside(region):
                    continue
                rv3d = region.data
                coord = (mx - region.x, my - region.y)
                origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
                depsgraph = context.evaluated_depsgraph_get()
                found, location, normal, _index, _obj, _matrix = context.scene.ray_cast(
                    depsgraph, origin, direction)
                if not found:
                    return None, None, rv3d.view_rotation
                return location, normal, rv3d.view_rotation
        return None


class SKETCH_OT_new(_FacePickMixin, bpy.types.Operator):
    bl_idname = "sketch.new"
    bl_label = "New Sketch"
    bl_description = "Create a sketch object on the chosen plane and start editing it"
    bl_options = {'REGISTER', 'UNDO'}

    plane: EnumProperty(name="Plane", items=PLANES, default='XY')

    @classmethod
    def poll(cls, context):
        if context.mode == 'OBJECT':
            return True
        return context.mode == 'EDIT_MESH' and context.active_object is not None

    def invoke(self, context, event):
        if self.plane == 'FACE':
            return self.start_pick(context)
        return self.execute(context)

    def execute(self, context):
        if self.plane == 'FACE':
            self.report({'ERROR'}, "The Face plane needs a click in the viewport")
            return {'CANCELLED'}
        cursor = context.scene.cursor.location
        if self.plane == 'VIEW':
            space = context.space_data
            rv3d = getattr(space, "region_3d", None) if space is not None and space.type == 'VIEW_3D' else None
            if rv3d is None:
                self.report({'ERROR'}, "The View plane needs a 3D viewport")
                return {'CANCELLED'}
            matrix = rv3d.view_rotation.to_matrix().to_4x4()
            matrix.translation = cursor
        else:
            matrix = axis_plane_matrix(self.plane, cursor)
        create_sketch(context, matrix)
        activate_draw_tool(context)
        return {'FINISHED'}

    def use_face(self, context, matrix):
        create_sketch(context, matrix)
        activate_draw_tool(context)
        return {'FINISHED'}


class SKETCH_OT_new_path_3d(bpy.types.Operator):
    bl_idname = "sketch.new_path_3d"
    bl_label = "New 3D Path"
    bl_description = "Create a 3D path object and start drawing on object surfaces with the Sketch tool"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.mode == 'OBJECT':
            return True
        return context.mode == 'EDIT_MESH' and context.active_object is not None

    def execute(self, context):
        create_path_3d(context)
        activate_draw_tool(context)
        return {'FINISHED'}


class SKETCH_OT_set_plane_from_face(_FacePickMixin, bpy.types.Operator):
    bl_idname = "sketch.set_plane_from_face"
    bl_label = "Set Plane From Face"
    bl_description = "Move the sketch plane onto a clicked face, keeping the drawing in place"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode in {'OBJECT', 'EDIT_MESH'}
                and build.is_planar_sketch(context.active_object))

    def invoke(self, context, event):
        self._sketch_name = context.active_object.name
        return self.start_pick(context)

    def execute(self, context):
        self.report({'ERROR'}, "Set Plane From Face needs a click in the viewport")
        return {'CANCELLED'}

    def use_face(self, context, matrix):
        obj = bpy.data.objects.get(self._sketch_name)
        if not build.is_sketch(obj):
            self.report({'ERROR'}, "The sketch no longer exists")
            return {'CANCELLED'}
        reorient_sketch(obj, matrix)
        return {'FINISHED'}


classes = (SKETCH_OT_new, SKETCH_OT_new_path_3d, SKETCH_OT_set_plane_from_face)
