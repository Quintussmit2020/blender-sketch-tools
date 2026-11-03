"""Phase 7 part B: 3D path objects in the add-on (DECISIONS C19, C20), headless.

The path points are written straight into the mesh with their normals; drawing them in the
viewport (ops/draw3d.py) is covered elsewhere."""

import io
import os
import tempfile
import types
from contextlib import redirect_stdout

import bmesh
import bpy
from mathutils import Vector

import sketch_tools
from sketch_tools import props
from sketch_tools.core import build
from sketch_tools.core.attributes import ensure_path_attributes
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.core.resolve3d import PATH_NORMAL_ATTR
from sketch_tools.ops import create, fillet, subdivide, trim
from sketch_tools.ui import panel

from .test_addon import (
    SQUARE,
    _FakeLayout,
    _RecordingLayout,
    addon,
    assert_closed_and_consistent,
    count_rebuilds,
    make_sketch,
    only_event_handlers,
    output_bm,
    signed_volume,
)

# Two faces of the cube (-1..1): the top (normal +Z) and the +X side (normal +X). The corner
# itself is left out, because a point on a cube edge has no single normal.
Z = (0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)
CUBE_PATH = [
    ((-0.5, 0.0, 1.0), Z),
    ((0.6, 0.0, 1.0), Z),
    ((1.0, 0.0, 0.6), X),
    ((1.0, 0.0, -0.5), X),
]


def write_path(obj, points, closed=False):
    """Replace the path's geometry with points = [(co, normal)] joined in order."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # bm.clear() would drop the attribute layers too.
    bmesh.ops.delete(bm, geom=list(bm.verts), context='VERTS')
    layer = bm.verts.layers.float_vector.get(PATH_NORMAL_ATTR)
    assert layer is not None, "path has no normal attribute"
    verts = []
    for co, normal in points:
        v = bm.verts.new(co)
        v[layer] = normal
        verts.append(v)
    count = len(verts) if closed else len(verts) - 1
    for i in range(count):
        bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def make_path(points=CUBE_PATH, mode='SWEEP', closed=False):
    """sketch.new_path_3d, back in object mode, holding the given points."""
    assert bpy.ops.sketch.new_path_3d() == {'FINISHED'}
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='OBJECT')
    write_path(obj, points, closed)
    if obj.sketch_tools.output_mode != mode:
        obj.sketch_tools.output_mode = mode
    build.rebuild(obj)
    return obj


def out_mesh(obj):
    return obj.sketch_tools.output_object.data


def max_x(obj):
    return max(v.co.x for v in out_mesh(obj).vertices)


def poll_context(mode, active):
    return types.SimpleNamespace(mode=mode, area=types.SimpleNamespace(type='VIEW_3D'), active_object=active)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_new_path_3d_object_setup():
    with addon():
        assert bpy.ops.sketch.new_path_3d() == {'FINISHED'}
        obj = bpy.context.active_object
        assert obj.name == "Path3D" and obj.type == 'MESH'
        st = obj.sketch_tools
        assert st.is_sketch and st.is_path_3d
        assert st.output_mode == 'SWEEP' and st.sw_profile == 'SQUARE'
        assert obj.display_type == 'WIRE' and obj.show_in_front
        assert tuple(obj.location) == (0.0, 0.0, 0.0) and tuple(obj.scale) == (1.0, 1.0, 1.0)
        assert all(abs(a - b) < 1e-9 for row, ident in zip(obj.matrix_world, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
                   for a, b in zip(row, ident))
        assert obj.select_get() and bpy.context.view_layer.objects.active == obj
        assert obj.mode == 'EDIT'
        assert tuple(bpy.context.tool_settings.mesh_select_mode) == (True, False, False)
        for name, kind in ((PATH_NORMAL_ATTR, 'FLOAT_VECTOR'), (FILLET_RADIUS_ATTR, 'FLOAT'), (FILLET_ID_ATTR, 'INT')):
            attr = obj.data.attributes.get(name)
            assert attr is not None and attr.data_type == kind and attr.domain == 'POINT', name

        # The output exists (empty), parented to the path with the usual conventions.
        out = st.output_object
        assert out is not None and out.parent == obj and out.hide_select
        assert out.name == "Path3D.out"
        assert len(out.data.vertices) == 0


def test_new_path_3d_replaces_a_selection_and_works_from_edit_mode():
    with addon():
        sketch = make_sketch([SQUARE])
        bpy.ops.object.mode_set(mode='EDIT')
        assert bpy.ops.sketch.new_path_3d() == {'FINISHED'}
        path = bpy.context.active_object
        assert path != sketch and path.sketch_tools.is_path_3d
        assert not sketch.select_get()
        assert not sketch.sketch_tools.is_path_3d


def test_new_path_3d_activates_the_draw_tool_after_entering_edit_mode():
    with addon():
        original = create.activate_draw_tool
        seen = []

        def spy(context):
            seen.append((context.mode, context.active_object.sketch_tools.is_path_3d))
        create.activate_draw_tool = spy
        try:
            assert bpy.ops.sketch.new_path_3d() == {'FINISHED'}
        finally:
            create.activate_draw_tool = original
        assert seen == [('EDIT_MESH', True)], seen


def test_ensure_path_attributes_is_idempotent():
    mesh = bpy.data.meshes.new("PathAttrs")
    try:
        ensure_path_attributes(mesh)
        ensure_path_attributes(mesh)
        names = sorted(a.name for a in mesh.attributes if a.name.startswith("st_"))
        assert names == sorted([PATH_NORMAL_ATTR, FILLET_RADIUS_ATTR, FILLET_ID_ATTR]), names
    finally:
        bpy.data.meshes.remove(mesh)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def test_path_over_two_cube_faces_sweeps_to_a_manifold_solid():
    with addon():
        path = make_path()
        assert path.sketch_tools.warnings == "", path.sketch_tools.warnings
        bm = output_bm(path)
        try:
            assert len(bm.faces) > 0
            assert_closed_and_consistent(bm)
            assert signed_volume(bm) > 0.0
            assert all(e.is_manifold for e in bm.edges)
        finally:
            bm.free()
        # The output lives in the path's space.
        assert path.sketch_tools.output_object.parent == path


def test_open_path_without_end_caps_is_not_closed():
    with addon():
        path = make_path()
        path.sketch_tools.sw_cap_ends = False
        bm = output_bm(path)
        try:
            assert any(len(e.link_faces) == 1 for e in bm.edges)
        finally:
            bm.free()


def test_profile_properties_rebuild_a_3d_path():
    with addon():
        path = make_path()
        bm = output_bm(path)
        try:
            volume = signed_volume(bm)
        finally:
            bm.free()
        path.sketch_tools.sw_width = 0.4
        bm = output_bm(path)
        try:
            assert signed_volume(bm) > volume * 1.5
        finally:
            bm.free()
        path.sketch_tools.sw_profile = 'CYLINDER'
        path.sketch_tools.sw_segments = 8
        assert len(out_mesh(path).vertices) > 0
        assert path.sketch_tools.warnings == ""


def test_object_mode_geometry_change_rebuilds_via_the_handler():
    with addon():
        path = make_path()
        bpy.context.view_layer.update()
        before = max_x(path)
        calls, restore = count_rebuilds()
        try:
            path.data.vertices[3].co = (3.0, 0.0, -0.5)
            path.data.update()
            bpy.context.view_layer.update()
            assert calls[0] >= 1, "the depsgraph handler did not rebuild the 3D path"
            assert max_x(path) > before + 1.5, (before, max_x(path))
            settled = calls[0]
            for _ in range(3):
                bpy.context.view_layer.update()
            assert calls[0] == settled, "the rebuild keeps re-triggering itself"
        finally:
            restore()


def test_leaving_edit_mode_rebuilds_a_3d_path():
    with addon():
        path = make_path()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.view_layer.update()
        before = max_x(path)
        bm = bmesh.from_edit_mesh(path.data)
        bm.verts.ensure_lookup_table()
        bm.verts[3].co = Vector((3.0, 0.0, -0.5))
        bmesh.update_edit_mesh(path.data)
        bpy.context.view_layer.update()
        assert abs(max_x(path) - before) < 1e-6, "rebuilt while still in edit mode"
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        assert max_x(path) > before + 1.5


def test_rebuild_in_edit_mode_reads_the_edit_bmesh():
    with addon():
        path = make_path()
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(path.data)
        bm.verts.ensure_lookup_table()
        bm.verts[3].co = Vector((3.0, 0.0, -0.5))
        before = max_x(path)
        build.rebuild(path)
        assert max_x(path) > before + 1.5


def test_extrude_on_a_3d_path_warns_and_removes_the_output():
    with addon():
        path = make_path()
        name = path.sketch_tools.output_object.name
        path.sketch_tools.output_mode = 'EXTRUDE'
        assert path.sketch_tools.output_object is None
        assert name not in bpy.data.objects
        assert path.sketch_tools.warnings == "Extrude needs a planar sketch"
        # The mode is not lost: sweeping again builds a fresh output and clears the warning.
        path.sketch_tools.output_mode = 'SWEEP'
        assert path.sketch_tools.output_object is not None
        assert len(out_mesh(path).polygons) > 0
        assert path.sketch_tools.warnings == ""
        path.sketch_tools.output_mode = 'NONE'
        assert path.sketch_tools.output_object is None
        assert path.sketch_tools.warnings == ""


def test_extrude_on_a_planar_sketch_is_unchanged():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        assert sketch.sketch_tools.warnings == ""
        assert len(out_mesh(sketch).polygons) == 6


def test_apply_works_on_a_3d_path_output():
    with addon():
        path = make_path()
        path.select_set(True)
        bpy.context.view_layer.objects.active = path
        faces = len(out_mesh(path).polygons)
        assert bpy.ops.sketch.apply() == {'FINISHED'}
        applied = bpy.context.active_object
        assert applied.name == "Path3D" and applied.parent is None and not applied.hide_select
        assert applied.sketch_tools.is_sketch is False
        assert len(applied.data.polygons) == faces
        kept = bpy.data.objects["Path3D.sketch"]
        assert kept.sketch_tools.is_path_3d and kept.sketch_tools.output_object is None
        assert kept.sketch_tools.output_mode == 'NONE'
        assert kept.hide_get()


def test_fillet_attributes_are_ignored_and_not_cleaned_on_a_3d_path():
    with addon():
        path = make_path()
        mesh = path.data
        # A radius with id 0 is what C10 would reset on a planar sketch.
        mesh.attributes[FILLET_RADIUS_ATTR].data[1].value = 0.5
        mesh.update()
        before = max_x(path)
        build.rebuild(path)
        assert abs(mesh.attributes[FILLET_RADIUS_ATTR].data[1].value - 0.5) < 1e-6
        assert abs(max_x(path) - before) < 1e-6
        path.sketch_tools.output_mode = 'NONE'
        assert abs(mesh.attributes[FILLET_RADIUS_ATTR].data[1].value - 0.5) < 1e-6


def test_fillet_radius_field_does_not_write_to_a_3d_path():
    with addon():
        path = make_path()
        path.select_set(True)
        bpy.context.view_layer.objects.active = path
        for v in path.data.vertices:
            v.select = True
        bpy.context.window_manager.sketch_tools_fillet_radius = 0.3
        try:
            assert all(abs(d.value) < 1e-9 for d in path.data.attributes[FILLET_RADIUS_ATTR].data)
        finally:
            bpy.context.window_manager.sketch_tools_fillet_radius = 0.0


def test_a_planar_sketch_is_not_a_3d_path():
    with addon():
        sketch = make_sketch([SQUARE])
        path = make_path()
        assert build.is_planar_sketch(sketch) and not build.is_path_3d(sketch)
        assert build.is_path_3d(path) and not build.is_planar_sketch(path)
        assert not build.is_path_3d(None) and not build.is_planar_sketch(None)


# ---------------------------------------------------------------------------
# Polls and the custom profile
# ---------------------------------------------------------------------------

def test_2d_tools_refuse_a_3d_path():
    with addon():
        sketch = make_sketch([SQUARE])
        path = make_path()
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.collection.objects.link(plain)
        for cls in (fillet.SKETCH_OT_fillet, trim.SKETCH_OT_trim, trim.SKETCH_OT_extend,
                    subdivide.SKETCH_OT_subdivide):
            assert cls.poll(poll_context('EDIT_MESH', sketch)), cls
            assert not cls.poll(poll_context('EDIT_MESH', path)), cls
            assert not cls.poll(poll_context('EDIT_MESH', plain)), cls
            assert not cls.poll(poll_context('EDIT_MESH', None)), cls
        assert create.SKETCH_OT_set_plane_from_face.poll(types.SimpleNamespace(mode='OBJECT', active_object=sketch))
        assert not create.SKETCH_OT_set_plane_from_face.poll(
            types.SimpleNamespace(mode='OBJECT', active_object=path))
        assert not create.SKETCH_OT_set_plane_from_face.poll(
            types.SimpleNamespace(mode='EDIT_MESH', active_object=path))
        bpy.data.objects.remove(plain)


def test_custom_profile_poll_refuses_a_3d_path():
    with addon():
        sketch = make_sketch([SQUARE])
        other = make_sketch([SQUARE])
        path = make_path()
        assert not props._poll_custom_profile(sketch.sketch_tools, path)
        assert props._poll_custom_profile(sketch.sketch_tools, other)
        # The 3D path can use a planar sketch as its own custom profile.
        assert props._poll_custom_profile(path.sketch_tools, sketch)
        assert not build.is_profile_source(path)


def test_a_3d_path_set_as_a_profile_by_script_warns_instead_of_building():
    with addon():
        profile_source = make_path()
        user = make_sketch([SQUARE], mode='SWEEP')
        user.sketch_tools.sw_profile = 'CUSTOM'
        user.sketch_tools.sw_custom_object = profile_source
        assert "3D path" in user.sketch_tools.warnings
        assert user.sketch_tools.output_object is None


def test_a_3d_path_can_sweep_a_planar_sketch_profile_and_follows_its_edits():
    with addon():
        profile = make_sketch([[(-0.1, 0.0), (0.1, 0.0), (0.1, 0.2), (-0.1, 0.2)]])
        path = make_path()
        path.sketch_tools.sw_profile = 'CUSTOM'
        path.sketch_tools.sw_custom_object = profile
        assert path.sketch_tools.warnings == "", path.sketch_tools.warnings
        before = signed_volume_of(path)
        assert before > 0.0
        assert [o.name for o in build.dependents(profile)] == [path.name]

        profile.data.vertices[1].co = (0.3, 0.0, 0.0)
        profile.data.vertices[2].co = (0.3, 0.2, 0.0)
        profile.data.update()
        bpy.context.view_layer.update()
        assert signed_volume_of(path) > before * 1.2


def signed_volume_of(obj):
    bm = output_bm(obj)
    try:
        return signed_volume(bm)
    finally:
        bm.free()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class _PropEnumLayout(_RecordingLayout):
    """Adds the calls the 3D path panel makes and records operators."""

    def __init__(self, seen=None, ops=None, labels=None):
        super().__init__(seen)
        self.ops = ops if ops is not None else []
        self.labels = labels if labels is not None else []

    def _child(self, *_args, **_kwargs):
        return _PropEnumLayout(self.seen, self.ops, self.labels)

    column = row = box = _child

    def prop_enum(self, data, name, value, **_kwargs):
        prop = data.bl_rna.properties[name]
        assert value in {item.identifier for item in prop.enum_items}, value
        self.seen.append(f"{name}={value}")

    def operator(self, idname, **kwargs):
        self.ops.append(idname)
        return super().operator(idname, **kwargs)

    def operator_menu_enum(self, idname, prop, **kwargs):
        self.operator(idname)

    def label(self, **kwargs):
        self.labels.append(kwargs.get("text", ""))


def draw(cls, ctx):
    layout = _PropEnumLayout()
    cls.draw(types.SimpleNamespace(layout=layout), ctx)
    return layout


def test_panels_for_a_3d_path_show_only_what_applies():
    with addon():
        path = make_path()
        original_resolve = (build.resolve_bmesh, build.resolve_path_bmesh)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("panel draw resolved the path")

        build.resolve_bmesh = forbidden
        build.resolve_path_bmesh = forbidden
        try:
            for mode in ('EDIT_MESH', 'OBJECT'):
                ctx = types.SimpleNamespace(
                    mode=mode, active_object=path, window_manager=bpy.context.window_manager)
                assert all(cls.poll(ctx) for cls in panel.classes)

                create_layout = draw(panel.SKETCH_PT_create, ctx)
                assert "sketch.new_path_3d" in create_layout.ops
                assert "sketch.new" in create_layout.ops

                sketch_layout = draw(panel.SKETCH_PT_sketch, ctx)
                # Only the shape picker, and only where drawing is possible.
                assert sketch_layout.seen == (["sketch_tools_shape"] if mode == 'EDIT_MESH' else []), sketch_layout.seen
                assert sketch_layout.ops == [], sketch_layout.ops
                note = " ".join(sketch_layout.labels)
                assert "Sketch tool" in note, sketch_layout.labels
                # Object mode cannot draw: the note says to enter edit mode first.
                assert ("edit mode" in note) == (mode == 'OBJECT'), (mode, sketch_layout.labels)

                path.sketch_tools.warnings = "one\ntwo"
                with_warnings = draw(panel.SKETCH_PT_sketch, ctx)
                assert [t for t in with_warnings.labels if t in ("one", "two")] == ["one", "two"]
                path.sketch_tools.warnings = ""

                out_layout = draw(panel.SKETCH_PT_output, ctx)
                modes = [n for n in out_layout.seen if n.startswith("output_mode")]
                assert modes == ["output_mode=NONE", "output_mode=SWEEP"], modes
                assert "sw_profile" in out_layout.seen and "sw_width" in out_layout.seen
                assert not any(n.startswith("ex_") for n in out_layout.seen)
                assert "sketch.apply" in out_layout.ops and "sketch.rebuild" in out_layout.ops
        finally:
            build.resolve_bmesh, build.resolve_path_bmesh = original_resolve


def test_the_subdivide_menu_entry_is_not_offered_for_a_3d_path():
    from sketch_tools.ui import menus

    class _Menu:
        def __init__(self):
            self.layout = _PropEnumLayout()

    with addon():
        path = make_path()
        planar = make_sketch([SQUARE])
        for obj, expected in ((path, []), (planar, ["sketch.subdivide"]), (None, [])):
            menu = _Menu()
            menus._draw_subdivide(menu, types.SimpleNamespace(active_object=obj))
            assert menu.layout.ops == expected, (obj, menu.layout.ops)


def test_the_output_panel_of_a_3d_path_set_to_extrude_shows_no_extrude_fields():
    with addon():
        path = make_path()
        path.sketch_tools.output_mode = 'EXTRUDE'
        ctx = types.SimpleNamespace(mode='OBJECT', active_object=path, window_manager=bpy.context.window_manager)
        layout = draw(panel.SKETCH_PT_output, ctx)
        assert not any(n.startswith(("ex_", "sw_")) for n in layout.seen), layout.seen
        sketch_layout = draw(panel.SKETCH_PT_sketch, ctx)
        assert "Extrude needs a planar sketch" in sketch_layout.labels


def test_planar_sketch_panels_are_unchanged():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        ctx = types.SimpleNamespace(
            mode='EDIT_MESH', active_object=sketch, window_manager=bpy.context.window_manager)
        sketch_layout = draw(panel.SKETCH_PT_sketch, ctx)
        assert sorted(sketch_layout.seen) == [
            "fillet_segments", "sketch_tools_fillet_radius", "sketch_tools_shape"], sketch_layout.seen
        assert sketch_layout.ops == ["sketch.fillet", "sketch.trim", "sketch.extend", "sketch.subdivide"]
        out_layout = draw(panel.SKETCH_PT_output, ctx)
        assert "ex_height" in out_layout.seen


# ---------------------------------------------------------------------------
# Registration, undo, load
# ---------------------------------------------------------------------------

def test_registration_twice_in_a_row_with_the_path_operator():
    for _ in range(2):
        sketch_tools.register()
        try:
            assert hasattr(bpy.types, "SKETCH_OT_new_path_3d")
            assert "is_path_3d" in props.SketchToolsProps.bl_rna.properties
        finally:
            sketch_tools.unregister()
        assert not hasattr(bpy.types, "SKETCH_OT_new_path_3d")
    assert not hasattr(bpy.types.Object, "sketch_tools")


def test_undo_and_redo_rebuild_a_3d_path():
    with addon():
        path = make_path()
        name = path.name
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="base")
        path.sketch_tools.sw_width = 0.6
        bpy.ops.ed.undo_push(message="wider")
        narrow = None

        calls, restore = count_rebuilds()
        try:
            with only_event_handlers():
                bpy.ops.ed.undo()
            assert calls[0] >= 1, "undo_post did not rebuild"
            path = bpy.data.objects[name]
            assert abs(path.sketch_tools.sw_width - 0.2) < 1e-6
            narrow = signed_volume_of(path)

            after_undo = calls[0]
            with only_event_handlers():
                bpy.ops.ed.redo()
            assert calls[0] > after_undo, "redo_post did not rebuild"
            path = bpy.data.objects[name]
            assert abs(path.sketch_tools.sw_width - 0.6) < 1e-6
            assert signed_volume_of(path) > narrow * 2.0
        finally:
            restore()


def test_load_post_rebuilds_a_3d_path():
    with addon():
        path = make_path()
        name = path.name
        faces = len(out_mesh(path).polygons)
        assert faces > 0
        file_path = os.path.join(tempfile.gettempdir(), "sketch_tools_path3d_load_post_test.blend")
        try:
            with only_event_handlers():
                # A stale (empty) output, as an older file could hold.
                out_mesh(path).clear_geometry()
                bpy.ops.wm.save_as_mainfile(filepath=file_path, check_existing=False)
                bpy.ops.wm.open_mainfile(filepath=file_path)
            path = bpy.data.objects[name]
            assert path.sketch_tools.is_path_3d
            assert len(out_mesh(path).polygons) == faces, "load_post did not rebuild the 3D path"
            assert path.data.attributes.get(PATH_NORMAL_ATTR) is not None
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


def test_a_failing_3d_path_does_not_stop_a_planar_sketch_rebuild():
    with addon():
        path = make_path()
        good = make_sketch([SQUARE], mode='EXTRUDE')
        original = build.resolve_path_bmesh

        def broken(_bm):
            raise RuntimeError("boom")

        build.resolve_path_bmesh = broken
        try:
            good.data.vertices[2].co = (3.0, 3.0, 0.0)
            good.data.update()
            path.data.update()
            build.clear_error_log()
            with redirect_stdout(io.StringIO()):
                bpy.context.view_layer.update()
        finally:
            build.resolve_path_bmesh = original
        assert max(v.co.y for v in out_mesh(good).vertices) > 2.9, "the planar sketch was not rebuilt"
        assert path.sketch_tools.warnings == build.REBUILD_FAILED
