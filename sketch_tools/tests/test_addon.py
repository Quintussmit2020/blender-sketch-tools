"""Blender integration tests: registration, operators, live rebuild, Apply.

Each test registers the add-on in-process and unregisters it again, even on failure,
so the geometry tests in the same run see a clean Blender."""

import io
import itertools
import math
import os
import tempfile
import types
from contextlib import contextmanager, redirect_stdout

import bmesh
import bpy
from mathutils import Matrix, Vector

import sketch_tools
from sketch_tools import handlers, props
from sketch_tools.core import attributes, build
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.ops import create, modal_common
from sketch_tools.ops.create import face_matrix, reorient_sketch

OUTER = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
INNER = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)


_REAL_FIND_AREA = modal_common.find_area
_REAL_PROJECTOR_FACTORY = modal_common.projector_factory
_fake_areas = {}
_fake_pointers = itertools.count(7001)


def fake_area(**attrs):
    """A stand-in Area with a unique as_pointer(), for the operator shells."""
    pointer = next(_fake_pointers)
    return types.SimpleNamespace(as_pointer=lambda: pointer, **attrs)


def install_fake_area(area, projector_factory=None):
    """Let modal_common find the stand-in area (and use the given projector factory). Returns the
    function that takes it out again; the real functions come back once no stand-in is left, so
    shells restored in any order cannot leave a patch behind."""
    pointer = area.as_pointer()
    _fake_areas[pointer] = area
    modal_common.find_area = _fake_areas.get
    if projector_factory is not None:
        modal_common.projector_factory = projector_factory

    def restore():
        _fake_areas.pop(pointer, None)
        if not _fake_areas:
            modal_common.find_area = _REAL_FIND_AREA
            modal_common.projector_factory = _REAL_PROJECTOR_FACTORY
    return restore


@contextmanager
def addon():
    _clear_scene()
    sketch_tools.register()
    try:
        yield
    finally:
        try:
            _clear_scene()
        finally:
            sketch_tools.unregister()


def make_sketch(loops, mode='NONE'):
    """New XY sketch at the 3D cursor, back in object mode, holding the given closed loops."""
    assert bpy.ops.sketch.new(plane='XY') == {'FINISHED'}
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for loop in loops:
        verts = [bm.verts.new((x, y, 0.0)) for x, y in loop]
        for i in range(len(verts)):
            bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj.sketch_tools.output_mode = mode
    return obj


def output_bm(obj):
    out = obj.sketch_tools.output_object
    assert out is not None, "no output object"
    bm = bmesh.new()
    bm.from_mesh(out.data)
    return bm


def signed_volume(bm):
    total = 0.0
    for f in bm.faces:
        vs = [v.co for v in f.verts]
        for i in range(1, len(vs) - 1):
            total += vs[0].dot(vs[i].cross(vs[i + 1]))
    return total / 6.0


def assert_closed_and_consistent(bm):
    """Every edge has two faces that walk it in opposite directions (outward or inward, not mixed)."""
    for e in bm.edges:
        assert len(e.link_faces) == 2, f"edge {e.index} has {len(e.link_faces)} faces"
        directions = []
        for f in e.link_faces:
            for loop in f.loops:
                if loop.edge == e:
                    directions.append(loop.vert == e.verts[0])
        assert directions[0] != directions[1], f"edge {e.index} has inconsistent winding"


def matrices_close(a, b, eps=1e-6):
    return all(abs(a[i][j] - b[i][j]) < eps for i in range(4) for j in range(4))


def select_only(mesh, indices):
    for v in mesh.vertices:
        v.select = v.index in indices


def count_rebuilds():
    """Patch build.rebuild to count calls. Returns (counter list, restore function)."""
    original = build.rebuild
    calls = [0]

    def counting(obj):
        calls[0] += 1
        original(obj)

    build.rebuild = counting
    return calls, lambda: setattr(build, "rebuild", original)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_register_unregister_twice():
    _clear_scene()
    for _ in range(2):
        sketch_tools.register()
        sketch_tools.unregister()
    assert not hasattr(bpy.types.Object, "sketch_tools")
    assert not hasattr(bpy.types.WindowManager, "sketch_tools_fillet_radius")
    for handler_list in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post,
                         bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        assert not any(getattr(h, "__module__", "") == handlers.__name__ for h in handler_list)


def test_new_sketch_axis_planes():
    expected_axes = {
        'XY': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        'XZ': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        'YZ': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    }
    with addon():
        bpy.context.scene.cursor.location = (1.0, 2.0, 3.0)
        for plane, (x, y, z) in expected_axes.items():
            assert bpy.ops.sketch.new(plane=plane) == {'FINISHED'}
            obj = bpy.context.active_object
            m = obj.matrix_world
            for col, axis in enumerate((x, y, z)):
                got = Vector((m[0][col], m[1][col], m[2][col]))
                assert (got - Vector(axis)).length < 1e-6, f"{plane} axis {col}: {got}"
            assert (m.translation - Vector((1, 2, 3))).length < 1e-6
            st = obj.sketch_tools
            assert st.is_sketch
            assert st.fillet_segments == 8
            assert obj.display_type == 'WIRE'
            assert obj.show_in_front
            assert obj.select_get() and bpy.context.view_layer.objects.active == obj
            assert obj.mode == 'EDIT'
            assert tuple(bpy.context.tool_settings.mesh_select_mode) == (True, False, False)
            attrs = obj.data.attributes
            assert attrs[FILLET_RADIUS_ATTR].data_type == 'FLOAT'
            assert attrs[FILLET_RADIUS_ATTR].domain == 'POINT'
            assert attrs[FILLET_ID_ATTR].data_type == 'INT'
            assert attrs[FILLET_ID_ATTR].domain == 'POINT'
            bpy.ops.object.mode_set(mode='OBJECT')
        assert len([o for o in bpy.data.objects if o.select_get()]) == 1


def test_new_sketch_uses_fillet_segments_preference():
    with addon():
        entry = bpy.context.preferences.addons.new()
        entry.module = "sketch_tools"
        try:
            entry.preferences.default_fillet_segments = 5
            bpy.ops.sketch.new(plane='XY')
            assert bpy.context.active_object.sketch_tools.fillet_segments == 5
        finally:
            bpy.context.preferences.addons.remove(entry)


def test_square_in_square_extrudes_with_hole():
    with addon():
        bpy.context.scene.cursor.location = (1.0, 2.0, 3.0)
        sketch = make_sketch([OUTER, INNER], mode='EXTRUDE')
        st = sketch.sketch_tools
        out = st.output_object
        assert out is not None and out.name == "Sketch.out"
        assert out.parent == sketch
        assert out.hide_select
        assert matrices_close(out.matrix_parent_inverse, Matrix.Identity(4))
        assert matrices_close(out.matrix_basis, Matrix.Identity(4))
        assert out.users_collection[0] == sketch.users_collection[0]
        bpy.context.view_layer.update()
        assert matrices_close(out.matrix_world, sketch.matrix_world)

        bm = output_bm(sketch)
        try:
            assert_closed_and_consistent(bm)
            # Euler characteristic of a closed surface with one hole (a torus) is 0.
            assert len(bm.verts) - len(bm.edges) + len(bm.faces) == 0
            volume = signed_volume(bm)
            assert abs(volume - (16 - 4) * 2.0) < 1e-4, volume
        finally:
            bm.free()
        assert st.warnings == ""


def test_changing_height_rebuilds():
    with addon():
        sketch = make_sketch([OUTER, INNER], mode='EXTRUDE')
        sketch.sketch_tools.ex_height = 3.0
        bm = output_bm(sketch)
        try:
            assert abs(max(v.co.z for v in bm.verts) - 3.0) < 1e-6
            assert abs(signed_volume(bm) - 36.0) < 1e-4
        finally:
            bm.free()


def test_object_mode_vertex_move_rebuilds_via_depsgraph():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        bpy.context.view_layer.update()
        calls, restore = count_rebuilds()
        try:
            mesh = sketch.data
            mesh.vertices[2].co = (3.0, 3.0, 0.0)
            mesh.update()
            bpy.context.view_layer.update()
            assert calls[0] >= 1
            out_mesh = sketch.sketch_tools.output_object.data
            assert abs(max(v.co.x for v in out_mesh.vertices) - 3.0) < 1e-6

            # The rebuild's own writes must not keep triggering rebuilds.
            for _ in range(3):
                bpy.context.view_layer.update()
            settled = calls[0]
            mesh.update()
            bpy.context.view_layer.update()
            bpy.context.view_layer.update()
            assert calls[0] == settled + 1, (settled, calls[0])
        finally:
            restore()


def test_leaving_edit_mode_rebuilds():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.view_layer.update()
        bm = bmesh.from_edit_mesh(sketch.data)
        bm.verts.ensure_lookup_table()
        bm.verts[2].co = Vector((4.0, 4.0, 0.0))
        bmesh.update_edit_mesh(sketch.data)
        bpy.context.view_layer.update()
        out_mesh = sketch.sketch_tools.output_object.data
        assert abs(max(v.co.x for v in out_mesh.vertices) - 1.0) < 1e-6, "rebuilt during edit mode"

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        out_mesh = sketch.sketch_tools.output_object.data
        assert abs(max(v.co.x for v in out_mesh.vertices) - 4.0) < 1e-6


def test_fillet_radius_field_adds_arc_points():
    with addon():
        sketch = make_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]], mode='EXTRUDE')
        st = sketch.sketch_tools
        st.fillet_segments = 4
        out_mesh = st.output_object.data
        assert len(out_mesh.vertices) == 8

        # Object mode: through the attribute data.
        select_only(sketch.data, {0})
        bpy.context.window_manager.sketch_tools_fillet_radius = 0.5
        out_mesh = st.output_object.data
        assert len(out_mesh.vertices) == 2 * (3 + st.fillet_segments + 1), len(out_mesh.vertices)
        radii = [d.value for d in sketch.data.attributes[FILLET_RADIUS_ATTR].data]
        ids = [d.value for d in sketch.data.attributes[FILLET_ID_ATTR].data]
        assert radii == [0.5, 0.0, 0.0, 0.0], radii
        assert ids[0] != 0 and ids[1:] == [0, 0, 0], ids

        # Edit mode: through the edit bmesh; the rebuild reads the edit mesh.
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(sketch.data)
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            v.select = v.index == 2
        bmesh.update_edit_mesh(sketch.data)
        bpy.context.window_manager.sketch_tools_fillet_radius = 0.25
        out_mesh = st.output_object.data
        assert len(out_mesh.vertices) == 2 * (2 + 2 * (st.fillet_segments + 1)), len(out_mesh.vertices)
        bpy.ops.object.mode_set(mode='OBJECT')
        ids = [d.value for d in sketch.data.attributes[FILLET_ID_ATTR].data]
        assert ids[2] != 0 and ids[2] != ids[0]

        # Radius 0 clears the id too.
        select_only(sketch.data, {0})
        bpy.context.window_manager.sketch_tools_fillet_radius = 0.0
        assert sketch.data.attributes[FILLET_ID_ATTR].data[0].value == 0
        assert len(st.output_object.data.vertices) == 2 * (3 + st.fillet_segments + 1)


def test_cleanup_resets_copied_fillet_ids():
    with addon():
        sketch = make_sketch([SQUARE], mode='NONE')
        mesh = sketch.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        rad, ids = attributes.ensure_bm_layers(bm)
        bm.verts[0][rad] = 0.3
        bm.verts[0][ids] = 777
        bm.verts[2][rad] = 0.2  # id 0: written by something other than the add-on
        first = bm.verts[0]
        copied = bm.verts.new((-1.0, 0.0, 0.0))  # e.g. Ctrl+RMB extrude from vertex 0
        copied[rad] = 0.3
        copied[ids] = 777
        bm.edges.new((first, copied))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        sketch.sketch_tools.output_mode = 'EXTRUDE'
        radii = [d.value for d in mesh.attributes[FILLET_RADIUS_ATTR].data]
        id_values = [d.value for d in mesh.attributes[FILLET_ID_ATTR].data]
        assert abs(radii[0] - 0.3) < 1e-6 and id_values[0] == 777
        assert radii[2] == 0.0 and id_values[2] == 0
        assert radii[4] == 0.0 and id_values[4] == 0

        bm = bmesh.new()
        bm.from_mesh(mesh)
        try:
            assert attributes.cleanup_fillet_ids(bm) is False
        finally:
            bm.free()


def test_apply_keeps_and_hides_sketch():
    with addon():
        bpy.context.scene.cursor.location = (1.0, 0.0, 0.0)
        sketch = make_sketch([OUTER, INNER], mode='EXTRUDE')
        bpy.ops.object.mode_set(mode='EDIT')
        assert bpy.ops.sketch.apply() == {'FINISHED'}

        applied = bpy.data.objects["Sketch"]
        assert sketch.name == "Sketch.sketch"
        assert not applied.sketch_tools.is_sketch
        assert applied.parent is None
        assert not applied.hide_select
        assert len(applied.data.polygons) > 0
        assert (applied.matrix_world.translation - Vector((1, 0, 0))).length < 1e-6
        assert sketch.mode == 'OBJECT'
        assert sketch.hide_get()
        assert sketch.sketch_tools.output_object is None
        assert sketch.sketch_tools.output_mode == 'NONE'
        assert bpy.context.view_layer.objects.active == applied

        build.rebuild_all()
        bpy.context.view_layer.update()
        assert not [o.name for o in bpy.data.objects if o.name.endswith(".out")]


def test_apply_delete_sketch():
    with addon():
        make_sketch([SQUARE], mode='EXTRUDE')
        assert bpy.ops.sketch.apply(delete_sketch=True) == {'FINISHED'}
        assert [o.name for o in bpy.data.objects] == ["Sketch"]
        assert not bpy.data.objects["Sketch"].sketch_tools.is_sketch


def test_output_recreated_after_deletion():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        bpy.data.objects.remove(sketch.sketch_tools.output_object)
        assert sketch.sketch_tools.output_object is None
        assert bpy.ops.sketch.rebuild() == {'FINISHED'}
        out = sketch.sketch_tools.output_object
        assert out is not None and out.name == "Sketch.out" and out.parent == sketch

        # Unlinked from every scene (but still in bpy.data): replaced by a linked one.
        for coll in list(out.users_collection):
            coll.objects.unlink(out)
        build.rebuild(sketch)
        out = sketch.sketch_tools.output_object
        assert out.users_scene and out.name == "Sketch.out" and out.parent == sketch


def test_mode_none_removes_output():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        out = sketch.sketch_tools.output_object
        out_name, mesh_name = out.name, out.data.name
        sketch.sketch_tools.output_mode = 'NONE'
        assert sketch.sketch_tools.output_object is None
        assert out_name not in bpy.data.objects
        assert mesh_name not in bpy.data.meshes


def test_sweep_mode_builds_output():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        sketch.sketch_tools.output_mode = 'SWEEP'
        assert sketch.sketch_tools.output_object is not None
        assert sketch.sketch_tools.warnings == ""


def test_duplicated_sketch_gets_its_own_output():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        original_out = sketch.sketch_tools.output_object
        copy = sketch.copy()
        copy.data = sketch.data.copy()
        sketch.users_collection[0].objects.link(copy)
        assert copy.sketch_tools.output_object == original_out

        build.rebuild(copy)
        copy_out = copy.sketch_tools.output_object
        assert copy_out != original_out
        assert copy_out.parent == copy
        assert original_out.parent == sketch
        assert sketch.sketch_tools.output_object == original_out


def test_output_material_slots_preserved():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        mat = bpy.data.materials.new("Paint")
        sketch.sketch_tools.output_object.data.materials.append(mat)
        sketch.sketch_tools.ex_height = 5.0
        out_mesh = sketch.sketch_tools.output_object.data
        assert list(out_mesh.materials) == [mat]
        assert all(p.material_index == 0 for p in out_mesh.polygons)


def test_set_plane_keeps_world_positions():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        world_before = [sketch.matrix_world @ v.co for v in sketch.data.vertices]

        # A coplanar frame, moved and turned in the plane: nothing may move.
        turn = Matrix.Rotation(math.radians(30.0), 3, 'Z')
        new_world = face_matrix((0.5, 0.5, 0.0), (0, 0, 1), turn.to_quaternion())
        reorient_sketch(sketch, new_world)
        assert matrices_close(sketch.matrix_world, new_world)
        for v, before in zip(sketch.data.vertices, world_before):
            assert abs(v.co.z) < 1e-9
            assert ((sketch.matrix_world @ v.co) - before).length < 1e-6

        # A face frame looking down -Y gives the XZ plane.
        m = face_matrix((0, 0, 0), (0, -1, 0), Matrix.Identity(3).to_quaternion())
        assert matrices_close(m, Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))


class _FakeLayout:
    """Records nothing; checks that every prop and operator the panel names exists."""

    def __init__(self):
        self.use_property_split = False
        self.use_property_decorate = True
        self.enabled = True

    def _child(self, *_args, **_kwargs):
        return _FakeLayout()

    column = row = box = _child

    def prop(self, data, name, **_kwargs):
        assert name in data.bl_rna.properties, f"missing property {name}"

    def operator(self, idname, **_kwargs):
        group, op = idname.split(".")
        assert hasattr(getattr(bpy.ops, group), op), f"missing operator {idname}"
        return types.SimpleNamespace()

    def operator_menu_enum(self, idname, prop, **_kwargs):
        self.operator(idname)

    def label(self, **_kwargs):
        pass

    def separator(self, **_kwargs):
        pass


def test_panels_draw_without_resolving():
    from sketch_tools.ui import panel

    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        original = build.resolve_bmesh

        def forbidden(*_args, **_kwargs):
            raise AssertionError("panel draw resolved the sketch")

        build.resolve_bmesh = forbidden
        try:
            sketch.sketch_tools.warnings = "first\nsecond"
            profiles = [item[0] for item in props.SWEEP_PROFILES]
            for mode, profile in [('NONE', 'CUSTOM'), ('EXTRUDE', 'CUSTOM')] + [('SWEEP', p) for p in profiles]:
                with build.suspended():
                    sketch.sketch_tools.output_mode = mode
                    sketch.sketch_tools.sw_profile = profile
                ctx = types.SimpleNamespace(
                    mode='OBJECT', active_object=sketch,
                    window_manager=bpy.context.window_manager,
                )
                for cls in panel.classes:
                    assert cls.poll(ctx)
                    cls.draw(types.SimpleNamespace(layout=_FakeLayout()), ctx)
            ctx = types.SimpleNamespace(mode='OBJECT', active_object=None,
                                        window_manager=bpy.context.window_manager)
            assert panel.SKETCH_PT_create.poll(ctx)
            assert not panel.SKETCH_PT_sketch.poll(ctx)
            assert not panel.SKETCH_PT_output.poll(ctx)
        finally:
            build.resolve_bmesh = original


def test_handler_errors_are_logged_once():
    with addon():
        original = build.rebuild_all

        def broken():
            raise RuntimeError("boom")

        build.rebuild_all = broken
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                for _ in range(3):
                    handlers._on_undo_redo_post(bpy.context.scene)
        finally:
            build.rebuild_all = original
        assert out.getvalue().count("RuntimeError: boom") == 1, out.getvalue()


@contextmanager
def only_event_handlers():
    """Take the depsgraph handler out so a rebuild after undo or load can only come from
    the handler under test (a depsgraph pass would otherwise mask a missing one)."""
    handler_list = bpy.app.handlers.depsgraph_update_post
    func = handlers._on_depsgraph_update_post
    handler_list.remove(func)
    try:
        yield
    finally:
        handler_list.append(func)


def _out_max_x(sketch_name):
    sketch = bpy.data.objects[sketch_name]
    return max(v.co.x for v in sketch.sketch_tools.output_object.data.vertices)


def test_handlers_are_persistent():
    with addon():
        lists = (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post,
                 bpy.app.handlers.undo_post, bpy.app.handlers.redo_post)
        lists += (bpy.app.handlers.save_pre, bpy.app.handlers.save_post, bpy.app.handlers.save_post_fail)
        ours = [h for handler_list in lists for h in handler_list
                if getattr(h, "__module__", "") == handlers.__name__]
        assert len(ours) == 7, ours
        # persistent() stores the marker as an attribute whose value is None.
        assert all(hasattr(h, "_bpy_persistent") for h in ours)


def test_undo_and_redo_rebuild_every_sketch():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        name = sketch.name
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="base")
        sketch.sketch_tools.ex_height = 5.0
        bpy.ops.ed.undo_push(message="taller")

        calls, restore = count_rebuilds()
        try:
            with only_event_handlers():
                bpy.ops.ed.undo()
            assert calls[0] >= 1, "undo_post did not rebuild"
            sketch = bpy.data.objects[name]
            assert sketch.sketch_tools.ex_height == 2.0
            assert abs(max(v.co.z for v in sketch.sketch_tools.output_object.data.vertices) - 2.0) < 1e-6

            after_undo = calls[0]
            with only_event_handlers():
                bpy.ops.ed.redo()
            assert calls[0] > after_undo, "redo_post did not rebuild"
            sketch = bpy.data.objects[name]
            assert sketch.sketch_tools.ex_height == 5.0
            assert abs(max(v.co.z for v in sketch.sketch_tools.output_object.data.vertices) - 5.0) < 1e-6
        finally:
            restore()


def test_load_post_rebuilds_output():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        name = sketch.name
        path = os.path.join(tempfile.gettempdir(), "sketch_tools_load_post_test.blend")
        try:
            with only_event_handlers():
                # Save a stale (empty) output, as a file from an older version could hold.
                sketch.sketch_tools.output_object.data.clear_geometry()
                bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
                bpy.ops.wm.open_mainfile(filepath=path)
            sketch = bpy.data.objects[name]
            out = sketch.sketch_tools.output_object
            assert out is not None and len(out.data.polygons) == 6, "load_post did not rebuild"
        finally:
            if os.path.exists(path):
                os.remove(path)


def test_one_failing_sketch_does_not_stop_the_others():
    with addon():
        broken = make_sketch([SQUARE], mode='EXTRUDE')
        good = make_sketch([SQUARE], mode='EXTRUDE')
        broken_name, good_name = broken.name, good.name
        good.data.vertices[2].co = (3.0, 3.0, 0.0)

        original = build._rebuild

        def fail_for_broken(sketch):
            if sketch.name == broken_name:
                raise RuntimeError("boom")
            original(sketch)

        build._rebuild = fail_for_broken
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                for _ in range(3):
                    handlers._on_undo_redo_post(bpy.context.scene)
        finally:
            build._rebuild = original
        assert abs(_out_max_x(good_name) - 3.0) < 1e-6, "sketch after the failing one was not rebuilt"
        assert out.getvalue().count("RuntimeError: boom") == 1, out.getvalue()


def test_unrelated_object_update_does_not_rebuild_sketches():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        other = bpy.data.objects.new("Other", bpy.data.meshes.new("Other"))
        bpy.context.collection.objects.link(other)
        bpy.context.view_layer.update()
        bpy.context.view_layer.update()
        calls, restore = count_rebuilds()
        try:
            other.location.x = 5.0
            bpy.context.view_layer.update()
            assert calls[0] == 0, calls[0]
        finally:
            restore()


def test_face_plane_without_a_viewport_cancels_cleanly():
    with addon():
        try:
            result = bpy.ops.sketch.new('INVOKE_DEFAULT', plane='FACE')
        except RuntimeError as exc:
            assert "Face" in str(exc), exc
        else:
            assert result == {'CANCELLED'}
        assert not any(build.is_sketch(o) for o in bpy.data.objects)

        # The click handler's final step, driven directly.
        matrix = face_matrix((1.0, 2.0, 3.0), (0, 0, 1), Matrix.Identity(3).to_quaternion())
        result = sketch_tools.ops.create.SKETCH_OT_new.use_face(None, bpy.context, matrix)
        assert result == {'FINISHED'}
        obj = bpy.context.active_object
        assert build.is_sketch(obj) and obj.mode == 'EDIT'
        assert matrices_close(obj.matrix_world, matrix)


def test_face_pick_ignores_hidden_regions():
    from sketch_tools.ops.create import _FacePickMixin

    def region(kind, width, height):
        return types.SimpleNamespace(type=kind, x=0, y=0, width=width, height=height, data=None)

    area = types.SimpleNamespace(type='VIEW_3D', regions=[region('WINDOW', 1, 1), region('UI', 1, 200)])
    context = types.SimpleNamespace(window=types.SimpleNamespace(screen=types.SimpleNamespace(areas=[area])))
    event = types.SimpleNamespace(mouse_x=0, mouse_y=0)
    # A one pixel overlay region used to swallow the click, a one pixel WINDOW used to be hit.
    assert _FacePickMixin._ray_cast(context, event) is None


def test_fillet_field_undo_push_is_debounced():
    with addon():
        sketch = make_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]], mode='EXTRUDE')
        select_only(sketch.data, {0})
        pushes = []
        real_has_window, real_push = props._has_window, props._push_undo
        props._has_window = lambda: True
        props._push_undo = lambda: pushes.append(1)
        try:
            # A drag fires the update on every tick; nothing is pushed while it goes on.
            for value in (0.1, 0.2, 0.3, 0.4, 0.5):
                bpy.context.window_manager.sketch_tools_fillet_radius = value
            assert pushes == []
            assert bpy.app.timers.is_registered(props._undo_timer)

            # The single pending timer fires once, after the last change.
            props._undo_timer()
            bpy.app.timers.unregister(props._undo_timer)
            assert pushes == [1]
            assert not bpy.app.timers.is_registered(props._undo_timer)

            # Scheduling again restarts, never stacks, the timer.
            props._schedule_undo_push()
            props._schedule_undo_push()
            assert bpy.app.timers.is_registered(props._undo_timer)
        finally:
            props._has_window, props._push_undo = real_has_window, real_push
    assert not bpy.app.timers.is_registered(props._undo_timer), "unregister left the timer behind"


def test_fillet_field_schedules_nothing_without_a_window():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        select_only(sketch.data, {0})
        assert not props._has_window()
        bpy.context.window_manager.sketch_tools_fillet_radius = 0.1
        assert not bpy.app.timers.is_registered(props._undo_timer)


def test_multi_object_edit_exit_rebuilds_every_sketch():
    # The non-active sketch of a multi-object edit session is rebuilt on exit too.
    with addon():
        first = make_sketch([SQUARE], mode='EXTRUDE')
        second = make_sketch([SQUARE], mode='EXTRUDE')
        for obj in (first, second):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = first
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.view_layer.update()
        bm = bmesh.from_edit_mesh(second.data)
        bm.verts.ensure_lookup_table()
        bm.verts[2].co = Vector((6.0, 6.0, 0.0))
        bmesh.update_edit_mesh(second.data)
        bpy.context.view_layer.update()
        assert abs(_out_max_x(second.name) - 1.0) < 1e-6, "rebuilt during edit mode"
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        assert abs(_out_max_x(second.name) - 6.0) < 1e-6
        assert abs(_out_max_x(first.name) - 1.0) < 1e-6


def test_failed_rebuild_shows_a_warning_and_does_not_loop():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        name = sketch.name
        bpy.context.view_layer.update()
        original = build._rebuild
        attempts = []

        def failing(obj):
            attempts.append(1)
            raise RuntimeError("boom")

        build._rebuild = failing
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                sketch.data.vertices[2].co = (3.0, 3.0, 0.0)
                sketch.data.update()
                for _ in range(6):
                    bpy.context.view_layer.update()
            assert sketch.sketch_tools.warnings == build.REBUILD_FAILED
            assert attempts, "the failing rebuild was never attempted"
            settled = len(attempts)
            for _ in range(4):
                bpy.context.view_layer.update()
            assert len(attempts) == settled, "the warning write keeps re-triggering the rebuild"
        finally:
            build._rebuild = original
        sketch = bpy.data.objects[name]
        build.rebuild(sketch)
        assert sketch.sketch_tools.warnings == ""


def test_output_follows_sketch_rename():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        old_out = sketch.sketch_tools.output_object
        assert old_out.name == sketch.name + ".out"
        sketch.name = "Renamed"
        bpy.context.view_layer.update()
        out = sketch.sketch_tools.output_object
        assert out is old_out and out.name == "Renamed.out", out.name

        # A taken name leaves a numbered variant, which is then left alone.
        bpy.data.objects.new("Other.out", bpy.data.meshes.new("Other.out"))
        sketch.name = "Other"
        build.rebuild(sketch)
        settled = out.name
        assert settled.startswith("Other.out"), settled
        build.rebuild(sketch)
        assert out.name == settled


def _failing_rebuild():
    original = build._rebuild

    def failing(obj):
        raise RuntimeError("boom")

    build._rebuild = failing
    return original


def test_property_update_failure_sets_the_rebuild_warning():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        select_only(sketch.data, {0})
        original = _failing_rebuild()
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                sketch.sketch_tools.ex_height = 5.0
            assert sketch.sketch_tools.warnings == build.REBUILD_FAILED
            assert "boom" in out.getvalue()

            build.clear_error_log()
            sketch.sketch_tools.warnings = ""
            with redirect_stdout(out):
                bpy.context.window_manager.sketch_tools_fillet_radius = 0.25
            assert sketch.sketch_tools.warnings == build.REBUILD_FAILED
        finally:
            build._rebuild = original
        build.rebuild(sketch)
        assert sketch.sketch_tools.warnings == ""


class _Area:
    def __init__(self, kind):
        self.type = kind


def _fake_window(*kinds):
    return types.SimpleNamespace(screen=types.SimpleNamespace(areas=[_Area(k) for k in kinds]))


def test_undo_push_prefers_the_window_with_a_3d_view():
    plain = _fake_window('PROPERTIES', 'OUTLINER')
    viewport = _fake_window('OUTLINER', 'VIEW_3D')
    assert props._undo_window([plain, viewport]) is viewport
    assert props._undo_window([plain, _fake_window('TIMELINE')]) is plain


def test_undo_push_failure_is_reported_once_and_never_raised():
    real_flag, real_window = props._undo_error_shown, props._undo_window

    def no_window(windows):
        raise IndexError("no window")

    props._undo_error_shown = False
    props._undo_window = no_window
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            props._push_undo()
            props._push_undo()
        assert out.getvalue().count("could not push the undo step") == 1
    finally:
        props._undo_error_shown, props._undo_window = real_flag, real_window


def test_undo_push_runs_under_a_window_override():
    # Without the override the operator's poll fails the way it does in a timer callback.
    with bpy.context.temp_override(window=None):
        try:
            bpy.ops.ed.undo_push(message="x")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the push to fail without a window")
    real_flag = props._undo_error_shown
    props._undo_error_shown = False
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            with bpy.context.temp_override(window=None):
                props._push_undo()
        assert out.getvalue() == "", out.getvalue()
    finally:
        props._undo_error_shown = real_flag


class _NameRecorder:
    def __init__(self, name):
        self.__dict__["name"] = name
        self.__dict__["renames"] = 0

    def __setattr__(self, key, value):
        assert key == "name"
        self.__dict__["renames"] += 1
        self.__dict__["name"] = value


def _name_lengths():
    limit = build._max_name_bytes()
    return limit, (limit - 4, limit - 3, limit - 1, limit)


def test_long_sketch_names_do_not_rename_the_output_on_every_rebuild():
    limit, lengths = _name_lengths()
    for length in lengths:
        sketch = types.SimpleNamespace(name="a" * length)
        stored = build._stored_name(sketch.name + ".out")
        assert len(stored.encode()) <= limit
        out = _NameRecorder(stored)
        build._follow_sketch_name(sketch, out)
        assert out.renames == 0, f"{length}: renamed an output that already has its stored name"
        # A real rename still happens.
        out = _NameRecorder("stale")
        build._follow_sketch_name(sketch, out)
        assert out.name == stored and out.renames == 1
    # Multi-byte characters are cut at a character boundary, like Blender does.
    assert build._stored_name("é" * limit) == "é" * (limit // 2)
    assert build._stored_name("x" * (limit + 40)) == "x" * limit
    assert build._stored_name("short.out") == "short.out"


def test_long_sketch_name_output_is_stable_in_blender():
    limit, lengths = _name_lengths()
    with addon():
        for length in lengths:
            sketch = make_sketch([SQUARE], mode='EXTRUDE')
            sketch.name = "s" * length
            out = sketch.sketch_tools.output_object
            build.rebuild(sketch)
            settled = out.name
            for _ in range(3):
                build.rebuild(sketch)
            assert sketch.sketch_tools.output_object is out
            assert out.name == settled and len(settled.encode()) <= limit
            # Either the cut name, or a numbered variant when that clashes with the sketch
            # itself (length == limit); both are accepted, so nothing renamed it.
            assert settled == build._stored_name(sketch.name + ".out") or settled.startswith("s" * (limit - 4))
            bpy.data.objects.remove(out)
            bpy.data.objects.remove(sketch)


def test_new_sketch_skips_the_tool_switch_without_a_3d_view():
    calls = []
    real = create._set_tool
    create._set_tool = lambda *args: calls.append(args)
    try:
        with addon():
            assert bpy.ops.sketch.new(plane='XY') == {'FINISHED'}
            assert bpy.context.object.mode == 'EDIT'
    finally:
        create._set_tool = real
    assert calls == []


class _FakeRegion:
    def __init__(self, kind):
        self.type = kind


def _fake_tool_context(area_type, window=True):
    area = None
    if area_type is not None:
        area = types.SimpleNamespace(type=area_type, regions=[_FakeRegion('HEADER'), _FakeRegion('WINDOW')])
    return types.SimpleNamespace(window=object() if window else None, area=area)


def test_draw_tool_is_set_only_for_a_3d_view_area():
    calls = []
    real_set, real_flag = create._set_tool, create._tool_error_shown
    create._set_tool = lambda *args: calls.append(args)
    try:
        ctx = _fake_tool_context('VIEW_3D')
        create.activate_draw_tool(ctx)
        assert len(calls) == 1
        window, area, region = calls[0]
        assert window is ctx.window and area is ctx.area and region.type == 'WINDOW'
        assert create.DRAW_TOOL == "sketch_tools.sketch"

        for other in (_fake_tool_context('PROPERTIES'), _fake_tool_context(None),
                      _fake_tool_context('VIEW_3D', window=False)):
            create.activate_draw_tool(other)
        assert len(calls) == 1
    finally:
        create._set_tool, create._tool_error_shown = real_set, real_flag


def test_draw_tool_failure_is_reported_once_and_never_raised():
    real_set, real_flag = create._set_tool, create._tool_error_shown

    def failing(*args):
        raise RuntimeError("no tool")

    create._set_tool = failing
    create._tool_error_shown = False
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
            create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
        assert out.getvalue().count(f"could not activate {create.DRAW_TOOL}") == 1
    finally:
        create._set_tool, create._tool_error_shown = real_set, real_flag


def test_draw_tool_cancelled_result_is_reported_once():
    real_bpy, real_flag = create.bpy, create._tool_error_shown
    calls = []

    def tool_set_by_id(name):
        calls.append(name)
        return {'CANCELLED'}

    @contextmanager
    def temp_override(**kwargs):
        yield

    create.bpy = types.SimpleNamespace(
        ops=types.SimpleNamespace(wm=types.SimpleNamespace(tool_set_by_id=tool_set_by_id)),
        context=types.SimpleNamespace(temp_override=temp_override))
    create._tool_error_shown = False
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
            create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
        assert calls == [create.DRAW_TOOL] * 2
        assert out.getvalue().count(f"could not activate {create.DRAW_TOOL}") == 1
        assert "CANCELLED" in out.getvalue()
    finally:
        create.bpy, create._tool_error_shown = real_bpy, real_flag


def test_draw_tool_target_lookup_failure_is_swallowed():
    real_target, real_flag = create._tool_target, create._tool_error_shown

    def broken(context):
        raise AttributeError("no area")

    create._tool_target = broken
    create._tool_error_shown = False
    try:
        with redirect_stdout(io.StringIO()):
            create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
    finally:
        create._tool_target, create._tool_error_shown = real_target, real_flag


def test_new_sketch_switches_tool_only_after_entering_edit_mode():
    modes = []
    real_set, real_target = create._set_tool, create._tool_target
    # The operator has no 3D view headless, so hand it a target to reach the tool switch.
    create._tool_target = lambda context: ("window", "area", "region")
    create._set_tool = lambda *args: modes.append(bpy.context.mode)
    try:
        with addon():
            assert bpy.ops.sketch.new(plane='XY') == {'FINISHED'}
    finally:
        create._set_tool, create._tool_target = real_set, real_target
    assert modes == ['EDIT_MESH']


def test_face_path_activates_the_draw_tool_once_after_creating_the_sketch():
    seen = []
    real_activate = create.activate_draw_tool

    def record(context):
        seen.append((bpy.context.mode, build.is_sketch(context.active_object)))

    create.activate_draw_tool = record
    try:
        with addon():
            matrix = face_matrix(Vector((0, 0, 1)), Vector((0, 0, 1)), Matrix.Identity(3).to_quaternion())
            assert create.SKETCH_OT_new.use_face(None, bpy.context, matrix) == {'FINISHED'}
    finally:
        create.activate_draw_tool = real_activate
    assert len(seen) == 1
    assert seen[0] == ('EDIT_MESH', True)


# ---------------------------------------------------------------------------
# Sweep (Phase 2, part C)
# ---------------------------------------------------------------------------

def make_open_sketch(points, mode='NONE'):
    """New XY sketch holding one open polyline."""
    obj = make_sketch([], mode)
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0.0)) for x, y in points]
    for a, b in zip(verts, verts[1:]):
        bm.edges.new((a, b))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def make_sweep(path_sketch, **props_to_set):
    st = path_sketch.sketch_tools
    with build.suspended():
        for name, value in props_to_set.items():
            setattr(st, name, value)
        st.output_mode = 'SWEEP'
    build.rebuild(path_sketch)
    return path_sketch


def out_bbox(sketch):
    verts = sketch.sketch_tools.output_object.data.vertices
    lo = Vector([min(v.co[i] for v in verts) for i in range(3)])
    hi = Vector([max(v.co[i] for v in verts) for i in range(3)])
    return lo, hi


def assert_bbox(sketch, lo, hi, eps=1e-5):
    got_lo, got_hi = out_bbox(sketch)
    assert (got_lo - Vector(lo)).length < eps and (got_hi - Vector(hi)).length < eps, (got_lo, got_hi, lo, hi)


def make_curve_circle(radius, name="Profile"):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '2D'
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(3)
    k = 0.5522847498 * radius
    corners = [(radius, 0), (0, radius), (-radius, 0), (0, -radius)]
    for i, (x, y) in enumerate(corners):
        tx, ty = -y / radius, x / radius
        point = spline.bezier_points[i]
        point.co = (x, y, 0.0)
        point.handle_left_type = point.handle_right_type = 'FREE'
        point.handle_left = (x - tx * k, y - ty * k, 0.0)
        point.handle_right = (x + tx * k, y + ty * k, 0.0)
    spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _rect_profile(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_sweep_library_profiles_on_a_closed_square():
    with addon():
        sketch = make_sketch([SQUARE])
        expected_top = {'SQUARE': 0.2, 'CYLINDER': 0.2, 'HALF_ROUND': 0.1, 'TUBE': 0.2}
        volumes = {}
        for profile in ('SQUARE', 'CYLINDER', 'HALF_ROUND', 'TUBE'):
            make_sweep(sketch, sw_profile=profile)
            assert sketch.sketch_tools.warnings == "", (profile, sketch.sketch_tools.warnings)
            bm = output_bm(sketch)
            try:
                assert len(bm.faces) > 0
                assert_closed_and_consistent(bm)
                volumes[profile] = signed_volume(bm)
            finally:
                bm.free()
            assert volumes[profile] > 0.0, f"{profile}: inward normals ({volumes[profile]})"
            lo, hi = out_bbox(sketch)
            assert abs(lo.z) < 1e-6 and abs(hi.z - expected_top[profile]) < 1e-2, (profile, lo, hi)
        # Bar 0.2 x 0.2 around a unit square: (1.2^2 - 0.8^2) * 0.2.
        assert abs(volumes['SQUARE'] - 0.16) < 1e-5, volumes['SQUARE']
        # The tube is hollow: well under the solid cylinder.
        assert volumes['TUBE'] < volumes['CYLINDER'] * 0.5


def test_sweep_square_profile_has_one_ring_of_sixteen_quads():
    with addon():
        sketch = make_sweep(make_sketch([SQUARE]), sw_profile='SQUARE')
        bm = output_bm(sketch)
        try:
            assert len(bm.verts) == 16 and len(bm.faces) == 16
            assert all(len(f.verts) == 4 for f in bm.faces)
        finally:
            bm.free()


def test_sweep_open_path_with_and_without_cap_ends():
    with addon():
        sketch = make_open_sketch([(0, 0), (2, 0), (2, 2)])
        make_sweep(sketch, sw_profile='SQUARE', sw_cap_ends=True)
        bm = output_bm(sketch)
        try:
            assert_closed_and_consistent(bm)
            # A mitred strip 0.2 wide and 0.2 tall along a centreline of length 4.
            assert abs(signed_volume(bm) - 0.16) < 1e-5
        finally:
            bm.free()

        make_sweep(sketch, sw_cap_ends=False)
        bm = output_bm(sketch)
        try:
            boundary = [e for e in bm.edges if len(e.link_faces) == 1]
            assert len(boundary) == 8, len(boundary)
            assert len(bm.faces) == 8
        finally:
            bm.free()

        make_sweep(sketch, sw_profile='TUBE', sw_cap_ends=True)
        bm = output_bm(sketch)
        try:
            assert_closed_and_consistent(bm)
            assert signed_volume(bm) > 0.0
        finally:
            bm.free()


def test_sweep_custom_profile_from_sketch_with_pivot_and_rotation():
    with addon():
        profile = make_sketch([_rect_profile(1.0, 1.0, 1.4, 1.2)])
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile, sw_pivot='BL')
        assert path.sketch_tools.warnings == ""
        # u (0.4 wide) runs to the left of travel (+y), v (0.2 tall) is the plane normal.
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.2))

        make_sweep(path, sw_pivot='MC')
        assert_bbox(path, (0, -0.2, -0.1), (2, 0.2, 0.1))

        make_sweep(path, sw_pivot='BL', sw_rotation=90.0)
        assert_bbox(path, (0, 0, 0), (2, 0.2, 0.4))

        make_sweep(path, sw_rotation=0.0, sw_pivot='BL', sw_offset_u=0.5, sw_offset_v=0.25)
        assert_bbox(path, (0, 0.5, 0.25), (2, 0.9, 0.45))

        make_sweep(path, sw_offset_u=0.0, sw_offset_v=0.0, sw_mirror_u=True)
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.2))
        bm = output_bm(path)
        try:
            assert_closed_and_consistent(bm)
            assert signed_volume(bm) > 0.0
        finally:
            bm.free()


def test_sweep_custom_profile_from_bezier_curve():
    with addon():
        curve = make_curve_circle(0.1)
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=curve, sw_pivot='BC')
        assert path.sketch_tools.warnings == "", path.sketch_tools.warnings
        assert_bbox(path, (0, -0.1, 0), (2, 0.1, 0.2))
        bm = output_bm(path)
        try:
            assert_closed_and_consistent(bm)
            assert signed_volume(bm) > 0.0
        finally:
            bm.free()


def test_sweep_custom_profile_from_poly_curve():
    with addon():
        curve = bpy.data.curves.new("Poly", 'CURVE')
        curve.dimensions = '2D'
        spline = curve.splines.new('POLY')
        spline.points.add(3)
        for point, (x, y) in zip(spline.points, _rect_profile(0, 0, 0.4, 0.2)):
            point.co = (x, y, 0.0, 1.0)
        spline.use_cyclic_u = True
        obj = bpy.data.objects.new("Poly", curve)
        bpy.context.scene.collection.objects.link(obj)
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=obj, sw_pivot='BL')
        assert path.sketch_tools.warnings == "", path.sketch_tools.warnings
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.2))


def test_custom_profile_object_poll():
    with addon():
        path = make_sketch([SQUARE])
        other = make_sketch([SQUARE])
        curve = make_curve_circle(0.1)
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.scene.collection.objects.link(plain)
        st = path.sketch_tools
        assert props._poll_custom_profile(st, other)
        assert props._poll_custom_profile(st, curve)
        assert not props._poll_custom_profile(st, path)
        assert not props._poll_custom_profile(st, plain)


def test_custom_profile_warnings():
    with addon():
        path = make_sketch([SQUARE])
        make_sweep(path, sw_profile='CUSTOM')
        assert "no profile object" in path.sketch_tools.warnings
        assert path.sketch_tools.output_object is None

        with build.suspended():
            path.sketch_tools.sw_custom_object = path
        build.rebuild(path)
        assert "cannot use itself" in path.sketch_tools.warnings
        assert path.sketch_tools.output_object is None

        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.scene.collection.objects.link(plain)
        with build.suspended():
            path.sketch_tools.sw_custom_object = plain
        build.rebuild(path)
        assert "not a sketch or a curve" in path.sketch_tools.warnings


def test_profile_sketch_edit_rebuilds_the_sweep():
    with addon():
        profile = make_sketch([_rect_profile(0, 0, 0.4, 0.2)])
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile, sw_pivot='BL')
        bpy.context.view_layer.update()
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.2))

        mesh = profile.data
        for v in mesh.vertices:
            if v.co.y > 0.1:
                v.co.y = 0.6
        mesh.update()
        bpy.context.view_layer.update()
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.6))

        # Settled: nothing keeps rebuilding.
        calls, restore = count_rebuilds()
        try:
            for _ in range(3):
                bpy.context.view_layer.update()
            assert calls[0] == 0, calls[0]
        finally:
            restore()


def test_profile_sketch_leaving_edit_mode_rebuilds_the_sweep():
    with addon():
        profile = make_sketch([_rect_profile(0, 0, 0.4, 0.2)])
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile, sw_pivot='BL')
        bpy.context.view_layer.objects.active = profile
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.view_layer.update()
        bm = bmesh.from_edit_mesh(profile.data)
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            if v.co.x > 0.2:
                v.co.x = 1.0
        bmesh.update_edit_mesh(profile.data)
        bpy.context.view_layer.update()
        assert_bbox(path, (0, 0, 0), (2, 0.4, 0.2))
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        assert_bbox(path, (0, 0, 0), (2, 1.0, 0.2))


def test_profile_sketch_fillet_segments_rebuilds_the_sweep():
    with addon():
        profile = make_sketch([SQUARE])
        select_only(profile.data, {0})
        attributes.write_fillet_radius(profile, 0.3)
        path = make_open_sketch([(0, 0), (4, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile, sw_cap_ends=False)
        before = len(path.sketch_tools.output_object.data.vertices)
        profile.sketch_tools.fillet_segments = 2
        after = len(path.sketch_tools.output_object.data.vertices)
        assert after < before, (before, after)
        profile.sketch_tools.fillet_segments = 16
        assert len(path.sketch_tools.output_object.data.vertices) > before


def test_profile_curve_edit_rebuilds_the_sweep():
    with addon():
        curve = make_curve_circle(0.1)
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=curve, sw_pivot='MC')
        bpy.context.view_layer.update()
        assert_bbox(path, (0, -0.1, -0.1), (2, 0.1, 0.1))

        for point in curve.data.splines[0].bezier_points:
            point.co.x *= 2.0
            point.handle_left.x *= 2.0
            point.handle_right.x *= 2.0
        curve.data.update_tag()
        bpy.context.view_layer.update()
        assert_bbox(path, (0, -0.2, -0.1), (2, 0.2, 0.1))


def test_profile_dependency_cycle_terminates():
    with addon():
        a = make_open_sketch([(0, 0), (2, 0)])
        b = make_sketch([_rect_profile(0, 0, 0.4, 0.2)])
        make_sweep(a, sw_profile='CUSTOM', sw_custom_object=b)
        make_sweep(b, sw_profile='CUSTOM', sw_custom_object=a)
        bpy.context.view_layer.update()

        calls, restore = count_rebuilds()
        try:
            build.rebuild_dependents(a)
            assert calls[0] == 1, calls[0]
            calls[0] = 0
            a.data.update()
            b.data.update()
            bpy.context.view_layer.update()
            assert 1 <= calls[0] <= 8, calls[0]
            settled = calls[0]
            bpy.context.view_layer.update()
            bpy.context.view_layer.update()
            assert calls[0] == settled
        finally:
            restore()


def test_deleting_the_custom_object_leaves_a_warning():
    with addon():
        profile = make_sketch([_rect_profile(0, 0, 0.4, 0.2)])
        path = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile)
        assert path.sketch_tools.output_object is not None
        bpy.data.objects.remove(profile)
        bpy.context.view_layer.update()
        assert path.sketch_tools.sw_custom_object is None
        assert "no profile object" in path.sketch_tools.warnings
        assert path.sketch_tools.output_object is None
        build.rebuild_all()


def test_switching_between_extrude_and_sweep():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        st = sketch.sketch_tools
        st.output_mode = 'SWEEP'
        assert st.output_object is not None and st.output_object.parent == sketch
        assert st.warnings == ""
        lo, hi = out_bbox(sketch)
        assert abs(hi.z - 0.2) < 1e-6
        st.output_mode = 'EXTRUDE'
        lo, hi = out_bbox(sketch)
        assert abs(hi.z - 2.0) < 1e-6 and abs(lo.z) < 1e-6
        assert len([o for o in bpy.data.objects if o.name.endswith(".out")]) == 1


def test_apply_on_a_sweep_output():
    with addon():
        sketch = make_sweep(make_sketch([SQUARE]), sw_profile='CYLINDER')
        face_count = len(sketch.sketch_tools.output_object.data.polygons)
        assert bpy.ops.sketch.apply() == {'FINISHED'}
        applied = bpy.data.objects["Sketch"]
        assert sketch.name == "Sketch.sketch"
        assert not applied.sketch_tools.is_sketch
        assert applied.parent is None and not applied.hide_select
        assert len(applied.data.polygons) == face_count > 0


def _sharp_count(sketch):
    return sum(1 for e in sketch.sketch_tools.output_object.data.edges if e.use_edge_sharp)


def test_sweep_smooth_flags_survive_into_the_output():
    with addon():
        # A straight round bar has nothing to keep hard.
        bar = make_open_sketch([(0, 0), (2, 0)])
        make_sweep(bar, sw_profile='CYLINDER', sw_smooth=True, sw_cap_ends=False)
        assert all(p.use_smooth for p in bar.sketch_tools.output_object.data.polygons)
        assert _sharp_count(bar) == 0

        # A 90 degree turn is hard even on a round profile (C5).
        sketch = make_sketch([SQUARE])
        make_sweep(sketch, sw_profile='CYLINDER', sw_smooth=True)
        assert all(p.use_smooth for p in sketch.sketch_tools.output_object.data.polygons)
        round_sharp = _sharp_count(sketch)
        assert round_sharp > 0

        # A square profile keeps its own corners hard on top of that.
        make_sweep(sketch, sw_profile='SQUARE')
        assert _sharp_count(sketch) > 0
        assert all(p.use_smooth for p in sketch.sketch_tools.output_object.data.polygons)

        make_sweep(sketch, sw_smooth=False)
        mesh = sketch.sketch_tools.output_object.data
        assert not any(p.use_smooth for p in mesh.polygons)
        assert not any(e.use_edge_sharp for e in mesh.edges)


def test_sweep_bmesh_is_freed_on_success_and_on_error():
    with addon():
        sketch = make_sketch([SQUARE])
        made = []
        original = build.build_sweep

        def tracking(*args):
            bm, warnings = original(*args)
            made.append(bm)
            return bm, warnings

        build.build_sweep = tracking
        try:
            make_sweep(sketch)
            assert made and not made[-1].is_valid

            # A failure after the bmesh exists (here: the output step) must still free it.
            original_output = build._get_or_create_output

            def failing(_sketch):
                raise RuntimeError("boom")

            build._get_or_create_output = failing
            try:
                try:
                    build.rebuild(sketch)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("expected the failure to propagate")
            finally:
                build._get_or_create_output = original_output
            assert len(made) == 2 and not made[-1].is_valid
        finally:
            build.build_sweep = original


class _RecordingLayout(_FakeLayout):
    def __init__(self, seen=None):
        super().__init__()
        self.seen = seen if seen is not None else []

    def _child(self, *_args, **_kwargs):
        return _RecordingLayout(self.seen)

    column = row = box = _child

    def prop(self, data, name, **kwargs):
        super().prop(data, name, **kwargs)
        self.seen.append(name)


def test_sweep_panel_shows_only_the_fields_of_the_profile():
    from sketch_tools.ui import panel

    with addon():
        sketch = make_sketch([SQUARE], mode='SWEEP')
        common = ["sw_pivot", "sw_offset_u", "sw_offset_v", "sw_rotation", "sw_mirror_u",
                  "sw_mitre_limit", "sw_smooth", "sw_cap_ends", "sw_union"]
        expected = {
            'SQUARE': ["sw_width", "sw_height"],
            'CYLINDER': ["sw_radius", "sw_segments"],
            'HALF_ROUND': ["sw_width", "sw_segments"],
            'TUBE': ["sw_radius", "sw_thickness", "sw_segments"],
            'CUSTOM': ["sw_custom_object"],
        }
        ctx = types.SimpleNamespace(
            mode='OBJECT', active_object=sketch, window_manager=bpy.context.window_manager,
        )
        for profile, fields in expected.items():
            with build.suspended():
                sketch.sketch_tools.sw_profile = profile
            layout = _RecordingLayout()
            panel.SKETCH_PT_output.draw(types.SimpleNamespace(layout=layout), ctx)
            shown = [n for n in layout.seen if n.startswith("sw_")]
            assert shown == ["sw_profile"] + fields + common, (profile, shown)


# ---------------------------------------------------------------------------
# Sketch draw tool (Phase 3)
# ---------------------------------------------------------------------------

def _registered_tool_ids():
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    cls = ToolSelectPanelHelper._tool_class_from_space_type('VIEW_3D')
    return {item.idname for item in ToolSelectPanelHelper._tools_flatten(cls._tools['EDIT_MESH']) if item}


def test_sketch_tool_registers_and_unregisters():
    from sketch_tools.ui import tool

    assert tool.TOOL_ID not in _registered_tool_ids()
    with addon():
        assert tool.TOOL_ID in _registered_tool_ids()
        assert create.DRAW_TOOL == tool.TOOL_ID
        keymap = tool.SKETCH_TOOL_sketch.bl_keymap
        assert [item[0] for item in keymap] == ["sketch.draw", "sketch.draw_3d", "sketch.draw", "sketch.draw"]
        lmb, lmb_3d = keymap[0], keymap[1]
        # Shift and Ctrl (-1 = any) may be held on the first click, Alt stays excluded for navigation.
        # The 3D path item has the same modifier rules; the operator polls decide which one runs.
        assert lmb[1] == {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": -1, "ctrl": -1}
        assert lmb_3d[1] == lmb[1]
        for item in keymap[2:]:
            assert set(item[1]) == {"type", "value"}
        assert {dict(i[2]["properties"])["mode"] for i in keymap if i[0] == "sketch.draw"} == {
            'SETTING', 'RECTANGLE', 'CIRCLE'}
        # The LMB item follows the shape chosen in the tool header (C21).
        assert dict(lmb[2]["properties"]) == {"mode": 'SETTING'}
        assert tool.SKETCH_TOOL_sketch.bl_context_mode == 'EDIT_MESH'
    assert tool.TOOL_ID not in _registered_tool_ids()


def test_registered_tool_keymap_lets_the_first_click_carry_shift_and_ctrl_but_not_alt():
    with addon():
        items = []
        for kc in bpy.context.window_manager.keyconfigs:
            for km in kc.keymaps:
                if km.name == "3D View Tool: Edit Mesh, Sketch":
                    items.extend(km.keymap_items)
        lmbs = {kmi.idname: kmi for kmi in items if kmi.type == 'LEFTMOUSE'}
        assert set(lmbs) == {"sketch.draw", "sketch.draw_3d"}
        assert {kmi.type for kmi in items} == {'LEFTMOUSE', 'R', 'O'}
        for lmb in lmbs.values():
            assert (lmb.shift, lmb.ctrl, lmb.alt) == (-1, -1, 0)
        for kmi in items:
            if kmi.type in ('R', 'O'):
                assert kmi.idname == "sketch.draw" and (kmi.shift, kmi.ctrl, kmi.alt) == (0, 0, 0)


def test_draw_operator_registers_with_an_undo_step_and_unregisters():
    from sketch_tools.ops import draw_tool

    with addon():
        assert hasattr(bpy.types, "SKETCH_OT_draw")
        assert 'UNDO' in draw_tool.SKETCH_OT_draw.bl_options
        assert hasattr(bpy.ops.sketch, "draw")
    assert not hasattr(bpy.types, "SKETCH_OT_draw")


def _draw_poll_context(mode, area_type, active):
    area = None if area_type is None else types.SimpleNamespace(type=area_type)
    return types.SimpleNamespace(mode=mode, area=area, active_object=active)


def test_draw_operator_poll_refuses_everything_but_a_sketch_in_a_3d_view_edit_mode():
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    with addon():
        sketch = make_sketch([SQUARE])
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.collection.objects.link(plain)
        empty = bpy.data.objects.new("Empty", None)
        poll = SKETCH_OT_draw.poll
        assert poll(_draw_poll_context('EDIT_MESH', 'VIEW_3D', sketch))
        assert not poll(_draw_poll_context('EDIT_MESH', 'VIEW_3D', plain))
        assert not poll(_draw_poll_context('EDIT_MESH', 'VIEW_3D', empty))
        assert not poll(_draw_poll_context('EDIT_MESH', 'VIEW_3D', None))
        assert not poll(_draw_poll_context('OBJECT', 'VIEW_3D', sketch))
        assert not poll(_draw_poll_context('EDIT_MESH', 'PROPERTIES', sketch))
        assert not poll(_draw_poll_context('EDIT_MESH', None, sketch))
        bpy.data.objects.remove(empty)
        bpy.data.objects.remove(plain)


def test_new_sketch_activates_the_sketch_tool_id():
    from sketch_tools.ui import tool

    real_bpy, real_flag = create.bpy, create._tool_error_shown
    names = []

    def tool_set_by_id(name):
        names.append(name)
        return {'FINISHED'}

    @contextmanager
    def temp_override(**kwargs):
        yield

    create.bpy = types.SimpleNamespace(
        ops=types.SimpleNamespace(wm=types.SimpleNamespace(tool_set_by_id=tool_set_by_id)),
        context=types.SimpleNamespace(temp_override=temp_override))
    try:
        create.activate_draw_tool(_fake_tool_context('VIEW_3D'))
    finally:
        create.bpy, create._tool_error_shown = real_bpy, real_flag
    assert names == [tool.TOOL_ID] == ["sketch_tools.sketch"]


class _TopView:
    """Orthographic top view: 100 px per sketch unit, sketch origin at pixel (500, 400)."""

    def __init__(self, edge_on=False, size=None, behind=False, overlays=()):
        self.edge_on = edge_on or behind
        self.miss = 'BEHIND' if behind else 'EDGE_ON'
        self.size = size
        self.overlays = overlays

    def contains(self, mouse, window):
        if any(x <= window[0] < x + w and y <= window[1] < y + h for x, y, w, h in self.overlays):
            return False
        if self.size is None:
            return True
        return 0.0 <= mouse[0] < self.size[0] and 0.0 <= mouse[1] < self.size[1]

    def ray_to_local(self, mouse):
        if self.edge_on:
            return None
        return ((mouse[0] - 500.0) / 100.0, (mouse[1] - 400.0) / 100.0)

    def to_screen(self, xy):
        return (xy[0] * 100.0 + 500.0, xy[1] * 100.0 + 400.0)


def _px(x, y):
    return (x * 100.0 + 500.0, y * 100.0 + 400.0)


# The fake region sits at this offset in the window.
_WINDOW_OFFSET = (30.0, 20.0)


def _event(etype, value='PRESS', at=(0.0, 0.0), shift=False, ctrl=False, alt=False):
    px = _px(*at)
    return types.SimpleNamespace(
        type=etype, value=value, mouse_region_x=px[0], mouse_region_y=px[1],
        mouse_x=px[0] + _WINDOW_OFFSET[0], mouse_y=px[1] + _WINDOW_OFFSET[1],
        shift=shift, ctrl=ctrl, alt=alt, oskey=False, is_repeat=False)


def _edit_sketch(loops=()):
    """New XY sketch left in edit mode, holding the given closed loops."""
    assert bpy.ops.sketch.new(plane='XY') == {'FINISHED'}
    obj = bpy.context.active_object
    bm = bmesh.from_edit_mesh(obj.data)
    for loop in loops:
        verts = [bm.verts.new((x, y, 0.0)) for x, y in loop]
        for i in range(len(verts)):
            bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
    bmesh.update_edit_mesh(obj.data)
    return obj


def _mesh_state(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.index_update()
    return (sorted((round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6)) for v in bm.verts),
            sorted(tuple(sorted(v.index for v in e.verts)) for e in bm.edges),
            sorted(v.index for v in bm.verts if v.select))


def _make_controller(obj, mode='POLYLINE', view=None, grid=0.1, segments=32, snap_px=12, region_pointer=None,
                     unit_scale=1.0):
    from sketch_tools.core.draw_logic import DrawSession
    from sketch_tools.ops.draw_tool import DrawController, SketchEditor

    view = view or _TopView()
    headers, redraws = [], [0]
    editor = SketchEditor(obj)
    controller = DrawController(
        editor, DrawSession(mode, grid=grid, circle_segments=segments, unit_scale=unit_scale), lambda: view, snap_px,
        header=headers.append, redraw=lambda: redraws.__setitem__(0, redraws[0] + 1),
        region_pointer=region_pointer)
    return controller, editor, headers, redraws


def _click_at(controller, x, y, **mods):
    return controller.handle(_event('LEFTMOUSE', 'PRESS', (x, y), **mods))


def _chains_after_leaving_edit_mode(obj):
    from sketch_tools.core import resolve

    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        resolved = resolve.resolve_bmesh(bm, 8)
    finally:
        bm.free()
    return [c for c in resolved.chains if c.closed], [c for c in resolved.chains if not c.closed]


def test_drawn_polyline_creates_sharp_vertices_and_edges():
    with addon():
        obj = _edit_sketch()
        c, editor, headers, _ = _make_controller(obj)
        assert c.start(_event('LEFTMOUSE', at=(0, 0))) == 'RUNNING_MODAL'
        assert _click_at(c, 2, 0) == 'RUNNING_MODAL'
        assert _click_at(c, 2, 1) == 'RUNNING_MODAL'
        assert c.handle(_event('RIGHTMOUSE', at=(2, 1))) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert [(round(v.co.x, 5), round(v.co.y, 5), v.co.z) for v in bm.verts] == [(0, 0, 0), (2, 0, 0), (2, 1, 0)]
        assert len(bm.edges) == 2
        rad = bm.verts.layers.float[FILLET_RADIUS_ATTR]
        ids = bm.verts.layers.int[FILLET_ID_ATTR]
        assert all(v[rad] == 0.0 and v[ids] == 0 for v in bm.verts)
        assert headers[-1] is None and c.closed


def test_new_vertices_next_to_a_filleted_corner_stay_sharp():
    with addon():
        obj = _edit_sketch([SQUARE])
        bm = bmesh.from_edit_mesh(obj.data)
        rad, ids = bm.verts.layers.float[FILLET_RADIUS_ATTR], bm.verts.layers.int[FILLET_ID_ATTR]
        bm.verts.ensure_lookup_table()
        bm.verts[2][rad] = 0.25
        bm.verts[2][ids] = 77
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(1, 1)))
        _click_at(c, 3, 3)
        assert c.handle(_event('RET', at=(3, 3))) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        rad, ids = bm.verts.layers.float[FILLET_RADIUS_ATTR], bm.verts.layers.int[FILLET_ID_ATTR]
        bm.verts.ensure_lookup_table()
        assert (bm.verts[2][rad], bm.verts[2][ids]) == (0.25, 77)
        assert all(v[rad] == 0.0 and v[ids] == 0 for v in list(bm.verts)[4:])
        assert len(bm.verts) == 6


def test_the_session_selects_what_it_draws_and_nothing_else():
    with addon():
        obj = _edit_sketch([SQUARE])
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(5, 5)))
        _click_at(c, 6, 5)
        c.handle(_event('RIGHTMOUSE'))
        bm = bmesh.from_edit_mesh(obj.data)
        assert sorted(v.index for v in bm.verts if v.select) == [4, 5]


def test_a_cancelled_chain_leaves_the_mesh_unchanged():
    with addon():
        obj = _edit_sketch([SQUARE])
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts[1].select_set(True)
        bmesh.update_edit_mesh(obj.data)
        before = _mesh_state(obj)
        assert before[2] == [1]
        c, editor, headers, _ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(5, 5)))
        _click_at(c, 6, 5)
        _click_at(c, 6, 6)
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 7
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert _mesh_state(obj) == before
        assert headers[-1] is None and c.closed


def test_cancel_removes_edges_between_existing_vertices_too():
    with addon():
        obj = _edit_sketch([SQUARE])
        before = _mesh_state(obj)
        c, *_ = _make_controller(obj)
        # Start on vertex 0 (ctrl snap), then snap to vertex 2: a new edge joins two old vertices.
        c.start(_event('LEFTMOUSE', at=(0, 0), ctrl=True))
        _click_at(c, 1, 1, ctrl=True)
        assert len(bmesh.from_edit_mesh(obj.data).edges) == 5
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert _mesh_state(obj) == before


def test_a_lone_click_followed_by_rmb_leaves_nothing():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(1, 1)))
        assert c.handle(_event('RIGHTMOUSE')) == 'CANCELLED'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 0


def test_drawing_back_onto_the_first_vertex_closes_the_loop():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 2, 0)
        _click_at(c, 2, 2)
        _click_at(c, 0, 2)
        # Ctrl and a few pixels off the first vertex: snaps to it, closes and ends.
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (0.05, 0.03), ctrl=True)) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 4 and len(bm.edges) == 4
        closed, opened = _chains_after_leaving_edit_mode(obj)
        assert len(closed) == 1 and not opened and len(closed[0].points) == 4


def test_c_closes_a_drawn_chain_into_one_loop():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 3, 0)
        _click_at(c, 3, 3)
        assert c.handle(_event('C')) == 'FINISHED'
        closed, opened = _chains_after_leaving_edit_mode(obj)
        assert len(closed) == 1 and not opened and len(closed[0].points) == 3


def test_snapping_onto_another_chain_reuses_its_vertex():
    with addon():
        obj = _edit_sketch([SQUARE])
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(4, 0)))
        assert _click_at(c, 1.02, 0.02, ctrl=True) == 'RUNNING_MODAL'
        assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 5 and len(bm.edges) == 5


def test_vertex_snap_outside_the_pixel_threshold_creates_a_new_vertex():
    with addon():
        obj = _edit_sketch([SQUARE])
        c, *_ = _make_controller(obj, snap_px=12)
        c.start(_event('LEFTMOUSE', at=(4, 0)))
        # 20 px from vertex (1, 0): outside the 12 px threshold, so the grid takes over.
        _click_at(c, 1.2, 0.0, ctrl=True)
        c.handle(_event('RIGHTMOUSE'))
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 6


def test_grid_snap_uses_the_local_increment():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, grid=0.25, unit_scale=2.0)
        c.start(_event('LEFTMOUSE', at=(0.1, 0.1), ctrl=True))
        _click_at(c, 2.27, 0.1, ctrl=True)
        c.handle(_event('RIGHTMOUSE'))
        bm = bmesh.from_edit_mesh(obj.data)
        assert sorted(round(v.co.x, 5) for v in bm.verts) == [0.0, 2.25]


def test_drawn_rectangle_is_a_closed_four_vertex_chain_that_extrudes():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='RECTANGLE')
        assert c.start(_event('LEFTMOUSE', at=(1, 1))) == 'RUNNING_MODAL'
        assert _click_at(c, 4, 3) == 'FINISHED'
        closed, opened = _chains_after_leaving_edit_mode(obj)
        assert len(closed) == 1 and not opened and len(closed[0].points) == 4
        obj.sketch_tools.output_mode = 'EXTRUDE'
        obj.sketch_tools.ex_height = 1.5
        bm = output_bm(obj)
        assert len(bm.faces) == 6
        assert abs(abs(signed_volume(bm)) - 3.0 * 2.0 * 1.5) < 1e-4
        assert_closed_and_consistent(bm)
        bm.free()


def test_rectangle_by_typing_width_and_height():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='RECTANGLE')
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        c.handle(_event('MOUSEMOVE', at=(1, 1)))
        for key in ('FOUR', 'TAB', 'TWO', 'PERIOD', 'FIVE'):
            assert c.handle(_event(key)) == 'RUNNING_MODAL'
        assert c.handle(_event('RET')) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert sorted((round(v.co.x, 5), round(v.co.y, 5)) for v in bm.verts) == [(0, 0), (0, 2.5), (4, 0), (4, 2.5)]


def test_rectangle_by_click_and_drag():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='RECTANGLE')
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (0.02, 0.0))) == 'RUNNING_MODAL'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (2, 1))) == 'FINISHED'
        assert len(bmesh.from_edit_mesh(obj.data).edges) == 4


def test_drawn_circle_has_the_preference_point_count_and_radius():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='CIRCLE', segments=20)
        c.start(_event('LEFTMOUSE', at=(1, 1)))
        assert _click_at(c, 3.5, 1) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 20 and len(bm.edges) == 20
        assert all(abs(math.hypot(v.co.x - 1, v.co.y - 1) - 2.5) < 1e-5 for v in bm.verts)
        closed, opened = _chains_after_leaving_edit_mode(obj)
        assert len(closed) == 1 and not opened


def test_edge_on_view_places_no_point_and_shows_a_hint():
    from sketch_tools.ops.draw_tool import EDGE_ON_HINT

    with addon():
        obj = _edit_sketch()
        c, editor, headers, _ = _make_controller(obj, view=_TopView(edge_on=True))
        assert c.start(_event('LEFTMOUSE', at=(1, 1))) == 'RUNNING_MODAL'
        assert _click_at(c, 2, 2) == 'RUNNING_MODAL'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 0
        assert EDGE_ON_HINT in headers[-1]
        assert c.handle(_event('ESC')) == 'CANCELLED'


def test_navigation_events_pass_through_and_numpad_types_only_while_typing():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        for etype in ('MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'NDOF_MOTION',
                      'NUMPAD_1', 'NUMPAD_7', 'NUMPAD_PERIOD', 'NUMPAD_PLUS'):
            assert c.handle(_event(etype)) == 'PASS_THROUGH', etype
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (1, 0), alt=True)) == 'PASS_THROUGH'
        assert c.handle(_event('LEFT_CTRL')) == 'RUNNING_MODAL'
        assert c.handle(_event('ONE')) == 'RUNNING_MODAL'
        assert c.session.numeric.texts[0] == "1"
        assert c.handle(_event('NUMPAD_2')) == 'RUNNING_MODAL'
        assert c.session.numeric.texts[0] == "12"
        assert c.handle(_event('F5')) == 'RUNNING_MODAL'
        assert c.handle(_event('C', ctrl=True)) == 'RUNNING_MODAL'
        c.abort()


def test_typed_length_and_angle_place_the_point():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        c.handle(_event('MOUSEMOVE', at=(1, 0.2)))
        for key in ('THREE', 'TAB', 'NINE', 'ZERO'):
            c.handle(_event(key))
        assert c.handle(_event('RET')) == 'RUNNING_MODAL'
        assert c.handle(_event('RET')) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert sorted((round(v.co.x, 5), round(v.co.y, 5)) for v in bm.verts) == [(0, 0), (0, 3)]


def test_every_exit_path_clears_the_header_and_runs_the_cleanups():
    from sketch_tools.ops import draw_tool

    exits = {
        'finish': lambda c: c.handle(_event('RIGHTMOUSE')),
        'cancel': lambda c: c.handle(_event('ESC')),
        'close': lambda c: c.handle(_event('C')),
        'abort': lambda c: c.abort(),
        'close_all': lambda c: draw_tool.close_all(),
    }
    with addon():
        for name, leave in exits.items():
            obj = _edit_sketch([SQUARE])
            before = _mesh_state(obj)
            c, editor, headers, _ = _make_controller(obj)
            cleaned = []
            c.add_cleanup(lambda: cleaned.append(1))
            c.start(_event('LEFTMOUSE', at=(5, 5)))
            _click_at(c, 6, 5)
            _click_at(c, 6, 6)
            leave(c)
            assert c.closed and cleaned == [1] and headers[-1] is None, name
            assert c not in draw_tool._live
            c.close()
            assert cleaned == [1]
            if name in {'cancel', 'abort', 'close_all'}:
                assert _mesh_state(obj) == before, name
            else:
                assert len(bmesh.from_edit_mesh(obj.data).verts) == 7, name
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_controller_stops_when_edit_mode_is_left():
    with addon():
        obj = _edit_sketch()
        c, editor, headers, _ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        bpy.ops.object.mode_set(mode='OBJECT')
        assert c.handle(_event('MOUSEMOVE', at=(1, 1))) == 'CANCELLED'
        assert c.closed and headers[-1] is None


def test_controller_stops_and_drops_uncommitted_work_when_the_view_is_lost():
    # The area was retyped, so there is no projector. Like leaving edit mode, the run ends.
    from sketch_tools.ops import draw_tool

    with addon():
        for etype in ('MOUSEMOVE', 'A', 'TIMER'):
            obj = _edit_sketch()
            before = _mesh_state(obj)
            c, editor, headers, _ = _make_controller(obj)
            c.start(_event('LEFTMOUSE', at=(0, 0)))
            assert _click_at(c, 2, 0) == 'RUNNING_MODAL'
            assert _mesh_state(obj) != before
            c._projector_factory = lambda: None
            assert c.handle(_event(etype, 'PRESS' if etype == 'A' else 'NOTHING', at=(1, 1))) == 'CANCELLED', etype
            assert c.closed and headers[-1] is None and not draw_tool._live, etype
            assert _mesh_state(obj) == before, etype
            assert c.snapshot() is None
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_overlay_handler_is_removed_on_every_path_and_on_unregister():
    from sketch_tools.ui import overlay

    with addon():
        shell = overlay.DrawOverlay(lambda: None)
        shell.add()
        shell.add()
        assert overlay._live == {shell}
        shell.remove()
        shell.remove()
        assert not overlay._live
        left_over = overlay.DrawOverlay(lambda: None)
        left_over.add()
    assert not overlay._live and left_over._handle is None


def test_overlay_geometry_projects_the_preview():
    from sketch_tools.core.draw_logic import DrawSession, Mods
    from sketch_tools.ui import overlay

    s = DrawSession('RECTANGLE', grid=0.1)
    s.handle(("CLICK", (0, 0)))
    s.handle(("MOVE", (2, 1), Mods(ctrl=True)))
    geo = overlay.screen_geometry(s.preview(), _TopView().to_screen)
    assert geo["band"][0] == [(500.0, 400.0), (700.0, 400.0), (700.0, 500.0), (500.0, 500.0), (500.0, 400.0)]
    assert len(geo["band"]) == 2 and not geo["guide"] and not geo["snap"]
    # A point behind the camera drops its strip.
    assert overlay.screen_geometry(s.preview(), lambda xy: None)["band"] == []


def _operator_shell(obj, mode='POLYLINE', view=None):
    """A stand-in `self` and context for calling SKETCH_OT_draw.invoke / modal directly."""
    from sketch_tools.ops import draw_tool, modal_common

    headers, modal_handlers = [], []
    area = fake_area(type='VIEW_3D', header_text_set=headers.append, tag_redraw=lambda: None)
    wm = types.SimpleNamespace(modal_handler_add=modal_handlers.append)
    context = types.SimpleNamespace(
        active_object=obj, area=area, window_manager=wm, scene=bpy.context.scene,
        preferences=bpy.context.preferences, region=types.SimpleNamespace(as_pointer=lambda: 4242))
    shell = types.SimpleNamespace(mode=mode, reports=[])
    shell.report = lambda kind, text: shell.reports.append((kind, text))
    restore = install_fake_area(area, lambda o: (lambda: view or _TopView()))
    return shell, context, headers, modal_handlers, restore


def test_operator_invoke_and_modal_paths_clean_up_after_themselves():
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw
    from sketch_tools.ui import overlay

    with addon():
        for how in ('finish', 'esc', 'error', 'cancel_callback'):
            obj = _edit_sketch([SQUARE])
            before = _mesh_state(obj)
            shell, ctx, headers, handlers, restore = _operator_shell(obj)
            try:
                assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(5, 5))) == {'RUNNING_MODAL'}
                assert handlers == [shell] and len(overlay._live) == 1 and headers[-1] is not None
                assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(6, 5))) == {'RUNNING_MODAL'}
                if how == 'finish':
                    assert SKETCH_OT_draw.modal(shell, ctx, _event('RIGHTMOUSE')) == {'FINISHED'}
                elif how == 'esc':
                    assert SKETCH_OT_draw.modal(shell, ctx, _event('ESC')) == {'CANCELLED'}
                elif how == 'error':
                    def broken(event):
                        raise ZeroDivisionError("boom")
                    shell._controller.session.handle = broken
                    with redirect_stdout(io.StringIO()):
                        assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(6, 6))) == {'CANCELLED'}
                    assert shell.reports and shell.reports[0][0] == {'ERROR'}
                else:
                    SKETCH_OT_draw.cancel(shell, ctx)
            finally:
                restore()
            assert not overlay._live and headers[-1] is None, how
            if how == 'finish':
                assert len(bmesh.from_edit_mesh(obj.data).verts) == 6
            else:
                assert _mesh_state(obj) == before, how
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_a_second_draw_invoke_passes_through_while_a_run_is_live():
    from sketch_tools.ops import draw_tool
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw
    from sketch_tools.ui import overlay

    with addon():
        obj = _edit_sketch()
        shell, ctx, headers, handlers, restore = _operator_shell(obj)
        second, ctx2, headers2, handlers2, restore2 = _operator_shell(obj)
        try:
            assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0))) == {'RUNNING_MODAL'}
            assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(1, 0))) == {'RUNNING_MODAL'}
            before = _mesh_state(obj)
            assert SKETCH_OT_draw.invoke(second, ctx2, _event('LEFTMOUSE', at=(3, 3))) == {'PASS_THROUGH'}
            assert not hasattr(second, "_controller") and not handlers2 and not headers2
            # Blender discards the reports of a passed-through invoke, so none is made.
            assert second.reports == []
            assert len(draw_tool._live) == 1 and len(overlay._live) == 1
            assert _mesh_state(obj) == before
            assert SKETCH_OT_draw.modal(shell, ctx, _event('RIGHTMOUSE')) == {'FINISHED'}
            # Once the first run is over a new one starts normally.
            assert SKETCH_OT_draw.invoke(second, ctx2, _event('LEFTMOUSE', at=(0, 4))) == {'RUNNING_MODAL'}
            SKETCH_OT_draw.cancel(second, ctx2)
        finally:
            restore()
            restore2()
        assert not draw_tool._live and not overlay._live


def test_the_draw_tool_never_calls_into_an_area_that_was_closed_mid_run():
    from sketch_tools.ops import draw_tool
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw
    from sketch_tools.ui import overlay

    for key in ('ESC', 'RET'):
        with addon():
            obj = _edit_sketch()
            shell, ctx, headers, handlers, restore = _operator_shell(obj)
            closed, late = [], []
            ctx.area.header_text_set = lambda text: late.append(text) if closed else headers.append(text)
            ctx.area.tag_redraw = lambda: late.append("redraw") if closed else None
            try:
                assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0))) == {'RUNNING_MODAL'}
                closed.append(True)
                modal_common.find_area = lambda pointer: None
                assert SKETCH_OT_draw.modal(shell, ctx, _event(key)) in ({'CANCELLED'}, {'FINISHED'})
            finally:
                restore()
            assert not late, (key, late)
            assert not draw_tool._live and not overlay._live


def test_the_starting_click_over_an_overlay_rectangle_passes_through():
    from sketch_tools.ops import draw_tool
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw
    from sketch_tools.ui import overlay

    sidebar = (_WINDOW_OFFSET[0] + 800.0, 0.0, 400.0, 2000.0)
    with addon():
        obj = _edit_sketch([SQUARE])
        before = _mesh_state(obj)
        view = _TopView(size=(1200.0, 900.0), overlays=(sidebar,))
        shell, ctx, headers, handlers, restore = _operator_shell(obj, view=view)
        try:
            # at=(4, 0) is pixel x 900 in the region, window x 930: inside the sidebar.
            assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(4, 0))) == {'PASS_THROUGH'}
            assert not hasattr(shell, "_controller") and not handlers and not headers
            assert not draw_tool._live and not overlay._live and _mesh_state(obj) == before
            # The same press over the open viewport starts a run.
            assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(1, 0))) == {'RUNNING_MODAL'}
            SKETCH_OT_draw.cancel(shell, ctx)
            # A key start has no press to test; it waits for the first click, which is tested then.
            assert SKETCH_OT_draw.invoke(shell, ctx, _event('R', at=(4, 0))) == {'RUNNING_MODAL'}
            assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(4, 0))) == {'PASS_THROUGH'}
            SKETCH_OT_draw.cancel(shell, ctx)
        finally:
            restore()
        assert not draw_tool._live and not overlay._live and _mesh_state(obj) == before


def test_rmb_outside_the_region_passes_its_release_and_click_through():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 2, 0)
        before = _mesh_state(obj)
        assert c.handle(_outside_event('RIGHTMOUSE', 'PRESS')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('RIGHTMOUSE', 'CLICK')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('RIGHTMOUSE', 'RELEASE')) == 'PASS_THROUGH'
        assert not c.closed and _mesh_state(obj) == before
        # An RMB press inside still ends the run.
        assert c.handle(_event('RIGHTMOUSE', 'PRESS', (2, 1))) == 'FINISHED'
        c.abort()
        # Without a passed press an RMB release is swallowed as before.
        obj2 = _edit_sketch()
        c, *_ = _make_controller(obj2, view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        assert c.handle(_event('RIGHTMOUSE', 'RELEASE', (1, 1))) == 'RUNNING_MODAL'
        c.abort()


def test_click_events_of_a_passed_through_press_pass_through_in_either_order():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 2, 0)
        before = _mesh_state(obj)
        for order in (('CLICK', 'RELEASE'), ('RELEASE', 'CLICK'), ('CLICK_DRAG',), ('RELEASE', 'CLICK_DRAG')):
            assert c.handle(_outside_event('LEFTMOUSE', 'PRESS')) == 'PASS_THROUGH'
            for value in order:
                assert c.handle(_outside_event('LEFTMOUSE', value)) == 'PASS_THROUGH', (order, value)
        assert c.handle(_outside_event('LEFTMOUSE', 'DOUBLE_CLICK')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFTMOUSE', 'CLICK')) == 'PASS_THROUGH'
        assert _mesh_state(obj) == before
        # A press inside ends the record: its own CLICK is swallowed (the press already acted).
        assert _click_at(c, 2, 1) == 'RUNNING_MODAL'
        assert c.handle(_event('LEFTMOUSE', 'CLICK', (2, 1))) == 'RUNNING_MODAL'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (2, 1))) == 'RUNNING_MODAL'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 3
        c.abort()


def test_a_passed_alt_lmb_press_passes_its_release_and_clicks_through_even_after_alt_is_let_go():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='RECTANGLE', view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        before = _mesh_state(obj)
        # Alt+LMB navigates; Alt comes up before LMB, so the release arrives without it, far away.
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (1, 1), alt=True)) == 'PASS_THROUGH'
        assert c.handle(_event('MOUSEMOVE', at=(3, 2), alt=True)) == 'RUNNING_MODAL'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (4, 3))) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'CLICK', (4, 3))) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'CLICK_DRAG', (4, 3))) == 'PASS_THROUGH'
        assert not c.closed and _mesh_state(obj) == before
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 0
        # A fresh plain press ends the record and the run goes on to place the second corner.
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (2, 1))) == 'FINISHED'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 4


def _gizmo_scene(region, header_h, ui=None, fac=1.5, dpi=108.0, nav=80, nav_ui=True, gizmo=True, navigate=True):
    """Fake region, area, space and preferences shaped like the ones the C15 measurements came from."""
    def part(kind, x, y, w, h, alignment):
        return types.SimpleNamespace(type=kind, x=x, y=y, width=w, height=h, alignment=alignment)

    rx, ry, rw, rh = region
    sidebar = part('UI', rx + rw - 1, ry, 1, 1, 'RIGHT') if ui is None else part('UI', ui[0], ry, ui[1], rh - 2 * header_h, 'RIGHT')
    regions = [part('WINDOW', rx, ry, rw, rh, 'NONE'),
               part('HEADER', rx, ry + rh - header_h, rw, header_h, 'TOP'),
               part('TOOL_HEADER', rx, ry + rh - 2 * header_h, rw, header_h, 'TOP'),
               part('TOOLS', rx, ry, 84, rh - 2 * header_h, 'LEFT'),
               part('HUD', rx + 84, ry, 1, 1, 'FLOAT'), sidebar]
    area = types.SimpleNamespace(regions=regions)
    space = types.SimpleNamespace(show_gizmo=gizmo, show_gizmo_navigate=navigate)
    prefs = types.SimpleNamespace(
        system=types.SimpleNamespace(ui_scale=fac, dpi=dpi),
        view=types.SimpleNamespace(gizmo_size_navigate_v3d=nav, show_navigate_ui=nav_ui))
    return regions[0], area, space, prefs


# (region, header height, sidebar (x, width) or None, UI scale factor, gizmo size, navigation buttons,
# measured drawn extent x0, x1, y0, y1). Measured on Blender 5.1 by diffing screenshots with the
# gizmo on and off, at 150 percent OS scaling, so the factor is 1.5 times the Resolution Scale.
_GIZMO_MEASURED = [
    ((3, 183, 3150, 1832), 38, None, 1.5, 80, True, (3020, 3139, 1618, 1920)),
    ((3, 183, 3150, 1832), 38, None, 1.5, 80, False, (3020, 3133, 1809, 1920)),
    ((3, 183, 3150, 1832), 38, (2733, 420), 1.5, 80, True, (2601, 2720, 1618, 1920)),
    ((3, 183, 3150, 1832), 38, (2733, 420), 1.5, 80, False, (2601, 2714, 1809, 1920)),
    ((4, 198, 3148, 1796), 58, None, 2.25, 80, True, (2954, 3132, 1397, 1850)),
    ((4, 198, 3148, 1796), 58, None, 2.25, 80, False, (2954, 3123, 1684, 1850)),
    ((4, 198, 3148, 1796), 58, (2521, 631), 2.25, 80, True, (2324, 2502, 1397, 1850)),
    ((4, 198, 3148, 1796), 58, (2521, 631), 2.25, 80, False, (2324, 2493, 1684, 1850)),
    ((2, 176, 3152, 1847), 31, None, 86 / 72, 80, True, (3048, 3143, 1706, 1946)),
    ((2, 176, 3152, 1847), 31, None, 86 / 72, 80, False, (3048, 3138, 1857, 1946)),
    ((2, 176, 3152, 1847), 31, (2819, 335), 86 / 72, 80, True, (2714, 2809, 1706, 1946)),
    ((2, 176, 3152, 1847), 31, (2819, 335), 86 / 72, 80, False, (2714, 2804, 1857, 1946)),
    ((3, 183, 3150, 1832), 38, None, 1.5, 120, True, (2962, 3139, 1552, 1918)),
    # Size 200, taken from the QA round 4 overlay of the drawn ball on the old rectangle (0.97 slope):
    # it stuck out 4.5 px on the left and 3.5 px at the bottom without buttons.
    ((3, 183, 3150, 1832), 38, None, 1.5, 200, False, (2837, 3139, 1624, 1918)),
]


def test_navigation_gizmo_rect_covers_every_measured_footprint_without_much_to_spare():
    from sketch_tools.ops.modal_common import navigation_gizmo_rect

    for region, header_h, ui, fac, nav, nav_ui, (x0, x1, y0, y1) in _GIZMO_MEASURED:
        scene = _gizmo_scene(region, header_h, ui, fac=fac, nav=nav, nav_ui=nav_ui)
        x, y, w, h = navigation_gizmo_rect(*scene)
        case = (region, ui, fac, nav, nav_ui)
        assert x <= x0 and y <= y0 and x + w >= x1 and y + h >= y1, case
        # Bounded slack: the margin, rounding and the 0.02 per size the ball slope was raised by, scaled.
        assert x0 - x <= 4 + (2 + 0.02 * nav) * fac and y0 - y <= 6 + (4 + 0.02 * nav) * fac, (case, x0 - x, y0 - y)


def test_navigation_gizmo_rect_tolerates_an_area_without_an_active_space():
    from sketch_tools.ops.modal_common import navigation_gizmo_rect

    region, area, _space, prefs = _gizmo_scene((3, 183, 3150, 1832), 38)
    # An area that is being closed reports no active space.
    area.spaces = types.SimpleNamespace(active=None)
    assert navigation_gizmo_rect(region, area, area.spaces.active, prefs) is None


def test_navigation_gizmo_rect_follows_the_visible_corner_and_the_switches():
    from sketch_tools.ops.modal_common import navigation_gizmo_rect

    region = (3, 183, 3150, 1832)
    closed = navigation_gizmo_rect(*_gizmo_scene(region, 38))
    # Right edge at the region's edge, top edge below the tool header (y 1939).
    assert closed[0] + closed[2] == 3153 and closed[1] + closed[3] == 1939
    opened = navigation_gizmo_rect(*_gizmo_scene(region, 38, ui=(2733, 420)))
    assert opened[0] + opened[2] == 2733 and opened[1] + opened[3] == 1939 and opened[2:] == closed[2:]
    # Region overlap off: the headers lie above the region, so its own top edge is the limit.
    scene = _gizmo_scene(region, 38)
    for part in scene[1].regions:
        if part.type in ('HEADER', 'TOOL_HEADER'):
            part.y = 183 + 1832
    plain = navigation_gizmo_rect(*scene)
    assert plain[1] + plain[3] == 183 + 1832
    # Bigger scale and the buttons grow it; without them only the ball is left.
    big = navigation_gizmo_rect(*_gizmo_scene(region, 38, fac=2.25))
    assert big[2] > closed[2] and big[3] > closed[3]
    without = navigation_gizmo_rect(*_gizmo_scene(region, 38, nav_ui=False))
    assert without[2] == closed[2] and without[3] < closed[3] / 2
    # Off switches: nothing to exclude.
    assert navigation_gizmo_rect(*_gizmo_scene(region, 38, gizmo=False)) is None
    assert navigation_gizmo_rect(*_gizmo_scene(region, 38, navigate=False)) is None
    # No window (background mode) reads a zero ui_scale: dpi / 72 stands in.
    assert navigation_gizmo_rect(*_gizmo_scene(region, 38, fac=0.0, dpi=108.0)) == closed
    # A region too small for the gizmo is clipped to itself.
    tiny = navigation_gizmo_rect(*_gizmo_scene((0, 0, 100, 150), 20))
    assert tiny[0] >= 0 and tiny[1] >= 0 and tiny[0] + tiny[2] == 100


def test_a_press_on_the_navigation_gizmo_passes_through_the_controller():
    from mathutils import Matrix
    from sketch_tools.ops.modal_common import ViewProjector, navigation_gizmo_rect

    class View(ViewProjector):
        """The real contains(), the fake top view for the rest."""

        def __init__(self, region, overlays):
            super().__init__(region, None, Matrix.Identity(4), overlays)
            self._top = _TopView()

        def ray_to_local(self, mouse):
            return self._top.ray_to_local(mouse)

        def to_screen(self, xy):
            return self._top.to_screen(xy)

    with addon():
        obj = _edit_sketch()
        scene = _gizmo_scene((30, 20, 1000, 840), 38)
        rect = navigation_gizmo_rect(*scene)
        c, *_ = _make_controller(obj, view=View(scene[0], [rect]))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 2, 0)
        before = _mesh_state(obj)
        window = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        for etype, value in (('LEFTMOUSE', 'PRESS'), ('LEFTMOUSE', 'RELEASE'), ('RIGHTMOUSE', 'PRESS'),
                             ('RIGHTMOUSE', 'RELEASE'), ('MOUSEMOVE', 'NOTHING')):
            assert c.handle(_overlay_event(etype, value, window)) == 'PASS_THROUGH', (etype, value)
        assert _mesh_state(obj) == before and not c.closed
        # Just outside the rectangle, drawing goes on.
        assert c.handle(_overlay_event('LEFTMOUSE', 'PRESS', (rect[0] - 2, window[1]))) == 'RUNNING_MODAL'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 3
        c.abort()


def test_operator_key_start_waits_for_the_first_click():
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    with addon():
        obj = _edit_sketch()
        shell, ctx, headers, handlers, restore = _operator_shell(obj, mode='RECTANGLE')
        try:
            assert SKETCH_OT_draw.invoke(shell, ctx, _event('R', at=(0, 0))) == {'RUNNING_MODAL'}
            assert len(bmesh.from_edit_mesh(obj.data).verts) == 0
            assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(0, 0))) == {'RUNNING_MODAL'}
            assert SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(2, 1))) == {'FINISHED'}
        finally:
            restore()
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 4 and headers[-1] is None


def test_operator_uses_preferences_for_grid_circle_and_snap():
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    with addon():
        entry = bpy.context.preferences.addons.new()
        entry.module = "sketch_tools"
        try:
            prefs = entry.preferences
            prefs.grid_increment = 0.5
            prefs.circle_segments = 12
            prefs.snap_pixel_threshold = 30
            bpy.context.scene.unit_settings.scale_length = 2.0
            obj = _edit_sketch()
            shell, ctx, headers, handlers, restore = _operator_shell(obj, mode='CIRCLE')
            try:
                SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0)))
                controller = shell._controller
                assert controller.session.circle_segments == 12
                assert controller._snap_pixels == 30
                # The DISTANCE preference is already in local units: no division by the scale.
                assert abs(controller.session.grid - 0.5) < 1e-9
                assert controller.session.unit_scale == 2.0
                assert controller.region_pointer == 4242
                SKETCH_OT_draw.modal(shell, ctx, _event('LEFTMOUSE', at=(1, 0)))
            finally:
                restore()
            assert len(bmesh.from_edit_mesh(obj.data).verts) == 12
        finally:
            bpy.context.scene.unit_settings.scale_length = 1.0
            bpy.context.preferences.addons.remove(entry)


def test_operator_unit_scale_follows_the_unit_system():
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    with addon():
        units = bpy.context.scene.unit_settings
        entry = bpy.context.preferences.addons.new()
        entry.module = "sketch_tools"
        try:
            entry.preferences.grid_increment = 0.5
            for system, scale, want in (('METRIC', 0.01, 0.01), ('NONE', 0.01, 1.0), ('NONE', 2.0, 1.0),
                                        ('METRIC', 2.0, 2.0)):
                units.system, units.scale_length = system, scale
                obj = _edit_sketch()
                shell, ctx, *_, restore = _operator_shell(obj)
                try:
                    SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0)))
                    session = shell._controller.session
                    assert abs(session.unit_scale - want) < 1e-6, (system, scale)
                    assert abs(session.grid - 0.5) < 1e-9, (system, scale)
                    SKETCH_OT_draw.cancel(shell, ctx)
                finally:
                    restore()
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.data.objects.remove(obj)
        finally:
            units.system, units.scale_length = 'METRIC', 1.0
            bpy.context.preferences.addons.remove(entry)


def test_operator_falls_back_without_preferences():
    from sketch_tools.core import draw_logic
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    with addon():
        obj = _edit_sketch()
        shell, ctx, *_rest, restore = _operator_shell(obj)
        try:
            SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0)))
            session = shell._controller.session
            assert session.circle_segments == draw_logic.DEFAULT_CIRCLE_SEGMENTS
            assert abs(session.grid - draw_logic.DEFAULT_GRID_INCREMENT) < 1e-9
            assert shell._controller._snap_pixels == draw_logic.DEFAULT_SNAP_PIXELS
            shell._controller.abort()
        finally:
            restore()


# -- QA round 1: region bounds, system events, abort, invoke safety, hints, overlay, units -----------

def _outside_event(etype, value='PRESS', **kwargs):
    event = _event(etype, value, **kwargs)
    event.mouse_region_x, event.mouse_region_y = 1500.0, 400.0
    event.mouse_x, event.mouse_y = 1500.0 + _WINDOW_OFFSET[0], 400.0 + _WINDOW_OFFSET[1]
    return event


def test_pointer_events_outside_the_region_never_add_points_and_presses_pass_through():
    with addon():
        obj = _edit_sketch()
        view = _TopView(size=(1000.0, 800.0))
        c, editor, headers, _ = _make_controller(obj, view=view)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        assert _click_at(c, 2, 0) == 'RUNNING_MODAL'
        before = _mesh_state(obj)
        header = headers[-1]
        # A click on the sidebar or header: pass through, press and release.
        assert c.handle(_outside_event('LEFTMOUSE', 'PRESS')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFTMOUSE', 'RELEASE')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFTMOUSE', 'DOUBLE_CLICK')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('RIGHTMOUSE')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('MOUSEMOVE')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFT_CTRL')) == 'PASS_THROUGH'
        assert _mesh_state(obj) == before and not c.closed and headers[-1] == header
        assert c.mouse is None
        # Back inside the run goes on, and the stray release flag did not stick.
        assert _click_at(c, 2, 1) == 'RUNNING_MODAL'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (2, 1))) == 'RUNNING_MODAL'
        assert c.handle(_event('RIGHTMOUSE', at=(2, 1))) == 'FINISHED'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 3


def test_a_drag_released_outside_the_region_commits_nothing():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, mode='RECTANGLE', view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        assert c.handle(_outside_event('LEFTMOUSE', 'RELEASE')) == 'RUNNING_MODAL'
        assert not c.closed and len(bmesh.from_edit_mesh(obj.data).verts) == 0
        c.abort()


def test_the_region_check_covers_every_edge_of_the_region():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, view=_TopView(size=(1000.0, 800.0)))
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        for x, y in ((-1.0, 100.0), (100.0, -1.0), (1000.0, 100.0), (100.0, 800.0)):
            event = _event('LEFTMOUSE')
            event.mouse_region_x, event.mouse_region_y = x, y
            event.mouse_x, event.mouse_y = x + _WINDOW_OFFSET[0], y + _WINDOW_OFFSET[1]
            assert c.handle(event) == 'PASS_THROUGH', (x, y)
        c.abort()


def _overlay_event(etype, value, window):
    """An event at a window position; the region-relative one follows from the offset."""
    event = _event(etype, value)
    event.mouse_x, event.mouse_y = window
    event.mouse_region_x = window[0] - _WINDOW_OFFSET[0]
    event.mouse_region_y = window[1] - _WINDOW_OFFSET[1]
    return event


def test_presses_over_header_sidebar_and_toolbar_pass_through_under_region_overlap():
    # The fake WINDOW region spans the whole area, as with Blender's default region overlap, so
    # the pointer is inside its bounds everywhere; only the overlay rectangles tell them apart.
    header = (30.0, 800.0, 970.0, 38.0)
    toolbar = (30.0, 120.0, 84.0, 680.0)
    sidebar = (860.0, 120.0, 170.0, 680.0)
    spots = {'header': (500.0, 820.0), 'toolbar': (60.0, 400.0), 'sidebar': (900.0, 400.0)}
    with addon():
        obj = _edit_sketch()
        view = _TopView(size=(1000.0, 840.0), overlays=(header, toolbar, sidebar))
        c, editor, headers, _ = _make_controller(obj, view=view)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        assert _click_at(c, 2, 0) == 'RUNNING_MODAL'
        before = _mesh_state(obj)
        for name, window in spots.items():
            for etype in ('LEFTMOUSE', 'RIGHTMOUSE'):
                assert c.handle(_overlay_event(etype, 'PRESS', window)) == 'PASS_THROUGH', (name, etype)
            assert c.handle(_overlay_event('LEFTMOUSE', 'RELEASE', window)) == 'PASS_THROUGH', name
            assert c.handle(_overlay_event('MOUSEMOVE', 'NOTHING', window)) == 'PASS_THROUGH', name
        assert _mesh_state(obj) == before and not c.closed and c.mouse is None
        # The open viewport between the overlays still draws.
        assert c.handle(_overlay_event('LEFTMOUSE', 'PRESS', (300.0, 400.0))) == 'RUNNING_MODAL'
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 3
        c.abort()


def test_view_projector_excludes_overlay_rectangles_in_window_coordinates():
    from mathutils import Matrix
    from sketch_tools.ops.modal_common import ViewProjector

    region = types.SimpleNamespace(width=800, height=600)
    projector = ViewProjector(region, None, Matrix.Identity(4), [(700.0, 100.0, 100.0, 400.0)])
    # The region-relative point is irrelevant to the overlay test; the sidebar sits at window x 700-800.
    assert projector.contains((10.0, 10.0), (110.0, 110.0))
    assert not projector.contains((10.0, 10.0), (700.0, 110.0))
    assert not projector.contains((10.0, 10.0), (799.9, 499.9))
    assert projector.contains((10.0, 10.0), (800.0, 110.0)) and projector.contains((10.0, 10.0), (700.0, 500.0))
    assert not projector.contains((800.0, 10.0), (110.0, 110.0))


def test_overlay_rects_take_header_toolbar_and_sidebar_regions_with_a_real_size():
    from sketch_tools.ops.modal_common import overlay_rects

    def region(kind, x, y, w, h):
        return types.SimpleNamespace(type=kind, x=x, y=y, width=w, height=h)

    area = types.SimpleNamespace(regions=[
        region('WINDOW', 0, 0, 1000, 800), region('HEADER', 0, 762, 1000, 38),
        region('TOOLS', 0, 0, 84, 700), region('UI', 580, 0, 420, 700),
        region('UI', 0, 0, 1, 1), region('HUD', 10, 10, 200, 1), region('ASSET_SHELF', 0, 0, 1000, 100),
        region('NAVIGATION_BAR', 0, 0, 50, 50)])
    assert overlay_rects(area) == [(0, 762, 1000, 38), (0, 0, 84, 700), (580, 0, 420, 700), (0, 0, 1000, 100)]


def test_non_input_events_pass_through_and_keyboard_is_still_swallowed():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        for etype in ('TIMER', 'TIMER0', 'TIMER_REPORT', 'TIMER_AUTOSAVE', 'WINDOW_DEACTIVATE',
                      'TEXTINPUT', 'NONE', 'EVT_DROP', 'ACTIONZONE_AREA'):
            assert c.handle(_event(etype, 'NOTHING')) == 'PASS_THROUGH', etype
        # Ctrl+Z, F-keys and letters without a meaning are swallowed.
        assert c.handle(_event('Z', ctrl=True)) == 'RUNNING_MODAL'
        assert c.handle(_event('F5')) == 'RUNNING_MODAL'
        assert c.handle(_event('Q')) == 'RUNNING_MODAL'
        c.abort()


def test_abort_rolls_back_uncommitted_work_even_when_the_session_is_marked_done():
    with addon():
        obj = _edit_sketch([SQUARE])
        before = _mesh_state(obj)
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(5, 5)))
        _click_at(c, 6, 5)
        assert _mesh_state(obj) != before
        # An edit that failed half way leaves the session done and the editor holding the work.
        c.session.done = True
        c.abort()
        assert c.closed and _mesh_state(obj) == before


def test_abort_after_a_committed_chain_keeps_it():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 1, 0)
        assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
        c.abort()
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 2


def test_invoke_cleans_up_when_start_or_the_overlay_or_the_handler_fails():
    from sketch_tools.ops import draw_tool
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw
    from sketch_tools.ui import overlay

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    with addon():
        for failing in ('start', 'overlay', 'handler'):
            obj = _edit_sketch([SQUARE])
            before = _mesh_state(obj)
            shell, ctx, headers, handlers, restore = _operator_shell(obj)
            real_start, real_add = draw_tool.DrawController.start, overlay.DrawOverlay.add
            if failing == 'start':
                draw_tool.DrawController.start = boom
            elif failing == 'overlay':
                overlay.DrawOverlay.add = boom
            else:
                ctx.window_manager.modal_handler_add = boom
            try:
                with redirect_stdout(io.StringIO()):
                    assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(5, 5))) == {'CANCELLED'}
            finally:
                draw_tool.DrawController.start, overlay.DrawOverlay.add = real_start, real_add
                restore()
            assert shell.reports and shell.reports[0][0] == {'ERROR'}, failing
            assert not draw_tool._live and not overlay._live, failing
            assert headers[-1] is None and shell._controller.closed, failing
            assert _mesh_state(obj) == before, failing
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_plane_behind_the_view_has_its_own_hint():
    from sketch_tools.ops.draw_tool import BEHIND_HINT, EDGE_ON_HINT

    assert BEHIND_HINT != EDGE_ON_HINT
    with addon():
        obj = _edit_sketch()
        c, editor, headers, _ = _make_controller(obj, view=_TopView(behind=True))
        c.start(_event('LEFTMOUSE', at=(1, 1)))
        _click_at(c, 2, 2)
        assert BEHIND_HINT in headers[-1] and EDGE_ON_HINT not in headers[-1]
        c.abort()


def test_view_projector_reports_the_miss_reason_and_the_region_bounds():
    from mathutils import Matrix
    from sketch_tools.ops import modal_common
    from sketch_tools.ops.modal_common import ViewProjector

    class Rv3d:
        is_perspective = True

    region = types.SimpleNamespace(width=800, height=600)
    rays = {}
    real = modal_common.view3d_utils
    modal_common.view3d_utils = types.SimpleNamespace(
        region_2d_to_origin_3d=lambda r, rv, m: rays["origin"],
        region_2d_to_vector_3d=lambda r, rv, m: rays["direction"])
    try:
        projector = ViewProjector(region, Rv3d(), Matrix.Identity(4))
        assert projector.contains((0.0, 0.0), (0.0, 0.0)) and projector.contains((799.0, 599.0), (799.0, 599.0))
        assert not projector.contains((800.0, 10.0), (800.0, 10.0))
        assert not projector.contains((10.0, -0.5), (10.0, -0.5))
        rays.update(origin=(0, 0, 5), direction=(1, 0, 0))
        assert projector.ray_to_local((1, 1)) is None and projector.miss == 'EDGE_ON'
        rays.update(origin=(0, 0, -5), direction=(0, 0, -1))
        assert projector.ray_to_local((1, 1)) is None and projector.miss == 'BEHIND'
        rays.update(origin=(1, 2, 5), direction=(0, 0, -1))
        assert projector.ray_to_local((1, 1)) == (1.0, 2.0) and projector.miss is None
    finally:
        modal_common.view3d_utils = real


def test_overlay_draws_only_in_the_tool_region_and_restores_the_blend_state():
    from sketch_tools.core.draw_logic import DrawSession
    from sketch_tools.ui import overlay

    session = DrawSession('POLYLINE', grid=0.1)
    session.handle(("CLICK", (0, 0)))
    session.handle(("MOVE", (2, 1)))
    state = (session.preview(), (10.0, 10.0), _TopView().to_screen)
    calls = {"state": 0, "blend": [], "batches": [], "text": 0}

    def get_state():
        calls["state"] += 1
        return state

    class Shader:
        def uniform_float(self, *args):
            pass

    fake_gpu = types.SimpleNamespace(
        shader=types.SimpleNamespace(from_builtin=lambda name: Shader()),
        state=types.SimpleNamespace(
            blend_get=lambda: 'ADDITIVE', blend_set=lambda mode: calls["blend"].append(mode)))

    class Batch:
        def __init__(self, data):
            calls["batches"].append(data["pos"])

        def draw(self, shader):
            pass

    fake_blf = types.SimpleNamespace(
        size=lambda *a: None, position=lambda *a: None, color=lambda *a: None,
        draw=lambda *a: calls.__setitem__("text", calls["text"] + 1))
    current = types.SimpleNamespace(as_pointer=lambda: 7, width=800, height=600)
    real = (overlay.bpy, overlay.gpu, overlay.blf, overlay.batch_for_shader)
    overlay.bpy = types.SimpleNamespace(context=types.SimpleNamespace(region=current))
    overlay.gpu, overlay.blf = fake_gpu, fake_blf
    overlay.batch_for_shader = lambda shader, kind, data: Batch(data)
    try:
        other = overlay.DrawOverlay(get_state, region_pointer=8)
        other._draw()
        assert calls["state"] == 0 and not calls["blend"] and not calls["text"]
        mine = overlay.DrawOverlay(get_state, region_pointer=7)
        mine._draw()
        assert calls["state"] == 1 and calls["text"] > 0 and not mine._failed
        assert calls["blend"] == ['ALPHA', 'ADDITIVE']
        assert calls["batches"] and all(len(p) == 3 and p[2] == 0.0 for pos in calls["batches"] for p in pos)
    finally:
        overlay.bpy, overlay.gpu, overlay.blf, overlay.batch_for_shader = real


def test_the_controller_keeps_the_region_pointer_for_the_overlay():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, region_pointer=99)
        assert c.region_pointer == 99
        c.abort()


def test_typed_length_through_the_controller_is_in_the_scene_base_unit():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, unit_scale=0.01)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        c.handle(_event('MOUSEMOVE', at=(1, 0)))
        c.handle(_event('TWO'))
        assert c.handle(_event('RET')) == 'RUNNING_MODAL'
        assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert sorted(round(v.co.x, 4) for v in bm.verts) == [0.0, 200.0]
