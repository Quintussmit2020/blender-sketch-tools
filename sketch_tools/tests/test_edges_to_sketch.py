"""Edges to Sketch (DECISIONS C25), headless: the plain-data maths in core/edges_to_sketch.py and the
operator on real edit-mode meshes."""

import math
import types

import bmesh
import bpy
from mathutils import Matrix, Vector

from sketch_tools.core import build, edges_to_sketch as e2s
from sketch_tools.core.resolve3d import PATH_NORMAL_ATTR
from sketch_tools.ops import create

from .test_addon import addon, only_event_handlers, output_bm

SQUARE = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
LOOP = [(0, 1), (1, 2), (2, 3), (3, 0)]


def close(a, b, eps=1e-5):
    return (Vector(a) - Vector(b)).length < eps


def make_obj(name, verts, edges=(), faces=(), matrix=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, list(edges), list(faces))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if matrix is not None:
        obj.matrix_world = matrix
    return obj


def edit(objs, select, mode=(True, False, False)):
    """Enter edit mode on objs (the first is active), select edges by index lists
    {obj: [edge indices]} (or faces when mode is faces only)."""
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.tool_settings.mesh_select_mode = mode
    for o in objs:
        bm = bmesh.from_edit_mesh(o.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for elem in list(bm.verts) + list(bm.edges) + list(bm.faces):
            elem.select_set(False)
        indices = select.get(o, [])
        if mode == (False, False, True):
            for i in indices:
                bm.faces[i].select_set(True)
        else:
            for i in indices:
                bm.edges[i].select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(o.data)


def run():
    assert bpy.ops.sketch.edges_to_sketch() == {'FINISHED'}
    return bpy.context.active_object


def world(obj):
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def test_planar_square_gives_an_extruded_planar_sketch():
    with addon():
        src = make_obj("Src", SQUARE, LOOP)
        edit([src], {src: [0, 1, 2, 3]})
        obj = run()
        assert obj.name == "Src_Sketch"
        st = obj.sketch_tools
        assert st.is_sketch and not st.is_path_3d and st.output_mode == 'EXTRUDE'
        # No faces, no view: the normal falls back to world +Z, X to world X.
        assert obj.matrix_world.to_3x3() == Matrix.Identity(3) or all(
            close(obj.matrix_world.to_3x3()[i], Matrix.Identity(3)[i]) for i in range(3))
        assert close(obj.matrix_world.translation, (1, 1, 0))
        assert all(v.co.z == 0.0 for v in obj.data.vertices)
        assert len(obj.data.vertices) == 4 and len(obj.data.edges) == 4
        for p in SQUARE:
            assert any(close(p, w) for w in world(obj))
        out = st.output_object
        assert out is not None and out.parent == obj and len(out.data.polygons) > 0
        assert obj.display_type == 'WIRE' and out.hide_select


def test_tilted_square_on_rotated_scaled_source_keeps_world_positions_and_normal():
    with addon():
        matrix = (Matrix.Translation((3, -2, 5))
                  @ Matrix.Rotation(0.7, 4, 'X') @ Matrix.Rotation(-0.4, 4, 'Z')
                  @ Matrix.Diagonal((2.0, 0.5, 3.0, 1.0)))
        # A face whose winding points the normal to local -Z.
        src = make_obj("Tilt", SQUARE, LOOP, [(3, 2, 1, 0)], matrix)
        edit([src], {src: [0, 1, 2, 3]})
        expected_points = [matrix @ Vector(p) for p in SQUARE]
        normal_matrix = matrix.to_3x3().inverted().transposed()
        expected_normal = (normal_matrix @ Vector((0, 0, -1))).normalized()
        obj = run()
        z = obj.matrix_world.to_3x3().col[2].normalized()
        assert close(z, expected_normal, 1e-4), (z, expected_normal)
        got = world(obj)
        assert len(got) == 4
        for p in expected_points:
            assert any(close(p, g, 1e-4) for g in got), p
        assert all(v.co.z == 0.0 for v in obj.data.vertices)
        # The frame is orthonormal and right handed.
        m3 = obj.matrix_world.to_3x3()
        assert abs(m3.determinant() - 1.0) < 1e-5


def test_open_planar_polyline_sweeps():
    with addon():
        src = make_obj("Line", [(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1), (1, 2)])
        edit([src], {src: [0, 1]})
        obj = run()
        st = obj.sketch_tools
        assert st.is_sketch and not st.is_path_3d and st.output_mode == 'SWEEP'
        assert st.output_object is not None and len(st.output_object.data.polygons) > 0


def _cube(name, matrix=None):
    s = 1.0
    verts = [(x * s, y * s, z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for v in verts:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=list(bm.verts))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.dissolve_limit(bm, angle_limit=1e-3, verts=list(bm.verts), edges=list(bm.edges))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if matrix is not None:
        obj.matrix_world = matrix
    return obj


def _corner_edges(obj, corner):
    return [e.index for e in obj.data.edges
            if any(close(obj.data.vertices[v].co, corner) for v in e.vertices)]


def test_cube_corner_edges_give_a_3d_path_with_vertex_normals():
    with addon():
        matrix = Matrix.Rotation(0.3, 4, 'Z') @ Matrix.Diagonal((2.0, 1.0, 0.5, 1.0))
        cube = _cube("Cube", matrix)
        assert len(cube.data.polygons) == 6
        corner = (1, 1, 1)
        edges = _corner_edges(cube, corner)
        assert len(edges) == 3
        edit([cube], {cube: edges})
        # Expected: the bmesh vertex normals of the source, to world.
        bm = bmesh.new()
        bm.from_mesh(cube.data)
        normal_matrix = matrix.to_3x3().inverted().transposed()
        expected = {}
        for v in bm.verts:
            expected[tuple(round(c, 4) for c in matrix @ v.co)] = (normal_matrix @ v.normal).normalized()
        bm.free()
        obj = run()
        st = obj.sketch_tools
        assert st.is_path_3d and st.is_sketch and st.output_mode == 'SWEEP'
        assert obj.name == "Cube_Path"
        assert obj.matrix_world == Matrix.Identity(4)
        assert len(obj.data.vertices) == 4 and len(obj.data.edges) == 3
        attr = obj.data.attributes[PATH_NORMAL_ATTR].data
        for v, a in zip(obj.data.vertices, attr):
            key = tuple(round(c, 4) for c in v.co)
            assert key in expected, key
            n = Vector(a.vector)
            assert abs(n.length - 1.0) < 1e-5
            assert close(n, expected[key], 1e-4)
        assert st.output_object is not None and len(st.output_object.data.polygons) > 0


def test_zigzag_wire_uses_tangent_fallback_normals():
    with addon():
        pts = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 3)]
        src = make_obj("Zig", pts, [(0, 1), (1, 2), (2, 3)])
        edit([src], {src: [0, 1, 2]})
        obj = run()
        assert obj.sketch_tools.is_path_3d
        verts = obj.data.vertices
        attr = obj.data.attributes[PATH_NORMAL_ATTR].data
        order = {tuple(round(c) for c in v.co): i for i, v in enumerate(verts)}
        # End point (0,0,0): tangent X, so the normal is the Z perpendicular: +Z.
        n0 = Vector(attr[order[(0, 0, 0)]].vector)
        assert close(n0, (0, 0, 1))
        for v, a in zip(verts, attr):
            n = Vector(a.vector)
            assert abs(n.length - 1.0) < 1e-5 and n.z > 0


def test_plain_data_fallback_normals():
    # A wire hanging off a face-bearing vertex copies that vertex's normal along the chain.
    data = e2s.EdgeData(
        [Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((2, 0, 0)), Vector((2, 1, 1))],
        [Vector((1, 0, 0)), None, None, None], [(0, 1), (1, 2), (2, 3)])
    normals = e2s._path_normals(data)
    assert all(close(n, (1, 0, 0)) for n in normals)
    # Opposite neighbour normals cancel: tangent fallback for that vertex only.
    data = e2s.EdgeData(
        [Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((2, 0, 0))],
        [Vector((0, 1, 0)), None, Vector((0, -1, 0))], [(0, 1), (1, 2)])
    normals = e2s._path_normals(data)
    assert close(normals[1], (0, 0, 1)) and close(normals[0], (0, 1, 0))
    # Tangent along Z: closest perpendicular to +Z does not exist, +X is used.
    data = e2s.EdgeData([Vector((0, 0, 0)), Vector((0, 0, 1))], [None, None], [(0, 1)])
    assert close(e2s._path_normals(data)[0], (1, 0, 0))


def test_face_mode_uses_the_boundary_of_the_selected_region():
    with addon():
        verts = [(x, y, 0) for y in range(3) for x in range(3)]
        faces = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)]
        edges = set()
        for f in faces:
            for i in range(4):
                edges.add(tuple(sorted((f[i], f[(i + 1) % 4]))))
        src = make_obj("Grid", verts, sorted(edges), faces)
        edit([src], {src: [0, 1]}, mode=(False, False, True))
        obj = run()
        # Two faces side by side: 6 boundary edges, 6 vertices (the shared edge is interior).
        assert len(obj.data.edges) == 6 and len(obj.data.vertices) == 6
        assert obj.sketch_tools.output_mode == 'EXTRUDE'
        # Face normal +Z decides the orientation.
        assert close(obj.matrix_world.to_3x3().col[2], (0, 0, 1))


def test_multi_object_edit_merges_into_one_world_space_result():
    with addon():
        a = make_obj("A", [(0, 0, 0), (1, 0, 0)], [(0, 1)])
        b = make_obj("B", [(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1), (1, 2)],
                     matrix=Matrix.Translation((1, 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'Z'))
        # B in world: (1,0,0), (1,1,0), (0,1,0); its first point meets A's end.
        edit([a, b], {a: [0], b: [0, 1]})
        obj = run()
        assert obj.name == "A_Sketch"
        assert len(obj.data.edges) == 3 and len(obj.data.vertices) == 4
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]:
            assert any(close(p, w, 1e-4) for w in world(obj)), p
        assert obj.sketch_tools.output_mode == 'SWEEP'


def test_source_is_untouched():
    with addon():
        cube = _cube("Cube", Matrix.Translation((1, 2, 3)))
        edges = _corner_edges(cube, (1, 1, 1))
        edit([cube], {cube: edges})
        bm = bmesh.from_edit_mesh(cube.data)
        before = ([tuple(v.co) for v in bm.verts], [tuple(v.index for v in e.verts) for e in bm.edges],
                  [v.select for v in bm.verts], [e.select for e in bm.edges], [f.select for f in bm.faces])
        run()
        assert cube.mode == 'OBJECT'
        mesh = cube.data
        assert len(mesh.vertices) == 8 and len(mesh.edges) == 12
        assert [tuple(v.co) for v in mesh.vertices] == before[0]
        assert [tuple(e.vertices) for e in mesh.edges] == before[1]
        assert [v.select for v in mesh.vertices] == before[2]
        assert [e.select for e in mesh.edges] == before[3]
        assert [p.select for p in mesh.polygons] == before[4]


def test_nothing_selected_cancels_with_a_report():
    with addon():
        src = make_obj("Src", SQUARE, LOOP)
        edit([src], {})
        messages = []
        fake = types.SimpleNamespace(report=lambda kind, text: messages.append((kind, text)))
        assert create.SKETCH_OT_edges_to_sketch.execute(fake, bpy.context) == {'CANCELLED'}
        assert messages and "No edges" in messages[0][1]
        assert bpy.ops.sketch.edges_to_sketch() == {'CANCELLED'}
        assert bpy.context.mode == 'EDIT_MESH' and len(bpy.data.objects) == 1


def test_poll_needs_edit_mode_on_a_non_sketch_mesh():
    with addon():
        src = make_obj("Src", SQUARE, LOOP)
        bpy.context.view_layer.objects.active = src
        src.select_set(True)
        assert not bpy.ops.sketch.edges_to_sketch.poll()
        edit([src], {src: [0, 1, 2, 3]})
        assert bpy.ops.sketch.edges_to_sketch.poll()
        sketch = run()
        assert not bpy.ops.sketch.edges_to_sketch.poll()
        bpy.ops.object.mode_set(mode='EDIT')
        assert bpy.context.active_object == sketch and build.is_sketch(sketch)
        assert not bpy.ops.sketch.edges_to_sketch.poll()


def test_result_is_the_only_selected_and_active_object_in_object_mode():
    with addon():
        a = make_obj("A", SQUARE, LOOP)
        b = make_obj("B", [(5, 5, 0), (6, 5, 0)], [(0, 1)])
        edit([a, b], {a: [0, 1, 2, 3], b: [0]})
        obj = run()
        assert bpy.context.mode == 'OBJECT'
        assert bpy.context.view_layer.objects.active == obj
        assert [o.name for o in bpy.context.selected_objects] == [obj.name]
        assert a.mode == 'OBJECT' and b.mode == 'OBJECT'
        assert obj.sketch_tools.output_object is not None
        out = obj.sketch_tools.output_object
        assert not out.select_get()


def test_single_edge_degenerate_cases():
    # Wire edge, no faces, no view: world Z plane through the line, X along the edge.
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((2, 0, 0))], [None, None], [(0, 1)]))
    r = e2s.build_result(data)
    assert r.planar and close(r.z, (0, 0, 1)) and close(r.x, (1, 0, 0)) and not r.closed_only
    assert all(c.z == 0.0 for c in r.coords)
    # A vertical edge: Z is parallel, so X is used for the plane normal.
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((0, 0, 2))], [None, None], [(0, 1)]))
    r = e2s.build_result(data)
    assert abs(abs(r.z.dot(Vector((1, 0, 0)))) - 1.0) < 1e-5
    # View beats the world fallback: viewer along -Y gives a plane facing -Y.
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((2, 0, 0))], [None, None], [(0, 1)]))
    r = e2s.build_result(data, view=Vector((0, -1, 0)))
    assert close(r.z, (0, -1, 0))
    # Faces beat the view and the plane is one face's plane, not the diagonal between two:
    # without a view the face closest to +Z wins, with a view the one facing it.
    faces = [Vector((0, 0, 1)), Vector((1, 0, 0))]
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((0, 2, 0))], [None, None], [(0, 1)], faces))
    assert close(e2s.build_result(data).z, (0, 0, 1))
    assert close(e2s.build_result(data, view=Vector((0, -1, 0))).z, (0, 0, 1))  # tie: closest to +Z
    assert close(e2s.build_result(data, view=Vector((1, 0, 0.2)).normalized()).z, (1, 0, 0))
    # Both sides of a sheet: the face towards the viewer.
    faces = [Vector((0, 0, 1)), Vector((0, 0, -1))]
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((2, 0, 0))], [None, None], [(0, 1)], faces))
    assert close(e2s.build_result(data, view=Vector((0, 0.3, -1)).normalized()).z, (0, 0, -1))
    assert close(e2s.build_result(data, view=Vector((0, 0.3, 1)).normalized()).z, (0, 0, 1))
    # A face whose normal runs along the edge is unusable: the view decides.
    faces = [Vector((1, 0, 0))]
    data = e2s.weld(e2s.EdgeData([Vector((0, 0, 0)), Vector((2, 0, 0))], [None, None], [(0, 1)], faces))
    assert close(e2s.build_result(data, view=Vector((0, -1, 0))).z, (0, -1, 0))


def test_cube_edge_gets_the_plane_of_one_face_not_the_diagonal():
    with addon():
        cube = _cube("Cube")
        # The edge between the top (+Z) and the +X face.
        edge = [e.index for e in cube.data.edges
                if all(abs(cube.data.vertices[v].co.x - 1) < 1e-5 and abs(cube.data.vertices[v].co.z - 1) < 1e-5
                       for v in e.vertices)]
        assert len(edge) == 1
        edit([cube], {cube: edge})
        obj = run()
        assert obj.sketch_tools.output_mode == 'SWEEP' and not obj.sketch_tools.is_path_3d
        z = obj.matrix_world.to_3x3().col[2]
        assert close(z, (0, 0, 1)), z


def _bm_of(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def test_dirty_indices_do_not_weld_separate_vertices():
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in [(0, 0, 0), (1, 0, 0), (5, 0, 0), (6, 0, 0)]]
    for a, b in ((0, 1), (2, 3)):
        e = bm.edges.new((verts[a], verts[b]))
        e.select_set(True)
    assert all(v.index == -1 for v in verts)
    data = e2s.gather_edges(bm, Matrix.Identity(4), False)
    bm.free()
    assert len(data.points) == 4 and data.edges == [(0, 1), (2, 3)]
    assert [tuple(p) for p in data.points] == [(0, 0, 0), (1, 0, 0), (5, 0, 0), (6, 0, 0)]


def test_normals_are_fresh_and_the_bmesh_is_not_modified():
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]]
    bm.faces.new(vs)
    bm.normal_update()
    # Tip the face after the cache was computed: the cached normal still says +Z.
    for v in vs:
        v.co.x, v.co.z = v.co.z, v.co.x
    assert abs(vs[0].normal.z - 1.0) < 1e-6
    for e in bm.edges:
        e.select_set(True)
    before = [tuple(v.co) for v in bm.verts]
    data = e2s.gather_edges(bm, Matrix.Identity(4), False)
    assert [tuple(v.co) for v in bm.verts] == before
    bm.free()
    assert all(abs(abs(n.x) - 1.0) < 1e-5 for n in data.normals)
    assert all(abs(abs(n.x) - 1.0) < 1e-5 for n in data.face_normals)


def test_face_mode_orientation_ignores_unselected_neighbour_faces():
    with addon():
        cube = _cube("Cube")
        bm = _bm_of(cube)
        top = [f for f in bm.faces if f.normal.z > 0.9][0]
        for f in bm.faces:
            f.select_set(False)
        top.select_set(True)
        data = e2s.gather_edges(bm, Matrix.Identity(4), True)
        bm.free()
        assert len(data.edges) == 4
        assert len(data.face_normals) == 1 and close(data.face_normals[0], (0, 0, 1))
        # An asymmetric edge-mode selection of the same edges sees all four side faces too.
        bm = _bm_of(cube)
        for e in bm.edges:
            e.select_set(False)
        for f in bm.faces:
            if f.normal.z > 0.9:
                for e in f.edges:
                    e.select_set(True)
        data = e2s.gather_edges(bm, Matrix.Identity(4), False)
        bm.free()
        assert len(data.face_normals) == 5


def test_mixed_face_only_selection_gives_the_region_outline():
    with addon():
        cube = _cube("Cube")
        bm = _bm_of(cube)
        picks = [f.index for f in bm.faces if f.normal.z > 0.9 or f.normal.x > 0.9]
        bm.free()
        assert len(picks) == 2
        edit([cube], {cube: picks}, mode=(False, False, True))
        obj = run()
        # An L shaped outline: 6 edges and 6 vertices, not planar.
        assert obj.sketch_tools.is_path_3d and obj.sketch_tools.output_mode == 'SWEEP'
        assert len(obj.data.edges) == 6 and len(obj.data.vertices) == 6


def test_view_towards_viewer_reads_the_region_rotation():
    rotation = Matrix.Rotation(math.pi / 2, 4, 'X').to_quaternion()
    space = types.SimpleNamespace(type='VIEW_3D', region_3d=types.SimpleNamespace(view_rotation=rotation))
    view = create._view_towards_viewer(types.SimpleNamespace(space_data=space))
    assert close(view, (0, -1, 0))
    assert create._view_towards_viewer(types.SimpleNamespace(space_data=None)) is None
    other = types.SimpleNamespace(type='PROPERTIES', region_3d=None)
    assert create._view_towards_viewer(types.SimpleNamespace(space_data=other)) is None
    nothing = types.SimpleNamespace(type='VIEW_3D', region_3d=None)
    assert create._view_towards_viewer(types.SimpleNamespace(space_data=nothing)) is None


def test_undo_removes_the_result_and_restores_both_sources():
    with addon():
        a = make_obj("A", SQUARE, LOOP)
        b = make_obj("B", [(5, 5, 0), (6, 5, 0)], [(0, 1)])
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="setup")
        edit([a, b], {a: [0, 1, 2, 3], b: [0]})
        bpy.ops.ed.undo_push(message="selected")
        run()
        bpy.ops.ed.undo_push(message="result")
        assert len(bpy.data.objects) == 4  # two sources, the sketch and its output
        with only_event_handlers():
            bpy.ops.ed.undo()
        names = sorted(o.name for o in bpy.data.objects)
        assert names == ["A", "B"], names
        assert len(bpy.data.objects["A"].data.edges) == 4 and len(bpy.data.objects["B"].data.edges) == 1
