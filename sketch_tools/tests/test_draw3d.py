"""sketch.draw_3d (DECISIONS C19, C20): the scene source, the editor, the controller and the operator
shell with a fake view and fake events, as in the planar draw tool's tests. The viewport itself
(real regions, the keymap dispatch) is checked by hand."""

import io
import math
import types
from contextlib import redirect_stdout

import bmesh
import bpy
from mathutils import Matrix, Vector

from sketch_tools.core import resolve3d, snap3d
from sketch_tools.ops import draw3d, draw_tool, modal_common
from sketch_tools.tests.test_addon import (
    _WINDOW_OFFSET, SQUARE, addon, fake_area, install_fake_area, make_sketch,
)
from sketch_tools.tests.test_snap3d import Pinhole
from sketch_tools.ui import overlay

EYE = (5.0, -6.0, 4.0)
# Points on the three faces the camera sees (+Z, +X, -Y), well away from every edge.
TOP, SIDE, FRONT = (0.2, 0.3, 1.0), (1.0, 0.2, 0.1), (0.3, -1.0, -0.2)


class View(Pinhole):
    """The Pinhole camera with the region bounds test of the projector."""

    def __init__(self, eye=EYE, overlays=(), **kw):
        super().__init__(eye, **kw)
        self.overlays = overlays

    def contains(self, mouse, window):
        if any(x <= window[0] < x + w and y <= window[1] < y + h for x, y, w, h in self.overlays):
            return False
        return 0.0 <= mouse[0] < self.size[0] and 0.0 <= mouse[1] < self.size[1]

    def to_screen(self, co):
        p = self.project(co)
        return None if p is None else (p[0], p[1])


def ev(etype, value='PRESS', px=(400.0, 300.0), shift=False, ctrl=False, alt=False):
    return types.SimpleNamespace(
        type=etype, value=value, mouse_region_x=px[0], mouse_region_y=px[1],
        mouse_x=px[0] + _WINDOW_OFFSET[0], mouse_y=px[1] + _WINDOW_OFFSET[1],
        shift=shift, ctrl=ctrl, alt=alt, oskey=False, is_repeat=False)


def at(view, co, **mods):
    """A left press where the world point co shows on screen."""
    return ev('LEFTMOUSE', 'PRESS', view.screen(co), **mods)


def cube_object(name="Cube", matrix=None):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if matrix is not None:
        obj.matrix_world = matrix
    return obj


def edit_path(matrix=None):
    """A new 3D path in edit mode, with a cube at the origin to draw on."""
    cube_object()
    assert bpy.ops.sketch.new_path_3d() == {'FINISHED'}
    obj = bpy.context.active_object
    if matrix is not None:
        obj.matrix_world = matrix
    bpy.context.view_layer.update()
    return obj


def make_controller(obj, view=None, snap_px=12, source=None):
    view = view or View()
    source = source or draw3d.build_scene_source(bpy.context, obj)
    headers, redraws = [], [0]
    controller = draw3d.PathController(
        draw3d.PathEditor(obj), snap3d.PathSession(), lambda: view, source, snap_px,
        header=headers.append, redraw=lambda: redraws.__setitem__(0, redraws[0] + 1))
    return controller, headers, redraws, view


def path_state(obj):
    """(sorted [(co, normal)], sorted edge index pairs, selected indices) of the edit mesh."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.index_update()
    layer = bm.verts.layers.float_vector.get(resolve3d.PATH_NORMAL_ATTR)
    points = sorted((tuple(round(c, 4) for c in v.co), tuple(round(c, 4) for c in v[layer])) for v in bm.verts)
    edges = sorted(tuple(sorted(v.index for v in e.verts)) for e in bm.edges)
    return points, edges, sorted(v.index for v in bm.verts if v.select)


def coords(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return [tuple(round(c, 4) for c in v.co) for v in bm.verts]


def close(a, b, eps=1e-3):
    return (Vector(a) - Vector(b)).length < eps


# ---------------------------------------------------------------------------
# The scene source
# ---------------------------------------------------------------------------

def test_scene_source_takes_visible_meshes_that_are_not_sketches_outputs_or_the_path():
    with addon():
        path = edit_path()
        # The path's own output has faces once it has a segment.
        bm = bmesh.from_edit_mesh(path.data)
        bm.verts.new((0, 0, 1))
        bm.verts.new((1, 0, 1))
        bm.edges.new(tuple(bm.verts))
        bmesh.update_edit_mesh(path.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        out = path.sketch_tools.output_object
        assert out is not None and len(out.data.polygons) > 0
        hidden = cube_object("Hidden")
        hidden.hide_viewport = True
        wire = bpy.data.objects.new("Wire", bpy.data.meshes.new("Wire"))
        bpy.context.collection.objects.link(wire)
        wire.data.from_pydata([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
        planar = make_sketch([SQUARE], mode='EXTRUDE')
        assert planar.sketch_tools.output_object is not None
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        # Only the one cube. Hidden, wire-only, sketches and outputs are out.
        assert len(source.trees) == 1 and len(source.vertices) == 8 and len(source.edges) == 12
        assert source.notice == ""
        hit = source.ray_cast(Vector((0.2, 0.3, 5)), Vector((0, 0, -1)))
        assert hit is not None and close(hit[0], (0.2, 0.3, 1.0))


def test_scene_source_is_in_world_space_and_uses_evaluated_meshes():
    with addon():
        matrix = Matrix.Translation((10, 0, 0)) @ Matrix.Rotation(math.radians(90), 4, 'Z')
        path = edit_path()
        moved = cube_object("Moved", matrix)
        sub = moved.modifiers.new("Sub", 'SUBSURF')
        sub.levels = 1
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        # The default cube at the origin and the moved one; the subdivided one has 26 vertices.
        assert len(source.trees) == 2 and len(source.vertices) == 8 + 26
        xs = [v.x for v in source.vertices[8:]]
        assert 8.9 < min(xs) and max(xs) < 11.1
        co, normal, _ = source.nearest(Vector((10, 0, 5)))
        assert abs(co.x - 10) < 0.2 and normal.z > 0.9
        assert source.reach > 10.0


def test_scene_source_records_the_faces_around_every_snap_vertex_and_edge_in_world_space():
    with addon():
        path = edit_path()
        matrix = (Matrix.Translation((3, 2, 1)) @ Matrix.Rotation(math.radians(30), 4, 'Z')
                  @ Matrix.Diagonal((2, 1, 0.5, 1)))
        box = cube_object("Box", matrix)
        sub = box.modifiers.new("Sub", 'SUBSURF')
        sub.levels = 1
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        assert len(source.vertex_normals) == len(source.vertices) == 8 + 26
        assert len(source.edge_normals) == len(source.edges)
        # The default cube at the origin: three faces per corner, two per edge.
        assert all(len(n) == 3 for n in source.vertex_normals[:8])
        assert all(len(n) == 2 for n in source.edge_normals[:12])
        # The subdivided box (Catmull-Clark of a cube: 8 corner vertices on three quads each, 12 edge
        # and 6 face vertices on four): unit vectors in world space, pointing away from the centre.
        centre = matrix @ Vector((0, 0, 0))
        counts = sorted(len(n) for n in source.vertex_normals[8:])
        assert counts == [3] * 8 + [4] * 18, counts
        for co, normals in zip(source.vertices[8:], source.vertex_normals[8:]):
            assert all(abs(n.length - 1.0) < 1e-5 and n.dot(co - centre) > 0.0 for n in normals)


def test_scene_source_records_a_probe_for_every_face_and_hides_a_buried_one():
    with addon():
        path = edit_path()
        cube_object("Left", Matrix.Translation((-1, 0, 0)))
        cube_object("Right", Matrix.Translation((1, 0, 0)))
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        assert [len(c) for c in source.vertex_probes] == [len(n) for n in source.vertex_normals]
        assert [len(c) for c in source.edge_probes] == [len(n) for n in source.edge_normals]
        # Each target lies in the plane of its face, inside the face's triangle at the vertex (a
        # point 0.9 to 1.5 units from the corner of a 2 unit square), and the size is the edge.
        for co, normals, probes in zip(source.vertices, source.vertex_normals, source.vertex_probes):
            for n, faces in zip(normals, probes):
                assert len(faces) == 1
                probe = faces[0]
                assert abs(n.dot(probe.target - co)) < 1e-5 and 0.9 < (probe.target - co).length < 1.5
                assert abs(probe.size - 2.0) < 1e-5
        for (a, b), normals, probes in zip(source.edges, source.edge_normals, source.edge_probes):
            mid = (source.vertices[a] + source.vertices[b]) * 0.5
            for n, faces in zip(normals, probes):
                probe = faces[0]
                assert abs(n.dot(probe.target - mid)) < 1e-5 and abs(probe.size - 2.0) < 1e-5
        # The cubes touch at x = 0. From +X, at the shared corner (0, -1, 1), the +X face of the
        # left cube is buried; the pick takes a face in view.
        view = View((8.0, -3.0, 2.0))
        x, y = view.screen((0, -1, 1))
        placement = snap3d.place(source, view, (x + 1, y + 1), 12)
        assert placement.kind == snap3d.VERTEX and len(placement.faces) >= 3
        assert tuple(round(c) for c in placement.normal) == (0, -1, 0)
        assert (1, 0, 0) not in {tuple(round(c) for c in n) for n in placement.front}


def pit_object(matrix):
    """The U shaped pit of the snap tests as a real mesh with an 8-gon floor (a concave n-gon, so
    the probes come from the loop triangles)."""
    outline = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    n = len(outline)
    verts = [(x, y, 0.0) for x, y in outline] + [(x, y, 1.0) for x, y in outline]
    polys = [tuple(range(n))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    polys.append((n + 4, n + 5, n + 6, n + 3))
    mesh = bpy.data.meshes.new("Pit")
    mesh.from_pydata(verts, [], polys)
    obj = bpy.data.objects.new("Pit", mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix
    return obj


def test_a_concave_floor_of_a_real_mesh_is_a_front_face_at_its_reflex_corner():
    with addon():
        path = edit_path()
        bpy.data.objects.remove(bpy.data.objects["Cube"])
        matrix = Matrix.Translation((40, -25, 3)) @ Matrix.Rotation(math.radians(20), 4, 'Z')
        pit_object(matrix)
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        corner = matrix @ Vector((1, 1, 0))
        index = next(i for i, v in enumerate(source.vertices) if close(v, corner, 1e-3))
        snap = snap3d.Snap(source.vertices[index], snap3d.VERTEX, None, 0.0, index)
        floor_normal = matrix.to_3x3() @ Vector((0, 0, 1))
        probe = next(p for n, p in zip(source.snap_normals(snap), source.snap_probes(snap)) if close(n, floor_normal))[0]
        # The centre of the 8-gon, (1.5, 1.75), is in the solid column; the target is on the floor.
        local = matrix.inverted() @ probe.target
        in_arm = 0 <= local.x <= 1 or 2 <= local.x <= 3 or (1 <= local.x <= 2 and local.y < 1)
        assert abs(local.z) < 1e-4 and in_arm and abs(probe.size - 1.0) < 1e-4, tuple(local)
        for eye in ((-2, -2, 5), (0.3, 0.3, 5), (-4, -1, 6)):
            view = View(matrix @ Vector(eye), target=corner)
            x, y = view.screen(corner)
            placement = snap3d.place(source, view, (x + 1, y + 1), 12)
            assert placement.kind == snap3d.VERTEX and close(placement.co, corner, 1e-2)
            assert any(close(n, floor_normal, 1e-4) for n in placement.front), (eye, placement.front)
            assert close(placement.normal, floor_normal, 1e-4), eye


def test_clicking_path_vertices_and_closing_loops_never_duplicates_on_a_translated_path():
    # A path object away from the origin holds large local coordinates, so a vertex snapped to a
    # scene vertex comes back from the float32 round trip a little off.
    rotate = Matrix.Rotation(math.radians(30), 4, 'Z') @ Matrix.Diagonal((0.5, 0.5, 0.5, 1))
    corners = [(1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    jitter = [(2, 1), (-3, 2), (1, -4), (5, 5), (-2, -2), (0, 3), (4, -1), (-5, 0)]
    for matrix in (Matrix.Translation((1e3, 5e2, 0)), Matrix.Translation((1e4, 5e3, 0)),
                   Matrix.Translation((1e4, 5e3, 0)) @ rotate):
        with addon():
            obj = edit_path(matrix)
            for k in range(len(jitter)):
                label = (tuple(matrix.translation), k)
                c, *_, view = make_controller(obj)

                def click(controller, corner, offset, kind='LEFTMOUSE'):
                    x, y = view.screen(corner)
                    return controller.handle(ev(kind, px=(x + offset[0], y + offset[1])))

                before = len(bmesh.from_edit_mesh(obj.data).verts)
                for i, corner in enumerate(corners):
                    click(c, corner, jitter[(k + i) % len(jitter)])
                # Closing onto the first vertex adds no vertex and ends the run.
                assert click(c, corners[0], jitter[(k + 3) % len(jitter)]) == 'FINISHED', label
                bm = bmesh.from_edit_mesh(obj.data)
                assert len(bm.verts) - before == (3 if k == 0 else 0), label
                assert len(bm.edges) == 3, label
                # A new run that clicks existing vertices reuses them and adds the one new corner.
                c2, *_ = make_controller(obj, view)
                click(c2, corners[0], jitter[k])
                click(c2, (-1, -1, 1), (2, 1))
                click(c2, corners[1], jitter[k][::-1])
                assert click(c2, corners[1], (0, 0), 'RIGHTMOUSE') == 'FINISHED', label
                bm = bmesh.from_edit_mesh(obj.data)
                assert len(bm.verts) == 4 and len(bm.edges) == 5, (label, len(bm.verts), len(bm.edges))
                # Back to the three vertex loop for the next round.
                for v in [v for v in bm.verts if (obj.matrix_world @ v.co - Vector((-1, -1, 1))).length < 0.05]:
                    bm.verts.remove(v)
                bmesh.update_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                assert len(bm.verts) == 3 and len(bm.edges) == 3, label


def test_scene_source_caps_point_snapping_but_keeps_surfaces_and_says_so():
    with addon():
        path = edit_path()
        cube_object("Second", Matrix.Translation((5, 0, 0)))
        bpy.context.view_layer.update()
        real = draw3d.MAX_SNAP_VERTICES
        draw3d.MAX_SNAP_VERTICES = 10
        try:
            source = draw3d.build_scene_source(bpy.context, path)
        finally:
            draw3d.MAX_SNAP_VERTICES = real
        assert len(source.trees) == 2 and len(source.vertices) == 8
        assert source.notice == "Snap limit: 1 object(s) surface only"
        # The unsnapped cube is still hit by rays.
        assert source.ray_cast(Vector((5, 0, 5)), Vector((0, 0, -1))) is not None
        controller, headers, _, _ = make_controller(path, source=source)
        assert "Snap limit" in headers[-1]
        controller.abort()


# ---------------------------------------------------------------------------
# Drawing through the controller
# ---------------------------------------------------------------------------

def test_points_land_on_the_cube_faces_with_the_face_normals():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        for co in (TOP, SIDE, FRONT):
            assert c.handle(at(view, co)) == 'RUNNING_MODAL'
        # The first click is not written before a second point exists.
        assert c.handle(ev('RIGHTMOUSE', px=view.screen(TOP))) == 'FINISHED'
        points, edges, selected = path_state(obj)
        assert len(points) == 3 and len(edges) == 2 and len(selected) == 3
        by_co = dict(points)
        assert close(by_co[tuple(round(x, 4) for x in TOP)], (0, 0, 1), 1e-4)
        assert close(by_co[tuple(round(x, 4) for x in SIDE)], (1, 0, 0), 1e-4)
        assert close(by_co[tuple(round(x, 4) for x in FRONT)], (0, -1, 0), 1e-4)
        for (co, _) in points:
            assert any(close(co, t, 1e-3) for t in (TOP, SIDE, FRONT))
        assert headers[-1] is None and c.closed


def test_new_vertices_are_sharp_and_unit_normals_are_written():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(ev('RET', px=view.screen(SIDE)))
        bm = bmesh.from_edit_mesh(obj.data)
        rad = bm.verts.layers.float.get("st_fillet_radius")
        ids = bm.verts.layers.int.get("st_fillet_id")
        nor = bm.verts.layers.float_vector.get(resolve3d.PATH_NORMAL_ATTR)
        assert len(bm.verts) == 2
        for v in bm.verts:
            assert v[rad] == 0.0 and v[ids] == 0 and abs(v[nor].length - 1.0) < 1e-5


def test_the_normal_layer_is_created_when_the_path_has_none():
    with addon():
        cube_object()
        mesh = bpy.data.meshes.new("Bare")
        obj = bpy.data.objects.new("Bare", mesh)
        bpy.context.collection.objects.link(obj)
        obj.sketch_tools.is_path_3d = True
        obj.sketch_tools.is_sketch = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(ev('RIGHTMOUSE', px=view.screen(SIDE)))
        assert path_state(obj)[0][0][1] != (0.0, 0.0, 0.0)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert mesh.attributes[resolve3d.PATH_NORMAL_ATTR].data_type == 'FLOAT_VECTOR'


def test_snapping_to_a_cube_vertex_and_an_edge_midpoint():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        corner, mid = (1.0, -1.0, 1.0), (0.0, -1.0, 1.0)
        c.handle(ev('LEFTMOUSE', px=(view.screen(corner)[0] + 5, view.screen(corner)[1] - 3)))
        c.handle(ev('LEFTMOUSE', px=(view.screen(mid)[0] - 2, view.screen(mid)[1] + 2)))
        c.handle(ev('RIGHTMOUSE', px=view.screen(mid)))
        points, edges, _ = path_state(obj)
        assert [close(co, corner, 1e-4) or close(co, mid, 1e-4) for co, _ in points] == [True, True]
        for co, normal in points:
            # From EYE the -Y face is the one that most faces the eye, at the corner and the midpoint.
            assert tuple(round(x) for x in normal) == (0, -1, 0)
        assert len(edges) == 1


def test_a_miss_adds_nothing_and_says_so_in_the_header():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        before = path_state(obj)
        assert c.handle(ev('LEFTMOUSE', px=(5.0, 5.0))) == 'RUNNING_MODAL'
        assert path_state(obj) == before and not c.session.points
        assert draw3d.NO_SURFACE_HINT in headers[-1]
        c.handle(at(view, TOP))
        c.handle(ev('MOUSEMOVE', px=(5.0, 5.0)))
        assert draw3d.NO_SURFACE_HINT in headers[-1]
        assert c.handle(ev('LEFTMOUSE', px=(5.0, 5.0))) == 'RUNNING_MODAL'
        assert len(c.session.points) == 1 and path_state(obj) == before
        # A lone point ends as a cancel and leaves no geometry.
        assert c.handle(ev('RIGHTMOUSE', px=view.screen(TOP))) == 'CANCELLED'
        assert path_state(obj) == before
        # The hint clears once the cursor is over a surface again.
        c2, headers2, _, _ = make_controller(obj)
        c2.handle(ev('MOUSEMOVE', px=(5.0, 5.0)))
        c2.handle(ev('MOUSEMOVE', px=view.screen(TOP)))
        assert draw3d.NO_SURFACE_HINT not in headers2[-1]
        c2.abort()


def test_a_scene_without_meshes_has_its_own_hint():
    with addon():
        mesh = bpy.data.meshes.new("Lonely")
        obj = bpy.data.objects.new("Lonely", mesh)
        bpy.context.collection.objects.link(obj)
        obj.sketch_tools.is_path_3d = True
        obj.sketch_tools.is_sketch = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        c, headers, _, view = make_controller(obj)
        c.handle(ev('MOUSEMOVE', px=(400.0, 300.0)))
        assert draw3d.NO_MESH_HINT in headers[-1]
        c.abort()


def test_escape_removes_the_whole_run_and_keeps_earlier_geometry_and_selection():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(ev('RIGHTMOUSE', px=view.screen(SIDE)))
        before = path_state(obj)
        c2, headers, _, _ = make_controller(obj, view)
        for co in (FRONT, TOP):
            c2.handle(at(view, co))
        c2.handle(at(view, (0.5, 0.5, 1.0)))
        assert path_state(obj) != before
        assert c2.handle(ev('ESC', px=view.screen(TOP))) == 'CANCELLED'
        assert path_state(obj) == before and headers[-1] is None and c2.closed


def test_c_closes_the_chain_onto_its_first_point():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        assert c.handle(ev('C', px=view.screen(SIDE))) == 'RUNNING_MODAL', "two points cannot close"
        c.handle(at(view, FRONT))
        assert c.handle(ev('C', px=view.screen(FRONT))) == 'FINISHED'
        points, edges, _ = path_state(obj)
        assert len(points) == 3 and len(edges) == 3
        bpy.ops.object.mode_set(mode='OBJECT')
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        try:
            resolved = resolve3d.resolve_path_bmesh(bm)
        finally:
            bm.free()
        assert len(resolved.chains) == 1 and resolved.chains[0].closed
        assert len(resolved.chains[0].points) == 3


def test_snapping_back_onto_the_first_point_reuses_it_and_closes():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        first = view.screen(TOP)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(at(view, FRONT))
        # 6 px from the first point: a path point is a snap target, no new vertex is made.
        assert c.handle(ev('LEFTMOUSE', px=(first[0] + 6, first[1] + 2))) == 'FINISHED'
        points, edges, _ = path_state(obj)
        assert len(points) == 3 and len(edges) == 3


def test_a_new_run_can_start_on_an_existing_path_vertex():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(ev('RIGHTMOUSE', px=view.screen(SIDE)))
        c2, *_ = make_controller(obj, view)
        pos = view.screen(SIDE)
        c2.handle(ev('LEFTMOUSE', px=(pos[0] + 3, pos[1] - 3)))
        c2.handle(at(view, FRONT))
        assert c2.handle(ev('RIGHTMOUSE', px=pos)) == 'FINISHED'
        points, edges, _ = path_state(obj)
        assert len(points) == 3 and len(edges) == 2
        # Only what the second run added is selected.
        assert len(path_state(obj)[2]) == 2


def test_shift_locks_the_angle_in_the_face_plane():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, (-0.5, -0.5, 1.0)))
        c.handle(at(view, (0.5, -0.4, 1.0), shift=True))
        c.handle(ev('RIGHTMOUSE', px=view.screen(TOP)))
        found = [co for co in coords(obj)]
        assert any(close(co, (-0.5, -0.5, 1.0)) for co in found)
        # About 5.7 degrees off sketch X locks to 0 degrees, the length is kept.
        assert any(close(co, (0.505, -0.5, 1.0), 2e-3) for co in found), found


def test_ctrl_places_on_the_surface_without_snapping_or_the_angle_lock():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        corner = view.screen((1.0, -1.0, 1.0))
        near = (corner[0] + 5, corner[1] - 3)
        c.handle(at(view, TOP))
        # Without Ctrl the cursor next to the corner snaps to it; with Ctrl it is a plain surface point.
        c.handle(ev('MOUSEMOVE', px=near))
        assert c.session.cursor.kind == snap3d.VERTEX
        c.handle(ev('MOUSEMOVE', px=near, ctrl=True))
        assert c.session.cursor.kind == snap3d.FACE and not c.session.cursor.locked
        assert headers[-1].startswith("Snapping off | ") and "hold Shift or Ctrl" in headers[-1]
        origin, direction = view.ray(near)
        expected = c.source.ray_cast(origin, direction)[0]
        c.handle(ev('LEFTMOUSE', px=near, ctrl=True))
        point = c.session.points[-1].co
        c.handle(ev('RIGHTMOUSE', px=near))
        assert close(point, expected, 1e-4)
        assert not close(point, (1.0, -1.0, 1.0), 1e-2)
        # Pressing and letting go of Ctrl re-evaluates the cursor (the point just placed is a path
        # vertex now, so the snapped cursor may be that one).
        c2, *_ = make_controller(obj)
        c2.handle(ev('MOUSEMOVE', px=near))
        c2.handle(ev('LEFT_CTRL', px=near, ctrl=True))
        assert c2.session.cursor.kind == snap3d.FACE
        c2.handle(ev('LEFT_CTRL', value='RELEASE', px=near))
        assert c2.session.cursor.kind != snap3d.FACE
        c2.abort()


def test_ctrl_and_shift_together_keep_the_angle_lock_and_skip_snapping():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        c.handle(at(view, (0.1, -0.75, 1.0)))
        # Next to the near corner, on the inside of it: without modifiers this snaps to the corner.
        near = view.screen((0.97, -0.96, 1.0))
        c.handle(ev('MOUSEMOVE', px=near))
        assert c.session.cursor.kind == snap3d.VERTEX
        c.handle(ev('MOUSEMOVE', px=near, shift=True, ctrl=True))
        both = c.session.cursor
        assert both.kind == snap3d.FACE and both.locked and both.ref is None
        c.handle(ev('MOUSEMOVE', px=near, shift=True))
        shift_only = c.session.cursor
        assert shift_only.locked and close(shift_only.co, both.co, 1e-5)
        c.handle(ev('MOUSEMOVE', px=near, ctrl=True))
        assert not c.session.cursor.locked and c.session.cursor.kind == snap3d.FACE
        c.handle(ev('LEFTMOUSE', px=near, shift=True, ctrl=True))
        point = c.session.points[-1].co
        c.handle(ev('RIGHTMOUSE', px=near))
        # The point sits on the 0 degree line from the first point, not on the corner.
        assert close(point, both.co, 1e-5) and abs(point.y + 0.75) < 1e-4 and abs(point.z - 1.0) < 1e-4
        assert not close(point, (1.0, -1.0, 1.0), 1e-2)


def lean_view(elevation, azimuth=(0.6, -0.8)):
    e = math.radians(elevation)
    return View(eye=(azimuth[0] * math.cos(e) * 9, azimuth[1] * math.cos(e) * 9, math.sin(e) * 9))


def stored_normals(obj):
    return {co: n for co, n in path_state(obj)[0]}


def test_re_picked_normals_are_written_to_the_run_only_and_escape_removes_them():
    with addon():
        obj = edit_path()
        view = lean_view(20)
        corners = [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]
        # Run A: a vertex on the corner, then a point on the front face - the corner stores the front face.
        c, *_ = make_controller(obj, view)
        c.handle(at(view, corners[0]))
        c.handle(at(view, (0.5, -1.0, 0.0)))
        c.handle(ev('RIGHTMOUSE', px=view.screen((0.5, -1.0, 0.0))))
        after_a = path_state(obj)
        assert stored_normals(obj)[(1.0, -1.0, 1.0)] == (0.0, -1.0, 0.0)

        # Run B starts on that existing vertex and goes round the rim, seen from +X at 20 degrees.
        updates, real = [], bmesh.update_edit_mesh
        bmesh.update_edit_mesh = lambda *a, **k: (updates.append(1), real(*a, **k))[1]
        try:
            c2, *_ = make_controller(obj, view)
            for co in corners:
                before = len(updates)
                c2.handle(at(view, co))
                assert len(updates) - before <= 1, "one edit mesh update per click, re-picks included"
            assert c2.handle(ev('ESC', px=view.screen(corners[0]))) == 'CANCELLED'
        finally:
            bmesh.update_edit_mesh = real
        # Escape after re-picks leaves the mesh exactly as it was before the run.
        assert path_state(obj) == after_a

        c3, *_ = make_controller(obj, view)
        for co in corners:
            c3.handle(at(view, co))
        assert c3.handle(ev('RIGHTMOUSE', px=view.screen(corners[3]))) == 'FINISHED'
        normals = stored_normals(obj)
        # The vertex the run reused keeps the normal run A gave it; the three the run created are
        # all on the top face (the first two were first picked on the +X face, the most in view).
        assert normals[(1.0, -1.0, 1.0)] == (0.0, -1.0, 0.0)
        for co in corners[1:]:
            assert normals[co] == (0.0, 0.0, 1.0), (co, normals)


def test_a_rim_loop_through_the_controller_stores_the_top_normal_leaning_either_way():
    for azimuth in ((0.6, -0.8), (-0.6, -0.8)):
        for elevation in (20, 10):
            with addon():
                obj = edit_path()
                view = lean_view(elevation, azimuth)
                c, *_ = make_controller(obj, view)
                corners = [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]
                for co in corners:
                    c.handle(at(view, co))
                c.handle(ev('RIGHTMOUSE', px=view.screen(corners[3])))
                normals = stored_normals(obj)
                assert len(normals) == 4 and all(n == (0.0, 0.0, 1.0) for n in normals.values()), (
                    azimuth, elevation, normals)


def test_the_tool_description_says_what_ctrl_does_for_planar_and_3d_paths():
    from sketch_tools.ui import tool
    text = tool.SKETCH_TOOL_sketch.bl_description
    assert "hold Ctrl to flip it while held" in text and "Ctrl toggles snapping" not in text
    assert "Snapping follows the snap magnet (Shift+Tab)" in text and "Ctrl turns snapping off" not in text
    assert ("Shift or Ctrl turn snapping off (Shift also locks the angle in a polyline)" in text
            and "Shift locks the angle in the face plane" not in text)
    assert "on a planar sketch" in text and "While drawing" in text
    labels = []
    layout = types.SimpleNamespace(label=lambda text: labels.append(text), prop=lambda *a, **k: None)
    context = types.SimpleNamespace(window_manager=types.SimpleNamespace())
    tool.SKETCH_TOOL_sketch.draw_settings(context, layout, None)
    assert len(labels) == 1 and "hold Ctrl to flip snapping to points" in labels[0] and "Ctrl snapping off" not in labels[0]
    assert "while drawing also P polyline, A arc" in labels[0]
    # On a 3D path R and O only switch shapes inside a run, so the hint says so and points at the header.
    labels.clear()
    with addon():
        obj = edit_path()
        context = types.SimpleNamespace(window_manager=types.SimpleNamespace(), active_object=obj)
        tool.SKETCH_TOOL_sketch.draw_settings(context, layout, None)
    assert len(labels) == 1 and "Header picks the shape" in labels[0] and "While drawing" in labels[0]
    assert "Shift or Ctrl turn snapping off (Shift also locks the angle in a polyline)" in labels[0]
    assert "Ctrl snapping off" not in labels[0] and "flip snapping" not in labels[0]


def test_the_stored_normal_faces_the_viewer_on_a_back_face_and_follows_continuity_at_a_tie():
    with addon():
        obj = edit_path()
        # Inside the cube the top face is seen from behind: the stored normal points down.
        inside = View(eye=(0.1, 0.1, 0.0), target=(0.0, 0.0, 1.0))
        c, *_ = make_controller(obj, inside)
        c.handle(ev('LEFTMOUSE', px=(400.0, 300.0)))
        c.handle(ev('LEFTMOUSE', px=(430.0, 300.0)))
        c.handle(ev('RIGHTMOUSE', px=(430.0, 300.0)))
        normals = [n for _, n in path_state(obj)[0]]
        assert len(normals) == 2 and all(close(n, (0, 0, -1), 1e-3) for n in normals), normals

    # A view that sees the top-front edge at 45 degrees from both faces: the second point keeps the
    # normal of the first, whichever face that was.
    for first, expected in ((TOP, (0, 0, 1)), (FRONT, (0, -1, 0))):
        with addon():
            obj = edit_path()
            view = View(eye=(0.0, -9.0, 9.0))
            c, *_ = make_controller(obj, view)
            c.handle(at(view, first))
            c.handle(at(view, (0.0, -1.0, 1.0)))
            assert close(c.session.points[1].normal, expected, 1e-3), (first, c.session.points[1].normal)
            c.abort()


def test_a_path_with_a_transform_stores_local_positions_and_normals():
    with addon():
        matrix = Matrix.Translation((5, 2, -1)) @ Matrix.Rotation(math.radians(90), 4, 'Z')
        obj = edit_path(matrix)
        c, *_, view = make_controller(obj)
        c.handle(at(view, SIDE))
        c.handle(at(view, TOP))
        c.handle(ev('RIGHTMOUSE', px=view.screen(TOP)))
        inverse = matrix.inverted()
        points = dict(path_state(obj)[0])
        local_side = tuple(round(x, 4) for x in inverse @ Vector(SIDE))
        assert close(next(co for co in points if close(co, local_side, 2e-3)), local_side, 2e-3)
        # The +X face normal in world space is -Y in the path's local space (rotated 90 about Z).
        normal = next(n for co, n in points.items() if close(co, local_side, 2e-3))
        assert close(normal, (0, -1, 0), 1e-3)
        top_local = next(n for co, n in points.items() if not close(co, local_side, 2e-3))
        assert close(top_local, (0, 0, 1), 1e-3)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_navigation_system_events_and_numpad_pass_through_other_keys_are_swallowed():
    with addon():
        obj = edit_path()
        c, *_, view = make_controller(obj)
        c.handle(at(view, TOP))
        before = path_state(obj)
        for etype in ('MIDDLEMOUSE', 'WHEELUPMOUSE', 'TRACKPADPAN', 'NDOF_MOTION', 'TIMER', 'WINDOW_DEACTIVATE',
                      'NUMPAD_4', 'NUMPAD_PERIOD'):
            assert c.handle(ev(etype)) == 'PASS_THROUGH', etype
        for etype in ('ONE', 'A', 'TAB', 'R', 'O', 'N', 'SPACE', 'BACK_SPACE', 'PERIOD'):
            assert c.handle(ev(etype)) == 'RUNNING_MODAL', etype
        assert c.handle(ev('C', ctrl=True)) == 'RUNNING_MODAL' and not c.session.done
        assert path_state(obj) == before
        c.abort()


def test_pointer_events_outside_the_region_or_over_overlays_pass_through_and_add_nothing():
    with addon():
        obj = edit_path()
        view = View(overlays=[(_WINDOW_OFFSET[0] + 700.0, _WINDOW_OFFSET[1], 100.0, 600.0)])
        c, *_ = make_controller(obj, view)
        c.handle(at(view, TOP))
        before = path_state(obj)
        for px in ((900.0, 300.0), (-4.0, 10.0), (750.0, 200.0)):
            assert c.handle(ev('LEFTMOUSE', 'PRESS', px)) == 'PASS_THROUGH', px
            assert c.handle(ev('LEFTMOUSE', 'RELEASE', px)) == 'PASS_THROUGH'
            assert c.handle(ev('LEFTMOUSE', 'CLICK', px)) == 'PASS_THROUGH'
            assert c.handle(ev('RIGHTMOUSE', 'PRESS', px)) == 'PASS_THROUGH'
            assert c.handle(ev('RIGHTMOUSE', 'RELEASE', px)) == 'PASS_THROUGH'
            assert c.handle(ev('MOUSEMOVE', 'NOTHING', px)) == 'PASS_THROUGH'
        assert not c.closed and len(c.session.points) == 1 and path_state(obj) == before
        # A press that went through is the last one: the next inside press places again.
        assert c.handle(at(view, SIDE)) == 'RUNNING_MODAL' and len(c.session.points) == 2
        assert c.handle(ev('LEFTMOUSE', 'RELEASE', view.screen(SIDE))) == 'RUNNING_MODAL'
        # Alt+LMB navigates, with its release.
        assert c.handle(ev('LEFTMOUSE', 'PRESS', view.screen(FRONT), alt=True)) == 'PASS_THROUGH'
        assert c.handle(ev('LEFTMOUSE', 'RELEASE', view.screen(FRONT))) == 'PASS_THROUGH'
        assert len(c.session.points) == 2
        c.abort()


def test_edit_mode_left_or_view_lost_ends_the_run_and_drops_uncommitted_points():
    with addon():
        for how in ('mode', 'view'):
            obj = edit_path()
            c, headers, _, view = make_controller(obj)
            before = path_state(obj)
            c.handle(at(view, TOP))
            c.handle(at(view, SIDE))
            assert path_state(obj) != before
            if how == 'view':
                c._projector_factory = lambda: None
                assert c.handle(ev('MOUSEMOVE', px=(1.0, 1.0))) == 'CANCELLED'
                assert path_state(obj) == before
            else:
                bpy.ops.object.mode_set(mode='OBJECT')
                assert c.handle(ev('MOUSEMOVE')) == 'CANCELLED'
            assert c.closed and headers[-1] is None and not draw3d._live
            assert c.snapshot() is None
            if how == 'view':
                bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_the_scene_source_is_dropped_on_every_exit_path():
    with addon():
        for how in ('finish', 'esc', 'abort'):
            obj = edit_path()
            c, *_, view = make_controller(obj)
            assert not c.source.empty
            c.handle(at(view, TOP))
            c.handle(at(view, SIDE))
            if how == 'finish':
                c.handle(ev('RIGHTMOUSE', px=view.screen(SIDE)))
            elif how == 'esc':
                c.handle(ev('ESC'))
            else:
                c.abort()
            assert c.source.empty and not c.source.vertices, how
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def test_the_overlay_projects_the_3d_preview_and_draws_only_in_its_region():
    with addon():
        obj = edit_path()
        c, _, _, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(at(view, SIDE))
        c.handle(ev('MOUSEMOVE', px=view.screen(FRONT)))
        preview, mouse, to_screen = c.snapshot()
        geo = overlay.screen_geometry(preview, to_screen)
        assert len(geo["band"]) == 2 + 2, "chain, rubber band and two point markers"
        assert geo["band"][0][0] == tuple(view.screen(TOP)) or close(geo["band"][0][0], view.screen(TOP), 1e-3)
        assert len(geo["snap"]) == 1 and preview.lines == ["On face"]
        # A point behind the camera drops its strip.
        assert overlay.screen_geometry(preview, lambda co: None)["band"] == []
        handler = overlay.DrawOverlay(c.snapshot, region_pointer=5)
        assert handler._region_pointer == 5
        c.abort()


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

def _gl_perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    return Matrix(((f / aspect, 0, 0, 0), (0, f, 0, 0),
                   (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)), (0, 0, -1, 0)))


def _fake_rv3d(eye, target, perspective=True):
    forward = (Vector(target) - Vector(eye)).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward)
    rotation = Matrix((right, up, -forward))
    view = Matrix.Translation(Vector(eye)) @ rotation.transposed().to_4x4()
    view = view.inverted()
    if perspective:
        proj = _gl_perspective(60.0, 800 / 600, 0.1, 1000.0)
    else:
        proj = Matrix(((1 / 4.0, 0, 0, 0), (0, 1 / 3.0, 0, 0), (0, 0, -0.001, 0), (0, 0, 0, 1)))
    return types.SimpleNamespace(perspective_matrix=proj @ view, view_matrix=view, is_perspective=perspective)


def test_the_projector_maps_world_points_to_pixels_and_builds_sight_lines():
    region = types.SimpleNamespace(width=800, height=600)
    rv3d = _fake_rv3d((0, -10, 0), (0, 0, 0))
    p = draw3d.PathProjector(region, rv3d, reach=5.0)
    x, y, w = p.project((0, 0, 0))
    assert abs(x - 400) < 1e-3 and abs(y - 300) < 1e-3 and abs(w - 10) < 1e-4
    right = p.project((1, 0, 0))
    up = p.project((0, 0, 1))
    assert right[0] > 400 and abs(right[1] - 300) < 1e-3 and up[1] > 300 and abs(up[0] - 400) < 1e-3
    assert p.project((0, -20, 0)) is None, "behind the camera"
    assert p.to_screen((0, 0, 0)) == (x, y)
    origin, direction, distance = p.sight(Vector((0, 0, 0)))
    assert close(origin, (0, -10, 0)) and close(direction, (0, 1, 0), 1e-5) and abs(distance - 10) < 1e-4
    other = draw3d.PathProjector(region, _fake_rv3d((0, -10, 1), (0, 0, 0)), reach=5.0)
    assert other.key != p.key

    ortho = draw3d.PathProjector(region, _fake_rv3d((0, -10, 0), (0, 0, 0), perspective=False), reach=5.0)
    x, y, w = ortho.project((1, 7, 0))
    assert abs(w - 1.0) < 1e-6 and abs(x - 500) < 1e-3 and abs(y - 300) < 1e-3
    origin, direction, distance = ortho.sight(Vector((1, 7, 0)))
    assert close(direction, (0, 1, 0), 1e-5)
    # Starts outside everything within reach: behind the farthest scene point on the view side.
    assert abs((Vector((1, 7, 0)) - origin).dot(direction) - distance) < 1e-4 and origin.y < -5.0
    assert ortho.contains((10, 10), (10, 10)) and not ortho.contains((900, 10), (900, 10))


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def operator_shell(obj, view=None, source=None):
    """A stand-in `self` and context for calling SKETCH_OT_draw_3d.invoke / modal directly, with
    the projector injected (and the scene source when given)."""
    view = view or View()
    headers, handlers = [], []
    area = fake_area(type='VIEW_3D', header_text_set=headers.append, tag_redraw=lambda: None)
    wm = types.SimpleNamespace(modal_handler_add=handlers.append)
    context = types.SimpleNamespace(
        active_object=obj, area=area, window_manager=wm, scene=bpy.context.scene,
        preferences=bpy.context.preferences, region=types.SimpleNamespace(as_pointer=lambda: 4242),
        evaluated_depsgraph_get=bpy.context.evaluated_depsgraph_get, view_layer=bpy.context.view_layer)
    shell = types.SimpleNamespace(reports=[])
    shell.report = lambda kind, text: shell.reports.append((kind, text))
    real = (draw3d.projector_factory, draw3d.build_scene_source)
    draw3d.projector_factory = lambda o, s: (lambda: view)
    if source is not None:
        draw3d.build_scene_source = lambda ctx, o: source
    restore_area = install_fake_area(area)

    def restore():
        draw3d.projector_factory, draw3d.build_scene_source = real
        restore_area()
    return shell, context, headers, handlers, restore, view


def test_the_operator_registers_with_an_undo_step_and_unregisters():
    with addon():
        assert hasattr(bpy.types, "SKETCH_OT_draw_3d") and hasattr(bpy.ops.sketch, "draw_3d")
        assert 'UNDO' in draw3d.SKETCH_OT_draw_3d.bl_options
        assert draw3d.SKETCH_OT_draw_3d.bl_idname == "sketch.draw_3d"
    assert not hasattr(bpy.types, "SKETCH_OT_draw_3d")


def _poll_context(mode, area_type, active):
    area = None if area_type is None else types.SimpleNamespace(type=area_type)
    return types.SimpleNamespace(mode=mode, area=area, active_object=active)


def test_draw_3d_polls_only_for_a_3d_path_and_draw_refuses_it():
    with addon():
        path = edit_path()
        planar = make_sketch([SQUARE])
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.collection.objects.link(plain)
        empty = bpy.data.objects.new("Empty", None)
        poll3, poll2 = draw3d.SKETCH_OT_draw_3d.poll, draw_tool.SKETCH_OT_draw.poll
        assert poll3(_poll_context('EDIT_MESH', 'VIEW_3D', path))
        assert not poll2(_poll_context('EDIT_MESH', 'VIEW_3D', path))
        assert poll2(_poll_context('EDIT_MESH', 'VIEW_3D', planar))
        assert not poll3(_poll_context('EDIT_MESH', 'VIEW_3D', planar))
        for ctx in (_poll_context('EDIT_MESH', 'VIEW_3D', plain), _poll_context('EDIT_MESH', 'VIEW_3D', empty),
                    _poll_context('EDIT_MESH', 'VIEW_3D', None), _poll_context('OBJECT', 'VIEW_3D', path),
                    _poll_context('EDIT_MESH', 'PROPERTIES', path), _poll_context('EDIT_MESH', None, path)):
            assert not poll3(ctx)
        bpy.data.objects.remove(empty)


def test_the_tool_keymap_runs_draw_3d_on_the_same_click_as_draw():
    from sketch_tools.ui import tool

    with addon():
        keymap = tool.SKETCH_TOOL_sketch.bl_keymap
        lmb = {item[0]: item for item in keymap if item[1]["type"] == 'LEFTMOUSE'}
        assert set(lmb) == {"sketch.draw", "sketch.draw_3d"}
        assert lmb["sketch.draw_3d"][1] == lmb["sketch.draw"][1]
        assert not any(i[0] == "sketch.draw_3d" for i in keymap if i[1]["type"] in ('R', 'O'))
        found = []
        for kc in bpy.context.window_manager.keyconfigs:
            for km in kc.keymaps:
                if km.name == "3D View Tool: Edit Mesh, Sketch":
                    found.extend((kmi.idname, kmi.type) for kmi in km.keymap_items)
        assert ("sketch.draw", 'LEFTMOUSE') in found and ("sketch.draw_3d", 'LEFTMOUSE') in found


def test_operator_paths_clean_up_after_themselves():
    with addon():
        for how in ('finish', 'esc', 'error', 'cancel_callback', 'close_all'):
            obj = edit_path()
            before = path_state(obj)
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            op = draw3d.SKETCH_OT_draw_3d
            try:
                assert op.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
                assert handlers == [shell] and len(overlay._live) == 1 and headers[-1] is not None
                assert modal_common.any_sketch_modal_live()
                assert op.modal(shell, ctx, at(view, SIDE)) == {'RUNNING_MODAL'}
                if how == 'finish':
                    assert op.modal(shell, ctx, ev('RIGHTMOUSE')) == {'FINISHED'}
                elif how == 'esc':
                    assert op.modal(shell, ctx, ev('ESC')) == {'CANCELLED'}
                elif how == 'error':
                    def broken(event):
                        raise ZeroDivisionError("boom")
                    shell._controller.session.handle = broken
                    with redirect_stdout(io.StringIO()):
                        assert op.modal(shell, ctx, at(view, FRONT)) == {'CANCELLED'}
                    assert shell.reports and shell.reports[0][0] == {'ERROR'}
                elif how == 'cancel_callback':
                    op.cancel(shell, ctx)
                else:
                    draw3d.close_all()
            finally:
                restore()
            assert not overlay._live and headers[-1] is None and not draw3d._live, how
            assert not modal_common.any_sketch_modal_live(), how
            if how == 'finish':
                assert len(path_state(obj)[0]) == 2
            else:
                assert path_state(obj) == before, how
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_the_starting_click_that_misses_does_not_start_a_run():
    with addon():
        obj = edit_path()
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        try:
            result = draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, ev('LEFTMOUSE', px=(5.0, 5.0)))
        finally:
            restore()
        assert result == {'CANCELLED'} and not handlers and not overlay._live
        assert shell.reports == [({'WARNING'}, draw3d.NO_SURFACE_HINT)]
        assert headers[-1] is None and not draw3d._live and not path_state(obj)[0]


def test_the_starting_click_over_an_overlay_passes_through_without_building_a_scene():
    with addon():
        obj = edit_path()
        view = View(overlays=[(_WINDOW_OFFSET[0] + 700.0, _WINDOW_OFFSET[1], 100.0, 600.0)])
        shell, ctx, headers, handlers, restore, _ = operator_shell(obj, view)
        built = []
        draw3d.build_scene_source = lambda c, o: built.append(1)
        try:
            result = draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, ev('LEFTMOUSE', px=(750.0, 100.0)))
        finally:
            restore()
        assert result == {'PASS_THROUGH'} and not built and not handlers and not draw3d._live


def test_a_second_invoke_passes_through_while_any_sketch_modal_is_live_in_both_directions():
    with addon():
        obj = edit_path()
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        second, ctx2, _, handlers2, restore2, _ = operator_shell(obj, view)
        try:
            op = draw3d.SKETCH_OT_draw_3d
            assert op.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
            assert op.invoke(second, ctx2, at(view, SIDE)) == {'PASS_THROUGH'}
            assert not handlers2 and second.reports == [] and len(draw3d._live) == 1
            # The planar tool refuses too while the 3D run is live (it returns before reading anything).
            assert draw_tool.SKETCH_OT_draw.invoke(types.SimpleNamespace(), None, None) == {'PASS_THROUGH'}
            assert op.modal(shell, ctx, ev('ESC')) == {'CANCELLED'}
            assert op.invoke(second, ctx2, at(view, SIDE)) == {'RUNNING_MODAL'}
            op.cancel(second, ctx2)
        finally:
            restore()
            restore2()
        assert not draw3d._live and not overlay._live


def test_a_running_planar_draw_blocks_draw_3d():
    with addon():
        obj = edit_path()
        controller = draw_tool.DrawController.__new__(draw_tool.DrawController)
        draw_tool._live.add(controller)
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        try:
            assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'PASS_THROUGH'}
        finally:
            restore()
            draw_tool._live.discard(controller)
        assert not handlers and not draw3d._live


def test_the_operator_never_calls_into_an_area_that_was_closed_mid_run():
    with addon():
        for key in ('ESC', 'RET'):
            obj = edit_path()
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            closed, late = [], []
            ctx.area.header_text_set = lambda text: late.append(text) if closed else headers.append(text)
            ctx.area.tag_redraw = lambda: late.append("redraw") if closed else None
            try:
                op = draw3d.SKETCH_OT_draw_3d
                assert op.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
                closed.append(True)
                modal_common.find_area = lambda pointer: None
                assert op.modal(shell, ctx, ev(key)) in ({'CANCELLED'}, {'FINISHED'})
            finally:
                restore()
            assert not late, (key, late)
            assert not draw3d._live and not overlay._live
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_operator_start_failures_clean_up():
    with addon():
        obj = edit_path()
        # The scene cannot be built.
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        def broken(context, o):
            raise RuntimeError("no depsgraph")
        draw3d.build_scene_source = broken
        try:
            with redirect_stdout(io.StringIO()):
                assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'CANCELLED'}
        finally:
            restore()
        assert shell.reports[0][0] == {'ERROR'} and not draw3d._live and not overlay._live
        # The modal handler cannot be added.
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        def refuse(op):
            raise RuntimeError("no window")
        ctx.window_manager.modal_handler_add = refuse
        try:
            with redirect_stdout(io.StringIO()):
                assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'CANCELLED'}
        finally:
            restore()
        assert shell.reports[0][0] == {'ERROR'} and headers[-1] is None
        assert not draw3d._live and not overlay._live and not path_state(obj)[0]


def test_unregistering_the_addon_winds_down_a_running_run():
    restore = None
    try:
        with addon():
            obj = edit_path()
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
            draw3d.SKETCH_OT_draw_3d.modal(shell, ctx, at(view, SIDE))
            assert draw3d._live and overlay._live
            # The add-on goes away mid-run, still in edit mode: the run is rolled back.
        assert not draw3d._live and not overlay._live and not modal_common.any_sketch_modal_live()
        assert headers[-1] is None
    finally:
        if restore is not None:
            restore()


def box_geometry(lo, hi):
    xs = list(zip(lo, hi))
    verts = [(xs[0][i], xs[1][j], xs[2][k]) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return verts, [], [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


def test_a_buried_top_with_a_vertex_on_its_edge_is_hidden_by_a_flush_box():
    with addon():
        path = edit_path()
        bpy.data.objects.remove(bpy.data.objects["Cube"])
        # A prism whose top (0,0) (1,0) (2,0) (1,0.5) has (1,0) on the side from (0,0) to (2,0), so
        # Blender's loop triangulation of the top contains a flat triangle. A box sits flush on the
        # top, covering the vertex K = (1, 0, 1).
        outline = [(0, 0), (1, 0), (2, 0), (1, 0.5)]
        n = len(outline)
        verts = [(x, y, 0.0) for x, y in outline] + [(x, y, 1.0) for x, y in outline]
        polys = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
        polys += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        prism = bpy.data.meshes.new("Prism")
        prism.from_pydata(verts, [], polys)
        prism.update()
        flat = 0
        for tri in prism.loop_triangles:
            a, b, c = (Vector(prism.vertices[i].co) for i in tri.vertices)
            flat += (b - a).cross(c - a).length < 1e-6
        assert flat == 1, "the repro needs Blender's flat loop triangle"
        bpy.context.collection.objects.link(bpy.data.objects.new("Prism", prism))
        box = bpy.data.meshes.new("Box")
        box.from_pydata(*box_geometry((0.6, 0.0, 1.0), (1.4, 0.3, 1.5)))
        box.update()
        bpy.context.collection.objects.link(bpy.data.objects.new("Box", box))
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, path)
        corner = Vector((1, 0, 1))
        index = next(i for i, v in enumerate(source.vertices)
                     if close(v, corner, 1e-5) and len(source.vertex_normals[i]) == 2)
        snap = snap3d.Snap(source.vertices[index], snap3d.VERTEX, None, 0.0, index)
        normals = source.snap_normals(snap)
        up = next(i for i, normal in enumerate(normals) if close(normal, (0, 0, 1)))
        # The top's targets are inside the top, off its boundary, not at the corner.
        for probe in source.snap_probes(snap)[up]:
            assert abs(probe.target.z - 1.0) < 1e-5 and probe.target.y > 0.05, probe
            assert (probe.target - corner).length > 0.05
        for eye in ((1, -3, 6), (1.5, -4, 7), (1, -6, 4)):
            for ortho in (False, True):
                view = View(eye, target=corner, ortho=ortho, reach=source.reach)
                placement = snap3d.snap_placement(source, view, snap)
                assert [tuple(round(c) for c in v) for v in placement.front] == [(0, -1, 0)], (eye, ortho)
                assert tuple(round(c) for c in placement.normal) == (0, -1, 0), (eye, ortho)
