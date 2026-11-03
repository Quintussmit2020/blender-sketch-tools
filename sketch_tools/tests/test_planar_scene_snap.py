"""Scene snapping in the planar draw tool (DECISIONS C22): Ctrl also snaps to the visible vertices,
edge midpoints and edge points of other mesh objects, projected onto the sketch plane.

Controller and operator shell with a fake camera and a fake plane view, as in the Phase 3 and
Phase 7 tests. The viewport itself (real regions, the overlay) is checked by hand."""

import io
import math
import types
from contextlib import redirect_stdout

import bmesh
import bpy
from mathutils import Matrix, Vector

from sketch_tools.core import draw_logic, snap3d
from sketch_tools.core.draw_logic import DrawSession
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.ops import draw3d, draw_tool, modal_common
from sketch_tools.ops.draw_tool import DrawController, SketchEditor
from sketch_tools.tests.test_addon import (
    SQUARE, _edit_sketch, _operator_shell, _WINDOW_OFFSET, addon, make_sketch,
)
from sketch_tools.tests.test_draw3d import View, cube_object, ev
from sketch_tools.ui import overlay

EYE = (4.0, -5.0, 6.0)
# Corners of the default cube (side 2, centred on the origin) as seen from EYE: the top face and the
# +X and -Y sides are in view, (-1, 1, -1) is behind the cube.
A = (1.0, -1.0, 1.0)
B = (1.0, 1.0, 1.0)
C = (-1.0, -1.0, 1.0)
D = (-1.0, 1.0, 1.0)
HIDDEN = (-1.0, 1.0, -1.0)


class PlaneView:
    """The projector the planar controller wants (contains, ray_to_local, miss, to_screen(xy)) over
    a fake camera and the sketch's matrix."""

    def __init__(self, camera, matrix):
        self.camera = camera
        self.world = matrix.copy()
        self.inverse = matrix.inverted_safe()
        self.miss = None

    def contains(self, mouse, window):
        return self.camera.contains(mouse, window)

    def ray_to_local(self, mouse):
        origin, direction = self.camera.ray(mouse)
        xy, self.miss = draw_logic.ray_plane_hit(
            origin, direction, self.inverse, allow_behind=self.camera.ortho)
        return xy

    def to_screen(self, xy):
        return self.camera.screen(self.world @ Vector((xy[0], xy[1], 0.0)))


class Scene:
    def __init__(self, obj, camera, source, view):
        self.obj, self.camera, self.source, self.view = obj, camera, source, view
        self.headers = []

    def px(self, co):
        return self.camera.screen(Vector(co))

    def local(self, co):
        """Where an orthogonal projection of a world point onto the sketch plane lands, computed
        independently of the tool: local x and y of the point."""
        local = self.obj.matrix_world.inverted() @ Vector(co)
        return (local.x, local.y)

    def controller(self, mode='POLYLINE', grid=0.1, segments=32, snap_px=12, scene=True, snap_toggle=None):
        session = DrawSession(mode, grid=grid, circle_segments=segments)
        if scene and self.source.empty:
            # A controller owns its scene and drops it on close; a test that runs several needs a
            # new one for each, as every invoke builds its own.
            self.source = draw3d.build_scene_source(bpy.context, self.obj, normals=False)
        return DrawController(
            SketchEditor(self.obj), session, lambda: self.view, snap_px,
            header=self.headers.append, redraw=lambda: None,
            scene=self.source if scene else None,
            scene_view_factory=(lambda: self.camera) if scene else None, snap_toggle=snap_toggle)

    def click(self, controller, co=None, px=None, first=False, **mods):
        event = ev('LEFTMOUSE', 'PRESS', px or self.px(co), **mods)
        return controller.start(event) if first else controller.handle(event)

    def move(self, controller, co=None, px=None, **mods):
        return controller.handle(ev('MOUSEMOVE', 'NOTHING', px or self.px(co), **mods))


def setup(matrix=None, eye=EYE, ortho=False, cube_matrix=None, loops=()):
    obj = _edit_sketch(loops)
    if matrix is not None:
        obj.matrix_world = matrix
    cube_object("Cube", cube_matrix)
    bpy.context.view_layer.update()
    source = draw3d.build_scene_source(bpy.context, obj, normals=False)
    camera = View(eye=eye, ortho=ortho, reach=source.reach)
    return Scene(obj, camera, source, PlaneView(camera, obj.matrix_world))


def local_verts(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return [(round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 6)) for v in bm.verts]


def edge_count(obj):
    return len(bmesh.from_edit_mesh(obj.data).edges)


def near(a, b, eps=1e-3):
    return len(a) == len(b) and all(abs(x - y) < eps for x, y in zip(a, b))


def points_near(actual, expected, eps=1e-3):
    return len(actual) == len(expected) and all(near(a[:2], e) for a, e in zip(actual, expected))


# ---------------------------------------------------------------------------
# The kinds of snap
# ---------------------------------------------------------------------------

def test_ctrl_on_a_cube_vertex_gives_the_vertex_projected_onto_the_plane():
    with addon():
        s = setup()
        c = s.controller()
        assert s.click(c, A, first=True, ctrl=True) == 'RUNNING_MODAL'
        assert s.click(c, D, ctrl=True) == 'RUNNING_MODAL'
        assert c.handle(ev('RIGHTMOUSE', px=s.px(D))) == 'FINISHED'
        # The cube corners are at z = 1: the new vertices lie on the plane below them.
        assert local_verts(s.obj) == [(1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)]
        assert edge_count(s.obj) == 1
        bm = bmesh.from_edit_mesh(s.obj.data)
        rad, ids = bm.verts.layers.float[FILLET_RADIUS_ATTR], bm.verts.layers.int[FILLET_ID_ATTR]
        assert all(v[rad] == 0.0 and v[ids] == 0 for v in bm.verts)


def test_ctrl_on_an_edge_midpoint_snaps_to_the_projected_midpoint():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, (1.0, 0.0, 1.0), first=True, ctrl=True)
        s.click(c, (0.0, 1.0, 1.0), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        assert points_near(local_verts(s.obj), [(1.0, 0.0), (0.0, 1.0)])


def test_ctrl_near_an_edge_snaps_to_the_nearest_point_on_it():
    with addon():
        s = setup()
        c = s.controller()
        # A quarter and three quarters along the top edges, 50 px from the vertices and midpoints.
        s.click(c, (1.0, -0.5, 1.0), first=True, ctrl=True)
        s.click(c, (-0.5, 1.0, 1.0), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        assert points_near(local_verts(s.obj), [(1.0, -0.5), (-0.5, 1.0)])


def test_a_vertex_in_reach_beats_the_edge_points_beside_it():
    with addon():
        s = setup()
        # Nudge the mouse 5 px off a vertex: the vertex wins over the edge points beside it.
        px = s.px(A)
        c = s.controller()
        s.click(c, px=(px[0] + 3.0, px[1] + 4.0), first=True, ctrl=True)
        s.click(c, D, ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        assert points_near(local_verts(s.obj), [(1.0, -1.0), (-1.0, 1.0)])

def test_the_kind_follows_the_priority():
    with addon():
        s = setup()
        kinds = []
        for co in (A, (1.0, 0.0, 1.0), (1.0, -0.5, 1.0)):
            c = s.controller()
            s.move(c, co, ctrl=True)
            kinds.append(c.session.mods.snap.kind)
            c.abort()
        assert kinds == ["VERTEX", "MIDPOINT", "EDGE"]


def test_an_occluded_vertex_is_not_snapped():
    with addon():
        s = setup()
        assert snap3d.find_snap(s.source, s.camera, s.px(HIDDEN), 12) is None
        c = s.controller()
        px = s.px(HIDDEN)
        s.click(c, px=px, first=True, ctrl=True)
        s.click(c, D, ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        first = local_verts(s.obj)[0]
        # The grid point under the cursor, not the hidden corner's projection (-1, 1).
        expected = draw_logic.grid_snap_point(s.view.ray_to_local(px), 0.1)
        assert near(first[:2], expected, 1e-4) and not near(first[:2], (-1.0, 1.0), 0.05)


def test_the_sketchs_own_vertex_beats_a_scene_vertex_at_the_same_pixel():
    with addon():
        s = setup()
        # A sketch vertex that shows exactly where the cube corner A does.
        under = s.view.ray_to_local(s.px(A))
        bm = bmesh.from_edit_mesh(s.obj.data)
        bm.verts.new((under[0], under[1], 0.0))
        bmesh.update_edit_mesh(s.obj.data)
        assert not near(under, (1.0, -1.0), 0.1)
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.click(c, D, ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        # Start reused the existing vertex: one old vertex plus one new, none at the projection of A.
        assert len(verts) == 2 and near(verts[0][:2], under, 1e-4)
        assert not any(near(v[:2], (1.0, -1.0), 0.05) for v in verts)


def test_the_grid_is_the_fallback_when_nothing_is_near():
    with addon():
        s = setup()
        px = (60.0, 540.0)
        assert snap3d.find_snap(s.source, s.camera, px, 12) is None
        c = s.controller()
        s.click(c, px=px, first=True, ctrl=True)
        s.click(c, px=(80.0, 500.0), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        expected = [draw_logic.grid_snap_point(s.view.ray_to_local(p), 0.1) for p in (px, (80.0, 500.0))]
        assert points_near(local_verts(s.obj), expected, 1e-4)


def test_nothing_snaps_without_ctrl():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, A, first=True)
        s.click(c, D)
        c.handle(ev('RIGHTMOUSE'))
        expected = [s.view.ray_to_local(s.px(co)) for co in (A, D)]
        assert points_near(local_verts(s.obj), expected, 1e-4)
        assert not near(local_verts(s.obj)[0][:2], (1.0, -1.0), 0.05)


def test_the_threshold_is_the_snap_pixel_preference():
    with addon():
        s = setup()
        px = s.px(A)
        offset = (px[0] + 20.0, px[1])
        for snap_px, snapped in ((12, False), (30, True)):
            c = s.controller(snap_px=snap_px)
            s.move(c, px=offset, ctrl=True)
            assert (c.session.mods.snap is not None) == snapped, snap_px
            c.abort()


def test_shift_turns_scene_snapping_off_and_keeps_the_angle_lock():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.click(c, D, ctrl=True, shift=True)
        c.handle(ev('RIGHTMOUSE'))
        first, second = local_verts(s.obj)
        assert near(first[:2], (1.0, -1.0))
        # Not the scene vertex's projection: a multiple of 45 degrees from sketch X, length rounded
        # to the grid.
        assert not near(second[:2], (-1.0, 1.0), 0.05)
        dx, dy = second[0] - first[0], second[1] - first[1]
        assert abs(math.remainder(math.atan2(dy, dx), math.pi / 4)) < 1e-3
        assert abs(math.hypot(dx, dy) / 0.1 - round(math.hypot(dx, dy) / 0.1)) < 1e-2


def test_shift_ctrl_still_snaps_to_the_sketchs_own_vertex():
    with addon():
        s = setup(loops=[SQUARE])
        c = s.controller()
        s.click(c, first=True, px=s.view.to_screen((0.0, 0.0)), ctrl=True)
        s.move(c, px=s.view.to_screen((1.0, 1.0)), ctrl=True, shift=True)
        assert c.session.mods.snap is not None and c.session.mods.snap.ref is not None
        c.abort()


# ---------------------------------------------------------------------------
# Sketch planes
# ---------------------------------------------------------------------------

def test_an_xz_plane_projects_into_local_coordinates():
    with addon():
        matrix = Matrix.Rotation(math.radians(90.0), 4, 'X')
        s = setup(matrix, eye=(3.0, -8.0, 3.0))
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.click(c, D, ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        # World (x, y, z) lies at local (x, z) on a plane that is the world XZ plane; local z is 0.
        assert points_near(verts, [(1.0, 1.0), (-1.0, 1.0)]) and all(v[2] == 0.0 for v in verts)
        world = [matrix @ Vector(v) for v in verts]
        assert all(abs(w.y) < 1e-4 for w in world)


def test_a_tilted_face_plane_projects_orthogonally_into_local_coordinates():
    with addon():
        matrix = (Matrix.Translation((0.5, 0.2, -0.3)) @ Matrix.Rotation(0.5, 4, 'X')
                  @ Matrix.Rotation(0.7, 4, 'Z'))
        s = setup(matrix, eye=(2.0, -6.0, 5.0))
        normal = matrix.to_3x3() @ Vector((0.0, 0.0, 1.0))
        c = s.controller()
        corners = (A, D)
        s.click(c, corners[0], first=True, ctrl=True)
        s.click(c, corners[1], ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        assert points_near(verts, [s.local(co) for co in corners], 1e-3)
        for vert, co in zip(verts, corners):
            on_plane = matrix @ Vector(vert)
            # The vertex lies on the plane and the line back to the cube corner is the plane normal.
            assert abs(vert[2]) < 1e-9
            assert (Vector(co) - on_plane).cross(normal).length < 1e-3


def test_a_scaled_and_moved_object_snaps_by_its_world_vertices():
    with addon():
        cube_matrix = Matrix.Translation((2.0, 1.0, 0.0)) @ Matrix.Diagonal((2.0, 1.0, 1.0, 1.0))
        s = setup(cube_matrix=cube_matrix)
        world = cube_matrix @ Vector((1.0, -1.0, 1.0))
        c = s.controller()
        s.click(c, world, first=True, ctrl=True)
        s.click(c, cube_matrix @ Vector((-1.0, 1.0, 1.0)), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        assert points_near(local_verts(s.obj), [(world.x, world.y), (0.0, 2.0)])


def test_an_orthographic_view_snaps_the_same_way():
    with addon():
        s = setup(eye=(5.0, -5.0, 8.0), ortho=True)
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.click(c, (0.0, 1.0, 1.0), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        assert points_near(local_verts(s.obj), [(1.0, -1.0), (0.0, 1.0)])


# ---------------------------------------------------------------------------
# Every shape
# ---------------------------------------------------------------------------

def test_a_rectangle_takes_its_corners_from_the_scene():
    with addon():
        s = setup()
        c = s.controller('RECTANGLE')
        assert s.click(c, A, first=True, ctrl=True) == 'RUNNING_MODAL'
        assert s.click(c, D, ctrl=True) == 'FINISHED'
        verts = local_verts(s.obj)
        assert points_near(verts, [(1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)])
        assert edge_count(s.obj) == 4


def test_a_circle_takes_its_centre_and_radius_point_from_the_scene():
    with addon():
        s = setup()
        c = s.controller('CIRCLE', segments=24)
        s.click(c, A, first=True, ctrl=True)
        assert s.click(c, B, ctrl=True) == 'FINISHED'
        verts = local_verts(s.obj)
        assert len(verts) == 24 and edge_count(s.obj) == 24
        # Centre at the projection of A, radius to the projection of B (2 units). The centre is not
        # a vertex.
        assert all(abs(math.hypot(v[0] - 1.0, v[1] + 1.0) - 2.0) < 1e-3 for v in verts)


def test_a_three_point_arc_takes_all_three_points_from_the_scene():
    with addon():
        s = setup()
        c = s.controller('ARC_3PT')
        s.click(c, A, first=True, ctrl=True)
        assert s.click(c, (0.0, 1.0, 1.0), ctrl=True) == 'RUNNING_MODAL'
        assert s.click(c, C, ctrl=True) == 'FINISHED'
        verts = local_verts(s.obj)
        assert near(verts[0][:2], (1.0, -1.0)) and near(verts[-1][:2], (-1.0, -1.0))
        assert edge_count(s.obj) == len(verts) - 1
        # The circle through (1, -1), (0, 1) and (-1, -1): centre (0, -0.25), radius 1.25.
        assert all(abs(math.hypot(v[0], v[1] + 0.25) - 1.25) < 1e-3 for v in verts)


def test_a_centre_arc_takes_the_centre_and_start_from_the_scene_and_only_a_direction_for_the_end():
    with addon():
        s = setup()
        c = s.controller('ARC_CENTER')
        s.click(c, A, first=True, ctrl=True)
        assert s.click(c, B, ctrl=True) == 'RUNNING_MODAL'
        # D projects to (-1, 1): 135 degrees from the centre (1, -1), 2.83 away. The arc keeps the
        # radius of 2 and sweeps from 90 to 135 degrees.
        assert s.click(c, D, ctrl=True) == 'FINISHED'
        verts = local_verts(s.obj)
        assert near(verts[0][:2], (1.0, 1.0))
        end = (1.0 + 2.0 * math.cos(math.radians(135)), -1.0 + 2.0 * math.sin(math.radians(135)))
        assert near(verts[-1][:2], end)
        assert not any(near(v[:2], (-1.0, 1.0), 0.05) for v in verts)
        assert all(abs(math.hypot(v[0] - 1.0, v[1] + 1.0) - 2.0) < 1e-3 for v in verts)


def test_a_polyline_can_chain_several_scene_points_and_close_by_c():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.click(c, B, ctrl=True)
        s.click(c, D, ctrl=True)
        assert c.handle(ev('C')) == 'FINISHED'
        assert points_near(local_verts(s.obj), [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        assert edge_count(s.obj) == 3


def test_a_sketch_vertex_is_reused_after_a_scene_point_in_the_same_run():
    with addon():
        s = setup(loops=[SQUARE])
        c = s.controller()
        # A cube corner far from the square on screen, then the square's own corner.
        s.click(c, A, first=True, ctrl=True)
        s.click(c, px=s.view.to_screen((1.0, 1.0)), ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        assert len(verts) == 5 and near(verts[4][:2], (1.0, -1.0))
        # The second click reused vertex 2 of the square: its edge runs from the new vertex to it.
        bm = bmesh.from_edit_mesh(s.obj.data)
        assert any({v.index for v in e.verts} == {2, 4} for e in bm.edges)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def test_the_preview_names_the_scene_snap_kind_and_marks_the_projected_point():
    with addon():
        s = setup()
        c = s.controller()
        expected = {A: ("Scene vertex", (1.0, -1.0)), (1.0, 0.0, 1.0): ("Scene edge midpoint", (1.0, 0.0)),
                    (1.0, -0.5, 1.0): ("Scene point on edge", (1.0, -0.5))}
        for co, (label, xy) in expected.items():
            s.move(c, co, ctrl=True)
            preview = c.session.preview()
            assert preview.lines[0] == label and near(preview.snap_xy, xy)
            geometry = overlay.screen_geometry(preview, s.view.to_screen)
            assert len(geometry["snap"]) == 1
        # No label without Ctrl, and none for a sketch vertex.
        s.move(c, A)
        assert not c.session.preview().lines
        c.abort()


def test_the_label_comes_before_the_readout_of_a_run():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, A, first=True, ctrl=True)
        s.move(c, D, ctrl=True)
        lines = c.session.preview().lines
        assert lines[0] == "Scene vertex" and lines[1].startswith("Length")
        c.abort()


def test_the_kinds_are_the_snap3d_kinds_and_labels_are_ascii():
    assert set(draw_logic.SCENE_SNAP_LABELS) == {snap3d.VERTEX, snap3d.MIDPOINT, snap3d.EDGE}
    assert all(label.isascii() for label in draw_logic.SCENE_SNAP_LABELS.values())


def test_a_scene_snap_session_event_places_a_free_point_and_a_sketch_snap_reuses_the_vertex():
    session = DrawSession()
    scene_snap = draw_logic.Snap(None, (2.0, 3.0), "VERTEX")
    session.handle(("CLICK", (2.1, 3.1), draw_logic.Mods(False, True, scene_snap)))
    commands = session.handle(("CLICK", (5.0, 3.0), draw_logic.Mods(False, False, None)))
    adds = [cmd for cmd in commands if cmd[0] == "ADD_VERT"]
    assert [cmd[2] for cmd in adds] == [(2.0, 3.0), (5.0, 3.0)]
    assert all(cmd[1] < 0 for cmd in adds)
    own = DrawSession()
    own.handle(("CLICK", (0.0, 0.0), draw_logic.Mods(False, True, draw_logic.Snap(7, (0.0, 0.0)))))
    commands = own.handle(("CLICK", (4.0, 0.0), draw_logic.Mods(False, False, None)))
    assert commands == [("ADD_VERT", -1, (4.0, 0.0)), ("ADD_EDGE", 7, -1)]


# ---------------------------------------------------------------------------
# The scene source
# ---------------------------------------------------------------------------

def test_the_planar_scene_has_the_points_but_no_face_normals():
    with addon():
        s = setup()
        full = draw3d.build_scene_source(bpy.context, s.obj)
        assert len(s.source.vertices) == len(full.vertices) == 8
        assert len(s.source.edges) == len(full.edges) == 12
        assert all(len(n) == 3 for n in full.vertex_normals)
        assert all(len(n) == 0 for n in s.source.vertex_normals + s.source.edge_normals)
        assert s.source.notice == ""


def test_sketches_outputs_hidden_objects_and_faceless_meshes_are_not_snap_targets():
    with addon():
        s = setup()
        other = make_sketch([[(5, 5), (7, 5), (7, 7), (5, 7)]], mode='EXTRUDE')
        assert other.sketch_tools.output_object is not None
        hidden = cube_object("Hidden")
        hidden.hide_viewport = True
        wire = bpy.data.objects.new("Wire", bpy.data.meshes.new("Wire"))
        bpy.context.collection.objects.link(wire)
        wire.data.from_pydata([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
        bpy.context.view_layer.update()
        source = draw3d.build_scene_source(bpy.context, s.obj, normals=False)
        assert len(source.trees) == 1 and len(source.vertices) == 8


def test_the_point_caps_and_their_notice_apply():
    with addon():
        s = setup()
        real = draw3d.MAX_SNAP_VERTICES
        draw3d.MAX_SNAP_VERTICES = 4
        try:
            capped = draw3d.build_scene_source(bpy.context, s.obj, normals=False)
        finally:
            draw3d.MAX_SNAP_VERTICES = real
        assert capped.vertices == [] and not capped.empty
        # No point snaps at all here (the object still occludes), so it must not say "surface only".
        assert capped.notice == "Snap limit: 1 object(s) not snapped"
        s.source = capped
        c = s.controller()
        assert capped.notice in s.headers[-1]
        s.move(c, A, ctrl=True)
        assert c.session.mods.snap is None
        c.abort()


# ---------------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------------

def assert_dropped(source):
    assert source.empty and not source.vertices and not source.edges


def test_the_scene_source_is_dropped_on_every_exit_path_of_the_controller():
    with addon():
        for how in ('finish', 'esc', 'abort', 'close_all', 'lost_view', 'edit_mode_left'):
            s = setup()
            c = s.controller()
            s.click(c, A, first=True, ctrl=True)
            s.click(c, D, ctrl=True)
            assert not s.source.empty
            if how == 'finish':
                assert c.handle(ev('RIGHTMOUSE')) == 'FINISHED'
            elif how == 'esc':
                assert c.handle(ev('ESC')) == 'CANCELLED'
            elif how == 'abort':
                c.abort()
            elif how == 'close_all':
                draw_tool.close_all()
            elif how == 'lost_view':
                c._projector_factory = lambda: None
                assert c.handle(ev('MOUSEMOVE', 'NOTHING')) == 'CANCELLED'
            else:
                bpy.ops.object.mode_set(mode='OBJECT')
                assert c.handle(ev('MOUSEMOVE', 'NOTHING')) == 'CANCELLED'
            assert c.closed and not draw_tool._live, how
            assert_dropped(s.source)
            if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for obj in list(bpy.data.objects):
                bpy.data.objects.remove(obj)
            for mesh in list(bpy.data.meshes):
                bpy.data.meshes.remove(mesh)


def test_a_cancelled_scene_snapped_run_leaves_the_mesh_alone():
    with addon():
        s = setup(loops=[SQUARE])
        before = local_verts(s.obj)
        c = s.controller('CIRCLE')
        s.click(c, A, first=True, ctrl=True)
        s.move(c, B, ctrl=True)
        assert c.handle(ev('ESC')) == 'CANCELLED'
        assert local_verts(s.obj) == before and edge_count(s.obj) == 4


def snap_operator(s, mode='POLYLINE'):
    """The operator shell over the fake camera: the real invoke builds the scene from the Blender
    scene and the patched factory supplies the view."""
    shell, ctx, headers, handlers, restore = _operator_shell(s.obj, mode, s.view)
    real_factory = draw3d.projector_factory
    draw3d.projector_factory = lambda obj, source: (lambda: s.camera)
    sources = []
    real_build = draw3d.build_scene_source

    def spy(context, obj, normals=True):
        source = real_build(context, obj, normals)
        sources.append(source)
        return source
    draw3d.build_scene_source = spy

    def undo():
        draw3d.projector_factory, draw3d.build_scene_source = real_factory, real_build
        restore()
    return shell, ctx, headers, handlers, sources, undo


def test_the_operator_builds_the_scene_at_invoke_and_snaps_with_it():
    with addon():
        s = setup()
        shell, ctx, headers, handlers, sources, undo = snap_operator(s)
        try:
            op = draw_tool.SKETCH_OT_draw
            assert op.invoke(shell, ctx, ev('LEFTMOUSE', px=s.px(A), ctrl=True)) == {'RUNNING_MODAL'}
            assert len(sources) == 1 and len(sources[0].vertices) == 8
            assert op.modal(shell, ctx, ev('LEFTMOUSE', px=s.px(D), ctrl=True)) == {'RUNNING_MODAL'}
            assert op.modal(shell, ctx, ev('RIGHTMOUSE')) == {'FINISHED'}
        finally:
            undo()
        assert points_near(local_verts(s.obj), [(1.0, -1.0), (-1.0, 1.0)])
        assert_dropped(sources[0]) and not overlay._live and headers[-1] is None


def test_the_operator_drops_the_scene_on_every_exit_path():
    with addon():
        for how in ('finish', 'esc', 'error', 'cancel_callback', 'close_all', 'start_failure'):
            s = setup()
            shell, ctx, headers, handlers, sources, undo = snap_operator(s)
            op = draw_tool.SKETCH_OT_draw
            real_start = DrawController.start
            try:
                if how == 'start_failure':
                    def boom(self, event):
                        raise ZeroDivisionError("boom")
                    DrawController.start = boom
                    with redirect_stdout(io.StringIO()):
                        assert op.invoke(shell, ctx, ev('LEFTMOUSE', px=s.px(A), ctrl=True)) == {'CANCELLED'}
                else:
                    assert op.invoke(shell, ctx, ev('LEFTMOUSE', px=s.px(A), ctrl=True)) == {'RUNNING_MODAL'}
                    op.modal(shell, ctx, ev('LEFTMOUSE', px=s.px(D), ctrl=True))
                    if how == 'finish':
                        assert op.modal(shell, ctx, ev('RIGHTMOUSE')) == {'FINISHED'}
                    elif how == 'esc':
                        assert op.modal(shell, ctx, ev('ESC')) == {'CANCELLED'}
                    elif how == 'error':
                        def broken(event):
                            raise ZeroDivisionError("boom")
                        shell._controller.session.handle = broken
                        with redirect_stdout(io.StringIO()):
                            assert op.modal(shell, ctx, ev('LEFTMOUSE', px=s.px(C), ctrl=True)) == {'CANCELLED'}
                    elif how == 'cancel_callback':
                        op.cancel(shell, ctx)
                    else:
                        draw_tool.close_all()
            finally:
                DrawController.start = real_start
                undo()
            assert len(sources) == 1, how
            assert_dropped(sources[0])
            assert not draw_tool._live and not overlay._live and not modal_common.any_sketch_modal_live(), how
            bpy.ops.object.mode_set(mode='OBJECT')
            for obj in list(bpy.data.objects):
                bpy.data.objects.remove(obj)
            for mesh in list(bpy.data.meshes):
                bpy.data.meshes.remove(mesh)


def test_a_start_over_an_overlay_builds_no_scene():
    with addon():
        s = setup()
        s.camera.overlays = [(_WINDOW_OFFSET[0] + 700.0, _WINDOW_OFFSET[1], 100.0, 600.0)]
        shell, ctx, headers, handlers, sources, undo = snap_operator(s)
        try:
            result = draw_tool.SKETCH_OT_draw.invoke(shell, ctx, ev('LEFTMOUSE', px=(750.0, 100.0)))
        finally:
            undo()
        assert result == {'PASS_THROUGH'} and not sources and not draw_tool._live


def test_a_scene_that_cannot_be_read_only_switches_scene_snapping_off():
    with addon():
        s = setup()
        shell, ctx, headers, handlers, sources, undo = snap_operator(s)
        draw3d.build_scene_source = lambda context, obj, normals=True: 1 / 0
        try:
            op = draw_tool.SKETCH_OT_draw
            with redirect_stdout(io.StringIO()):
                assert op.invoke(shell, ctx, ev('LEFTMOUSE', px=s.px(A), ctrl=True)) == {'RUNNING_MODAL'}
            assert shell.reports and shell.reports[0][0] == {'WARNING'}
            op.modal(shell, ctx, ev('LEFTMOUSE', px=s.px(D), ctrl=True))
            assert op.modal(shell, ctx, ev('RIGHTMOUSE')) == {'FINISHED'}
        finally:
            undo()
        # Grid points under the cursor, not the cube corners' projections.
        assert not any(near(v[:2], (1.0, -1.0), 0.05) for v in local_verts(s.obj))
        assert len(local_verts(s.obj)) == 2 and not draw_tool._live


def test_the_controller_without_a_scene_behaves_as_in_phase_3():
    with addon():
        s = setup(loops=[SQUARE])
        c = s.controller(scene=False)
        s.move(c, A, ctrl=True)
        assert c.session.mods.snap is None
        c.abort()
        c = s.controller(scene=False)
        s.move(c, px=s.view.to_screen((1.0, 1.0)), ctrl=True)
        assert c.session.mods.snap is not None and c.session.mods.snap.ref == 2
        c.abort()
