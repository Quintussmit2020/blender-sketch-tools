"""Phase 8 part B: rectangle, circle and arcs on 3D paths (DECISIONS C21), with the fake view and
the fake events of test_draw3d. The viewport itself (real regions, the header buttons) is checked by
hand."""

import math
import types

import bmesh
import bpy
from mathutils import Matrix, Vector

from sketch_tools.core import build, resolve3d, snap3d
from sketch_tools.ops import draw3d, modal_common
from sketch_tools.tests.test_shapes_addon import draw_sketch_panel
from sketch_tools.tests.test_addon import addon, assert_closed_and_consistent, output_bm, signed_volume
from sketch_tools.tests.test_draw3d import (
    FRONT, SIDE, TOP, View, _fake_rv3d, at, edit_path, ev, operator_shell, path_state,
)
from sketch_tools.ui import overlay

RECTANGLE, CIRCLE = snap3d.RECTANGLE, snap3d.CIRCLE
ARC_3PT, ARC_CENTER, POLYLINE = snap3d.ARC_3PT, snap3d.ARC_CENTER, snap3d.POLYLINE
UP = Vector((0, 0, 1))
# Corners of the default cube, named front/back, top/bottom, left/right.
CORNERS = {"fbl": (-1, -1, -1), "fbr": (1, -1, -1), "ftl": (-1, -1, 1), "ftr": (1, -1, 1),
           "btr": (1, 1, 1), "btl": (-1, 1, 1)}


def shape_controller(obj, shape, view=None, segments=16, snap_px=12):
    view = view or View()
    source = draw3d.build_scene_source(bpy.context, obj)
    headers = []
    controller = draw3d.PathController(
        draw3d.PathEditor(obj), snap3d.PathSession(shape, segments), lambda: view, source, snap_px,
        header=headers.append, redraw=lambda: None)
    return controller, headers, view


def chains(obj):
    return resolve3d.resolve_path_bmesh(bmesh.from_edit_mesh(obj.data)).chains


def only_chain(obj):
    found = chains(obj)
    assert len(found) == 1, len(found)
    return found[0]


def draw(obj, shape, clicks, view=None, segments=16):
    """A run through the controller: one left press per world point; the last one must finish."""
    c, _, view = shape_controller(obj, shape, view, segments)
    for co in clicks[:-1]:
        assert c.handle(at(view, co)) == 'RUNNING_MODAL'
    assert c.handle(at(view, clicks[-1])) == 'FINISHED'
    assert c.closed and not draw3d._live
    return only_chain(obj)


def polyline(obj, clicks, view=None):
    c, _, view = shape_controller(obj, POLYLINE, view)
    for co in clicks:
        assert c.handle(at(view, co)) == 'RUNNING_MODAL'
    assert c.handle(ev('RIGHTMOUSE')) == 'FINISHED'
    return only_chain(obj)


def rotated(offset, degrees):
    """offset turned counter-clockwise about +Z."""
    return Matrix.Rotation(math.radians(degrees), 3, 'Z') @ Vector(offset)


def plane_axes(view, normal):
    """The plane's x and y axes the way the tool defines them: the view's right vector projected."""
    x = view.right - normal * view.right.dot(normal)
    x.normalize()
    return x, normal.cross(x)


def near(a, b, eps=2e-3):
    return (Vector(a) - Vector(b)).length < eps


def contains_point(points, wanted, eps=2e-3):
    return any(near(p, wanted, eps) for p in points)


def vertex_count(obj):
    return len(bmesh.from_edit_mesh(obj.data).verts)


def edge_count(obj):
    return len(bmesh.from_edit_mesh(obj.data).edges)


def sweep(obj):
    """Leave edit mode and rebuild: the output as a bmesh and the path's warnings."""
    bpy.ops.object.mode_set(mode='OBJECT')
    build.rebuild(obj)
    return output_bm(obj), obj.sketch_tools.warnings


# ---------------------------------------------------------------------------
# Plane helpers
# ---------------------------------------------------------------------------

def test_plane_basis_is_right_handed_and_uses_the_right_vector_projected():
    for normal in (UP, Vector((0, -1, 0)), Vector((1, 2, 3)).normalized(), Vector((-3, 1, -1)).normalized()):
        right = Vector((0.6, 0.8, 0.0))
        x, y = snap3d.plane_basis(normal, right)
        n = normal.normalized()
        assert abs(x.length - 1) < 1e-6 and abs(y.length - 1) < 1e-6
        assert abs(x.dot(n)) < 1e-6 and abs(y.dot(n)) < 1e-6 and (x.cross(y) - n).length < 1e-6
        projected = right - n * right.dot(n)
        assert (x - projected.normalized()).length < 1e-6
    # A right vector along the normal has no projection: world X, or Y when that is along it too.
    assert near(snap3d.plane_basis(UP, UP)[0], (1, 0, 0), 1e-6)
    assert near(snap3d.plane_basis(UP, None)[0], (1, 0, 0), 1e-6)
    x, y = snap3d.plane_basis(Vector((1, 0, 0)), None)
    assert near(x, (0, 1, 0), 1e-6) and near(y, (0, 0, 1), 1e-6)


def test_a_ray_meets_a_plane_unless_parallel_grazing_or_behind():
    origin, normal = Vector((0, 0, 1)), UP
    hit = snap3d.ray_plane_point(Vector((2, 3, 5)), Vector((0, 0, -2)), origin, normal)
    assert near(hit, (2, 3, 1), 1e-6)
    assert snap3d.ray_plane_point(Vector((0, 0, 5)), Vector((1, 0, 0)), origin, normal) is None
    assert snap3d.ray_plane_point(Vector((0, 0, 5)), Vector((1, 0, -1e-4)), origin, normal) is None
    assert snap3d.ray_plane_point(Vector((0, 0, 5)), Vector((0, 0, 1)), origin, normal) is None


# ---------------------------------------------------------------------------
# Rectangle and circle
# ---------------------------------------------------------------------------

def test_a_rectangle_started_on_the_top_face_lies_in_the_top_plane():
    with addon():
        obj = edit_path()
        c, _, view = shape_controller(obj, RECTANGLE)
        assert c.handle(at(view, TOP)) == 'RUNNING_MODAL'
        assert not path_state(obj)[0], "nothing is written before the shape is complete"
        target = Vector((0.6, 0.5, 1.0))
        assert c.handle(at(view, target)) == 'FINISHED'
        chain = only_chain(obj)
        assert chain.closed and len(chain.points) == 4 and edge_count(obj) == 4
        assert all(abs(p.z - 1.0) < 1e-4 for p in chain.points)
        assert all(near(n, (0, 0, 1), 1e-5) for n in chain.normals)
        # Corners are the first point, the clicked one and the two between, along the axes of the
        # plane (the view's right vector projected into it).
        x, y = plane_axes(view, UP)
        origin = Vector(TOP)
        u, v = (target - origin).dot(x), (target - origin).dot(y)
        for corner in (origin, origin + x * u, target, origin + y * v):
            assert contains_point(chain.points, corner), corner
        assert abs(x.z) < 1e-6, "the right vector of a camera without roll is horizontal"


def test_a_rectangle_started_on_the_front_face_lies_in_the_front_plane():
    with addon():
        obj = edit_path()
        target = Vector((-0.5, -1.0, 0.5))
        chain = draw(obj, RECTANGLE, [FRONT, target])
        assert chain.closed and len(chain.points) == 4
        assert all(abs(p.y + 1.0) < 1e-4 for p in chain.points)
        assert all(near(n, (0, -1, 0), 1e-5) for n in chain.normals)
        assert contains_point(chain.points, FRONT) and contains_point(chain.points, target)


def test_a_circle_on_the_top_face_has_the_preference_segments_around_the_first_point():
    with addon():
        obj = edit_path()
        chain = draw(obj, CIRCLE, [TOP, (0.7, 0.3, 1.0)], segments=12)
        assert chain.closed and len(chain.points) == 12 and edge_count(obj) == 12 and vertex_count(obj) == 12
        centre = Vector(TOP)
        assert all(abs((p - centre).length - 0.5) < 1e-3 and abs(p.z - 1.0) < 1e-4 for p in chain.points)
        assert all(near(n, (0, 0, 1), 1e-5) for n in chain.normals)
        # The centre is not a vertex.
        assert not contains_point(chain.points, centre, 0.1)


def test_a_circle_on_the_front_face_lies_in_the_front_plane_and_may_leave_the_face():
    with addon():
        obj = edit_path()
        # A radius of 2 reaches far past the 2 unit face: the shape is in the plane, not on the cube.
        chain = draw(obj, CIRCLE, [FRONT, (-1.7, -1.0, -0.2)], segments=8)
        assert len(chain.points) == 8
        assert all(abs(p.y + 1.0) < 1e-4 and abs((p - Vector(FRONT)).length - 2.0) < 1e-3 for p in chain.points)
        assert all(near(n, (0, -1, 0), 1e-5) for n in chain.normals)


def test_a_later_click_whose_ray_hits_another_face_still_lands_in_the_first_plane():
    with addon():
        obj = edit_path()
        c, _, view = shape_controller(obj, RECTANGLE)
        c.handle(at(view, TOP))
        # The pixel shows the +X face; the ray passes the z = 1 plane in front of it, off the cube.
        assert c.handle(at(view, SIDE)) == 'FINISHED'
        chain = only_chain(obj)
        assert all(abs(p.z - 1.0) < 1e-4 for p in chain.points)
        far = max(chain.points, key=lambda p: (p - Vector(TOP)).length)
        assert far.x > 1.5 and far.y < -1.0, tuple(far)
        assert all(near(n, (0, 0, 1), 1e-5) for n in chain.normals)


def test_the_shape_plane_keeps_the_axis_of_the_view_at_the_first_click():
    with addon():
        obj = edit_path()
        holder = [View()]
        source = draw3d.build_scene_source(bpy.context, obj)
        c = draw3d.PathController(draw3d.PathEditor(obj), snap3d.PathSession(RECTANGLE), lambda: holder[0],
                                  source, 12, header=lambda t: None, redraw=lambda: None)
        first = holder[0]
        c.handle(at(first, TOP))
        holder[0] = View((-6.0, -5.0, 4.0))
        target = Vector((0.4, 0.5, 1.0))
        assert c.handle(at(holder[0], target)) == 'FINISHED'
        x, y = plane_axes(first, UP)
        chain = only_chain(obj)
        origin = Vector(TOP)
        for corner in (origin + x * (target - origin).dot(x), origin + y * (target - origin).dot(y)):
            assert contains_point(chain.points, corner), corner


def test_snapping_in_a_shape_plane_projects_the_snap_point():
    with addon():
        # The bottom corner snaps and lands straight above, in the top plane.
        obj = edit_path()
        chain = draw(obj, RECTANGLE, [TOP, (1.0, -1.0, -1.0)])
        assert contains_point(chain.points, (1.0, -1.0, 1.0))


def test_ctrl_in_a_shape_plane_turns_snapping_off():
    with addon():
        corner = Vector((1.0, -1.0, -1.0))
        # The point where the ray through that pixel meets the plane.
        obj = edit_path()
        c, _, view = shape_controller(obj, RECTANGLE)
        c.handle(at(view, TOP))
        assert c.handle(at(view, corner, ctrl=True)) == 'FINISHED'
        origin, direction = view.ray(view.screen(corner))
        expected = snap3d.ray_plane_point(origin, direction, Vector(TOP), UP)
        assert contains_point(only_chain(obj).points, expected) and expected.x > 2.0


def test_a_zero_size_shape_is_not_placed():
    with addon():
        obj = edit_path()
        for shape in (RECTANGLE, CIRCLE):
            c, _, view = shape_controller(obj, shape)
            c.handle(at(view, TOP))
            assert c.handle(at(view, TOP)) == 'RUNNING_MODAL' and not c.session.done
            c.abort()
        assert not path_state(obj)[0]


def test_a_rectangle_or_arc_can_start_on_an_existing_path_vertex_and_reuses_it():
    with addon():
        obj = edit_path()
        # Two path vertices from a polyline run.
        first, second = Vector(TOP), Vector((0.6, -0.4, 1.0))
        polyline(obj, [first, second])
        assert vertex_count(obj) == 2 and edge_count(obj) == 1

        c, _, view = shape_controller(obj, RECTANGLE)
        c.handle(at(view, first))
        assert c.session.points[0].ref >= 0, "the click reuses the path vertex"
        c.handle(at(view, (0.0, 0.7, 1.0)))
        assert vertex_count(obj) == 2 + 3 and edge_count(obj) == 1 + 4
        bm = bmesh.from_edit_mesh(obj.data)
        degree = {tuple(round(c, 4) for c in v.co): len(v.link_edges) for v in bm.verts}
        assert degree[tuple(round(c, 4) for c in first)] == 3

        # An arc from one path vertex to another through a surface point: both are reused.
        before_v, before_e = vertex_count(obj), edge_count(obj)
        c, _, view = shape_controller(obj, ARC_3PT)
        for co in (second, (-0.4, -0.5, 1.0), first):
            c.handle(at(view, co))
        assert c.session.done
        arc_points = len(c.editor.bm.verts) - before_v + 2
        assert arc_points >= 4
        assert vertex_count(obj) == before_v + arc_points - 2 and edge_count(obj) == before_e + arc_points - 1


# ---------------------------------------------------------------------------
# Arcs
# ---------------------------------------------------------------------------

def test_a_centre_arc_runs_counter_clockwise_in_the_plane_from_start_to_end():
    with addon():
        obj = edit_path()
        centre, start = Vector((0.0, 0.0, 1.0)), Vector((0.5, 0.0, 1.0))
        end = centre + rotated(start - centre, 80)
        chain = draw(obj, ARC_CENTER, [centre, start, end])
        points = chain.points
        # 80 degrees at 16 segments per circle: ceil(3.56) = 4 segments, 5 points, an open chain.
        assert not chain.closed and len(points) == 5 and edge_count(obj) == 4
        assert near(points[0], start) and near(points[-1], end)
        assert all(abs(p.z - 1.0) < 1e-4 and abs((p - centre).length - 0.5) < 1e-3 for p in points)
        assert all(near(n, (0, 0, 1), 1e-5) for n in chain.normals)
        # Counter-clockwise seen against the normal: every step turns about +Z.
        assert all((a - centre).cross(b - centre).z > 0.0 for a, b in zip(points, points[1:]))


def test_a_centre_arc_to_an_end_clockwise_of_the_start_goes_the_long_way_round():
    with addon():
        obj = edit_path()
        centre, start = Vector((0.0, 0.0, 1.0)), Vector((0.5, 0.0, 1.0))
        end = centre + rotated(start - centre, -100)
        chain = draw(obj, ARC_CENTER, [centre, start, end], segments=16)
        # 260 degrees: ceil(16 * 0.72) = 12 segments. It passes the far side of the circle.
        assert len(chain.points) == 13 and not chain.closed
        assert contains_point(chain.points, centre - (start - centre), 0.1)
        assert all((a - centre).cross(b - centre).z > 0.0 for a, b in zip(chain.points, chain.points[1:]))


def test_a_centre_arc_without_an_angle_waits_for_an_end_point():
    with addon():
        obj = edit_path()
        centre, start = Vector((0.0, 0.0, 1.0)), Vector((0.5, 0.0, 1.0))
        c, headers, view = shape_controller(obj, ARC_CENTER)
        c.handle(at(view, centre))
        c.handle(at(view, start))
        assert c.handle(at(view, start)) == 'RUNNING_MODAL'
        assert "no angle" in headers[-1] and not path_state(obj)[0]
        assert c.handle(at(view, centre + rotated(start - centre, 80))) == 'FINISHED'
        assert len(only_chain(obj).points) == 5


def circumcentre(a, b, c):
    ab, ac = b - a, c - a
    n = ab.cross(ac)
    return a + (n.cross(ab) * ac.length_squared + ac.cross(n) * ab.length_squared) / (2.0 * n.length_squared)


def test_a_three_point_arc_through_three_snapped_points_on_one_face():
    with addon():
        obj = edit_path()
        p0, p1, p2 = Vector((1, -1, 1)), Vector((1, 1, 1)), Vector((-1, 1, 1))
        c, _, view = shape_controller(obj, ARC_3PT)
        c.handle(at(view, p0))
        c.handle(at(view, p1))
        assert c.handle(at(view, p2)) == 'FINISHED'
        chain = only_chain(obj)
        # A half circle about the centre of the face: 8 segments of the 16 per circle.
        assert not chain.closed and len(chain.points) == 9
        assert near(chain.points[0], p0) and near(chain.points[-1], p2)
        assert all(abs(p.z - 1.0) < 1e-4 and abs((p - Vector((0, 0, 1))).length - math.sqrt(2.0)) < 1e-3
                   for p in chain.points)
        assert contains_point(chain.points, p1)
        assert all(near(n, (0, 0, 1), 1e-5) for n in chain.normals)


def test_a_three_point_arc_across_two_faces_lies_in_the_plane_of_its_three_points():
    with addon():
        obj = edit_path()
        # Front-top-left and front-bottom-right share the front face; the third is on the back.
        p0, p1, p2 = (Vector(CORNERS[k]) for k in ("ftl", "fbr", "btr"))
        c, _, view = shape_controller(obj, ARC_3PT, segments=24)
        c.handle(at(view, p0))
        c.handle(at(view, p1))
        assert c.handle(at(view, p2)) == 'FINISHED'
        chain = only_chain(obj)
        normal = (p1 - p0).cross(p2 - p0).normalized()
        centre = circumcentre(p0, p1, p2)
        radius = (p0 - centre).length
        assert near(chain.points[0], p0) and near(chain.points[-1], p2) and len(chain.points) > 4
        assert all(abs(normal.dot(p - p0)) < 1e-4 and abs((p - centre).length - radius) < 1e-3 for p in chain.points)
        assert abs(radius - (p1 - centre).length) < 1e-4
        # Every generated point carries the first point's normal, not the plane's: the front face,
        # which holds the first segment.
        assert all(near(n, (0, -1, 0), 1e-5) for n in chain.normals)
        assert abs(Vector((0, -1, 0)).dot(normal)) > 0.1


def test_a_three_point_arc_takes_the_face_its_points_share_for_the_first_normal():
    with addon():
        obj = edit_path()
        # The first corner touches three faces and the eye prefers the front one when it is placed;
        # the other two points are on the top, so the top is the face of the arc, as for a polyline.
        c, _, view = shape_controller(obj, ARC_3PT)
        c.handle(at(view, (1, -1, 1)))
        assert near(c.session.points[0].normal, (0, -1, 0), 1e-5)
        c.handle(at(view, (1, 1, 1)))
        assert c.handle(at(view, (-1, 1, 1))) == 'FINISHED'
        assert all(near(n, (0, 0, 1), 1e-5) for n in only_chain(obj).normals)


def test_three_collinear_points_make_no_arc_and_the_run_keeps_waiting():
    with addon():
        obj = edit_path()
        c, headers, view = shape_controller(obj, ARC_3PT)
        c.handle(at(view, (-1, -1, 1)))
        c.handle(at(view, (0, -1, 1)))
        before = path_state(obj)
        assert c.handle(at(view, (1, -1, 1))) == 'RUNNING_MODAL'
        assert not c.session.done and path_state(obj) == before and not before[0]
        assert "in a line" in headers[-1]
        preview = c.session.preview()
        assert len(preview.strips) == 1 and len(preview.strips[0]) == 3, "the polyline shows the points"
        # A third point off the line completes it, and the note is gone.
        assert c.handle(at(view, (1, 1, 1))) == 'FINISHED'
        assert len(only_chain(obj).points) > 3 and headers[-1] is None


def test_escape_removes_everything_of_every_shape_and_keeps_earlier_geometry():
    with addon():
        obj = edit_path()
        polyline(obj, [(0.2, 0.3, 1.0), (0.6, -0.4, 1.0)])
        before = path_state(obj)
        runs = {
            RECTANGLE: [TOP], CIRCLE: [TOP], ARC_CENTER: [(0, 0, 1), (0.5, 0, 1)],
            ARC_3PT: [(-0.8, -0.4, 1), (0.0, -0.7, 1)],
        }
        for shape, clicks in runs.items():
            for stop in range(len(clicks) + 1):
                c, _, view = shape_controller(obj, shape)
                c.handle(at(view, clicks[0]))
                for co in clicks[1:stop + 1]:
                    c.handle(at(view, co))
                assert c.handle(ev('ESC')) == 'CANCELLED', (shape, stop)
                assert path_state(obj) == before and c.closed and not draw3d._live
        # RMB and Enter keep nothing of an incomplete shape either.
        for key in ('RIGHTMOUSE', 'RET'):
            c, _, view = shape_controller(obj, ARC_3PT)
            c.handle(at(view, (-0.8, -0.4, 1)))
            c.handle(at(view, (0.0, -0.7, 1)))
            assert c.handle(ev(key)) == 'CANCELLED' and path_state(obj) == before


# ---------------------------------------------------------------------------
# Keys and the starting shape
# ---------------------------------------------------------------------------

def test_shape_keys_switch_while_fewer_than_two_points_exist():
    with addon():
        obj = edit_path()
        c, headers, view = shape_controller(obj, POLYLINE)
        s = c.session
        c.handle(at(view, TOP))
        steps = [('R', RECTANGLE), ('O', CIRCLE), ('A', ARC_3PT), ('A', ARC_CENTER), ('A', ARC_3PT),
                 ('P', POLYLINE), ('A', ARC_3PT), ('R', RECTANGLE), ('A', ARC_3PT), ('O', CIRCLE)]
        for key, shape in steps:
            assert c.handle(ev(key)) == 'RUNNING_MODAL'
            assert s.shape == shape, (key, s.shape)
            assert snap3d._MODE_HELP[shape] in headers[-1]
        # Ctrl and Alt keep their meaning (the key is swallowed, nothing switches).
        assert c.handle(ev('R', ctrl=True)) == 'RUNNING_MODAL' and s.shape == CIRCLE
        assert c.handle(ev('A', alt=True)) == 'RUNNING_MODAL' and s.shape == CIRCLE
        c.abort()


def test_shape_keys_are_ignored_once_a_second_point_exists():
    with addon():
        obj = edit_path()
        for shape, second in ((ARC_3PT, (0.0, -0.7, 1.0)), (ARC_CENTER, (0.5, 0.0, 1.0)), (POLYLINE, (0.5, 0.0, 1.0))):
            c, _, view = shape_controller(obj, shape)
            c.handle(at(view, (0.0, 0.0, 1.0)))
            c.handle(at(view, second))
            assert len(c.session.points) == 2
            for key in 'ROPA':
                c.handle(ev(key))
                assert c.session.shape == shape, (shape, key)
            c.abort()
        # With nothing placed yet the session accepts them too.
        session = snap3d.PathSession()
        session.handle(("KEY", "O"))
        assert session.shape == CIRCLE


def test_the_header_names_the_shape_keys_and_says_typed_values_are_unavailable():
    with addon():
        obj = edit_path()
        c, headers, view = shape_controller(obj, POLYLINE)
        c.handle(at(view, TOP))
        assert headers[-1].startswith("Snapping on | ")
        assert "R rect" in headers[-1] and "A arc" in headers[-1] and "no typed values" in headers[-1]
        c.handle(ev('R'))
        assert "hold Shift or Ctrl to turn snapping off" in headers[-1] and "no typed values" in headers[-1]
        assert headers[-1].endswith("Esc cancels, Ctrl+Z off")
        c.handle(at(view, (0.6, 0.5, 1.0)))
        assert headers[-1] is None
        assert chr(0x2014) not in snap3d.PathSession(ARC_3PT).header_text()


def test_the_header_enum_chooses_the_shape_the_run_starts_with():
    with addon():
        obj = edit_path()
        wm = bpy.context.window_manager
        assert wm.sketch_tools_shape == POLYLINE
        for shape in snap3d.SHAPES:
            wm.sketch_tools_shape = shape
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            ctx.window_manager = types.SimpleNamespace(
                modal_handler_add=handlers.append, sketch_tools_shape=wm.sketch_tools_shape)
            try:
                assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
                assert shell._controller.session.shape == shape
            finally:
                draw3d.close_all()
                restore()
        # A circle through the operator: the circle_segments preference sets the count.
        wm.sketch_tools_shape = CIRCLE
        shell, ctx, headers, handlers, restore, view = operator_shell(obj)
        ctx.window_manager = types.SimpleNamespace(
            modal_handler_add=handlers.append, sketch_tools_shape=wm.sketch_tools_shape)
        real_prefs = draw3d.get_prefs
        draw3d.get_prefs = lambda context: types.SimpleNamespace(circle_segments=10, snap_pixel_threshold=12)
        try:
            op = draw3d.SKETCH_OT_draw_3d
            assert op.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
            assert op.modal(shell, ctx, at(view, (0.7, 0.3, 1.0))) == {'FINISHED'}
        finally:
            draw3d.get_prefs = real_prefs
            restore()
        assert vertex_count(obj) == 10 and only_chain(obj).closed
        wm.sketch_tools_shape = POLYLINE


def test_the_panel_picker_sets_the_shape_a_3d_path_run_starts_with():
    with addon():
        obj = edit_path()
        wm = bpy.context.window_manager
        # The Sketch panel offers the enum for a 3D path in edit mode; the property it edits is the one the tool reads.
        layout = draw_sketch_panel(obj, 'EDIT_MESH')
        assert len(layout.picks) == 1
        for shape in snap3d.SHAPES:
            wm.sketch_tools_shape = shape
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            ctx.window_manager = types.SimpleNamespace(
                modal_handler_add=handlers.append, sketch_tools_shape=wm.sketch_tools_shape)
            try:
                assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
                assert shell._controller.session.shape == shape
            finally:
                draw3d.close_all()
                restore()
        wm.sketch_tools_shape = POLYLINE


def test_a_missing_or_unknown_shape_setting_starts_a_polyline():
    with addon():
        obj = edit_path()
        for setting in ({}, {"sketch_tools_shape": "SPIRAL"}):
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            ctx.window_manager = types.SimpleNamespace(modal_handler_add=handlers.append, **setting)
            try:
                assert draw3d.SKETCH_OT_draw_3d.invoke(shell, ctx, at(view, TOP)) == {'RUNNING_MODAL'}
                assert shell._controller.session.shape == POLYLINE
            finally:
                draw3d.close_all()
                restore()


def test_a_shape_is_one_undo_step_written_by_the_closing_click_alone():
    assert 'UNDO' in draw3d.SKETCH_OT_draw_3d.bl_options
    for shape, start, clicks in (
            (RECTANGLE, TOP, [(0.6, 0.5, 1.0)]), (CIRCLE, TOP, [(0.6, 0.5, 1.0)]),
            (ARC_CENTER, (0.0, 0.0, 1.0), [(0.5, 0.0, 1.0), (0.1, 0.5, 1.0)]),
            (ARC_3PT, TOP, [(0.0, -0.7, 1.0), (0.6, 0.5, 1.0)])):
        with addon():
            obj = edit_path()
            shell, ctx, headers, handlers, restore, view = operator_shell(obj)
            ctx.window_manager = types.SimpleNamespace(modal_handler_add=handlers.append, sketch_tools_shape=shape)
            batches = []
            real_apply = draw3d.PathEditor.apply

            def spy(self, commands):
                if commands:
                    batches.append([cmd[0] for cmd in commands])
                return real_apply(self, commands)

            draw3d.PathEditor.apply = spy
            try:
                op = draw3d.SKETCH_OT_draw_3d
                assert op.invoke(shell, ctx, at(view, start)) == {'RUNNING_MODAL'}
                for co in clicks[:-1]:
                    assert op.modal(shell, ctx, at(view, co)) == {'RUNNING_MODAL'}
                assert not batches and not path_state(obj)[0], shape
                assert op.modal(shell, ctx, at(view, clicks[-1])) == {'FINISHED'}
            finally:
                draw3d.PathEditor.apply = real_apply
                restore()
            # The whole shape is one edit-mesh update, so Blender's undo step for the operator
            # holds exactly it; the run is over and nothing is left live.
            assert len(batches) == 1 and batches[0].count("ADD_VERT") >= 3 and batches[0][-1] == "END", (shape, batches)
            assert not modal_common.any_sketch_modal_live() and not handlers[1:]


# ---------------------------------------------------------------------------
# Feedback and the projector
# ---------------------------------------------------------------------------

def test_the_overlay_previews_the_outline_before_the_closing_click():
    with addon():
        obj = edit_path()
        c, _, view = shape_controller(obj, RECTANGLE)
        c.handle(at(view, TOP))
        target = Vector((0.6, 0.5, 1.0))
        c.handle(ev('MOUSEMOVE', px=view.screen(target)))
        preview, mouse, to_screen = c.snapshot()
        ring = preview.strips[0]
        assert len(ring) == 5 and near(ring[0], ring[-1]) and all(abs(p.z - 1.0) < 1e-4 for p in ring)
        assert contains_point(ring, target) and preview.lines == ["In shape plane"]
        assert len(overlay.screen_geometry(preview, to_screen)["band"]) == 1 + 1, "outline and the marker"
        assert not path_state(obj)[0]
        c.abort()

        c, _, view = shape_controller(obj, CIRCLE, segments=12)
        c.handle(at(view, TOP))
        c.handle(ev('MOUSEMOVE', px=view.screen((0.7, 0.3, 1.0))))
        preview = c.snapshot()[0]
        assert len(preview.strips[0]) == 13 and len(preview.strips[1]) == 2, "ring and radius line"
        c.abort()

        c, _, view = shape_controller(obj, ARC_CENTER)
        c.handle(at(view, (0, 0, 1)))
        c.handle(at(view, (0.5, 0, 1)))
        c.handle(ev('MOUSEMOVE', px=view.screen(rotated((0.5, 0.0, 1.0), 80))))
        strips = c.snapshot()[0].strips
        assert [len(s) for s in strips] == [2, 2, 5], "start line, end line and the arc"
        c.abort()


def test_the_cursor_is_hinted_when_its_ray_misses_the_shape_plane():
    with addon():
        obj = edit_path()
        # Looking almost level, the top of the region is above the horizon: those rays rise away
        # from the plane below them.
        c, headers, view = shape_controller(obj, RECTANGLE, View((5.0, -6.0, 4.0), target=(0, 0, 3.5)))
        c.handle(at(view, TOP))
        assert c.handle(ev('MOUSEMOVE', px=(400.0, 580.0))) == 'RUNNING_MODAL'
        assert draw3d.NO_PLANE_HINT in headers[-1]
        assert c.handle(ev('LEFTMOUSE', px=(400.0, 580.0))) == 'RUNNING_MODAL'
        assert not c.session.done and not path_state(obj)[0]
        c.handle(ev('MOUSEMOVE', px=view.screen((0.6, 0.5, 1.0))))
        assert draw3d.NO_PLANE_HINT not in headers[-1]
        c.abort()


def test_the_projector_gives_the_view_right_vector():
    region = types.SimpleNamespace(width=800, height=600)
    for eye in ((0, -10, 0), (5, -6, 4), (-7, 3, 2)):
        rv3d = _fake_rv3d(eye, (0, 0, 0))
        projector = draw3d.PathProjector(region, rv3d)
        forward = (Vector((0, 0, 0)) - Vector(eye)).normalized()
        assert near(projector.right, forward.cross(UP).normalized(), 1e-5), eye


# ---------------------------------------------------------------------------
# Sweeping the results
# ---------------------------------------------------------------------------

def test_closed_shapes_sweep_to_manifold_outputs_and_arcs_to_capped_ones():
    results = {}
    for name, shape, clicks in (
            ("rectangle", RECTANGLE, [TOP, (0.6, 0.5, 1.0)]),
            ("circle", CIRCLE, [FRONT, (0.4, -1.0, -0.2)]),
            ("centre arc", ARC_CENTER, [(0, 0, 1), (0.5, 0, 1), (0, -0.5, 1)]),
            ("three-point arc", ARC_3PT, [CORNERS["ftl"], CORNERS["fbr"], CORNERS["btr"]])):
        with addon():
            obj = edit_path()
            chain = draw(obj, shape, clicks)
            bm, warnings = sweep(obj)
            try:
                assert warnings == "", (name, warnings)
                assert len(bm.faces) > 0 and signed_volume(bm) > 0.0, name
                assert_closed_and_consistent(bm)
                assert all(e.is_manifold for e in bm.edges), name
                results[name] = chain.closed
            finally:
                bm.free()
    assert results == {"rectangle": True, "circle": True, "centre arc": False, "three-point arc": False}
