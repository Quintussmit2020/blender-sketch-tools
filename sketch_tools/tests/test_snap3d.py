"""Pure snapping and placement logic of the 3D path draw tool (core/snap3d.py)."""

import math
import random
import time

import bmesh
from mathutils import Matrix, Vector

from sketch_tools.core import snap3d
from sketch_tools.core.snap3d import (
    EDGE, FACE, MIDPOINT, PATH_VERTEX, VERTEX, PathSession, Placement, SceneSource,
)

CUBE_VERTS = [(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
# Outward counter-clockwise faces: -Z, +Z, -Y, +Y, -X, +X.
CUBE_POLYS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
NORMALS = {(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)}


class Pinhole:
    """A perspective (or orthographic) camera that implements the view snap3d.place wants. An
    orthographic sight line starts outside the scene like the real projector's: reach (the scene's
    SceneSource.reach) + |co| + 1 behind the point."""

    def __init__(self, eye, target=(0, 0, 0), size=(800, 600), focal=800.0, ortho=False, reach=0.0):
        self.eye = Vector(eye)
        self.forward = (Vector(target) - self.eye).normalized()
        self.right = self.forward.cross(Vector((0, 0, 1))).normalized()
        self.up = self.right.cross(self.forward)
        self.size, self.focal, self.ortho, self.reach = size, focal, ortho, reach
        self.key = ("pinhole", tuple(eye), tuple(target), size, focal, ortho)
        self.projects = 0

    def project(self, co):
        self.projects += 1
        d = Vector(co) - self.eye
        x, y, z = d.dot(self.right), d.dot(self.up), d.dot(self.forward)
        if self.ortho:
            return (self.size[0] / 2 + self.focal * x / 8.0, self.size[1] / 2 + self.focal * y / 8.0, 1.0)
        if z <= 1e-6:
            return None
        return (self.size[0] / 2 + self.focal * x / z, self.size[1] / 2 + self.focal * y / z, z)

    def screen(self, co):
        p = self.project(co)
        return (p[0], p[1])

    def ray(self, mouse):
        dx, dy = mouse[0] - self.size[0] / 2, mouse[1] - self.size[1] / 2
        if self.ortho:
            origin = self.eye + (self.right * dx + self.up * dy) * (8.0 / self.focal)
            return origin, self.forward.copy()
        direction = self.forward * self.focal + self.right * dx + self.up * dy
        return self.eye.copy(), direction.normalized()

    def sight(self, co):
        co = Vector(co)
        if self.ortho:
            distance = self.reach + co.length + 1.0
            return co - self.forward * distance, self.forward.copy(), distance
        offset = co - self.eye
        return self.eye.copy(), offset.normalized(), offset.length


def cube_scene(matrix=None, extra=()):
    scene = SceneSource()
    scene.add_polygons(CUBE_VERTS, CUBE_POLYS, matrix or Matrix.Identity(4))
    for verts, polys, mat in extra:
        scene.add_polygons(verts, polys, mat)
    return scene


def close(a, b, eps=1e-4):
    return (Vector(a) - Vector(b)).length < eps


def normal_of(placement):
    return tuple(round(c) for c in placement.normal)


# ---------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------

def test_ray_hits_the_cube_with_the_face_normal():
    scene = cube_scene()
    for origin, direction, expected in (
        ((0.2, 0.3, 5), (0, 0, -1), (0, 0, 1)),
        ((5, 0.1, 0.2), (-1, 0, 0), (1, 0, 0)),
        ((0.1, -5, -0.4), (0, 1, 0), (0, -1, 0)),
        ((0, 0.5, -5), (0, 0, 1), (0, 0, -1)),
    ):
        co, normal, dist = scene.ray_cast(Vector(origin), Vector(direction))
        assert normal_of(Placement(co, normal, FACE, None)) == expected
        assert abs(dist - 4.0) < 1e-4
        assert abs(co.dot(Vector(expected)) - 1.0) < 1e-5
    assert scene.ray_cast(Vector((5, 5, 5)), Vector((1, 0, 0))) is None


def test_scene_follows_the_object_matrix_and_keeps_normals_outward_under_mirroring():
    rotate = Matrix.Rotation(math.radians(90), 4, 'Z')
    for matrix in (Matrix.Translation((10, 0, 0)) @ rotate @ Matrix.Diagonal((2, 1, 1, 1)),
                   Matrix.Diagonal((-1, 1, 1, 1))):
        scene = cube_scene(matrix)
        origin = matrix @ Vector((0.2, 0.3, 5))
        top = (matrix @ Vector((0.2, 0.3, 1))) - origin
        co, normal, dist = scene.ray_cast(origin, top.normalized())
        assert close(co, matrix @ Vector((0.2, 0.3, 1)))
        # +Z stays +Z for these matrices: the normal is outward, not flipped by the mirror.
        assert close(normal, (0, 0, 1))
    scene = cube_scene(Matrix.Diagonal((-1, 1, 1, 1)))
    co, normal, _ = scene.nearest(Vector((5, 0.2, 0.1)))
    assert close(co, (1, 0.2, 0.1)) and close(normal, (1, 0, 0))


def test_nearest_picks_the_closest_object():
    far = Matrix.Translation((10, 0, 0))
    scene = cube_scene(extra=[(CUBE_VERTS, CUBE_POLYS, far)])
    co, normal, dist = scene.nearest(Vector((5, 0, 0)))
    assert close(co, (1, 0, 0)) and abs(dist - 4.0) < 1e-4
    co, _, dist = scene.nearest(Vector((8, 0, 0)))
    assert close(co, (9, 0, 0)) and abs(dist - 1.0) < 1e-4
    hit = scene.ray_cast(Vector((5, 0, 5)), Vector((0, 0, -1)))
    assert hit is None


def test_clear_drops_everything():
    scene = cube_scene()
    scene.projected(Pinhole((5, -6, 4)))
    scene.clear()
    assert scene.empty and not scene.vertices and not scene.edges and scene.ray_cast(
        Vector((0, 0, 5)), Vector((0, 0, -1))) is None


def test_projection_is_cached_while_the_view_key_stays_the_same():
    scene = cube_scene()
    view = Pinhole((5, -6, 4))
    scene.projected(view)
    count = view.projects
    scene.projected(view)
    assert view.projects == count
    scene.projected(Pinhole((5, -6, 4.5)))
    assert view.projects == count


# ---------------------------------------------------------------------------
# Snap targets
# ---------------------------------------------------------------------------

EYE = (5.0, -6.0, 4.0)


def snap_at(scene, view, co, dx=0.0, dy=0.0, threshold=12, path=(), **kw):
    x, y = view.screen(co)
    return snap3d.place(scene, view, (x + dx, y + dy), threshold, path, **kw)


def test_a_vertex_within_the_threshold_snaps_and_takes_the_nearest_face_normal():
    scene, view = cube_scene(), Pinhole(EYE)
    placement = snap_at(scene, view, (1, -1, 1), 5, 3)
    assert placement.kind == VERTEX and close(placement.co, (1, -1, 1))
    assert placement.ref is None
    # Three faces meet at the corner; from EYE the -Y face is the one that most faces the eye.
    assert normal_of(placement) == (0, -1, 0)


def test_a_vertex_beyond_the_threshold_falls_back_to_the_face_under_the_cursor():
    scene, view = cube_scene(), Pinhole(EYE)
    corner = view.screen((1, -1, 1))
    # 30 px to the inside of the +Z face's corner.
    toward = view.screen((0.4, -0.4, 1))
    direction = Vector(toward) - Vector(corner)
    mouse = Vector(corner) + direction.normalized() * 30
    placement = snap3d.place(scene, view, tuple(mouse), 12)
    assert placement.kind == FACE and normal_of(placement) == (0, 0, 1)
    assert abs(placement.co.z - 1.0) < 1e-4
    # A larger threshold would have snapped.
    assert snap3d.place(scene, view, tuple(mouse), 40).kind == VERTEX


def test_midpoint_beats_the_point_on_the_edge():
    scene, view = cube_scene(), Pinhole(EYE)
    placement = snap_at(scene, view, (0, -1, 1), 2, 2)
    assert placement.kind == MIDPOINT and close(placement.co, (0, -1, 1))
    assert normal_of(placement) == (0, -1, 0)


def test_the_nearest_point_on_an_edge_is_exact_in_perspective():
    scene, view = cube_scene(), Pinhole((3.0, -4.0, 2.5), focal=500.0)
    # Top edge on y=-1 from (-1,-1,1) to (1,-1,1): the point at t=0.3 along the edge.
    target = Vector((-1, -1, 1)).lerp(Vector((1, -1, 1)), 0.3)
    placement = snap_at(scene, view, tuple(target), 0, 0)
    assert placement.kind == EDGE and close(placement.co, target, 1e-5)
    # The pixel nearest to the edge line, 3 px off it, gives the point whose projection is nearest.
    placement = snap_at(scene, view, tuple(target), 0, 3)
    assert placement.kind == EDGE
    assert abs(placement.co.y + 1) < 1e-5 and abs(placement.co.z - 1) < 1e-5
    assert math.dist(view.screen(placement.co), view.screen(target)) < 4.0


def test_priority_is_vertex_then_midpoint_then_edge_when_all_are_in_range():
    # A small, far cube: its vertex, the edge midpoint and the edge all fall within 12 px.
    scene, view = cube_scene(), Pinhole((30.0, -35.0, 24.0), focal=300.0)
    top_edge = (Vector((-1, -1, 1)), Vector((1, -1, 1)))
    ends = [view.screen(p) for p in top_edge]
    assert math.dist(*ends) < 24
    corner = top_edge[1]
    assert snap_at(scene, view, tuple(corner)).kind == VERTEX
    mid = (top_edge[0] + top_edge[1]) / 2
    # Cursor at the midpoint: the vertex is within 12 px too, and wins.
    assert snap_at(scene, view, tuple(mid)).kind == VERTEX
    # With a threshold that excludes the vertices, the midpoint wins over the edge under it.
    assert snap_at(scene, view, tuple(mid), threshold=math.dist(*ends) / 2 - 1).kind == MIDPOINT
    quarter = top_edge[0].lerp(top_edge[1], 0.25)
    assert snap_at(scene, view, tuple(quarter), threshold=math.dist(*ends) / 4 - 0.5).kind == EDGE


def test_nearest_candidate_of_a_kind_wins():
    scene, view = cube_scene(), Pinhole((30.0, -35.0, 24.0), focal=300.0)
    a, b = Vector((-1, -1, 1)), Vector((1, -1, 1))
    near_a = view.screen(a)
    placement = snap3d.place(scene, view, (near_a[0] + 1, near_a[1] + 1), 12)
    assert placement.kind == VERTEX and close(placement.co, a)


def test_hidden_candidates_are_skipped():
    scene, view = cube_scene(), Pinhole((0.01, -0.02, 9.0), focal=500.0)
    # The bottom vertex projects inside the top face from straight above; it is behind the cube.
    hidden = (1, 1, -1)
    mouse = view.screen(hidden)
    assert snap3d.find_snap(scene, view, mouse, 6) is None
    placement = snap3d.place(scene, view, mouse, 6)
    assert placement.kind == FACE and abs(placement.co.z - 1.0) < 1e-4
    # The hidden bottom edge midpoint is skipped too.
    assert snap3d.find_snap(scene, view, view.screen((0, 1, -1)), 6) is None
    # The visible top vertex above it is found.
    assert snap3d.find_snap(scene, view, view.screen((1, 1, 1)), 6).kind == VERTEX


def test_visible_vertices_on_the_silhouette_are_not_taken_for_hidden():
    scene, view = cube_scene(), Pinhole(EYE)
    # Every corner on the front-facing side of the cube is visible.
    for corner in ((1, -1, 1), (1, -1, -1), (-1, -1, 1), (1, 1, 1)):
        snap = snap3d.find_snap(scene, view, view.screen(corner), 4)
        assert snap is not None and snap.kind == VERTEX and close(snap.co, corner), corner
    # The corner at the back, bottom is behind the cube.
    assert snap3d.find_snap(scene, view, view.screen((-1, 1, -1)), 4) is None


def test_orthographic_views_test_occlusion_along_the_view_direction():
    scene = cube_scene()
    view = Pinhole((4.0, -5.0, 20.0), ortho=True, reach=scene.reach)
    assert snap3d.find_snap(scene, view, view.screen((1, -1, 1)), 6).kind == VERTEX
    assert snap3d.is_visible(scene, view, Vector((1, -1, 1)))
    assert not snap3d.is_visible(scene, view, Vector((-1, 1, -1)))
    snap = snap3d.find_snap(scene, view, view.screen((-1, 1, -1)), 6)
    assert snap is None or not close(snap.co, (-1, 1, -1))
    placement = snap3d.place(scene, view, view.screen((0.3, 0.2, 1)), 6)
    assert placement.kind == FACE and close(placement.co, (0.3, 0.2, 1))


def test_edges_with_an_end_behind_the_camera_are_ignored():
    scene, view = cube_scene(), Pinhole((0.0, 0.0, 0.5), target=(0, 0, -5), focal=500.0)
    # Must not raise, and finds no edge snap for the ones crossing the camera plane.
    snap3d.find_snap(scene, view, (400, 300), 12)


def test_snapping_onto_a_path_vertex_reuses_it():
    scene, view = cube_scene(), Pinhole(EYE)
    path = [(3, Vector((1, -1, 1))), (-2, Vector((1, 1, 1)))]
    placement = snap_at(scene, view, (1, -1, 1), 2, 2, path=path)
    assert placement.ref == 3 and placement.kind == PATH_VERTEX and close(placement.co, (1, -1, 1), 1e-9)
    # A snap onto the midpoint of an edge where the path has a vertex reuses it as well.
    placement = snap_at(scene, view, (0, -1, 1), path=[(5, Vector((0, -1, 1)))])
    assert placement.ref == 5
    # Not close enough to count as the same point (a pixel away at this zoom): a new vertex.
    placement = snap_at(scene, view, (1, -1, 1), path=[(3, Vector((1.02, -1, 1)))])
    assert placement.ref is None and placement.kind == VERTEX


def test_a_path_vertex_off_the_scene_geometry_is_a_snap_target_too():
    scene, view = cube_scene(), Pinhole(EYE)
    floating = Vector((3, -3, 0.5))
    placement = snap_at(scene, view, tuple(floating), 3, 3, path=[(9, floating)])
    assert placement.ref == 9 and placement.kind == PATH_VERTEX and close(placement.co, floating)
    # Hidden behind the cube: skipped like any other candidate.
    behind = Vector((-3, 3, -0.5))
    mouse = view.screen(behind)
    snap = snap3d.find_snap(scene, view, mouse, 6, [(9, behind)])
    assert snap is None or snap.ref != 9


def test_nothing_under_the_cursor_places_nothing():
    scene, view = cube_scene(), Pinhole(EYE)
    assert snap3d.place(scene, view, (5, 5), 12) is None
    assert snap3d.place(SceneSource(), view, (400, 300), 12) is None


# ---------------------------------------------------------------------------
# Normals of snapped points (C19)
# ---------------------------------------------------------------------------

TOP_CORNERS = [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]


def test_the_four_top_corners_seen_from_above_all_get_the_top_normal():
    scene = cube_scene()
    for view in (Pinhole((0.01, -0.4, 9.0)), Pinhole((0.7, 0.9, 9.0)), Pinhole((0.0, -0.5, 30.0), ortho=True, reach=scene.reach)):
        for corner in TOP_CORNERS:
            placement = snap_at(scene, view, corner, 2, 2)
            assert placement.kind == VERTEX and close(placement.co, corner), (view.eye, corner)
            assert tuple(placement.normal) == (0.0, 0.0, 1.0), (view.eye, corner, tuple(placement.normal))


def test_a_corner_seen_from_the_side_gets_the_side_normal():
    scene = cube_scene()
    placement = snap_at(scene, Pinhole((9.0, 0.3, 0.4)), (1, -1, 1), 2, 2)
    assert placement.kind == VERTEX and normal_of(placement) == (1, 0, 0)


def test_the_top_front_edge_midpoint_takes_the_face_that_faces_the_eye():
    scene = cube_scene()
    mid = (0, -1, 1)
    above = snap_at(scene, Pinhole((0.0, -0.4, 9.0)), mid)
    assert above.kind == MIDPOINT and close(above.co, mid) and normal_of(above) == (0, 0, 1)
    front = snap_at(scene, Pinhole((0.0, -9.0, 0.3)), mid)
    assert front.kind == MIDPOINT and normal_of(front) == (0, -1, 0)


def test_a_point_on_an_edge_takes_the_face_that_faces_the_eye():
    scene, mid = cube_scene(), Vector((-1, -1, 1)).lerp(Vector((1, -1, 1)), 0.3)
    above = snap_at(scene, Pinhole((0.0, -0.4, 9.0), focal=500.0), tuple(mid))
    assert above.kind == EDGE and normal_of(above) == (0, 0, 1)
    front = snap_at(scene, Pinhole((0.0, -9.0, 0.3), focal=500.0), tuple(mid))
    assert front.kind == EDGE and normal_of(front) == (0, -1, 0)


def test_equally_facing_faces_are_settled_by_the_previous_normal():
    scene, view, mid = cube_scene(), Pinhole((0.0, -9.0, 9.0)), (0, -1, 1)
    top, front = Vector((0, 0, 1)), Vector((0, -1, 0))
    assert normal_of(snap3d.place(scene, view, view.screen(mid), 12, previous_normal=top)) == (0, 0, 1)
    assert normal_of(snap3d.place(scene, view, view.screen(mid), 12, previous_normal=front)) == (0, -1, 0)
    # Clearly not a tie: the previous normal does not override the face that faces the eye.
    view = Pinhole((0.0, -9.0, 3.0))
    assert normal_of(snap3d.place(scene, view, view.screen(mid), 12, previous_normal=top)) == (0, -1, 0)


def draw_run(scene, clicks):
    """Click through (view, world point, jitter) in order with a PathSession, the way the controller
    does, and return the normals the session wrote (as rounded tuples) in the order of the points."""
    session, written = PathSession(), {}
    for view, co, kind in clicks:
        x, y = view.screen(co)
        mouse = (x + 2, y + 1) if kind == "snap" else (x, y)
        placement = snap3d.place(scene, view, mouse, 12, session.points[-1:] and [(p.ref, p.co) for p in session.points],
                                 previous_normal=session.last_normal(), previous_point=session.last_point(),
                                 earlier_points=session.earlier_points())
        assert placement is not None and (placement.kind != FACE) == (kind == "snap"), (co, placement)
        for cmd in session.handle(("CLICK", placement)):
            if cmd[0] == "ADD_VERT":
                written[cmd[1]] = tuple(round(c, 3) for c in cmd[3])
            elif cmd[0] == "SET_NORMAL":
                assert cmd[1] in written, cmd
                written[cmd[1]] = tuple(round(c, 3) for c in cmd[2])
    return [written.get(p.ref) for p in session.points]


def front_eye(elevation, azimuth=(0.0, -1.0)):
    e = math.radians(elevation)
    return Vector((azimuth[0] * math.cos(e), azimuth[1] * math.cos(e), math.sin(e))) * 8


def test_a_rim_loop_on_a_cube_gets_the_top_normal_at_all_four_corners():
    scene = cube_scene()
    loop = [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]
    for elevation in (20, 35, 60):
        view = Pinhole(front_eye(elevation))
        normals = draw_run(scene, [(view, c, "snap") for c in loop])
        assert normals == [(0.0, 0.0, 1.0)] * 4, (elevation, normals)
    # The eye leans towards +X as well: still the top at the steeper angles.
    for elevation in (35, 60):
        view = Pinhole(front_eye(elevation, (0.6, -0.8)))
        normals = draw_run(scene, [(view, c, "snap") for c in loop])
        assert normals == [(0.0, 0.0, 1.0)] * 4, (elevation, normals)
    # Leaning towards either side at a low angle a side face is the one most in view along the first
    # or the last segment; the later points (and the earlier ones) show the loop lies on the top.
    for azimuth in ((-0.6, -0.8), (0.6, -0.8)):
        for elevation in (10, 20):
            view = Pinhole(front_eye(elevation, azimuth))
            normals = draw_run(scene, [(view, c, "snap") for c in loop])
            assert normals == [(0.0, 0.0, 1.0)] * 4, (azimuth, elevation, normals)


def test_a_rim_loop_gets_the_top_normal_from_every_start_corner_direction_and_side():
    scene = cube_scene()
    base = [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]
    for backwards in (False, True):
        for start in range(4):
            loop = base[start:] + base[:start]
            loop = loop[::-1] if backwards else loop
            for azimuth in range(0, 360, 45):
                a = math.radians(azimuth)
                for elevation in (10, 20, 35, 60):
                    view = Pinhole(front_eye(elevation, (math.sin(a), -math.cos(a))))
                    normals = draw_run(scene, [(view, c, "snap") for c in loop])
                    assert normals == [(0.0, 0.0, 1.0)] * 4, (backwards, start, azimuth, elevation, normals)


def test_later_points_rewrite_only_the_normals_of_vertices_the_run_created():
    scene = cube_scene()
    view = Pinhole(front_eye(20, (0.6, -0.8)))
    both = (Vector((0, 0, 1)), Vector((1, 0, 0)))
    eye = Vector((0.6, -0.8, 0.4)).normalized()

    def placement(co, normal, ref=None, faces=both):
        return Placement(Vector(co), Vector(normal), VERTEX, ref, False, faces, eye)

    # A vertex that existed before the run (ref 5) is never rewritten, in either direction.
    session = PathSession()
    session.handle(("CLICK", placement((1, -1, 1), (1, 0, 0), ref=5)))
    cmds = session.handle(("CLICK", placement((1, 1, 1), (1, 0, 0))))
    cmds += session.handle(("CLICK", placement((-1, 1, 1), (0, 0, 1))))
    assert [c[0] for c in cmds] == ["ADD_VERT", "ADD_EDGE", "SET_NORMAL", "ADD_VERT", "ADD_EDGE"]
    assert cmds[2][1] == -1
    assert session.points[0].normal == Vector((1, 0, 0))
    # The run's own vertex (ref -1) in the middle is: its two neighbours lie in the top plane.
    assert session.points[1].ref == -1 and session.points[1].normal == Vector((0, 0, 1))
    # A second visit to a vertex of the run is not rewritten, and the closing click rewrites nothing.
    cmds = session.handle(("CLICK", placement((-1, -1, 1), (0, 0, 1))))
    cmds += session.handle(("CLICK", placement((1, 1, 1), (1, 0, 0), ref=-1)))
    assert not any(c[0] == "SET_NORMAL" and c[1] == 5 for c in cmds)
    assert session.points[-1].normal == Vector((1, 0, 0)) and session.points[1].normal == Vector((0, 0, 1))
    # The rewrite is a command for a vertex that is already written, with the new normal.
    session = PathSession()
    session.handle(("CLICK", placement((1, -1, 1), (0, -1, 0))))
    session.handle(("CLICK", placement((1, 1, 1), (1, 0, 0))))
    cmds = session.handle(("CLICK", placement((-1, 1, 1), (0, 0, 1))))
    assert [c[0] for c in cmds] == ["SET_NORMAL", "SET_NORMAL", "ADD_VERT", "ADD_EDGE"], cmds
    assert {c[1]: tuple(c[2]) for c in cmds[:2]} == {-1: (0.0, 0.0, 1.0), -2: (0.0, 0.0, 1.0)}


def test_a_path_up_a_side_face_and_across_the_top_keeps_the_face_it_lies_on():
    scene = cube_scene()
    # Seen mostly from above, with the +X face still in view.
    view = Pinhole((3.0, -1.5, 6.0))
    clicks = [
        (view, (1.0, 0.3, -0.6), "face"),
        (view, (1.0, 0.1, 0.1), "face"),
        # The midpoint of the top +X edge: the eye alone would take the top face.
        (view, (1.0, 0.0, 1.0), "snap"),
        (view, (0.3, 0.1, 1.0), "face"),
        # The midpoint of the top +Y edge: the segment from the previous point lies on the top.
        (view, (0.0, 1.0, 1.0), "snap"),
    ]
    normals = draw_run(scene, clicks)
    assert normals == [(1.0, 0.0, 0.0)] * 3 + [(0.0, 0.0, 1.0)] * 2, normals
    # Control: the same snap without a previous point takes the top face.
    x, y = view.screen((1.0, 0.0, 1.0))
    assert normal_of(snap3d.place(scene, view, (x + 2, y + 1), 12)) == (0, 0, 1)


def test_a_face_the_viewer_cannot_see_is_not_preferred_over_one_in_view():
    scene = cube_scene()
    # The previous point is on the +X face, but this view is from x < 1: that face is a back face,
    # the top is the face in view and keeps the normal pointing at the viewer.
    view = Pinhole((0.5, -0.4, 9.0))
    x, y = view.screen((1.0, 0.0, 1.0))
    placement = snap3d.place(scene, view, (x + 2, y + 1), 12, previous_point=Vector((1.0, 0.3, -0.6)))
    assert placement.kind == MIDPOINT and normal_of(placement) == (0, 0, 1)
    # Inside a room every face is a back face: the face the segment lies on is still the one taken,
    # flipped towards the viewer.
    inside = Pinhole((0.0, -0.5, 0.2), target=(1, 1, 1))
    x, y = inside.screen((1, 1, 1))
    placement = snap3d.place(scene, inside, (x + 2, y + 1), 12, previous_point=Vector((1.0, 1.0, -0.5)))
    assert placement.kind == VERTEX and normal_of(placement) in ((-1, 0, 0), (0, -1, 0))
    assert abs(placement.normal.dot(Vector((0, 0, 1)))) < 1e-6


def test_the_first_point_takes_the_face_the_first_segment_lies_on():
    scene = cube_scene()
    view = Pinhole(front_eye(20))
    # Alone, the corner seen from the front takes the front face.
    x, y = view.screen((1, -1, 1))
    assert normal_of(snap3d.place(scene, view, (x + 2, y + 1), 12)) == (0, -1, 0)
    # A second point on the top face: the top is the only face of the corner that contains it.
    assert draw_run(scene, [(view, (1, -1, 1), "snap"), (view, (0.2, -0.5, 1.0), "face")])[0] == (0.0, 0.0, 1.0)
    # On the front face: the front face.
    assert draw_run(scene, [(view, (1, -1, 1), "snap"), (view, (0.5, -1.0, 0.0), "face")])[0] == (0.0, -1.0, 0.0)
    # On the side face the viewer cannot see well: still that face.
    side = Pinhole(front_eye(20, (1.0, -0.2)))
    assert draw_run(scene, [(side, (1, -1, 1), "snap"), (side, (1.0, 0.4, 0.2), "face")])[0] == (1.0, 0.0, 0.0)
    # On none of the corner's faces: the eye decides, as before.
    assert draw_run(scene, [(view, (1, -1, 1), "snap"), (view, (-1.0, 0.3, 0.2), "face")])[0] == (0.0, -1.0, 0.0)


def test_a_path_vertex_that_exists_already_is_not_repicked():
    scene, view = cube_scene(), Pinhole(front_eye(20))
    session = PathSession()
    existing = snap3d.Placement(Vector((1, -1, 1)), Vector((0, -1, 0)), PATH_VERTEX, 5, False,
                                (Vector((0, 0, 1)), Vector((0, -1, 0))), Vector((0, -1, 0.3)).normalized())
    session.handle(("CLICK", existing))
    cmds = session.handle(("CLICK", Placement(Vector((0.2, -0.5, 1.0)), Vector((0, 0, 1)), FACE, None)))
    assert [c[0] for c in cmds] == ["ADD_VERT", "ADD_EDGE"] and cmds[0][3] == Vector((0, 0, 1))
    assert session.points[0].normal == Vector((0, -1, 0))


def test_snap_normals_follow_a_rotated_and_scaled_object():
    # +Z of the object points along world +X; seen from +X the corner takes that face.
    matrix = Matrix.Rotation(math.radians(90), 4, 'Y') @ Matrix.Diagonal((2, 1, 3, 1))
    scene = cube_scene(matrix)
    corner = matrix @ Vector((1, -1, 1))
    placement = snap_at(scene, Pinhole((30.0, 0.5, 0.5), focal=300.0), tuple(corner), 1, 1)
    assert placement.kind == VERTEX and close(placement.co, corner)
    assert close(placement.normal, (1, 0, 0))


# ---------------------------------------------------------------------------
# Exact adjacency (QA round 2, W1)
# ---------------------------------------------------------------------------

def icosphere(subdivisions=3):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    bm.verts.ensure_lookup_table()
    verts = [tuple(v.co) for v in bm.verts]
    polys = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    return verts, polys


def test_scene_records_the_faces_around_every_vertex_and_edge():
    scene = cube_scene()
    assert len(scene.vertex_normals) == len(scene.vertices) == 8
    assert len(scene.edge_normals) == len(scene.edges) == 12
    for co, normals in zip(scene.vertices, scene.vertex_normals):
        assert {tuple(round(c) for c in n) for n in normals} == {
            (int(co.x), 0, 0), (0, int(co.y), 0), (0, 0, int(co.z))}
    for (a, b), normals in zip(scene.edges, scene.edge_normals):
        shared = {tuple(round(c) for c in n) for n in scene.vertex_normals[a]} & {
            tuple(round(c) for c in n) for n in scene.vertex_normals[b]}
        assert len(normals) == 2 and {tuple(round(c) for c in n) for n in normals} == shared


def test_adjacent_normals_follow_the_matrix_and_a_flat_patch_lists_one_normal():
    matrix = Matrix.Rotation(math.radians(90), 4, 'Y') @ Matrix.Diagonal((2, 1, 3, 1))
    scene = cube_scene(matrix)
    for co, normals in zip(scene.vertices, scene.vertex_normals):
        assert len(normals) == 3 and all(abs(n.length - 1.0) < 1e-6 for n in normals)
    # A 2 x 1 grid of coplanar quads: the shared edge and its ends see the one normal.
    grid = SceneSource()
    grid.add_polygons([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)],
                      [(0, 1, 4, 3), (1, 2, 5, 4)], Matrix.Identity(4))
    assert all(len(n) == 1 for n in grid.vertex_normals + grid.edge_normals)


def sphere_checks(matrix, eye_offset, label):
    """Place on 40 visible vertices of an icosphere and return the worst angle (degrees) between
    the stored normal and the radial direction, plus the slowest query in seconds."""
    verts, polys = icosphere(5)
    scene = SceneSource()
    scene.add_polygons(verts, polys, matrix)
    centre = matrix @ Vector((0, 0, 0))
    view = Pinhole(centre + eye_offset, target=centre)
    worst, slowest, done = 0.0, 0.0, 0
    for world in scene.vertices:
        radial = (world - centre).normalized()
        if (view.eye - world).normalized().dot(radial) < 0.5:
            continue
        x, y = view.screen(world)
        start = time.perf_counter()
        placement = snap3d.place(scene, view, (x + 1, y + 1), 6)
        slowest = max(slowest, time.perf_counter() - start)
        assert placement.kind == VERTEX and close(placement.co, world, 1e-3 * max(1.0, world.length) * 1e-1), label
        worst = max(worst, math.degrees(placement.normal.angle(radial)))
        done += 1
        if done == 40:
            break
    assert done == 40, label
    return worst, slowest


def test_snapped_normals_are_adjacent_faces_far_from_the_origin_and_at_tiny_scale():
    # An icosphere with 5120 faces: a face normal is at most about 2.5 degrees off the radial direction.
    for label, matrix, offset in (
        ("origin", Matrix.Identity(4), Vector((3, -4, 2))),
        ("1e3", Matrix.Translation((1e3, 0, 0)), Vector((3, -4, 2))),
        ("1e4", Matrix.Translation((1e4, 0, 0)), Vector((3, -4, 2))),
        ("1e-3 scale", Matrix.Diagonal((1e-3, 1e-3, 1e-3, 1)), Vector((3, -4, 2)) * 1e-3),
        ("1e3 and scale 1e-1", Matrix.Translation((1e3, 0, 0)) @ Matrix.Diagonal((0.1, 0.1, 0.1, 1)),
         Vector((3, -4, 2)) * 0.1),
    ):
        worst, _ = sphere_checks(matrix, offset, label)
        assert worst < 5.0, (label, worst)


def test_a_snap_query_far_from_the_origin_stays_fast():
    for matrix in (Matrix.Identity(4), Matrix.Translation((1e4, 0, 0))):
        _, slowest = sphere_checks(matrix, Vector((3, -4, 2)), "timing")
        # The distance search this replaced took about 50 ms per query at 1e4.
        assert slowest < 0.02, slowest


def test_the_fallback_search_is_tight_and_capped_for_points_without_adjacency():
    verts, polys = icosphere(5)
    for matrix in (Matrix.Identity(4), Matrix.Translation((1e4, 0, 0)),
                   Matrix.Diagonal((1e-3, 1e-3, 1e-3, 1))):
        scene = SceneSource()
        scene.add_polygons(verts, polys, matrix)
        for index in (0, 100, 1000):
            found = scene.normals_at(scene.vertices[index])
            expected = scene.vertex_normals[index]
            assert 1 <= len(found) <= snap3d.MAX_FALLBACK_NORMALS
            assert len(found) == len(expected), (index, len(found), len(expected))
        # A snap without an index (a path vertex on the surface) uses the fallback and still works.
        view = Pinhole(matrix @ Vector((0, 0, 0)) + (matrix.to_scale().x * Vector((3, -4, 2))),
                       target=matrix @ Vector((0, 0, 0)))
        co = scene.vertices[0]
        radial = (co - matrix @ Vector((0, 0, 0))).normalized()
        placement = snap3d.snap_placement(scene, view, snap3d.Snap(co, PATH_VERTEX, 7, 0.0), None)
        assert placement.kind == PATH_VERTEX and abs(placement.normal.length - 1.0) < 1e-5
    # Many coincident faces are capped.
    stack = SceneSource()
    stack.add_polygons([(0, 0, 0), (1, 0, 0), (0, 1, 0)] * 1, [(0, 1, 2)] * 100, Matrix.Identity(4))
    assert len(stack.normals_at(Vector((0, 0, 0)))) == 1


def test_loose_vertices_fall_back_to_the_nearest_surface():
    scene = SceneSource()
    scene.add_polygons(CUBE_VERTS, CUBE_POLYS, Matrix.Identity(4))
    scene.add_points([Vector((0.3, 0.2, 1.0))], [])
    view = Pinhole(EYE)
    placement = snap3d.snap_placement(scene, view, snap3d.Snap(Vector((0.3, 0.2, 1.0)), VERTEX, None, 0.0, 8))
    assert normal_of(placement) == (0, 0, 1)


# ---------------------------------------------------------------------------
# Normals face the viewer (C19)
# ---------------------------------------------------------------------------

def test_a_front_face_hit_keeps_its_normal_and_a_back_face_hit_is_flipped_to_the_viewer():
    scene = cube_scene()
    co, normal, _ = scene.ray_cast(Vector((0.2, 0.3, 5)), Vector((0, 0, -1)))
    assert close(normal, (0, 0, 1))
    # From inside the cube the ray meets the back of the top face: the normal faces the ray's origin.
    co, normal, _ = scene.ray_cast(Vector((0.2, 0.3, 0.0)), Vector((0, 0, 1)))
    assert close(co, (0.2, 0.3, 1.0)) and close(normal, (0, 0, -1))
    co, normal, _ = scene.ray_cast(Vector((0.0, 0.1, 0.2)), Vector((-1, 0, 0)))
    assert close(normal, (1, 0, 0))


def test_a_single_sided_plane_reports_the_side_the_viewer_sees():
    scene = SceneSource()
    scene.add_polygons([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [(0, 1, 2, 3)], Matrix.Identity(4))
    assert close(scene.ray_cast(Vector((0.1, 0.1, 4)), Vector((0, 0, -1)))[1], (0, 0, 1))
    assert close(scene.ray_cast(Vector((0.1, 0.1, -4)), Vector((0, 0, 1)))[1], (0, 0, -1))
    view = Pinhole((0.3, 0.2, -5.0), target=(0, 0, 0))
    placement = snap3d.place(scene, view, (400, 300), 12)
    assert placement.kind == FACE and close(placement.normal, (0, 0, -1))


def test_placing_inside_a_room_stores_the_normal_of_the_visible_side():
    scene = cube_scene()
    view = Pinhole((0.1, 0.1, 0.0), target=(0.0, 0.0, 1.0))
    placement = snap3d.place(scene, view, (400, 300), 12)
    assert placement.kind == FACE and close(placement.co, (0.0, 0.0, 1.0), 1e-3)
    assert close(placement.normal, (0, 0, -1))
    # A snapped corner seen from inside: the three back faces are flipped, the most frontal wins.
    view = Pinhole((0.0, -0.5, 0.2), target=(1, 1, 1))
    placement = snap_at(scene, view, (1, 1, 1), 2, 2)
    assert placement.kind == VERTEX and normal_of(placement) == (0, -1, 0)


def test_ctrl_turns_snapping_off_without_the_angle_lock():
    scene, view = cube_scene(), Pinhole(EYE)
    corner = view.screen((1, -1, 1))
    mouse = (corner[0] + 3, corner[1] + 2)
    assert snap3d.place(scene, view, mouse, 12).kind == VERTEX
    placement = snap3d.place(scene, view, mouse, 12, ctrl=True)
    assert placement.kind == FACE and not placement.locked and placement.ref is None
    # The point is exactly where the cursor ray meets the cube, and stays so with an anchor.
    origin, direction = view.ray(mouse)
    assert close(placement.co, scene.ray_cast(origin, direction)[0])
    anchored = snap3d.place(scene, view, mouse, 12, ctrl=True, anchor=Vector((0, 0, 1)), previous_dir=Vector((1, 0, 0)))
    assert close(anchored.co, placement.co) and not anchored.locked


def test_the_header_says_shift_and_ctrl_turn_snapping_off():
    assert "hold Shift or Ctrl: snapping off, Shift locks 45 degrees" in PathSession().header_text()
    for shape in (snap3d.RECTANGLE, snap3d.CIRCLE, snap3d.ARC_3PT, snap3d.ARC_CENTER):
        assert "hold Shift or Ctrl to turn snapping off" in PathSession(shape).header_text(), shape


# ---------------------------------------------------------------------------
# Angle lock
# ---------------------------------------------------------------------------

def polar(angle_deg, length=1.0):
    a = math.radians(angle_deg)
    return Vector((math.cos(a) * length, math.sin(a) * length, 0.0))


def angle_of(v):
    return math.degrees(math.atan2(v.y, v.x))


def test_lock_rounds_to_45_degree_steps_from_the_previous_segment():
    origin, up = Vector((0, 0, 0)), Vector((0, 0, 1))
    for target, expected in ((20, 0), (30, 45), (80, 90), (100, 90), (170, 180), (-40, -45)):
        out = snap3d.lock_in_plane(origin, polar(target, 2.0), up, Vector((1, 0, 0)))
        assert abs(out.length - 2.0) < 1e-6 and abs(out.z) < 1e-9
        assert abs(angle_of(out) - expected) < 1e-4, (target, angle_of(out))
    # Relative to a previous segment at 10 degrees the steps are 10, 55, 100 ...
    out = snap3d.lock_in_plane(origin, polar(60), up, polar(10))
    assert abs(angle_of(out) - 55) < 1e-4
    out = snap3d.lock_in_plane(origin, polar(-30), up, polar(10))
    assert abs(angle_of(out) - (-35)) < 1e-4


def test_lock_is_measured_in_the_plane_of_the_given_normal():
    # A wall facing -Y: the plane is XZ. The previous segment runs up the wall at 10 degrees.
    anchor, wall = Vector((1, 2, 3)), Vector((0, -1, 0))
    previous = Vector((math.cos(math.radians(10)), 0, math.sin(math.radians(10))))
    target = anchor + Vector((math.cos(math.radians(50)), 0.3, math.sin(math.radians(50)))) * 1.5
    out = snap3d.lock_in_plane(anchor, target, wall, previous)
    # Off-plane part of the target (y) is dropped; the angle is the nearest 45 degree step from 10.
    assert abs(out.y - anchor.y) < 1e-9
    step = out - anchor
    assert abs(math.degrees(math.atan2(step.z, step.x)) - 55) < 1e-4
    assert abs(step.length - math.hypot(target.x - anchor.x, target.z - anchor.z)) < 1e-6


def test_lock_without_a_usable_previous_segment_uses_world_x_then_world_y():
    origin = Vector((0, 0, 0))
    # First segment on the floor: relative to world X.
    out = snap3d.lock_in_plane(origin, polar(30), Vector((0, 0, 1)))
    assert abs(angle_of(out) - 45) < 1e-4
    # A previous segment along the normal carries no direction.
    out = snap3d.lock_in_plane(origin, polar(30), Vector((0, 0, 1)), Vector((0, 0, 1)))
    assert abs(angle_of(out) - 45) < 1e-4
    # Plane normal along X: world X is no reference, world Y is.
    out = snap3d.lock_in_plane(origin, Vector((0, 1, 0.3)), Vector((1, 0, 0)))
    assert abs(out.x) < 1e-9 and abs(out.z) < 1e-9 and out.y > 0.9
    assert snap3d.lock_in_plane(origin, Vector((0, 0, 5)), Vector((0, 0, 1))) is None


def test_lock_to_surface_casts_the_locked_point_back_onto_the_face():
    scene = cube_scene()
    anchor, previous = Vector((0, 0, 1)), Vector((1, 0, 0))
    hit_co = Vector((0.8, 0.3, 1))
    placement = snap3d.lock_to_surface(scene, anchor, previous, hit_co, Vector((0, 0, 1)))
    length = math.hypot(0.8, 0.3)
    assert placement.locked and placement.kind == FACE
    assert close(placement.co, (length, 0, 1)) and normal_of(placement) == (0, 0, 1)


def test_lock_that_leaves_the_face_is_skipped():
    scene = cube_scene()
    anchor, hit = Vector((0.5, 0, 1)), Vector((0.9, 0.9, 1))
    # The locked point lands at 45 degrees past the cube's edge: no surface below it.
    assert snap3d.lock_to_surface(scene, anchor, Vector((1, 0, 0)), hit, Vector((0, 0, 1))) is None


def test_place_with_shift_locks_the_surface_point_and_turns_snapping_off():
    scene, view = cube_scene(), Pinhole(EYE)
    anchor, previous = Vector((-0.5, -0.5, 1)), Vector((1, 0, 0))
    # Cursor on the top face, 10 degrees off the previous direction, close to a vertex on screen.
    target = anchor + polar(10, 1.3)
    placement = snap_at(scene, view, tuple(target), shift=True, anchor=anchor, previous_dir=previous)
    assert placement.kind == FACE and placement.locked
    assert close(placement.co, anchor + polar(0, 1.3), 1e-3)
    # Near a vertex, Shift still does not snap to it.
    near_corner = (0.95, -0.95, 1)
    assert snap_at(scene, view, near_corner).kind == VERTEX
    placement = snap_at(scene, view, near_corner, shift=True, anchor=anchor, previous_dir=previous)
    assert placement.kind == FACE
    # Without an anchor (first point) Shift only places the surface point.
    placement = snap_at(scene, view, tuple(target), shift=True)
    assert placement.kind == FACE and not placement.locked and close(placement.co, target, 1e-3)


def test_place_with_shift_uses_the_hit_when_the_lock_leaves_the_surface():
    scene, view = cube_scene(), Pinhole(EYE)
    anchor = Vector((0.5, 0, 1))
    hit = Vector((0.9, 0.9, 1))
    placement = snap_at(scene, view, tuple(hit), shift=True, anchor=anchor, previous_dir=Vector((1, 0, 0)))
    assert placement is not None and not placement.locked and close(placement.co, hit, 1e-3)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

Z = Vector((0, 0, 1))


def at(x, y, z=1.0, ref=None, kind=FACE):
    return Placement(Vector((x, y, z)), Z.copy(), kind, ref)


def test_session_first_click_writes_nothing_and_the_second_writes_both():
    s = PathSession()
    assert s.handle(("CLICK", at(0, 0))) == []
    cmds = s.handle(("CLICK", at(1, 0)))
    assert [c[0] for c in cmds] == ["ADD_VERT", "ADD_VERT", "ADD_EDGE"]
    assert cmds[0][1] == -1 and cmds[1][1] == -2 and cmds[2][1:] == (-1, -2)
    assert cmds[0][3] == Z and not s.done
    cmds = s.handle(("CLICK", at(1, 1)))
    assert [c[0] for c in cmds] == ["ADD_VERT", "ADD_EDGE"] and cmds[1][1:] == (-2, -3)


def test_session_misses_and_zero_length_segments_add_nothing():
    s = PathSession()
    assert s.handle(("CLICK", None)) == [] and not s.points
    s.handle(("CLICK", at(0, 0)))
    assert s.handle(("CLICK", None)) == []
    assert s.handle(("CLICK", at(0, 0))) == []
    assert len(s.points) == 1


def test_session_ends_with_rmb_or_enter_and_cancels_a_lone_point():
    for key in ("RMB", "ENTER"):
        s = PathSession()
        s.handle(("CLICK", at(0, 0)))
        s.handle(("CLICK", at(1, 0)))
        assert s.handle(("KEY", key)) == [("END",)] and s.done and not s.cancelled
        assert s.handle(("CLICK", at(5, 5))) == []
        lone = PathSession()
        lone.handle(("CLICK", at(0, 0)))
        assert lone.handle(("KEY", key)) == [("CANCEL",)] and lone.cancelled
    empty = PathSession()
    assert empty.handle(("KEY", "RMB")) == [("CANCEL",)]


def test_session_escape_cancels_everything():
    s = PathSession()
    s.handle(("CLICK", at(0, 0)))
    s.handle(("CLICK", at(1, 0)))
    assert s.handle(("KEY", "ESC")) == [("CANCEL",)] and s.cancelled and s.done


def test_session_c_closes_with_three_points_or_more():
    s = PathSession()
    s.handle(("CLICK", at(0, 0)))
    s.handle(("CLICK", at(1, 0)))
    assert s.handle(("KEY", "C")) == [] and not s.done
    s.handle(("CLICK", at(1, 1)))
    cmds = s.handle(("KEY", "C"))
    assert cmds == [("ADD_EDGE", -3, -1), ("END",)] and s.done and not s.cancelled


def test_session_snapping_onto_the_first_vertex_closes_the_loop():
    s = PathSession()
    s.handle(("CLICK", at(0, 0, ref=4, kind=PATH_VERTEX)))
    s.handle(("CLICK", at(1, 0)))
    # Two points: the first vertex is not a close target yet.
    assert s.handle(("CLICK", at(0, 0, ref=4, kind=PATH_VERTEX))) == []
    s.handle(("CLICK", at(1, 1)))
    cmds = s.handle(("CLICK", at(0, 0, ref=4, kind=PATH_VERTEX)))
    assert cmds == [("ADD_EDGE", -2, 4), ("END",)] and s.done


def test_session_reuses_existing_vertices_without_adding_them():
    s = PathSession()
    assert s.handle(("CLICK", at(0, 0, ref=7, kind=PATH_VERTEX))) == []
    cmds = s.handle(("CLICK", at(1, 0)))
    # The first point exists already: only the new vertex and the edge are written.
    assert cmds == [("ADD_VERT", -1, Vector((1, 0, 1)), Z), ("ADD_EDGE", 7, -1)]
    # Snapping onto the point just placed again adds nothing.
    assert s.handle(("CLICK", at(1, 0, ref=-1, kind=PATH_VERTEX))) == []


def test_session_preview_shows_the_chain_the_rubber_band_and_the_candidate():
    s = PathSession()
    assert s.preview().strips == [] and s.preview().lines == []
    s.handle(("CLICK", at(0, 0)))
    s.handle(("MOVE", at(2, 0, kind=VERTEX)))
    p = s.preview()
    assert p.strips == [[Vector((0, 0, 1)), Vector((2, 0, 1))]] and p.snap_xy == Vector((2, 0, 1))
    assert p.lines == ["Vertex"]
    s.handle(("CLICK", at(2, 0)))
    s.handle(("MOVE", Placement(Vector((3, 0, 1)), Z.copy(), FACE, None, True)))
    p = s.preview()
    assert len(p.strips) == 2 and p.strips[0] == [Vector((0, 0, 1)), Vector((2, 0, 1))]
    assert p.lines == ["On face (angle locked)"] and p.markers == [Vector((0, 0, 1)), Vector((2, 0, 1))]
    s.handle(("MOVE", None))
    assert s.preview().snap_xy is None and len(s.preview().strips) == 1


def test_session_anchor_gives_the_last_point_and_the_previous_direction():
    s = PathSession()
    assert s.anchor() == (None, None)
    s.handle(("CLICK", at(0, 0)))
    assert s.anchor() == (Vector((0, 0, 1)), None)
    s.handle(("CLICK", at(1, 2)))
    last, previous = s.anchor()
    assert last == Vector((1, 2, 1)) and previous == Vector((1, 2, 0))


def test_header_says_typed_values_are_unavailable():
    text = PathSession().header_text()
    assert "no typed values" in text and chr(0x2014) not in text


# ---------------------------------------------------------------------------
# QA round 3: evidence that fits none of the faces, abutting objects, far objects
# ---------------------------------------------------------------------------

def test_evidence_that_lies_on_none_of_the_faces_is_skipped_not_the_end_of_the_evidence():
    scene = cube_scene()
    rim = Vector((1.0, 0.0, 1.0))
    both = (Vector((0, 0, 1)), Vector((1, 0, 0)))
    eye = Vector((0.2, -0.3, 1.0)).normalized()
    on_x = Vector((1.0, 0.4, -0.5))
    on_neither = Vector((-0.5, -1.0, 0.3))
    # The eye alone takes the top.
    assert tuple(snap3d._pick_normal(both, eye)) == (0.0, 0.0, 1.0)
    # The previous point lies on neither face, the one before it on the +X face: that face.
    picked = snap3d._pick_normal(both, eye, None, rim, on_neither, [on_x])
    assert tuple(picked) == (1.0, 0.0, 0.0)
    # The same with the previous point on the +X face and an earlier one on neither: still +X.
    picked = snap3d._pick_normal(both, eye, None, rim, on_x, [on_neither])
    assert tuple(picked) == (1.0, 0.0, 0.0)
    # Evidence on none of the faces at all leaves the eye to decide.
    assert tuple(snap3d._pick_normal(both, eye, None, rim, on_neither, [on_neither])) == (0.0, 0.0, 1.0)

    # Through a run: the point on the rim is placed with the previous point on the -Y face (on
    # neither of its faces) and re-picked once the next point, on the +X face, is known.
    view = Pinhole((4.0, -4.0, 6.0))
    normals = draw_run(scene, [(view, (0.0, -1.0, 0.3), "face"), (view, (1.0, 0.0, 1.0), "snap"),
                               (view, (1.0, 0.3, -0.4), "face")])
    assert normals == [(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], normals


def test_a_run_that_follows_a_face_keeps_it_when_a_point_in_between_lies_on_none_of_the_faces():
    scene = cube_scene()
    # The reviewer's repro: the third point's two nearest neighbours do not all lie on its faces,
    # the 2-back point does, and it still decides.
    for eye in ((4, -4, 6), (5, -5, 8), (3, -5, 6)):
        view = Pinhole(eye)
        normals = draw_run(scene, [(view, (0.0, -1.0, 0.3), "face"), (view, (-0.4, -1.0, 1.0), "snap"),
                                   (view, (0.3, -1.0, 1.0), "snap"), (view, (1.0, 0.2, 0.3), "face")])
        assert normals == [(0.0, -1.0, 0.0)] * 3 + [(1.0, 0.0, 0.0)], (eye, normals)


def touching_cubes():
    return cube_scene(extra=[(CUBE_VERTS, CUBE_POLYS, Matrix.Translation((2, 0, 0)))])


def test_a_face_buried_by_an_abutting_object_is_not_front_facing():
    scene = touching_cubes()
    # The shared corner (1, -1, 1) is visible; the +X face of the left cube is under the right one.
    for eye, expected in (((8, -3, 2), (0, -1, 0)), ((8, -1, 6), (0, 0, 1)), ((-6, -4, 5), (0, 0, 1)),
                          ((3, -8, 1), (0, -1, 0))):
        view = Pinhole(eye)
        placement = snap_at(scene, view, (1, -1, 1), 1, 1)
        assert placement.kind == VERTEX and len(placement.faces) == 3
        assert normal_of(placement) == expected, (eye, normal_of(placement))
        assert (1, 0, 0) not in {tuple(round(c) for c in n) for n in placement.front}, eye
    # Without the neighbour the +X face of a lone cube is a front face, and the eye takes it.
    lone = cube_scene()
    view = Pinhole((8, -3, 2))
    placement = snap_at(lone, view, (1, -1, 1), 1, 1)
    assert (1, 0, 0) in {tuple(round(c) for c in n) for n in placement.front}
    assert normal_of(placement) == (1, 0, 0)


def test_a_buried_face_does_not_win_the_pick_along_a_segment_that_lies_on_it():
    scene = touching_cubes()
    view = Pinhole((8, -3, 2))
    # The segment along the shared top edge lies on the buried +X face and on the top: the visible
    # top is taken, at the placement and after the next point re-picks the first.
    normals = draw_run(scene, [(view, (1.0, -1.0, 1.0), "snap"), (view, (1.0, 0.0, 1.0), "snap")])
    assert normals[0] == (0.0, 0.0, 1.0), normals
    placement = snap_at(scene, view, (1, -1, 1), 1, 1, previous_point=Vector((1.0, 0.0, 1.0)))
    assert normal_of(placement) == (0, 0, 1)


def test_the_visibility_test_is_bounded_and_skipped_without_a_choice():
    scene = touching_cubes()
    view = Pinhole((8, -3, 2))
    snap = snap3d.find_snap(scene, view, view.screen((1, -1, 1)), 12)
    normals = list(scene.snap_normals(snap))
    probes = scene.snap_probes(snap)
    to_eye = (view.eye - snap.co).normalized()
    front = snap3d.front_faces(scene, view, snap.co, normals, probes, to_eye)
    assert {tuple(round(c) for c in n) for n in front} == {(0, -1, 0), (0, 0, 1)}
    # One face towards the eye: nothing to decide, no ray needed.
    assert snap3d.front_faces(scene, view, snap.co, normals, probes, Vector((-1, 0, 0))) == ()
    # No probes (a scene built from plain points): the old behaviour.
    assert snap3d.front_faces(scene, view, snap.co, normals, (), to_eye) == ()


def test_clicks_that_are_close_but_not_the_same_point_are_not_merged_far_from_the_origin():
    offset = Vector((3.7e4, -2.0e4, 1.0e3))
    scene = cube_scene(Matrix.Translation(offset))
    view = Pinhole(offset + Vector((5.0, -6.0, 4.0)), target=offset)
    corner = offset + Vector((1, -1, 1))
    # A path vertex 0.3 units from the corner is another point (it was merged within 3.7 units).
    near = [(4, corner + Vector((0.3, 0.0, 0.0)))]
    placement = snap_at(scene, view, tuple(corner), 1, 1, path=near)
    assert placement.kind == VERTEX and placement.ref is None and close(placement.co, corner, 1e-2)
    # The float32 round trip of the same point (a few ulp) still is the vertex.
    ulp = 3.7e4 * 2.0 ** -23
    same = [(4, corner + Vector((3 * ulp, 0.0, 0.0)))]
    placement = snap_at(scene, view, tuple(corner), 1, 1, path=same)
    assert placement.ref == 4
    # Directly: the limit is about 1e-6 of the coordinates (0.04 here).
    ref, co = snap3d._reuse(corner, [(1, corner + Vector((0.3, 0, 0)))])
    assert ref is None
    ref, co = snap3d._reuse(corner, [(1, corner)])
    assert ref == 1


# ---------------------------------------------------------------------------
# Concave faces, translated paths, big faces (QA round 4)
# ---------------------------------------------------------------------------

# A pit with a notch column: the floor is the 8-gon U, whose centre (1.5, 1.75) lies in the solid
# column, and the walls face into the pit.
PIT_U = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
PIT_EYES = ((-2, -2, 5), (0.3, 0.3, 5), (-4, -1, 6))


def pit_scene(matrix=None):
    n = len(PIT_U)
    verts = [(x, y, 0.0) for x, y in PIT_U] + [(x, y, 1.0) for x, y in PIT_U]
    polys = [tuple(range(n))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    polys.append((n + 4, n + 5, n + 6, n + 3))
    scene = SceneSource()
    scene.add_polygons(verts, polys, matrix or Matrix.Identity(4))
    return scene


def in_pit_floor(x, y):
    """Even-odd test against the U-shaped floor outline."""
    hit = False
    for i, (ax, ay) in enumerate(PIT_U):
        bx, by = PIT_U[(i + 1) % len(PIT_U)]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            hit = not hit
    return hit


def vertex_snap(scene, co):
    index = next(i for i, v in enumerate(scene.vertices) if close(v, co))
    return snap3d.Snap(scene.vertices[index], VERTEX, None, 0.0, index)


def test_the_probe_target_of_a_concave_face_lies_inside_the_face():
    scene = pit_scene()
    snap = vertex_snap(scene, (1, 1, 0))
    normals, probes = scene.snap_normals(snap), scene.snap_probes(snap)
    floor = next(p for n, p in zip(normals, probes) if close(n, (0, 0, 1)))[0]
    # The reflex corner of the floor: the face centre is in the solid column, the target is not.
    assert not in_pit_floor(1.5, 1.75)
    assert in_pit_floor(floor.target.x, floor.target.y) and abs(floor.target.z) < 1e-6
    assert abs(floor.size - 1.0) < 1e-6
    # An edge uses the triangle that has the edge: the notch's inner edge (1, 1) - (1, 3).
    edge = next(e for e, (a, b) in enumerate(scene.edges)
                if {tuple(scene.vertices[a]), tuple(scene.vertices[b])} == {(1.0, 1.0, 0.0), (1.0, 3.0, 0.0)})
    probe = next(p for n, p in zip(scene.edge_normals[edge], scene.edge_probes[edge]) if close(n, (0, 0, 1)))[0]
    assert in_pit_floor(probe.target.x, probe.target.y) and abs(probe.size - 2.0) < 1e-6


def test_the_floor_of_a_concave_pit_stays_a_front_face_at_the_notch_corner():
    for matrix, ortho in ((Matrix.Identity(4), False), (Matrix.Identity(4), True),
                          (Matrix.Translation((1e3, -2e3, 50)), False)):
        scene = pit_scene(matrix)
        corner = matrix @ Vector((1, 1, 0))
        for eye in PIT_EYES:
            view = Pinhole(matrix @ Vector(eye), target=corner, ortho=ortho, reach=scene.reach)
            placement = snap_at(scene, view, tuple(corner), 1, 1)
            assert placement.kind == VERTEX and close(placement.co, corner, 1e-2) and len(placement.faces) == 3
            fronts = {tuple(round(c) for c in n) for n in placement.front}
            assert (0, 0, 1) in fronts, (eye, ortho, fronts)
            assert normal_of(placement) == (0, 0, 1), (eye, ortho)
            # The front faces are stored with the point, for the re-pick of a run.
            session = PathSession()
            session.handle(("CLICK", placement))
            assert any(close(n, (0, 0, 1)) for n in session.points[0].front)


def test_a_face_that_only_the_visibility_test_removed_stays_when_the_path_lies_on_it():
    floor, wall, side = Vector((0, 0, 1)), Vector((0, -1, 0)), Vector((-1, 0, 0))
    to_eye = Vector((0, -1, 1)).normalized()
    here = Vector((0, 0, 0))
    # The path came along the floor, not along the wall: the floor stays although the probe hid it.
    assert close(snap3d._pick_normal([floor, wall], to_eye, None, here, Vector((0, 1, 0)), (), (wall,)), floor)
    # Without evidence the visibility test decides.
    assert close(snap3d._pick_normal([floor, wall], to_eye, None, here, None, (), (wall,)), wall)
    # A previous point on both faces (the segment lies on their shared edge) is no evidence.
    assert close(snap3d._pick_normal([floor, wall], to_eye, None, here, Vector((1, 0, 0)), (), (wall,)), wall)
    # Two faces removed: nothing is restored.
    to_eye3 = Vector((-1, -1, 1)).normalized()
    assert close(snap3d._pick_normal([floor, wall, side], to_eye3, None, here, Vector((0, 1, 0)), (), (wall,)), wall)


def block_and_floating_box():
    """A 40 unit block whose top corner is at the origin and a box floating 4.2 above the ground
    next to it: seen from (1, 1, 1) the box covers the block's top from 0.2 to 2 units from the
    corner, the corner itself is clear."""
    def box(lo, hi):
        xs = list(zip(lo, hi))
        verts = [(xs[0][i], xs[1][j], xs[2][k]) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
        return verts, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]

    scene = SceneSource()
    for lo, hi in (((-40, -40, -40), (0, 0, 0)), ((3, 3, 4.2), (4, 4, 5))):
        scene.add_polygons(*box(lo, hi), Matrix.Identity(4))
    return scene


def test_the_top_of_a_big_block_stays_visible_at_its_corner_in_an_orthographic_view():
    # The sight distance of an orthographic view is the size of the scene, so a fixed share of it
    # carried the probe 0.7 units off, behind the floating box.
    scene = block_and_floating_box()
    corner = Vector((0, 0, 0))
    snap = vertex_snap(scene, corner)
    assert snap.index == 7
    view = Pinhole(Vector((1, 1, 1)) * 30, ortho=True, reach=scene.reach)
    assert snap3d.is_visible(scene, view, corner)
    _, direction, _ = view.sight(corner)
    front = snap3d.front_faces(scene, view, corner, list(scene.snap_normals(snap)), scene.snap_probes(snap),
                               -direction.normalized())
    assert {tuple(round(c) for c in n) for n in front} == {(0, 0, 1), (0, 1, 0), (1, 0, 0)}
    placement = snap3d.place(scene, view, view.screen(corner), 6)
    assert placement.kind == VERTEX and (0, 0, 1) in {tuple(round(c) for c in n) for n in placement.front}


def probe_offsets(scene, view, snap):
    """(world distance, screen distance) from the snap point of every probe point front_faces tests."""
    seen = []
    real = snap3d.is_visible

    def spy(s, v, co):
        a, b = v.project(snap.co), v.project(co)
        seen.append(((co - snap.co).length, math.hypot(a[0] - b[0], a[1] - b[1])))
        return real(s, v, co)

    snap3d.is_visible = spy
    try:
        _, direction, _ = view.sight(snap.co)
        snap3d.front_faces(scene, view, snap.co, list(scene.snap_normals(snap)), scene.snap_probes(snap),
                           -direction.normalized())
    finally:
        snap3d.is_visible = real
    return seen


def test_the_probe_step_is_capped_by_the_shortest_edge_and_by_the_screen():
    # A bar 1 wide and 40 long seen from far away: the face's far corner is 40 units off, but the
    # corner has a 1 unit edge, so the probe stays within a tenth of it (times the diagonal of the
    # sideways and the upward part).
    scene = SceneSource()
    scene.add_polygons([(x, y, z) for z in (0, 1) for y in (0, 40) for x in (0, 1)],
                       [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)],
                       Matrix.Identity(4))
    snap = vertex_snap(scene, (1, 0, 1))
    for ortho in (False, True):
        view = Pinhole(Vector((90, -80, 70)), target=(0.5, 20, 0.5), ortho=ortho, reach=scene.reach, focal=200.0)
        offsets = probe_offsets(scene, view, snap)
        assert offsets, ortho
        assert max(w for w, _ in offsets) <= 0.1 * math.sqrt(2.0) + 1e-6, (ortho, offsets)
    # Faces of 40 units on a scene-sized sight line: the screen limit rules (a tenth of the edge
    # would be 4 units). The pixel count is linear, so a perspective view is a hair over.
    block = block_and_floating_box()
    snap = vertex_snap(block, (0, 0, 0))
    for ortho in (False, True):
        view = Pinhole(Vector((1, 1, 1)) * 30, ortho=ortho, reach=block.reach)
        offsets = probe_offsets(block, view, snap)
        assert offsets and max(px for _, px in offsets) <= snap3d.PROBE_PIXELS * 1.01, (ortho, offsets)
        assert max(w for w, _ in offsets) < 4.0, (ortho, offsets)


def test_a_path_vertex_beside_a_scene_vertex_on_screen_is_reused_whatever_the_float_error():
    scene = cube_scene()
    view = Pinhole(EYE)
    corner = Vector((1, -1, 1))
    # 0.0008 units are far beyond the reuse distance of a unit sized point and under a tenth of a
    # pixel at this zoom (about 100 px per unit): the same point on screen.
    for error in (Vector((0.0008, 0, 0)), Vector((0, 0.0005, 0.0005)), Vector((-0.0008, 0.0, 0.0))):
        for dx, dy in ((0, 0), (3, -2), (-4, 5)):
            placement = snap_at(scene, view, tuple(corner), dx, dy, path=[(7, corner + error)])
            assert placement.ref == 7 and close(placement.co, corner + error, 1e-9), (error, dx, dy)
            assert placement.kind == PATH_VERTEX
    # The path vertex wins even when the scene vertex is closer to the cursor.
    x, y = view.screen(corner)
    placement = snap3d.place(scene, view, (x - 0.3, y), 12, [(7, corner + Vector((0.0008, 0, 0)))])
    assert placement.ref == 7
    # The float round trip of a path at an offset of 1e4 is a few thousandths: the object's offset
    # widens the world distance, still under half a pixel on screen.
    for error in (Vector((0.004, 0, 0)), Vector((0, 0.003, 0.003))):
        placement = snap_at(scene, view, tuple(corner), 1, 1, path=[(7, corner + error)], reuse_scale=1e4)
        assert placement.ref == 7 and placement.kind == PATH_VERTEX, error
    # A path vertex a few pixels away is another point.
    placement = snap_at(scene, view, tuple(corner), 0, 0, path=[(7, corner + Vector((0.05, 0, 0)))])
    assert placement.ref is None and placement.kind == VERTEX
    # A path vertex inside the cube, on the sight line through the corner, shows at the same pixel
    # but is hidden: it does not capture the visible scene vertex.
    behind = corner + (corner - view.eye).normalized() * 0.5
    placement = snap_at(scene, view, tuple(corner), 1, 1, path=[(7, behind)])
    assert placement.ref is None and placement.kind == VERTEX


def test_the_reuse_distance_scales_with_the_offset_of_the_path_object():
    corner = Vector((1, -1, 1))
    off_by = corner + Vector((0.003, 0.0, 0.0))
    # 3e-3 is a float32 round trip at an offset of 1e4 but a new point near the origin.
    assert snap3d._reuse(corner, [(4, off_by)])[0] is None
    assert snap3d._reuse(corner, [(4, off_by)], 1e4)[0] == 4
    assert snap3d._reuse(corner, [(4, corner + Vector((0.2, 0, 0)))], 1e4)[0] is None


# ---------------------------------------------------------------------------
# Probes of faces with a vertex on an edge, coplanar faces, depth and far coordinates
# ---------------------------------------------------------------------------

def inside_2d(polygon, x, y):
    hit = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            hit = not hit
    return hit


def test_the_targets_of_a_face_with_a_vertex_on_an_edge_skip_the_flat_triangle():
    # (1, 0) lies on the side from (0, 0) to (2, 0): the triangulation has the flat triangle
    # (0, 1, 2) next to the real one (0, 2, 3). The flat one has the largest angle (180 degrees) at
    # the middle corner and its centroid is on the boundary.
    outline = [(0, 0), (1, 0), (2, 0), (1, 0.5)]
    points = [Vector((x, y, 0.0)) for x, y in outline]
    for triangles in ([(0, 1, 2), (0, 2, 3)], snap3d.triangulate(points)):
        corners, sides = snap3d.face_probes(points, triangles)
        for probe in corners + sides:
            assert probe.target.y > 0.1 and inside_2d(outline, probe.target.x, probe.target.y), probe
    corners, sides = snap3d.face_probes(points, [(0, 1, 2), (0, 2, 3)])
    assert all(close(p.target, (1, 1 / 6, 0)) for p in corners + sides)
    # Sizes stay the polygon's own edges.
    assert abs(corners[1].size - 1.0) < 1e-9 and abs(sides[0].size - 1.0) < 1e-9
    # A face with nothing but flat triangles falls back to its vertex mean.
    line = [Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((2, 0, 0))]
    corners, sides = snap3d.face_probes(line, [(0, 1, 2)])
    assert all(close(p.target, (1, 0, 0)) for p in corners + sides)


def test_a_flat_triangle_of_short_edges_is_flat_whatever_the_float32_error_of_its_corners():
    # Three points on a line 0.02 long at coordinates near 1: the float32 rounding of the corners
    # alone makes a triangle that is 1e-7 thick, above 1e-6 of the squared side.
    first, last = Vector((-0.981, -0.1939, 0.0)), Vector((-0.9788, -0.2049, 0.0))
    middle = first.lerp(last, 0.37) + Vector((0.0, 1e-7, 0.0))
    points = [first, middle, last, Vector((0.5, 0.5, 0.0))]
    corners, sides = snap3d.face_probes(points, [(0, 1, 2), (0, 2, 3)])
    for probe in corners + sides:
        assert probe.target.y > 0.01, probe


def test_the_corner_target_is_in_the_triangle_with_the_largest_angle_there():
    # A convex quad cut from corner 0: the first triangle has a 14 degree angle at the corner, the
    # second one 76 degrees.
    points = [Vector(p) for p in ((0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 3, 0))]
    corners, _ = snap3d.face_probes(points, [(0, 1, 2), (0, 2, 3)])
    assert close(corners[0].target, (4 / 3, 4 / 3, 0), 1e-6), corners[0].target
    # The same cut listed the other way round, and corner 2, where the first triangle has the
    # larger angle (76 degrees against 41).
    corners, _ = snap3d.face_probes(points, [(0, 2, 3), (0, 1, 2)])
    assert close(corners[0].target, (4 / 3, 4 / 3, 0), 1e-6)
    assert close(corners[2].target, (8 / 3, 1 / 3, 0), 1e-6), corners[2].target


def box_mesh(lo, hi):
    xs = list(zip(lo, hi))
    verts = [(xs[0][i], xs[1][j], xs[2][k]) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return verts, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


# A slab 2 x 1 x 1 whose top is two quads (Q1 over x 0 to 1, Q2 over x 1 to 2) and whose front is
# two quads as well; the snap point is the vertex (1, 0, 1) shared by all four.
SLAB_VERTS = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (2, 0, 1), (2, 1, 1), (1, 1, 1), (0, 1, 1)]
SLAB_Q1, SLAB_Q2 = (6, 7, 10, 11), (7, 8, 9, 10)
SLAB_SIDES = [(0, 1, 7, 6), (1, 2, 8, 7), (0, 5, 4, 3, 2, 1), (3, 5, 11, 9), (5, 0, 6, 11), (2, 3, 9, 8)]


def slab_scene(top, box_hi_x):
    scene = SceneSource()
    scene.add_polygons(SLAB_VERTS, list(top) + SLAB_SIDES, Matrix.Identity(4))
    scene.add_polygons(*box_mesh((0, 0, 1), (box_hi_x, 1, 2)), Matrix.Identity(4))
    return scene


def test_a_normal_shared_by_coplanar_faces_is_visible_when_any_of_them_is():
    corner = Vector((1, 0, 1))
    for top in ((SLAB_Q1, SLAB_Q2), (SLAB_Q2, SLAB_Q1)):
        # A box flush on Q1 only: Q2's top is in view, whatever the polygon order.
        scene = slab_scene(top, 1.0)
        snap = vertex_snap(scene, corner)
        normals = scene.snap_normals(snap)
        up = next(i for i, n in enumerate(normals) if close(n, (0, 0, 1)))
        assert len(scene.snap_probes(snap)[up]) == 2, top
        for eye in ((2, -2, 5), (4, -1, 5), (3, -3, 3)):
            for ortho in (False, True):
                view = Pinhole(eye, target=corner, ortho=ortho, reach=scene.reach)
                placement = snap3d.snap_placement(scene, view, snap)
                fronts = {tuple(round(c) for c in n) for n in placement.front}
                assert fronts == {(0, 0, 1), (0, -1, 0)}, (top, eye, ortho, fronts)
        # A box over the whole top buries both faces of the normal.
        scene = slab_scene(top, 2.0)
        snap = vertex_snap(scene, corner)
        for eye in ((2, -2, 5), (3, -3, 3)):
            for ortho in (False, True):
                view = Pinhole(eye, target=corner, ortho=ortho, reach=scene.reach)
                placement = snap3d.snap_placement(scene, view, snap)
                assert {tuple(round(c) for c in n) for n in placement.front} == {(0, -1, 0)}, (top, eye, ortho)
                assert normal_of(placement) == (0, -1, 0)


def test_a_path_vertex_far_behind_a_scene_vertex_on_the_same_pixel_does_not_capture_the_click():
    # A thin tall post and a wall behind it: the wall point shows 0.4 px beside the post's corner.
    scene = SceneSource()
    scene.add_polygons(CUBE_VERTS, CUBE_POLYS, Matrix.Diagonal((0.1, 0.1, 1.0, 1)))
    scene.add_polygons([(-5, 5, -5), (5, 5, -5), (5, 5, 5), (-5, 5, 5)], [(0, 1, 2, 3)], Matrix.Identity(4))
    view = Pinhole((0.0, -10, 0.0), target=(0, 0, 0), ortho=True, reach=scene.reach)
    corner, wall = Vector((0.1, -0.1, 1.0)), Vector((0.104, 5.0, 1.0))
    assert snap3d.is_visible(scene, view, corner) and snap3d.is_visible(scene, view, wall)
    x, y = view.screen(corner)
    wx, wy = view.screen(wall)
    assert math.hypot(wx - x, wy - y) <= snap3d.OWN_PIXELS
    # The cursor is nearer to the corner than to the wall point on screen.
    mouse = (x - 1.0, y)
    placement = snap3d.place(scene, view, mouse, 12, [(5, wall)])
    assert placement.kind == VERTEX and placement.ref is None and close(placement.co, corner, 1e-3)
    # A vertex a hair beside the corner (within the world distance) is still reused.
    near = corner + Vector((0.0004, 0.0004, 0.0))
    placement = snap3d.place(scene, view, mouse, 12, [(5, near)])
    assert placement.kind == PATH_VERTEX and placement.ref == 5
    # The allowed distance grows with the path object's offset from the origin.
    deep = corner + Vector((0.004, 0.5, 0.0))
    assert snap3d.place(scene, view, mouse, 12, [(5, deep)]).ref is None
    assert snap3d.place(scene, view, mouse, 12, [(5, deep)], reuse_scale=1e4).ref == 5


def test_a_close_up_far_from_the_origin_does_not_call_a_visible_face_hidden():
    import random
    rng = random.Random(3)
    cases = bad = 0
    # A cube of 0.01 units 1e4 from the origin seen from 0.06 to 0.12 units: the sight line is
    # shorter than the float32 error of the coordinates (about 1e-3).
    matrix = Matrix.Translation((1e4, 5e3, -1e4)) @ Matrix.Diagonal((0.01, 0.01, 0.01, 1))
    scene = cube_scene(matrix)
    centre = matrix @ Vector((0, 0, 0))
    for _ in range(200):
        elevation, azimuth = math.radians(rng.uniform(10, 80)), rng.uniform(0, 6.28)
        direction = Vector((math.cos(azimuth) * math.cos(elevation), math.sin(azimuth) * math.cos(elevation),
                            math.sin(elevation)))
        view = Pinhole(centre + direction * rng.uniform(0.06, 0.12), target=centre)
        i = rng.randrange(len(scene.vertices))
        snap = snap3d.Snap(scene.vertices[i], VERTEX, None, 0.0, i)
        _, sight, _ = view.sight(snap.co)
        to_eye = -sight.normalized()
        normals = list(scene.snap_normals(snap))
        facing = [n for n in normals if n.dot(to_eye) > 1e-6]
        if len(facing) < 2 or not snap3d.is_visible(scene, view, snap.co):
            continue
        cases += 1
        front = snap3d.front_faces(scene, view, snap.co, normals, scene.snap_probes(snap), to_eye)
        bad += len(front) != len(facing)
    assert cases > 40 and bad == 0, (cases, bad)


# ---------------------------------------------------------------------------
# Probe selection, ray budget and constants (QA round 6)
# ---------------------------------------------------------------------------

def counted_rays(scene):
    """A one item list that counts the ray casts the scene answers from now on."""
    count = [0]
    cast = scene.ray_cast

    def counting(origin, direction):
        count[0] += 1
        return cast(origin, direction)

    scene.ray_cast = counting
    return count


def front_of(scene, view, snap):
    """(normals, front faces) of a scene vertex snap, as snap_placement asks for them."""
    _, sight, _ = view.sight(snap.co)
    normals = list(scene.snap_normals(snap))
    return normals, snap3d.front_faces(scene, view, snap.co, normals, scene.snap_probes(snap),
                                       -sight.normalized())


def key_of(normal):
    return tuple(round(c, 5) for c in normal)


def fan_scene(count, rim_radii, buried, order_seed):
    """A fan of `count` coplanar triangles over 270 degrees around the origin (normal +Z), a tilted
    triangle over the remaining 90 degrees (a second normal that faces an eye above), and a lid
    at z = 0.5 over the fan sectors listed in buried. The polygons are listed in a shuffled order.
    The snap vertex is scene vertex 0, the origin."""
    rim = [Vector((math.cos(1.5 * math.pi * k / count), math.sin(1.5 * math.pi * k / count), 0.0))
           * rim_radii[k] for k in range(count + 1)]
    verts = [(0.0, 0.0, 0.0)] + [tuple(p) for p in rim] + [(2.0, 0.0, -1.0)]
    polys = [(0, 1 + k, 2 + k) for k in range(count)] + [(0, count + 1, count + 2)]
    random.Random(order_seed).shuffle(polys)
    scene = SceneSource()
    scene.add_polygons(verts, polys, Matrix.Identity(4))
    for k in buried:
        a, b = rim[k] * 1.2, rim[k + 1] * 1.2
        scene.add_polygons([(0.0, 0.0, 0.5), (a.x, a.y, 0.5), (b.x, b.y, 0.5)], [(0, 1, 2)],
                           Matrix.Identity(4))
    return scene


def up_in(front):
    """Whether the +Z normal of the fan is among the front faces."""
    return any(n.z > 0.999 for n in front)


FAN_EYE = (0.001, -0.002, 12.0)


def test_the_probes_tested_for_a_normal_do_not_depend_on_the_order_of_its_faces():
    # Most of a flat patch is buried: with more than MAX_PROBES_PER_NORMAL faces, the first four in
    # polygon order can all be under the lid. Two neighbouring sectors, nearer to the vertex than
    # the rest, are in view; the normal is visible whatever the order.
    view = Pinhole(FAN_EYE, target=(0, 0, 0))
    for count in (8, 12):
        for visible in range(count - 1):
            radii = [1.0 if visible <= k <= visible + 2 else 2.0 for k in range(count + 1)]
            buried = [k for k in range(count) if k not in (visible, visible + 1)]
            assert len(buried) == count - 2
            for seed in range(12):
                scene = fan_scene(count, radii, buried, seed)
                snap = vertex_snap(scene, (0, 0, 0))
                normals, front = front_of(scene, view, snap)
                up = next(i for i, n in enumerate(normals) if n.z > 0.999)
                assert len(scene.snap_probes(snap)[up]) == count
                assert len(front) == 2 and up_in(front), (count, visible, seed)
        # Everything buried: hidden whatever the order, and the tilted normal stays.
        for seed in range(12):
            scene = fan_scene(count, [2.0] * (count + 1), range(count), seed)
            normals, front = front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
            assert len(front) == 1 and not up_in(front), (count, seed)


def test_equidistant_probes_are_ordered_by_their_coordinates_not_by_the_order_of_the_faces():
    # All sectors at the same distance: the probes are told apart by their coordinates alone, and
    # the verdict (visible or hidden) is the same for every order of the polygons.
    view = Pinhole(FAN_EYE, target=(0, 0, 0))
    for count, visible in ((8, 3), (12, 0), (12, 5), (12, 10)):
        verdicts = set()
        for seed in range(40):
            buried = [k for k in range(count) if k not in (visible, visible + 1)]
            scene = fan_scene(count, [2.0] * (count + 1), buried, seed)
            normals, front = front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
            verdicts.add(up_in(front))
        assert len(verdicts) == 1, (count, visible, verdicts)


def test_a_normal_gets_at_most_four_probes_and_the_nearest_is_always_one_of_them():
    view = Pinhole(FAN_EYE, target=(0, 0, 0))
    for count in (8, 12):
        # All buried: the up normal is tested with exactly 4 rays, the tilted one with 1.
        scene = fan_scene(count, [2.0] * (count + 1), range(count), 0)
        rays = counted_rays(scene)
        front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
        assert rays[0] == 4 + 1, (count, rays[0])
    # Fewer probes than the cap are all used: 3 sectors, none in view, 3 rays plus 1.
    scene = fan_scene(3, [2.0] * 4, range(3), 0)
    rays = counted_rays(scene)
    front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
    assert rays[0] == 3 + 1, rays[0]
    # Of 12 sectors only the nearest two are in view: one of them is among the probes tested,
    # whichever sector and order, and its being visible ends the tests (1 ray plus the tilted one).
    for visible in range(11):
        radii = [2.0] * 13
        radii[visible] = radii[visible + 1] = radii[visible + 2] = 1.0
        scene = fan_scene(12, radii, [k for k in range(12) if k not in (visible, visible + 1)], visible)
        rays = counted_rays(scene)
        normals, front = front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
        assert up_in(front) and rays[0] == 1 + 1, (visible, rays[0])


def test_the_probes_tested_include_the_farthest_when_the_visible_sectors_are_the_farthest():
    # The mirror of the first test: the three rim points of the visible pair are the farthest from
    # the vertex, so the pair's probes are the farthest two of the normal. A selection of the
    # nearest four only (or of the nearest plus the three farthest, or the three nearest plus the
    # farthest) misses them.
    view = Pinhole(FAN_EYE, target=(0, 0, 0))
    for count in (8, 12):
        for visible in range(count - 1):
            radii = [3.0 if visible <= k <= visible + 2 else 1.0 for k in range(count + 1)]
            buried = [k for k in range(count) if k not in (visible, visible + 1)]
            for seed in range(6):
                scene = fan_scene(count, radii, buried, seed)
                normals, front = front_of(scene, view, vertex_snap(scene, (0, 0, 0)))
                assert len(front) == 2 and up_in(front), (count, visible, seed)


def test_the_probes_tested_are_spread_over_the_distance_order_at_the_rounded_ranks():
    # Sector k is the k-th nearest to the vertex (the radii grow with k), and only sector s is in
    # view. Of n probes the ones tested are those at the sorted indices round(j * (n - 1) / 3) for
    # j in 0..3: the nearest, the farthest and two evenly spaced between. So the normal is found
    # exactly when s is one of those ranks, which also pins the known limit that a lone visible
    # probe at any other rank is missed (and that the ranks are rounded, not truncated).
    view = Pinhole(FAN_EYE, target=(0, 0, 0))
    for count in (8, 12):
        radii = [1.0 + 0.1 * k for k in range(count + 1)]
        ranks = {round(j * (count - 1) / 3) for j in range(4)}
        assert len(ranks) == 4 and ranks != {int(j * (count - 1) / 3) for j in range(4)}
        for sector in range(count):
            for seed in range(4):
                buried = [k for k in range(count) if k != sector]
                scene = fan_scene(count, radii, buried, seed)
                snap = vertex_snap(scene, (0, 0, 0))
                normals = list(scene.snap_normals(snap))
                up = next(i for i, n in enumerate(normals) if n.z > 0.999)
                # The precondition: the probes sorted by distance are the sectors in order.
                targets = sorted((p.target for p in scene.snap_probes(snap)[up]), key=lambda t: t.length)
                assert [int((math.atan2(t.y, t.x) % math.tau) / (1.5 * math.pi / count))
                        for t in targets] == list(range(count)), (count, seed)
                _, front = front_of(scene, view, snap)
                assert up_in(front) == (sector in ranks), (count, sector, seed, sorted(ranks))


def cone_scene(count, order_seed):
    """An upright cone with `count` triangles, tip at (0, 0, 1) as scene vertex 0, and a lid at
    z = 1.5 over its y > 0 half."""
    verts = [(0.0, 0.0, 1.0)] + [(math.cos(2 * math.pi * k / count), math.sin(2 * math.pi * k / count), 0.0)
                                 for k in range(count)]
    polys = [(0, 1 + k, 1 + (k + 1) % count) for k in range(count)]
    random.Random(order_seed).shuffle(polys)
    scene = SceneSource()
    scene.add_polygons(verts, polys, Matrix.Identity(4))
    scene.add_polygons([(-3, 0.0, 1.5), (3, 0.0, 1.5), (3, 3, 1.5), (-3, 3, 1.5)], [(0, 1, 2, 3)],
                       Matrix.Identity(4))
    return scene


def test_the_ray_budget_goes_to_the_most_frontal_normals_whatever_the_order():
    view = Pinhole((0.3, 0.2, 12.0), target=(0, 0, 0))
    for count in (24, 40):
        results = set()
        reference = None
        for seed in range(25):
            scene = cone_scene(count, seed)
            snap = vertex_snap(scene, (0, 0, 1))
            _, sight, _ = view.sight(snap.co)
            to_eye = -sight.normalized()
            if reference is None:
                # The same scene with a budget that never runs out.
                budget = snap3d.MAX_PROBE_RAYS
                snap3d.MAX_PROBE_RAYS = 1000
                try:
                    normals, full = front_of(scene, view, snap)
                finally:
                    snap3d.MAX_PROBE_RAYS = budget
                reference = {key_of(n) for n in full}
                frontal = sorted(normals, key=lambda n: (-n.dot(to_eye), n.x, n.y, n.z))
                top = [key_of(n) for n in frontal[:16]]
                assert len(normals) == count and all(n.z > 0.0 and n.dot(to_eye) > 0.0 for n in normals)
                assert 3 < sum(1 for k in top if k not in reference) < 16, "the lid buries some of them"
            rays = counted_rays(scene)
            normals, front = front_of(scene, view, snap)
            assert rays[0] == 16, (count, seed, rays[0])
            shown = {key_of(n) for n in front}
            # The 16 most frontal normals are tested: buried ones are out, visible ones are in.
            assert all((k in shown) == (k in reference) for k in top), (count, seed)
            # The rest was not tested and counts as visible.
            assert all(key_of(n) in shown for n in normals if key_of(n) not in top)
            results.add(frozenset(shown))
        assert len(results) == 1, (count, len(results))


def test_a_thin_sliver_face_keeps_its_triangles():
    # 40 long and 1e-3 or 1e-4 wide: thin but real triangles (area above 1e-6 of the squared
    # longest side plus the float32 part), so the targets are the centroids of the two triangles
    # and not the vertex mean, which is also inside the strip.
    for width in (1e-3, 1e-4):
        points = [Vector(p) for p in ((0, 0, 0), (40, 0, 0), (40, width, 0), (0, width, 0))]
        corners, sides = snap3d.face_probes(points, [(0, 1, 2), (0, 2, 3)])
        lower, upper = (80 / 3, width / 3), (40 / 3, 2 * width / 3)
        for probe, want in zip(corners + sides, [upper, lower, lower, upper, lower, lower, upper, upper]):
            assert abs(probe.target.x - want[0]) < 1e-4 and abs(probe.target.y - want[1]) < width * 1e-2, \
                (width, probe.target, want)
            assert 0.0 < probe.target.y < width


def test_a_flat_triangle_is_dropped_by_its_area_alone():
    # Three points 2 apart with the middle one 1e-6 off the line: the triangle's area (1e-6) is far
    # below 1e-6 of the squared side (4e-6) and above the float32 part (8e-7), so only the area
    # rule drops it. Kept, its centroid on the boundary would be the target of the middle corner.
    points = [Vector(p) for p in ((0, 0, 0), (1, 1e-6, 0), (2, 0, 0), (1, 1, 0))]
    corners, sides = snap3d.face_probes(points, [(0, 1, 2), (0, 2, 3)])
    for probe in corners + sides:
        assert abs(probe.target.y - 1 / 3) < 1e-6, probe


def test_a_path_vertex_beside_a_scene_vertex_is_captured_only_within_the_world_distance():
    scene = SceneSource()
    scene.add_polygons(CUBE_VERTS, CUBE_POLYS, Matrix.Diagonal((0.1, 0.1, 1.0, 1)))
    scene.add_polygons([(-5, 5, -5), (5, 5, -5), (5, 5, 5), (-5, 5, 5)], [(0, 1, 2, 3)], Matrix.Identity(4))
    view = Pinhole((0.0, -10, 0.0), target=(0, 0, 0), ortho=True, reach=scene.reach)
    corner = Vector((0.1, -0.1, 1.0))
    x, y = view.screen(corner)
    mouse = (x - 1.0, y)
    # Beside the post, 0.4 px from the corner on screen and 6.4e-3 from it in the world.
    for offset, scale, captured in (((0.004, 0.005, 0.0), 0.0, False),     # 1e-3 at unit scale
                                    ((0.004, 0.005, 0.0), 1e3, True),      # 1e-4 of the offset: 0.1
                                    ((0.004, 0.05, 0.0), 1e3, True),
                                    ((0.004, 0.5, 0.0), 1e3, False),
                                    ((0.004, 0.5, 0.0), 1e4, True),        # 1.0
                                    ((0.0004, 0.0004, 0.0), 0.0, True)):
        near = corner + Vector(offset)
        assert snap3d.is_visible(scene, view, near)
        assert math.hypot(*(a - b for a, b in zip(view.screen(near), (x, y)))) <= snap3d.OWN_PIXELS
        placement = snap3d.place(scene, view, mouse, 12, [(5, near)], reuse_scale=scale)
        assert (placement.ref == 5) == captured, (offset, scale, placement.ref)
        assert placement.kind == (PATH_VERTEX if captured else VERTEX)
