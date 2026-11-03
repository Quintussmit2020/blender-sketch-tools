import math
from types import SimpleNamespace

from mathutils import Matrix, Vector

from sketch_tools.builders.sweep import build_sweep
from sketch_tools.builders.sweep3d import build_sweep_3d
from sketch_tools.tests.test_extrude import signed_volume
from sketch_tools.tests.test_resolve import square
from sketch_tools.tests.test_sweep import (
    check_closed, circle, half_round, path, rect, ring, tube,
)

S2 = math.sqrt(0.5)
UP = (0.0, 0.0, 1.0)


def path3d(points, normals=None, closed=False):
    """Plain-data ResolvedPath. normals default to +Z for every point."""
    normals = normals if normals is not None else [UP] * len(points)
    chain = SimpleNamespace(
        points=[Vector(p) for p in points], normals=[Vector(n) for n in normals], closed=closed
    )
    return SimpleNamespace(chains=[chain], warnings=[])


def sweep3d(resolved, rings, limit=4.0, smooth=False, caps=True):
    bm, warnings = build_sweep_3d(resolved, rings, limit, smooth, caps)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.normal_update()
    return bm, warnings


def planar_pair(points, closed, rings, limit=4.0, smooth=False, caps=True):
    """The planar sweep and the 3D sweep of the same XY path."""
    a, wa = build_sweep(path(points, closed), rings, limit, smooth, caps)
    b, wb = build_sweep_3d(path3d([(x, y, 0.0) for x, y in points], closed=closed), rings, limit, smooth, caps)
    for bm in (a, b):
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
    return a, wa, b, wb


def canonical(face):
    """Vertex indices rotated to start at the smallest: the cyclic order without the start."""
    idx = [v.index for v in face.verts]
    k = idx.index(min(idx))
    return tuple(idx[k:] + idx[:k])


def station_map(rings, station_count):
    """Station number of every vertex of a sweep that has all rings at each station."""
    per_station = sum(len(r.points) for r in rings)
    return [s for s in range(station_count) for _ in range(per_station)]


def assert_same_mesh(a, b, stations, tol=1e-6):
    """Same vertices, wall faces and shading. stations[i] is vertex i's station. Cap faces lie in
    one station; the diagonals that dissolve_limit leaves in a holed cap differ from call to
    call even in the planar builder, so caps are compared by the area they cover and their
    smooth flag."""
    assert (len(a.verts), len(b.verts)) == (len(stations), len(stations))
    for va, vb in zip(a.verts, b.verts):
        assert (va.co - vb.co).length < tol, (va.index, tuple(va.co), tuple(vb.co))

    def split(bm):
        walls, caps = {}, {}
        for f in bm.faces:
            ids = {stations[v.index] for v in f.verts}
            if len(ids) == 1:
                caps.setdefault(next(iter(ids)), []).append(f)
            else:
                walls[canonical(f)] = f.smooth
        return walls, caps

    walls_a, caps_a = split(a)
    walls_b, caps_b = split(b)
    assert walls_a == walls_b
    assert set(caps_a) == set(caps_b)
    for k in caps_a:
        assert abs(sum(f.calc_area() for f in caps_a[k]) - sum(f.calc_area() for f in caps_b[k])) < 1e-5
        assert {f.smooth for f in caps_a[k]} == {f.smooth for f in caps_b[k]}

    def wall_edges(bm):
        return {
            tuple(sorted(v.index for v in e.verts)): e.smooth
            for e in bm.edges
            if any(len({stations[v.index] for v in f.verts}) > 1 for f in e.link_faces)
        }

    assert wall_edges(a) == wall_edges(b)


PROFILES = {
    "square": [rect(-0.5, 0.5, 0.0, 1.0)],
    "centred_square": [ring([(0.4, -0.3), (0.4, 0.3), (-0.4, 0.3), (-0.4, -0.3)])],
    "cylinder": [circle(0.4, 12, cv=0.4)],
    "half_round": [half_round(0.5, 8)],
    "tube": tube(0.5, 0.3, 12, 0.5),
    "open_strip": [ring([(-1.0, 0.0), (0.0, 0.5), (1.0, 0.0)], closed=False, depth=-1)],
}

PATHS = {
    "square_closed": (square(0, 0, 4), True),
    "square_closed_cw": (square(0, 0, 4, ccw=False), True),
    "l_open": ([(0, 0), (3, 0), (3, 3)], False),
    "zigzag_open": ([(0, 0), (2, 1), (4, -1), (6, 0), (6, 3)], False),
    "straight_open": ([(0, 0), (1, 0), (2, 0), (5, 0)], False),
    "acute_open": ([(0, 0), (5, 0), (5 + 5 * math.cos(math.radians(160)), 5 * math.sin(math.radians(160)))], False),
    "acute_closed": ([(0, 0), (6, 0), (0.5, 0.4)], True),
    "reversal_open": ([(0, 0), (3, 0), (1, 0)], False),
    "two_point": ([(0, 0), (4, 1)], False),
    "pentagon_closed": ([(3 * math.cos(2 * math.pi * k / 5), 3 * math.sin(2 * math.pi * k / 5)) for k in range(5)], True),
}


def test_planar_equivalence_all_profiles_and_paths():
    checked = 0
    for pname, (pts, closed) in PATHS.items():
        for rname, rings in PROFILES.items():
            for smooth in (False, True):
                for caps in (False, True):
                    a, wa, b, wb = planar_pair(pts, closed, rings, smooth=smooth, caps=caps)
                    assert wa == wb, (pname, rname)
                    assert_same_mesh(a, b, station_map(rings, len(pts)))
                    a.free()
                    b.free()
                    checked += 1
    assert checked == len(PATHS) * len(PROFILES) * 4


def test_planar_equivalence_mitre_limits():
    for limit in (1.0, 1.2, 1.5, 2.0, 4.0, 10.0):
        for pname in ("acute_open", "acute_closed", "square_closed", "zigzag_open"):
            pts, closed = PATHS[pname]
            a, wa, b, wb = planar_pair(pts, closed, PROFILES["square"], limit=limit)
            assert wa == wb, (pname, limit)
            assert_same_mesh(a, b, station_map(PROFILES["square"], len(pts)))
            a.free()
            b.free()


def test_planar_equivalence_multiple_chains():
    resolved = SimpleNamespace(chains=[
        SimpleNamespace(points=[Vector(p) for p in square(0, 0, 4)], closed=True, depth=0, parent=None, children=[]),
        SimpleNamespace(points=[Vector((10, 0)), Vector((13, 0)), Vector((13, 2))], closed=False, depth=-1, parent=None, children=[]),
    ], warnings=[])
    resolved3 = SimpleNamespace(chains=[
        SimpleNamespace(points=[Vector((x, y, 0)) for x, y in square(0, 0, 4)], normals=[Vector(UP)] * 4, closed=True),
        SimpleNamespace(points=[Vector((10, 0, 0)), Vector((13, 0, 0)), Vector((13, 2, 0))], normals=[Vector(UP)] * 3, closed=False),
    ], warnings=[])
    a, wa = build_sweep(resolved, PROFILES["tube"], 4.0, True, True)
    b, wb = build_sweep_3d(resolved3, PROFILES["tube"], 4.0, True, True)
    for bm in (a, b):
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
    assert wa == wb
    assert_same_mesh(a, b, station_map(PROFILES["tube"], 7))
    a.free()
    b.free()


def test_rotated_planar_path_matches_rotated_planar_sweep():
    """A path in any plane, with normals along that plane's normal, is the planar sweep rotated."""
    rotations = (
        Matrix.Rotation(math.radians(37.0), 3, Vector((1.0, 2.0, 0.5))),
        Matrix.Rotation(math.radians(90.0), 3, 'X'),
        Matrix.Rotation(math.radians(-123.0), 3, Vector((0.3, -1.0, 0.7))),
    )
    for rot in rotations:
        for pname in ("acute_open", "zigzag_open", "square_closed", "acute_closed"):
            pts, closed = PATHS[pname]
            for rname in ("square", "tube", "cylinder"):
                for limit in (1.5, 4.0):
                    a, wa = build_sweep(path(pts, closed), PROFILES[rname], limit, True, True)
                    resolved = path3d(
                        [rot @ Vector((x, y, 0.0)) for x, y in pts],
                        [rot @ Vector(UP)] * len(pts),
                        closed,
                    )
                    b, wb = build_sweep_3d(resolved, PROFILES[rname], limit, True, True)
                    assert wa == wb, (pname, rname, limit)
                    assert len(a.verts) == len(b.verts) and len(a.faces) == len(b.faces)
                    a.verts.ensure_lookup_table()
                    b.verts.ensure_lookup_table()
                    for va, vb in zip(a.verts, b.verts):
                        assert (rot @ va.co - vb.co).length < 1e-5
                    a.free()
                    b.free()


# Cube [-1, 1]^3. The profile's bottom (v = 0) is the surface, 0.4 wide, 0.3 tall.
PROFILE_ON_SURFACE = [rect(-0.2, 0.2, 0.0, 0.3)]


def bottom_vertices(bm, station):
    base = station * 4
    return [bm.verts[base], bm.verts[base + 1]]


def test_path_up_a_face_and_across_the_top():
    points = [(0, -1, -0.6), (0, -1, 0.3), (0, -1, 1.0), (0, 0, 1.0), (0, 0.6, 1.0)]
    front, top = (0, -1, 0), UP
    for edge_normal in ((0, -S2, S2), front, top):
        normals = [front, front, edge_normal, top, top]
        bm, warnings = sweep3d(path3d(points, normals), PROFILE_ON_SURFACE, caps=True)
        assert warnings == []
        check_closed(bm)
        assert signed_volume(bm) > 0.0
        for station in (0, 1):
            for v in bottom_vertices(bm, station):
                assert abs(v.co.y + 1.0) < 1e-6
        for v in bottom_vertices(bm, 2):
            assert abs(v.co.y + 1.0) < 1e-6 and abs(v.co.z - 1.0) < 1e-6
        for station in (3, 4):
            for v in bottom_vertices(bm, station):
                assert abs(v.co.z - 1.0) < 1e-6
        # The top ring follows the outer offset surface: y = -1.3 on the front, z = 1.3 on top.
        for station, axis, value in ((0, "y", -1.3), (1, "y", -1.3), (3, "z", 1.3), (4, "z", 1.3)):
            for i in (2, 3):
                assert abs(getattr(bm.verts[station * 4 + i].co, axis) - value) < 1e-6
        corner_top = bm.verts[2 * 4 + 2].co
        assert abs(corner_top.y + 1.3) < 1e-6 and abs(corner_top.z - 1.3) < 1e-6
        # Front strip (z -0.6 to 1.0) + the 0.3 square corner + top strip (y -1.0 to 0.6), times 0.4.
        expected = 0.4 * (0.3 * 1.6 + 0.09 + 0.3 * 1.6)
        assert abs(signed_volume(bm) - expected) < 1e-5, (edge_normal, signed_volume(bm))
        bm.free()


def test_closed_loop_around_four_cube_faces():
    items = [
        ((1, 0, 0), (1, 0, 0)), ((1, 1, 0), (S2, S2, 0)), ((0, 1, 0), (0, 1, 0)), ((-1, 1, 0), (-S2, S2, 0)),
        ((-1, 0, 0), (-1, 0, 0)), ((-1, -1, 0), (-S2, -S2, 0)), ((0, -1, 0), (0, -1, 0)), ((1, -1, 0), (S2, -S2, 0)),
    ]
    for start in range(8):
        rotated = items[start:] + items[:start]
        points = [p for p, _n in rotated]
        normals = [n for _p, n in rotated]
        for smooth in (False, True):
            bm, warnings = sweep3d(path3d(points, normals, closed=True), PROFILE_ON_SURFACE, smooth=smooth)
            assert warnings == []
            assert (len(bm.verts), len(bm.faces)) == (8 * 4, 8 * 4)
            check_closed(bm)
            # The band between the cube's square (half width 1) and the 1.3 square, 0.4 tall.
            assert abs(signed_volume(bm) - 0.4 * (2.6 ** 2 - 2.0 ** 2)) < 1e-4
            for k in range(8):
                for v in bottom_vertices(bm, k):
                    assert max(abs(v.co.x), abs(v.co.y)) - 1.0 < 1e-6 and max(abs(v.co.x), abs(v.co.y)) > 1.0 - 1e-6
            bm.free()


def test_closed_loop_does_not_twist_at_the_closing_station():
    # Start the loop mid-face and at a corner: every station frame comes from its own normal.
    points = [(1, 0, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0)]
    normals = [(1, 0, 0), (S2, S2, 0), (-S2, S2, 0), (-S2, -S2, 0), (S2, -S2, 0)]
    bm, _w = sweep3d(path3d(points, normals, closed=True), PROFILE_ON_SURFACE)
    check_closed(bm)
    assert signed_volume(bm) > 0.0
    first = [bm.verts[i].co for i in range(4)]
    # Station 0 is on the face x = 1 with the profile standing along +x, spanning z.
    assert all(abs(v.x - 1.0) < 1e-6 or abs(v.x - 1.3) < 1e-6 for v in first)
    assert abs(first[0].z + 0.2) < 1e-6 and abs(first[1].z - 0.2) < 1e-6
    bm.free()


def test_acute_3d_corner_respects_the_mitre_limit():
    a = math.radians(160.0)
    rot = Matrix.Rotation(math.radians(55.0), 3, Vector((1.0, 0.3, -0.4)))
    pts2 = [Vector((0, 0, 0)), Vector((5, 0, 0)), Vector((5 + 5 * math.cos(a), 5 * math.sin(a), 0))]
    pts = [rot @ p for p in pts2]
    normal = rot @ Vector(UP)
    profile = [rect(-0.5, 0.5, 0.0, 1.0)]
    bm, warnings = sweep3d(path3d(pts, [normal] * 3), profile, limit=4.0)
    assert any("Mitre limit clamped 1 corner" in w for w in warnings)
    for i in range(4, 8):
        u = profile[0].points[i - 4][0]
        v = profile[0].points[i - 4][1]
        # The in-plane part of the offset is capped at limit * |u|; the height part is untouched.
        offset = bm.verts[i].co - pts[1]
        height = offset.dot(normal)
        assert abs(height - v) < 1e-5
        planar_part = (offset - normal * height).length
        assert abs(planar_part - 4.0 * abs(u)) < 1e-5
    bm.free()

    bm, warnings = sweep3d(path3d(pts, [normal] * 3), profile, limit=10.0)
    assert not any("Mitre" in w for w in warnings)
    offset = bm.verts[6].co - pts[1]
    planar_part = (offset - normal * offset.dot(normal)).length
    assert abs(planar_part - 0.5 / math.cos(math.radians(80.0))) < 1e-4
    bm.free()


def finite_and_bounded(bm, bound=100.0):
    for v in bm.verts:
        assert all(math.isfinite(c) for c in v.co)
        assert max(abs(c) for c in v.co) < bound


def test_normal_parallel_to_the_tangent():
    pts = [(0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0), (6, 3, 0)]
    expected, _w = build_sweep(path([(x, y) for x, y, _z in pts], False), PROFILES["square"], 4.0, False, True)
    expected.verts.ensure_lookup_table()
    # The middle point's normal lies along the path: its up is carried from the previous one.
    for bad in ((1, 0, 0), (-1, 0, 0), (1, 0, 1e-5), (0, 0, 0)):
        normals = [UP, UP, bad, UP, UP]
        bm, warnings = sweep3d(path3d(pts, normals), PROFILES["square"])
        assert warnings == []
        finite_and_bounded(bm)
        check_closed(bm)
        assert len(bm.verts) == len(expected.verts)
        for va, vb in zip(expected.verts, bm.verts):
            assert (va.co - vb.co).length < 1e-5
        bm.free()
    expected.free()

    # Every normal along the path (or the first one): stable perpendicular, no NaN, still a solid.
    for normals in ([(1, 0, 0)] * 5, [(1, 0, 0), UP, UP, UP, UP]):
        bm, _w = sweep3d(path3d(pts, normals), PROFILES["square"])
        finite_and_bounded(bm)
        check_closed(bm)
        assert signed_volume(bm) > 0.0
        bm.free()

    # Closed loop where the first corner normal lies along the bisecting tangent, and a vertical path with +Z normals.
    bm, _w = sweep3d(path3d([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)], [(S2, -S2, 0), UP, UP, UP], closed=True), PROFILES["square"])
    finite_and_bounded(bm)
    check_closed(bm)
    assert abs(signed_volume(bm) - 16.0) < 1e-4
    bm.free()

    bm, _w = sweep3d(path3d([(0, 0, 0), (0, 0, 3), (0, 2, 3)]), PROFILES["square"])
    finite_and_bounded(bm)
    check_closed(bm)
    assert signed_volume(bm) > 0.0
    bm.free()


def test_reversal_in_3d_has_no_nan():
    cases = (
        ([(0, 0, 0), (3, 0, 0), (1, 0, 0)], False),
        ([(0, 0, 0), (3, 0, 0), (1, 0, 0), (0, 0.0001, 0)], True),
        ([(0, 0, 0), (0, 2, 2), (0, 1, 1)], False),
        ([(0, 0, 0), (0, 2, 2), (0, 1, 1), (0, 0, 0.0001)], True),
    )
    for pts, closed in cases:
        bm, _w = sweep3d(path3d(pts, closed=closed), PROFILES["square"], limit=4.0)
        assert len(bm.faces) > 0
        finite_and_bounded(bm)
        bm.free()


def station_ring_edges(bm, station, per_station):
    return [
        e for e in bm.edges
        if e.verts[0].index // per_station == station and e.verts[1].index // per_station == station
    ]


def test_smooth_sharp_rings_use_the_3d_turn():
    profile = [circle(0.3, 12, cv=0.3)]
    up_y = [(0, 1, 0)] * 3
    # Tilted path, normal perpendicular to both segments: 45 degrees is hard, 25 degrees smooth.
    for elevation, hard in ((0.0, True), (20.0, False)):
        a = math.radians(elevation)
        pts = [(0, 0, 0), (3, 0, 3), (3 + 3 * math.cos(a), 0, 3 + 3 * math.sin(a))]
        d0 = Vector(pts[1]).normalized()
        d1 = (Vector(pts[2]) - Vector(pts[1])).normalized()
        assert (math.degrees(d0.angle(d1)) > 30.0) == hard
        bm, _w = sweep3d(path3d(pts, up_y), profile, smooth=True, caps=False)
        ring_edges = station_ring_edges(bm, 1, 12)
        assert ring_edges and all((not e.smooth) == hard for e in ring_edges)
        bm.free()

    # A 90 degree turn out of the XY plane is measured in 3D too.
    bm, _w = sweep3d(path3d([(0, 0, 0), (0, 0, 3), (0, 3, 3)], [(1, 0, 0)] * 3), profile, smooth=True, caps=False)
    ring_edges = station_ring_edges(bm, 1, 12)
    assert ring_edges and all(not e.smooth for e in ring_edges)
    bm.free()


def test_tube_with_caps_on_an_open_3d_path():
    points = [(0, -1, 0), (0, -1, 1), (0, 1, 1)]
    normals = [(0, -1, 0), (0, -S2, S2), UP]
    profile = tube(0.3, 0.2, 16, 0.3)
    bm, warnings = sweep3d(path3d(points, normals), profile, caps=True)
    assert warnings == []
    check_closed(bm)
    assert signed_volume(bm) > 0.0
    # Two segments of 32 wall faces each, then the cap faces.
    walls = 2 * 32
    assert len(bm.faces) > walls
    # The start cap faces against travel (-z), the end cap along it (+y).
    ends_start = [f for f in bm.faces if f.index >= walls and all(abs(v.co.z) < 1e-6 for v in f.verts)]
    ends_end = [f for f in bm.faces if f.index >= walls and all(abs(v.co.y - 1.0) < 1e-6 for v in f.verts)]
    assert ends_start and ends_end
    assert all(f.normal.z < -0.99 for f in ends_start)
    assert all(f.normal.y > 0.99 for f in ends_end)
    annulus = 0.5 * 16 * (0.3 ** 2 - 0.2 ** 2) * math.sin(2.0 * math.pi / 16)
    assert abs(sum(f.calc_area() for f in ends_start) - annulus) < 1e-5
    bm.free()

    bm, _w = sweep3d(path3d(points, normals), profile, caps=False)
    assert len(bm.faces) == walls
    assert sum(1 for e in bm.edges if len(e.link_faces) == 1) == 4 * 16
    bm.free()


def test_reversing_surface_normals_warn_and_are_not_flipped():
    square = PROFILES["square"]
    flipped = [(0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1)]
    bm, warnings = sweep3d(path3d([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], flipped), square)
    assert warnings == ["Surface normals reverse once along the path; the profile twists there"], warnings
    # The normals are kept as drawn: the first station stands on +Z, the last hangs from -Z.
    first = [v.co.z for v in bm.verts if abs(v.co.x) < 1e-6]
    last = [v.co.z for v in bm.verts if abs(v.co.x - 3) < 1e-6]
    assert max(first) > 0.0
    assert max(last) <= 1e-6 and min(last) < 0.0
    finite_and_bounded(bm)
    bm.free()

    # A closed path that goes over and back: two reversals, counted per pair of neighbouring stations.
    ring = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]
    bm, warnings = sweep3d(path3d(ring, [(0, 0, 1), (0, 0, -1), (0, 0, -1), (0, 0, 1)], closed=True), square)
    assert warnings == ["Surface normals reverse 2 times along the path; the profile twists there"], warnings
    bm.free()


def test_normals_that_turn_less_than_120_degrees_do_not_warn():
    square = PROFILES["square"]
    for angle in (45.0, 100.0, 119.0):
        a = math.radians(angle)
        normals = [(0, 0, 1), (0, math.sin(a), math.cos(a)), (0, math.sin(a), math.cos(a))]
        bm, warnings = sweep3d(path3d([(0, 0, 0), (2, 0, 0), (4, 0, 0)], normals), square)
        assert not any("reverse" in w for w in warnings), (angle, warnings)
        bm.free()
    a = math.radians(121.0)
    bm, warnings = sweep3d(path3d([(0, 0, 0), (2, 0, 0)], [(0, 0, 1), (0, math.sin(a), math.cos(a))]), PROFILES["square"])
    assert warnings == ["Surface normals reverse once along the path; the profile twists there"], warnings
    bm.free()
    # Flat paths and a path with a rotated plane never warn.
    bm, warnings = sweep3d(path3d([(0, 0, 0), (4, 0, 0), (4, 4, 0)]), square)
    assert warnings == []
    bm.free()


def test_no_profile_and_short_chains_3d():
    bm, warnings = sweep3d(path3d([(0, 0, 0), (4, 0, 0), (4, 4, 0)], closed=True), [])
    assert warnings == ["No profile to sweep"]
    assert len(bm.verts) == 0
    bm.free()

    bm, warnings = sweep3d(path3d([(1, 1, 1)]), PROFILES["square"])
    assert warnings == [] and len(bm.verts) == 0
    bm.free()
