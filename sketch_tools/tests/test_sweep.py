import math
from types import SimpleNamespace

from mathutils import Vector

from sketch_tools.builders.sweep import build_sweep
from sketch_tools.tests.test_extrude import signed_volume
from sketch_tools.tests.test_resolve import resolve_loops, square


def ring(points, closed=True, depth=0, parent=None):
    return SimpleNamespace(points=list(points), closed=closed, depth=depth, parent=parent)


def rect(u0, u1, v0, v1):
    return ring([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def circle(radius, segments, cw=False, cv=0.0, depth=0, parent=None):
    pts = [
        (radius * math.cos(2.0 * math.pi * k / segments), cv + radius * math.sin(2.0 * math.pi * k / segments))
        for k in range(segments)
    ]
    return ring(pts[::-1] if cw else pts, depth=depth, parent=parent)


def tube(radius, inner, segments, cv):
    return [circle(radius, segments, cv=cv), circle(inner, segments, cw=True, cv=cv, depth=1, parent=0)]


def half_round(radius, segments):
    arc = [(radius * math.cos(math.pi * k / segments), radius * math.sin(math.pi * k / segments)) for k in range(segments + 1)]
    return ring(arc)


def path(points, closed):
    chain = SimpleNamespace(points=[Vector(p) for p in points], closed=closed, depth=0 if closed else -1, parent=None, children=[])
    return SimpleNamespace(chains=[chain], warnings=[])


def sweep(resolved, rings, limit=4.0, smooth=False, caps=True):
    bm, warnings = build_sweep(resolved, rings, limit, smooth, caps)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.normal_update()
    return bm, warnings


def check_closed(bm):
    """Manifold, and every directed edge used once, so no face is flipped against a neighbour."""
    directed = set()
    for e in bm.edges:
        assert len(e.link_faces) == 2, f"edge {e.index} has {len(e.link_faces)} faces"
    for f in bm.faces:
        for loop in f.loops:
            key = (loop.vert.index, loop.link_loop_next.vert.index)
            assert key not in directed, f"inconsistent winding at {key}"
            directed.add(key)


def check_outward_open(bm, centre_path, hole_radius=0.0):
    """Open shell: every wall faces away from the profile's centre line, except walls closer
    than hole_radius, which are a tube's inner surface and face it."""
    for f in bm.faces:
        c = f.calc_center_median()
        best = None
        for a, b in zip(centre_path, centre_path[1:]):
            ab = b - a
            t = max(0.0, min(1.0, (c - a).dot(ab) / ab.length_squared))
            q = a + ab * t
            d = (c - q).length
            if best is None or d < best[0]:
                best = (d, q)
        outward = f.normal.dot(c - best[1]) > 0.0
        assert outward == (best[0] > hole_radius), f"face {f.index} is oriented wrongly"


def test_closed_square_path_square_profile():
    bm, warnings = sweep(path(square(0, 0, 4), True), [rect(-0.5, 0.5, 0.0, 1.0)])
    assert warnings == []
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (16, 32, 16)
    check_closed(bm)
    # Outer 5x5 minus inner 3x3 square, one unit tall.
    assert abs(signed_volume(bm) - 16.0) < 1e-4
    corners = square(0, 0, 4)
    for k, c in enumerate(corners):
        stations = [bm.verts[k * 4 + i] for i in range(4)]
        for v in stations:
            assert abs(math.hypot(v.co.x - c[0], v.co.y - c[1]) - math.sqrt(0.5)) < 1e-5
    bm.free()


def test_closed_square_path_clockwise_input():
    bm, _w = sweep(path(square(0, 0, 4, ccw=False), True), [rect(-0.5, 0.5, 0.0, 1.0)])
    assert len(bm.faces) == 16
    check_closed(bm)
    assert abs(signed_volume(bm) - 16.0) < 1e-4
    bm.free()


def test_closed_square_path_round_profiles():
    sq = path(square(0, 0, 4), True)
    for profile, rings_n in (
        (circle(0.3, 16, cv=0.3), 16),
        ([half_round(0.4, 8)], 9),
    ):
        profile = profile if isinstance(profile, list) else [profile]
        bm, warnings = sweep(sq, profile)
        assert warnings == []
        assert len(bm.faces) == 4 * rings_n
        assert len(bm.verts) == 4 * rings_n
        check_closed(bm)
        assert signed_volume(bm) > 0.0
        bm.free()


def test_closed_square_path_tube():
    sq = path(square(0, 0, 4), True)
    outer_only, _w = sweep(sq, [circle(0.4, 16, cv=0.4)])
    inner_only, _w = sweep(sq, [circle(0.3, 16, cv=0.4)])
    bm, warnings = sweep(sq, tube(0.4, 0.3, 16, 0.4))
    assert warnings == []
    assert (len(bm.verts), len(bm.faces)) == (4 * 32, 4 * 32)
    check_closed(bm)
    # The inner surface is present and faces the tube axis: it subtracts its own volume.
    assert abs(signed_volume(bm) - (signed_volume(outer_only) - signed_volume(inner_only))) < 1e-4
    inner_z = [bm.verts[k * 32 + 16 + i].co.z for k in range(4) for i in range(16)]
    assert abs(min(inner_z) - 0.1) < 1e-5 and abs(max(inner_z) - 0.7) < 1e-5
    for b in (bm, outer_only, inner_only):
        b.free()


def test_open_path_caps():
    pts = [(0, 0), (3, 0), (3, 3)]
    profile = [rect(-0.5, 0.5, 0.0, 1.0)]
    bm, _w = sweep(path(pts, False), profile, caps=True)
    assert len(bm.faces) == 2 * 4 + 2
    assert len(bm.verts) == 12
    check_closed(bm)
    assert signed_volume(bm) > 0.0
    bm.free()

    bm, _w = sweep(path(pts, False), profile, caps=False)
    assert len(bm.faces) == 8
    assert sum(1 for e in bm.edges if len(e.link_faces) == 1) == 8
    check_outward_open(bm, [Vector((x, y, 0.5)) for x, y in pts])
    bm.free()

    # A path that goes the other way round still faces outward.
    bm, _w = sweep(path(pts[::-1], False), profile, caps=False)
    check_outward_open(bm, [Vector((x, y, 0.5)) for x, y in pts[::-1]])
    bm.free()


def test_open_path_tube_ring_caps():
    pts = [(0, 0), (4, 0)]
    bm, warnings = sweep(path(pts, False), tube(0.5, 0.3, 16, 0.5), caps=True)
    assert warnings == []
    check_closed(bm)
    annulus = 0.5 * 16 * (0.5 ** 2 - 0.3 ** 2) * math.sin(2.0 * math.pi / 16)
    assert abs(signed_volume(bm) - annulus * 4.0) < 1e-4
    end_caps = [f for f in bm.faces if all(abs(v.co.x - 4.0) < 1e-6 for v in f.verts)]
    assert end_caps and all(f.normal.x > 0.99 for f in end_caps)
    assert abs(sum(f.calc_area() for f in end_caps) - annulus) < 1e-4
    start_caps = [f for f in bm.faces if all(abs(v.co.x) < 1e-6 for v in f.verts)]
    assert start_caps and all(f.normal.x < -0.99 for f in start_caps)
    bm.free()

    bm, _w = sweep(path(pts, False), tube(0.5, 0.3, 16, 0.5), caps=False)
    assert len(bm.faces) == 32
    check_outward_open(bm, [Vector((0, 0, 0.5)), Vector((4, 0, 0.5))], hole_radius=0.4)
    bm.free()


def test_open_path_island_and_touching_hole_caps():
    island = [
        ring([(-2, 0), (2, 0), (2, 4), (-2, 4)]),
        ring([(-1.5, 0.5), (-1.5, 3.5), (1.5, 3.5), (1.5, 0.5)], depth=1, parent=0),
        ring([(-0.5, 1.5), (0.5, 1.5), (0.5, 2.5), (-0.5, 2.5)], depth=2, parent=1),
    ]
    bm, warnings = sweep(path([(0, 0), (5, 0)], False), island, caps=True)
    assert warnings == []
    check_closed(bm)
    assert abs(signed_volume(bm) - (16.0 - 9.0 + 1.0) * 5.0) < 1e-3
    bm.free()

    # The hole shares a section of the outer ring's bottom edge: the cap moves the hole points
    # and the walls must follow, otherwise the shell cracks.
    touching = [rect(-1.0, 1.0, 0.0, 2.0), ring([(-0.5, 0.0), (-0.5, 1.0), (0.5, 1.0), (0.5, 0.0)], depth=1, parent=0)]
    bm, warnings = sweep(path([(0, 0), (5, 0)], False), touching, caps=True)
    assert warnings == []
    check_closed(bm)
    assert abs(signed_volume(bm) - (4.0 - 1.0) * 5.0) < 0.01
    bm.free()


def test_filleted_square_path_smooth():
    resolved = resolve_loops([square(0, 0, 4)], radius=0.8, segments=6)
    bm, warnings = sweep(resolved, [rect(-0.3, 0.3, 0.0, 0.5)], smooth=True)
    assert warnings == []
    check_closed(bm)
    assert signed_volume(bm) > 0.0
    assert all(f.smooth for f in bm.faces)
    # A mitred band of width 2h around a closed polyline has area 2h * perimeter exactly.
    points = resolved.chains[0].points
    perimeter = sum((points[i] - points[i - 1]).length for i in range(len(points)))
    assert abs(signed_volume(bm) - 0.6 * perimeter * 0.5) < 1e-3
    bm.free()


def test_acute_corner_is_bounded_by_the_mitre_limit():
    a = math.radians(160.0)
    corner = Vector((5.0, 0.0))
    pts = [(0, 0), (5, 0), (5 + 5 * math.cos(a), 5 * math.sin(a))]
    profile = [rect(-0.5, 0.5, 0.0, 1.0)]
    bm, warnings = sweep(path(pts, False), profile, limit=4.0)
    assert any("Mitre limit clamped 1 corner" in w for w in warnings)
    n0 = Vector((0.0, 1.0))
    n1 = Vector((-math.sin(a), math.cos(a)))
    bisector = (n0 + n1).normalized()
    for i in range(4, 8):
        u = profile[0].points[i - 4][0]
        v = bm.verts[i].co
        offset = Vector((v.x, v.y)) - corner
        assert offset.length <= 4.0 * abs(u) + 1e-5
        assert abs(offset.length - 4.0 * abs(u)) < 1e-5
        # Left of travel is the inside of this left turn; right-side points stay outside.
        assert offset.dot(bisector) * u > 0.0
    bm.free()

    bm, warnings = sweep(path(pts, False), profile, limit=10.0)
    assert not any("Mitre" in w for w in warnings)
    offset = Vector((bm.verts[6].co.x, bm.verts[6].co.y)) - corner
    assert abs(offset.length - 0.5 / math.cos(math.radians(80.0))) < 1e-4
    bm.free()


def test_reversal_has_no_nan():
    for closed_pts, closed in (([(0, 0), (3, 0), (1, 0)], False), ([(0, 0), (3, 0), (1, 0), (0, 0.0001)], True)):
        bm, _w = sweep(path(closed_pts, closed), [rect(-0.5, 0.5, 0.0, 1.0)], limit=4.0)
        assert len(bm.faces) > 0
        for v in bm.verts:
            assert all(math.isfinite(c) for c in v.co)
            assert max(abs(c) for c in v.co) < 100.0
        bm.free()


def _classify(bm, per_station):
    """(along, ring) edges of a single ring sweep whose stations hold per_station vertices."""
    along, ring_edges = [], []
    for e in bm.edges:
        a, b = e.verts[0].index // per_station, e.verts[1].index // per_station
        (ring_edges if a == b else along).append(e)
    return along, ring_edges


def test_smooth_shading_profile_corners():
    sq = path(square(0, 0, 4), True)
    bm, _w = sweep(sq, [rect(-0.5, 0.5, 0.0, 1.0)], smooth=True)
    assert all(f.smooth for f in bm.faces)
    along, _ring = _classify(bm, 4)
    assert along and all(not e.smooth for e in along)
    bm.free()

    bm, _w = sweep(sq, [circle(0.4, 16, cv=0.4)], smooth=True)
    along, ring_edges = _classify(bm, 16)
    assert along and all(e.smooth for e in along)
    # The 90 degree path corners keep hard rings.
    assert ring_edges and all(not e.smooth for e in ring_edges)
    bm.free()

    bm, _w = sweep(sq, [circle(0.4, 16, cv=0.4)], smooth=False)
    assert all(not f.smooth for f in bm.faces)
    assert all(e.smooth for e in bm.edges)
    bm.free()


def test_smooth_fillet_arc_rings_stay_smooth():
    resolved = resolve_loops([square(0, 0, 4)], radius=0.8, segments=6)
    bm, _w = sweep(resolved, [circle(0.2, 16, cv=0.2)], smooth=True)
    along, ring_edges = _classify(bm, 16)
    assert all(e.smooth for e in along)
    assert all(e.smooth for e in ring_edges)
    bm.free()


def test_smooth_open_path_keeps_caps_flat():
    bm, _w = sweep(path([(0, 0), (3, 0), (3, 3)], False), [circle(0.4, 16, cv=0.4)], smooth=True, caps=True)
    along, ring_edges = _classify(bm, 16)
    assert all(e.smooth for e in along)
    # Three stations: the two ends (cap borders) and the 90 degree corner are hard.
    assert ring_edges and all(not e.smooth for e in ring_edges)
    bm.free()


def test_open_profile_ring_makes_open_strip():
    strip = ring([(-1.0, 0.0), (0.0, 0.5), (1.0, 0.0)], closed=False, depth=-1)
    bm, warnings = sweep(path([(0, 0), (3, 0), (3, 3)], False), [strip], caps=True)
    assert warnings == []
    assert (len(bm.verts), len(bm.faces)) == (9, 4)
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    assert len(boundary) == 8
    # Ring order picks the side: walking left to right is the underside of a CCW solid.
    assert all(f.normal.z < 0.0 for f in bm.faces)
    bm.free()


def test_no_profile_and_short_chains():
    bm, warnings = sweep(path(square(0, 0, 4), True), [])
    assert warnings == ["No profile to sweep"]
    assert len(bm.verts) == 0
    bm.free()

    bm, warnings = sweep(path([(1, 1)], False), [rect(-0.5, 0.5, 0.0, 1.0)])
    assert warnings == [] and len(bm.verts) == 0
    bm.free()


def test_multiple_chains_merge():
    resolved = resolve_loops([square(0, 0, 4)], polylines=[[(10, 0), (13, 0)]])
    bm, _w = sweep(resolved, [rect(-0.5, 0.5, 0.0, 1.0)], caps=True)
    # 16 walls + 4 walls and 2 end caps on the open chain.
    assert len(bm.faces) == 16 + 4 + 2
    check_closed(bm)
    assert abs(signed_volume(bm) - (16.0 + 3.0)) < 1e-4
    bm.free()
