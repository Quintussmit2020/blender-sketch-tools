import math

import bmesh
from mathutils import Vector

from sketch_tools.core import geometry2d as g2d
from sketch_tools.core.resolve import FILLET_RADIUS_ATTR, resolve_bmesh


class Sketch:
    """Builds a wire bmesh the way a user would: separate vertices per polyline."""

    def __init__(self, with_layer=True):
        self.bm = bmesh.new()
        self.layer = self.bm.verts.layers.float.new(FILLET_RADIUS_ATTR) if with_layer else None

    def vert(self, x, y, z=0.0, radius=0.0):
        v = self.bm.verts.new((x, y, z))
        if self.layer is not None:
            v[self.layer] = radius
        return v

    def edge(self, a, b):
        return self.bm.edges.new((a, b))

    def poly(self, pts, closed=False, radius=0.0):
        verts = [self.vert(*p, radius=radius) for p in pts]
        for i in range(len(verts) - 1):
            self.edge(verts[i], verts[i + 1])
        if closed:
            self.edge(verts[-1], verts[0])
        return verts

    def resolve(self, segments=8):
        return resolve_bmesh(self.bm, segments)

    def free(self):
        self.bm.free()


def square(x0, y0, size, ccw=True):
    pts = [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
    return pts if ccw else pts[::-1]


def resolve_loops(loops, polylines=(), segments=8, radius=0.0):
    s = Sketch()
    for pts in loops:
        s.poly(pts, closed=True, radius=radius)
    for pts in polylines:
        s.poly(pts)
    try:
        return s.resolve(segments)
    finally:
        s.free()


def closed_chains(resolved):
    return [c for c in resolved.chains if c.closed]


def _has_point(points, x, y, eps=1e-6):
    return any((p - Vector((x, y))).length < eps for p in points)


def test_square():
    r = resolve_loops([square(0, 0, 2, ccw=False)])
    assert len(r.chains) == 1
    c = r.chains[0]
    assert c.closed and c.depth == 0 and c.parent is None and c.children == []
    assert len(c.points) == 4
    assert g2d.signed_area(c.points) > 0.0
    assert all(isinstance(p, Vector) and len(p) == 2 for p in c.points)
    assert r.warnings == []


def test_square_with_hole():
    # Both drawn CCW: the resolver must flip the hole.
    r = resolve_loops([square(0, 0, 4), square(1, 1, 2)])
    outer, hole = r.chains
    assert outer.depth == 0 and hole.depth == 1
    assert hole.parent == 0 and outer.children == [1]
    assert g2d.signed_area(outer.points) > 0.0
    assert g2d.signed_area(hole.points) < 0.0
    assert r.warnings == []


def test_three_level_nesting():
    # Listed innermost first so the order of chains does not drive the result.
    r = resolve_loops([square(4, 4, 2), square(2, 2, 6), square(0, 0, 10)])
    depth = {round(abs(g2d.signed_area(c.points))): c for c in r.chains}
    island, hole, outer = depth[4], depth[36], depth[100]
    assert (outer.depth, hole.depth, island.depth) == (0, 1, 2)
    assert r.chains[hole.parent] is outer
    assert r.chains[island.parent] is hole
    assert outer.parent is None
    assert g2d.signed_area(island.points) > 0.0
    assert g2d.signed_area(hole.points) < 0.0


def test_open_polyline():
    s = Sketch()
    pts = [(0, 0), (1, 0), (1, 1), (2, 1)]
    verts = s.poly(pts)
    # Endpoint radii are ignored; open chains are never filleted at their ends.
    verts[0][s.layer] = 0.3
    verts[-1][s.layer] = 0.3
    r = s.resolve()
    s.free()
    assert len(r.chains) == 1
    c = r.chains[0]
    assert not c.closed and c.depth == -1 and c.parent is None
    got = [(round(p.x, 6), round(p.y, 6)) for p in c.points]
    assert got == pts or got == pts[::-1]


def test_junction_splits_chains():
    s = Sketch()
    centre = s.vert(0, 0)
    for x, y in ((1, 0), (-1, 0), (0, 1)):
        s.edge(centre, s.vert(x, y))
    r = s.resolve()
    s.free()
    assert len(r.chains) == 3
    for c in r.chains:
        assert not c.closed and len(c.points) == 2
        assert _has_point(c.points, 0, 0)


def test_filleted_square():
    rad, segs = 0.25, 4
    r = resolve_loops([square(0, 0, 2)], segments=segs, radius=rad)
    c = r.chains[0]
    assert c.closed and c.depth == 0
    assert len(c.points) == 4 * (segs + 1)
    assert not _has_point(c.points, 2, 2)
    centre = Vector((2 - rad, 2 - rad))
    near = [p for p in c.points if p.x > 1.5 and p.y > 1.5]
    assert len(near) == segs + 1
    for p in near:
        assert abs((p - centre).length - rad) < 1e-6
    assert g2d.signed_area(c.points) > 0.0
    assert r.warnings == []


def test_rectangle_with_tail_c8():
    s = Sketch()
    rect = s.poly([(0, 0), (3, 0), (3, 2), (0, 2)], closed=True)
    s.edge(rect[2], s.vert(5, 4))
    r = s.resolve()
    s.free()
    loops = closed_chains(r)
    opens = [c for c in r.chains if not c.closed]
    assert len(loops) == 1 and len(opens) == 1
    assert len(loops[0].points) == 4 and loops[0].depth == 0
    assert abs(g2d.signed_area(loops[0].points) - 6.0) < 1e-6
    assert len(opens[0].points) == 2


def test_loose_endpoints_weld_c9():
    s = Sketch()
    a = s.poly([(0, 0), (2, 0), (2, 2)])
    b = s.poly([(2, 2), (0, 2), (0, 0)])
    # The two copies of corner (0, 0) carry different radii: the larger one wins.
    a[0][s.layer] = 0.0
    b[-1][s.layer] = 0.5
    before = [(tuple(v.co), v[s.layer]) for v in s.bm.verts]
    r = s.resolve(segments=4)
    after = [(tuple(v.co), v[s.layer]) for v in s.bm.verts]
    edges_after = len(s.bm.edges)
    s.free()
    assert before == after and edges_after == 4, "resolver must not modify the input bmesh"
    assert len(r.chains) == 1
    c = r.chains[0]
    assert c.closed
    assert len(c.points) == 3 + 5
    assert not _has_point(c.points, 0, 0)
    assert _has_point(c.points, 0.5, 0) and _has_point(c.points, 0, 0.5)


def test_touching_loops_c11():
    # Inner loop has a vertex on the outer boundary and shares part of an edge: nested, no crossing.
    r = resolve_loops([square(0, 0, 4), [(0, 1), (2, 1), (2, 3), (0, 3)]])
    outer, inner = r.chains
    assert outer.depth == 0 and inner.depth == 1 and inner.parent == 0
    assert not any("cross" in w for w in r.warnings)

    r = resolve_loops([square(0, 0, 4), [(0, 2), (2, 1), (3, 2), (2, 3)]])
    assert r.chains[1].depth == 1
    assert not any("cross" in w for w in r.warnings)


def test_loops_sharing_a_corner():
    # Two squares meeting at one welded corner: a valence-4 junction, two junction loops (C8).
    r = resolve_loops([square(0, 0, 1), square(1, 1, 1)])
    loops = closed_chains(r)
    assert len(loops) == 2 and len(r.chains) == 2
    assert all(c.depth == 0 for c in loops)
    assert not any("cross" in w for w in r.warnings)


def test_shared_corner_at_float32_precision_is_not_a_crossing():
    # Both loops end an edge in (0, 0); the edge parameters read 0.99999997 in float32 maths.
    r = resolve_loops([[(0, 0), (10, 0), (5, 1)], [(0, 0), (4, 0.2), (3, 0.1)]])
    assert not any("cross" in w for w in r.warnings), r.warnings
    assert [c.depth for c in r.chains if c.closed] == [0, 1]


def test_crossing_loops_c11():
    r = resolve_loops([square(0, 0, 2), square(1, 1, 2)])
    assert "Loops 1 and 2 cross; extrude result may be wrong" in r.warnings


def test_fillet_clamp_warning():
    r = resolve_loops([square(0, 0, 1)], segments=4, radius=5.0)
    c = r.chains[0]
    assert any("clamped" in w for w in r.warnings)
    assert all(-1e-9 <= p.x <= 1 + 1e-9 and -1e-9 <= p.y <= 1 + 1e-9 for p in c.points)
    # Arcs meet at the edge midpoints; the shared tangent points are deduplicated.
    assert len(c.points) == 4 * 4
    assert _has_point(c.points, 0.5, 0)


def _corner_loop(theta, ox, oy, rot):
    """Triangle with a corner of theta at (ox, oy) between legs of 1 and 3, turned by rot."""
    legs = [(math.cos(rot), math.sin(rot)), (3 * math.cos(rot + theta), 3 * math.sin(rot + theta))]
    return (ox, oy), [(ox + legs[0][0], oy + legs[0][1]), (ox + legs[1][0], oy + legs[1][1])]


def _fillet_one_corner(theta, ox, oy, rot, scale):
    """Resolves a triangle whose only filleted corner has radius scale * the exact-fit radius.
    Vertex coordinates are float32 as in a real mesh."""
    corner, others = _corner_loop(theta, ox, oy, rot)
    s = Sketch()
    verts = [s.vert(*corner, radius=0.5 * math.tan(theta / 2.0) * scale)]
    verts += [s.vert(*q) for q in others]
    for i in range(3):
        s.edge(verts[i], verts[(i + 1) % 3])
    try:
        return s.resolve(8)
    finally:
        s.free()


def test_fillet_that_fits_exactly_is_not_reported_as_clamped():
    # The live repro: 4 x 3 rectangle, corner (4, 3) at r = 0.5, top edge cut into 1 unit pieces.
    s = Sketch()
    pts = [(0, 0), (4, 0), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]
    verts = [s.vert(x, y, radius=0.5 if (x, y) == (4, 3) else 0.0) for x, y in pts]
    for i in range(len(verts)):
        s.edge(verts[i], verts[(i + 1) % len(verts)])
    try:
        r = s.resolve(8)
    finally:
        s.free()
    assert r.warnings == []
    c = r.chains[0]
    assert _has_point(c.points, 4, 2.5) and _has_point(c.points, 3.5, 3)
    k = 0.5 * math.sqrt(0.5)
    assert _has_point(c.points, 3.5 + k, 2.5 + k, 1e-5)
    assert len(c.points) == 6 + 9


def test_exact_fit_is_not_clamped_at_any_angle_offset_or_rotation():
    for degrees in (30, 60, 90, 120, 150):
        theta = math.radians(degrees)
        for offset in (0.0, 100.0, 1000.0):
            for rot in (0.0, 0.7, 2.3):
                r = _fillet_one_corner(theta, offset, -offset * 0.5, rot, 1.0)
                assert r.warnings == [], (degrees, offset, rot, r.warnings)


def test_a_radius_one_percent_over_the_limit_is_still_clamped():
    for degrees in (30, 90, 150):
        theta = math.radians(degrees)
        for offset in (0.0, 1000.0):
            r = _fillet_one_corner(theta, offset, offset, 0.4, 1.01)
            assert any("1 corner" in w and "clamped" in w for w in r.warnings), (degrees, offset)


def _fillet_one_stored_corner(degrees, offset, rot, over):
    """Resolves a triangle whose filleted corner has a radius over * the exact fit worked out from
    the stored (float32) vertex coordinates."""
    theta = math.radians(degrees)
    corner, others = _corner_loop(theta, offset, -0.5 * offset, rot)
    pv, pp, nx = Vector(others[0]), Vector(corner), Vector(others[1])
    ax, ay = pp.x - pv.x, pp.y - pv.y
    bx, by = nx.x - pp.x, nx.y - pp.y
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    angle = math.atan2(abs(ax * by - ay * bx), -(ax * bx + ay * by))
    s = Sketch()
    verts = [s.vert(*corner, radius=over * 0.5 * min(la, lb) * math.tan(angle * 0.5))]
    verts += [s.vert(*q) for q in others]
    for i in range(3):
        s.edge(verts[i], verts[(i + 1) % 3])
    try:
        return s.resolve(8)
    finally:
        s.free()


def test_resolver_does_not_warn_for_exact_fits_at_sharp_and_near_flat_corners():
    for degrees in (5, 12, 20, 160, 172, 179):
        for offset in (0.0, 1000.0):
            for rot in (0.0, 0.7, 2.3):
                r = _fillet_one_stored_corner(degrees, offset, rot, 1.0)
                assert r.warnings == [], (degrees, offset, rot, r.warnings)


def test_resolver_warns_for_a_tenth_of_a_percent_over_at_offset_1000():
    for degrees in (5, 20, 90, 160, 175):
        for rot in (0.0, 0.7, 2.3):
            r = _fillet_one_stored_corner(degrees, 1000.0, rot, 1.001)
            assert any("1 corner" in w and "clamped" in w for w in r.warnings), (degrees, rot)


def test_fillet_skips_straight_and_spike():
    s = Sketch()
    a = math.radians(0.2)
    s.poly([(-1, 0), (0, 0), (math.cos(a), math.sin(a))], radius=0.2)
    r = s.resolve()
    s.free()
    assert len(r.chains[0].points) == 3
    assert r.warnings == []


def test_zero_length_segments():
    s = Sketch()
    v = s.poly([(0, 0), (1, 0), (1, 1), (0, 1)])
    dup = s.vert(0, 1)         # same spot as the last vertex, joined by a zero-length edge
    s.edge(v[-1], dup)
    s.edge(dup, v[0])
    s.vert(0.5, 0.5)           # loose vertex, no edges
    r = s.resolve()
    s.free()
    assert len(r.chains) == 1
    c = r.chains[0]
    assert c.closed and len(c.points) == 4


def test_degenerate_loop_becomes_open():
    s = Sketch()
    s.poly([(0, 0), (1, 0), (1, 0.000001)], closed=True)
    r = s.resolve()
    s.free()
    assert len(r.chains) == 1
    assert not r.chains[0].closed and r.chains[0].depth == -1


def test_off_plane_projected_with_warning():
    s = Sketch()
    s.poly([(0, 0, 0.5), (1, 0, 0.0), (1, 1, 0.00005), (0, 1, 0.0)], closed=True)
    r = s.resolve()
    zs = sorted(v.co.z for v in s.bm.verts)
    s.free()
    assert zs[-1] == 0.5, "input bmesh must keep its z values"
    assert len(r.warnings) == 1 and "off the sketch plane" in r.warnings[0]
    assert r.warnings[0].startswith("1 vertex ")
    assert r.chains[0].closed and len(r.chains[0].points) == 4


def test_missing_radius_layer():
    s = Sketch(with_layer=False)
    s.poly(square(0, 0, 1), closed=True)
    r = s.resolve()
    s.free()
    assert len(r.chains) == 1 and len(r.chains[0].points) == 4


def test_faces_ignored():
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0)) for x, y in square(0, 0, 1)]
    bm.faces.new(verts)
    r = resolve_bmesh(bm, 8)
    assert len(bm.faces) == 1
    bm.free()
    assert len(r.chains) == 1 and r.chains[0].closed


def test_fillet_on_reflex_corner_of_l_shape():
    s = Sketch()
    pts = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    verts = [s.vert(x, y, radius=0.5 if (x, y) == (2, 2) else 0.0) for x, y in pts]
    for i in range(len(verts)):
        s.edge(verts[i], verts[(i + 1) % len(verts)])
    segments = 6
    try:
        r = s.resolve(segments)
    finally:
        s.free()
    assert len(r.chains) == 1 and r.chains[0].closed
    points = r.chains[0].points
    assert len(points) == 5 + segments + 1
    assert r.warnings == []

    # The arc sits outside the L: centre at (2.5, 2.5), radius 0.5, tangent to both edges.
    centre = Vector((2.5, 2.5))
    arc = [p for p in points if (p - centre).length < 0.6 and p.x < 2.6 and p.y < 2.6]
    assert len(arc) == segments + 1, len(arc)
    for p in arc:
        assert abs((p - centre).length - 0.5) < 1e-6
    assert _has_point(points, 2.5, 2.0) and _has_point(points, 2.0, 2.5)
    assert not _has_point(points, 2.0, 2.0), "sharp reflex corner survived"

    # Rounding a reflex corner adds the sliver between the corner and the arc.
    assert abs(abs(g2d.signed_area(points)) - (12.0 + 0.25 * (1.0 - math.pi / 4.0))) < 0.02
