import math
import struct
import sys
import time

from mathutils import Vector

from sketch_tools.builders import caps as caps_module
from sketch_tools.builders.extrude import build_extrude
from sketch_tools.core.geometry2d import cross2
from sketch_tools.tests.test_resolve import resolve_loops, square

TOL = caps_module.MIN_COLLINEAR_TOL


def extrude(loops, polylines=(), height=2.0, offset=0.0, flip=False, caps=True, radius=0.0, segments=8):
    resolved = resolve_loops(loops, polylines, segments=segments, radius=radius)
    return build_extrude(resolved, height, offset, flip, caps)


def assert_manifold(bm):
    for e in bm.edges:
        assert len(e.link_faces) == 2, f"edge {e.index} has {len(e.link_faces)} faces"


def signed_volume(bm):
    total = 0.0
    for f in bm.faces:
        vs = [v.co for v in f.verts]
        for i in range(1, len(vs) - 1):
            total += vs[0].dot(vs[i].cross(vs[i + 1]))
    return total / 6.0


def components(bm):
    seen = set()
    count = 0
    for start in bm.verts:
        if start.index in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start.index)
        while stack:
            v = stack.pop()
            for e in v.link_edges:
                o = e.other_vert(v)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
    return count


def cap_faces(bm, z, up):
    out = []
    for f in bm.faces:
        if all(abs(v.co.z - z) < 1e-6 for v in f.verts):
            if (f.normal.z > 0.5) == up:
                out.append(f)
    return out


def prep(bm):
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.normal_update()


def test_plain_square():
    bm, warnings = extrude([square(0, 0, 2)], height=3.0)
    prep(bm)
    assert warnings == []
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (8, 12, 6)
    assert_manifold(bm)
    assert abs(signed_volume(bm) - 12.0) < 1e-6
    top = cap_faces(bm, 3.0, up=True)
    bottom = cap_faces(bm, 0.0, up=False)
    assert len(top) == 1 and len(top[0].verts) == 4
    assert len(bottom) == 1 and len(bottom[0].verts) == 4
    bm.free()


def test_square_with_hole():
    bm, warnings = extrude([square(0, 0, 4), square(1, 1, 2)], height=1.0)
    prep(bm)
    assert warnings == []
    assert len(bm.verts) == 16
    assert_manifold(bm)
    # Torus-like solid: Euler characteristic 0.
    assert len(bm.verts) - len(bm.edges) + len(bm.faces) == 0
    assert abs(signed_volume(bm) - 12.0) < 1e-6
    for z, up in ((1.0, True), (0.0, False)):
        caps = cap_faces(bm, z, up)
        assert caps
        assert abs(sum(f.calc_area() for f in caps) - 12.0) < 1e-6, "hole missing from cap"
        assert all(len(f.verts) > 3 for f in caps), "cap triangles left after dissolve"
        assert len(caps) <= 2
    bm.free()


def test_ngon_with_hole_dissolves_to_ngons():
    ring = [(5 * math.cos(2 * math.pi * k / 12), 5 * math.sin(2 * math.pi * k / 12)) for k in range(12)]
    bm, _ = extrude([ring, square(-1, -1, 2)], height=1.0)
    prep(bm)
    assert_manifold(bm)
    caps = cap_faces(bm, 1.0, True)
    assert all(len(f.verts) > 3 for f in caps)
    assert len(caps) <= 2
    bm.free()


def test_island_two_solids():
    bm, _ = extrude([square(0, 0, 10), square(2, 2, 6), square(4, 4, 2)], height=1.0)
    prep(bm)
    assert_manifold(bm)
    assert components(bm) == 2
    assert abs(signed_volume(bm) - (100 - 36 + 4)) < 1e-6
    bm.free()


def test_flip_and_offset():
    bm, _ = extrude([square(0, 0, 1)], height=2.0, offset=0.5, flip=True)
    prep(bm)
    zs = sorted({round(v.co.z, 6) for v in bm.verts})
    assert zs == [-1.5, 0.5]
    assert_manifold(bm)
    assert abs(signed_volume(bm) - 2.0) < 1e-6
    bm.free()

    bm, _ = extrude([square(0, 0, 4), square(1, 1, 2)], height=1.0, flip=True)
    prep(bm)
    assert_manifold(bm)
    assert abs(signed_volume(bm) - 12.0) < 1e-6
    bm.free()


def test_negative_height_outward():
    bm, _ = extrude([square(0, 0, 1)], height=-2.0)
    prep(bm)
    assert abs(signed_volume(bm) - 2.0) < 1e-6
    bm.free()


def test_no_caps():
    bm, _ = extrude([square(0, 0, 1)], caps=False)
    assert len(bm.faces) == 4
    bm.free()


def test_open_chains_ignored():
    bm, warnings = extrude([square(0, 0, 1)], polylines=[[(3, 0), (4, 0)], [(5, 0), (6, 1)]])
    assert "2 open chains ignored by Extrude" in warnings
    assert len(bm.faces) == 6
    bm.free()


def test_zero_height():
    bm, warnings = extrude([square(0, 0, 1)], height=0.0)
    assert len(bm.verts) == 0 and len(bm.faces) == 0
    assert any("zero" in w for w in warnings)
    bm.free()


def test_filleted_square_extrude():
    bm, _ = extrude([square(0, 0, 2)], radius=0.25, segments=4)
    prep(bm)
    assert len(bm.faces) == 20 + 2
    assert_manifold(bm)
    assert signed_volume(bm) > 0.0
    bm.free()


def test_touching_loops_stay_separate():
    # Two loops meeting at one corner keep their own vertices (C3): still manifold.
    bm, _ = extrude([square(0, 0, 1), square(1, 1, 1)])
    prep(bm)
    assert_manifold(bm)
    assert components(bm) == 2
    assert len(bm.verts) == 16
    bm.free()


def _inside_loop(p, loop):
    x, y = p
    inside = False
    n = len(loop)
    for i in range(n):
        (x0, y0), (x1, y1) = loop[i], loop[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def assert_outward(bm, loops, z_top):
    """Every face points out of the material: walls are probed in plan view with an
    even-odd test, caps by the sign of their normal."""
    for f in bm.faces:
        zs = [v.co.z for v in f.verts]
        if max(zs) - min(zs) < 1e-9:
            expected = 1.0 if abs(zs[0] - z_top) < 1e-9 else -1.0
            assert f.normal.z * expected > 0.5, f"cap at z={zs[0]} faces the wrong way"
            continue
        centre = f.calc_center_median()
        probe = (centre.x + f.normal.x * 1e-3, centre.y + f.normal.y * 1e-3)
        inside = sum(_inside_loop(probe, loop) for loop in loops) % 2 == 1
        assert not inside, f"wall face {f.index} points into the material"


def test_walls_outward_without_caps_and_with_flip():
    loops = [square(0, 0, 10), square(2, 2, 6), square(4, 4, 2)]
    for caps in (True, False):
        for flip in (False, True):
            bm, _ = extrude(loops, height=1.5, flip=flip, caps=caps)
            prep(bm)
            assert_outward(bm, loops, z_top=0.0 if flip else 1.5)
            bm.free()


COLLINEAR_OUTER = [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (2, 4), (0, 4), (0, 2)]


def test_collinear_outer_points_with_hole_are_manifold():
    bm, warnings = extrude([COLLINEAR_OUTER, square(1, 1, 2)], height=1.0)
    prep(bm)
    assert warnings == []
    assert_manifold(bm)
    assert abs(signed_volume(bm) - 12.0) < 1e-6
    for z, up in ((1.0, True), (0.0, False)):
        caps = cap_faces(bm, z, up)
        assert abs(sum(f.calc_area() for f in caps) - 12.0) < 1e-6
    bm.free()


def test_many_collinear_points_in_a_row_and_on_the_hole():
    outer = [(x, 0) for x in range(0, 6)] + [(6, y) for y in range(0, 6)] \
        + [(x, 6) for x in range(6, 0, -1)] + [(0, y) for y in range(6, 0, -1)]
    outer = [p for k, p in enumerate(outer) if p not in outer[:k]]
    hole = [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]
    for flip in (False, True):
        bm, warnings = extrude([outer, hole], height=1.0, flip=flip)
        prep(bm)
        assert warnings == []
        assert_manifold(bm)
        assert abs(signed_volume(bm) - (36 - 4)) < 1e-6
        assert_outward(bm, [outer, hole], z_top=0.0 if flip else 1.0)
        bm.free()


def test_collinear_points_in_a_filleted_hole_cap():
    # Fillet arcs plus an extra collinear point on a straight side of the hole.
    hole = [(1, 1), (2, 1), (3, 1), (3, 3), (1, 3)]
    bm, warnings = extrude([square(0, 0, 4), hole], height=1.0, radius=0.2, segments=4)
    prep(bm)
    assert warnings == []
    assert_manifold(bm)
    assert signed_volume(bm) > 0.0
    bm.free()



# Holed caps where tessellate_polygon alone overlaps a hole (seed 130, area 141.27 against
# 140.0) or returns a duplicate sliver pair (seed 1823). Rotated random shapes from the QA fuzz.
BAD_TESSELLATION_ROTATED_RECTS = [
    [
        (0, 0), (5.46861, 5.46861), (14.1421, 14.1421), (7.91188, 20.3724), (7.07107, 21.2132),
        (-0.660169, 13.482), (-2.47632, 11.6658), (-4.34163, 9.80051), (-7.07107, 7.07107),
    ],
    [
        (0, 2.82843), (2.82843, 5.65685), (-1.41421, 9.89949), (-3.86058, 7.45313), (-4.24264, 7.07107),
        (-1.59636, 4.42479),
    ],
    [
        (4.94975, 7.77817), (6.00276, 8.83118), (6.8747, 9.70312), (7.77817, 10.6066), (3.53553, 14.8492),
        (3.1485, 14.4622), (2.06679, 13.3805), (1.48201, 12.7957), (0.707107, 12.0208), (4.25564, 8.47228),
    ],
    [
        (8.48528, 12.7279), (10.6066, 14.8492), (10.0617, 15.3941), (9.97326, 15.4826), (7.77817, 17.6777),
        (6.18774, 16.0872), (5.65685, 15.5563), (6.92823, 14.285), (7.95295, 13.2603),
    ],
]

BAD_TESSELLATION_STAR = [
    [
        (6.30452, 6.30452), (-3.85935, 7.5744), (-6.62948, -1.05001), (-0.982149, -6.20104),
        (1.14097, -5.45847), (1.90908, -5.18982), (6.81677, -3.47332),
    ],
    [
        (0.959795, 0.959795), (-0.57799, 1.13437), (-2.05537, -0.325538), (-0.263774, -1.6654),
        (1.11787, -0.569582),
    ],
]


def region_area(loops):
    """Even-odd area of nested loops: outer minus holes, from the loops as given."""
    total = 0.0
    for k, loop in enumerate(loops):
        n = len(loop)
        a = abs(sum(loop[i][0] * loop[(i + 1) % n][1] - loop[(i + 1) % n][0] * loop[i][1] for i in range(n))) * 0.5
        total += a if k == 0 else -a
    return total


def assert_clean_holed_solid(loops, height=1.0, flip=False):
    bm, warnings = extrude(loops, height=height, flip=flip)
    prep(bm)
    assert warnings == [], warnings
    assert_manifold(bm)
    area = region_area(loops)
    assert abs(signed_volume(bm) - area * height) < 1e-3 * area
    assert not [f for f in bm.faces if f.calc_area() < 1e-9], "zero-area face"
    assert_outward(bm, loops, z_top=0.0 if flip else height)
    bm.free()


def _resolved_closed(loops):
    return [c for c in resolve_loops(loops).chains if c.closed]


def _loop_edge_keys(rings):
    keys = set()
    start = 0
    for ring in rings:
        n = len(ring)
        keys.update(caps_module._edge_key(start + i, start + (i + 1) % n) for i in range(n))
        start += n
    return keys


def _first_tessellation_is_valid(loops):
    rings = [c.points for c in _resolved_closed(loops)]
    tol = caps_module._collinear_tol(rings)
    pts = [p for ring in rings for p in ring]
    edges = _loop_edge_keys(rings)
    first_try = caps_module._clean_triangles(caps_module._tessellate(rings, 0.0), pts, tol)
    first_try = caps_module._split_at_collinear_points(first_try, pts, edges, tol)
    return caps_module._cap_is_valid(first_try, pts, edges, region_area(rings), tol)


def test_tessellation_that_overlaps_a_hole_is_rescued():
    assert not _first_tessellation_is_valid(BAD_TESSELLATION_ROTATED_RECTS), \
        "the shape no longer breaks the first tessellation; pick a new regression shape"
    for flip in (False, True):
        assert_clean_holed_solid(BAD_TESSELLATION_ROTATED_RECTS, flip=flip)


def test_duplicate_sliver_pair_is_rescued():
    assert not _first_tessellation_is_valid(BAD_TESSELLATION_STAR), \
        "the shape no longer breaks the first tessellation; pick a new regression shape"
    assert_clean_holed_solid(BAD_TESSELLATION_STAR)


def test_inconsistent_tessellation_warns_and_keeps_the_walls():
    real = caps_module.tessellate_polygon
    # One triangle never covers the region, so every attempt fails validation.
    caps_module.tessellate_polygon = lambda coords: [(0, 1, 2)]
    try:
        bm, warnings = extrude([square(0, 0, 4), square(1, 1, 2)], height=1.0)
    finally:
        caps_module.tessellate_polygon = real
    assert warnings.count("Cap triangulation is inconsistent") == 1
    walls = [f for f in bm.faces if max(v.co.z for v in f.verts) - min(v.co.z for v in f.verts) > 0.5]
    assert len(walls) == 8
    bm.free()


def test_failed_first_tessellation_retries_on_rotated_points():
    real = caps_module.tessellate_polygon
    calls = []

    def flaky(coords):
        calls.append(coords[0][0].copy())
        return [(0, 1, 2)] if len(calls) == 1 else real(coords)

    caps_module.tessellate_polygon = flaky
    try:
        bm, warnings = extrude([square(0, 0, 4), square(1, 1, 2)], height=1.0)
    finally:
        caps_module.tessellate_polygon = real
    prep(bm)
    assert warnings == []
    assert len(calls) == 2
    assert (calls[0] - calls[1]).length > 1e-3, "retry must tessellate rotated points"
    assert_manifold(bm)
    assert abs(signed_volume(bm) - 12.0) < 1e-6
    bm.free()


def test_duplicate_triangles_are_dropped():
    pts = [Vector(p) for p in ((0, 0), (4, 0), (4, 4), (0, 4))]
    cleaned = caps_module._clean_triangles([(0, 1, 2), (2, 1, 0), (0, 2, 3)], pts, TOL)
    assert cleaned == [(0, 1, 2), (0, 2, 3)]


def test_near_collinear_triangle_is_a_sliver():
    # Height 1e-6 over a long edge is below the shared tolerance; 1e-4 is a real triangle.
    pts = [Vector(p) for p in ((0, 0), (10, 0), (5, 1e-6), (5, 1e-4))]
    assert caps_module._clean_triangles([(0, 1, 2)], pts, TOL) == []
    assert caps_module._clean_triangles([(0, 1, 3)], pts, TOL) == [(0, 1, 3)]
    # The same tolerance decides whether a point splits a segment.
    assert caps_module._points_on_segment(pts, 0, 1, exclude=(0, 1, 3), tol=TOL) == [2]
    assert caps_module._points_on_segment(pts, 0, 1, exclude=(0, 1, 2), tol=TOL) == []


def test_split_handles_a_triangle_spanning_two_edges():
    # Both xy and yz of the triangle skip a loop point, so one fan is not enough.
    pts = [Vector(p) for p in ((0, 0), (4, 0), (0, 4), (2, 0), (2, 2))]
    loop_edges = {caps_module._edge_key(a, b) for a, b in ((0, 3), (3, 1), (1, 4), (4, 2), (2, 0))}
    result = caps_module._split_at_collinear_points([(0, 1, 2)], pts, loop_edges, TOL)
    assert len(result) == 3
    assert abs(sum(cross2(pts[b] - pts[a], pts[c] - pts[a]) for a, b, c in result) * 0.5 - 8.0) < 1e-9
    assert all(cross2(pts[b] - pts[a], pts[c] - pts[a]) > 0.0 for a, b, c in result)
    used = {}
    for t in result:
        for i in range(3):
            key = caps_module._edge_key(t[i], t[(i + 1) % 3])
            used[key] = used.get(key, 0) + 1
    assert all(used[key] == 1 for key in loop_edges)
    assert all(n == 2 for key, n in used.items() if key not in loop_edges)


def _rotated(loop, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in loop]


def _subdivided(loop, per_edge=2):
    out = []
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        out.append(a)
        for k in range(1, per_edge + 1):
            t = k / (per_edge + 1) + 0.013 * k
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _star(n, r_out, r_in):
    return [((r_out if k % 2 == 0 else r_in) * math.cos(math.pi * k / n),
             (r_out if k % 2 == 0 else r_in) * math.sin(math.pi * k / n)) for k in range(2 * n)]


def test_rotated_and_star_shaped_collinear_points():
    for angle in (0.0, 0.3, 0.7853981633974483, 2.1):
        rect = _rotated(_subdivided([(0, 0), (10, 0), (10, 8), (0, 8)], 3), angle)
        hole = _rotated(_subdivided([(2, 2), (6, 2), (6, 5), (2, 5)], 2), angle)
        assert_clean_holed_solid([rect, hole])
        spiky = _subdivided(_star(6, 9.0, 5.0), 2)
        small = _subdivided(_star(4, 2.5, 1.2), 1)
        assert_clean_holed_solid([_rotated(spiky, angle), _rotated(small, angle)])


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


# Rotated rectangles with collinear points, stored as float32 like sketch vertices. With the old
# absolute 1e-14 area skip these left zero-area cap faces.
FLOAT32_ROTATED_RECTS = [
    [
        [
            (0.0, 0.0), (1.2855243682861328, -0.4712546765804291), (5.228764057159424, -1.9167895317077637),
            (7.351800918579102, -2.6950643062591553), (9.389009475708008, -3.441875457763672),
            (12.142509460449219, 4.069332122802734), (2.75350022315979, 7.511207580566406),
        ],
        [
            (2.5661768913269043, 1.1894267797470093), (5.165477752685547, 0.23656050860881805),
            (6.321780681610107, -0.18732339143753052), (6.764681816101074, 1.0208567380905151),
            (7.354343414306641, 2.6293792724609375), (3.5987396240234375, 4.006129741668701),
            (2.8622944355010986, 1.9971988201141357),
        ],
    ],
    [
        [
            (0.0, 0.0), (1.1358362436294556, 9.935284614562988), (-5.341297149658203, 10.675772666931152),
            (-6.81239128112793, 10.843953132629395), (-7.948227405548096, 0.9086690545082092),
            (-6.920451641082764, 0.7911701202392578), (-6.597410202026367, 0.7542389035224915),
            (-6.022637844085693, 0.6885289549827576),
        ],
        [
            (-1.7598896026611328, 2.214224100112915), (-1.3055551052093506, 6.188337802886963),
            (-3.0873777866363525, 6.39204216003418), (-3.4093339443206787, 6.428849220275879),
            (-4.286140441894531, 6.529088973999023), (-4.740474700927734, 2.5549750328063965),
        ],
    ],
]


def test_float32_collinear_cases_have_no_zero_area_faces():
    for loops in FLOAT32_ROTATED_RECTS:
        assert_clean_holed_solid(loops)


def test_float32_quantised_collinear_points_leave_no_zero_area_faces():
    # Sketch vertices are stored as float32; points that were exactly collinear in double
    # precision are only nearly so afterwards and must not become zero-area triangles.
    for angle in (0.3, 0.7853981633974483, 1.9):
        rect = _rotated(_subdivided([(0, 0), (10, 0), (10, 8), (0, 8)], 3), angle)
        hole = _rotated(_subdivided([(2, 2), (6, 2), (6, 5), (2, 5)], 3), angle)
        assert_clean_holed_solid([[(_f32(x), _f32(y)) for x, y in loop] for loop in (rect, hole)])


# Rectangles with collinear points and a hole, 300 to 450 units from the origin and stored as
# float32 (cases 1 and 3 are rotated, case 2 is axis aligned). There the points are collinear
# only to about 2e-5, outside the old absolute 2e-6, and left zero-area cap faces.
FLOAT32_LARGE_COORDINATE_CASES = [
    [
        [
            (-114.38220977783203, 273.49932861328125), (-109.12158203125, 276.384765625),
            (-103.86095428466797, 279.27020263671875), (-98.60032653808594, 282.1556396484375),
            (-93.3396987915039, 285.0411071777344), (-88.07907104492188, 287.9265441894531),
            (-77.55780792236328, 293.6974182128906), (-72.29718017578125, 296.5828857421875),
            (-75.18262481689453, 301.843505859375), (-80.9535140991211, 312.36474609375),
            (-83.83895874023438, 317.6253967285156), (-89.60984802246094, 328.1466369628906),
            (-94.87047576904297, 325.2611999511719), (-100.131103515625, 322.3757629394531),
            (-110.65235900878906, 316.6048583984375), (-121.17361450195312, 310.833984375),
            (-131.6948699951172, 305.0630798339844), (-128.80943298339844, 299.8024597167969),
            (-125.92398834228516, 294.5418395996094), (-123.03854370117188, 289.2812194824219),
        ],
        [
            (-94.36021423339844, 311.8544921875), (-89.0995864868164, 314.7399597167969),
            (-91.98503112792969, 320.0005798339844), (-97.24565887451172, 317.1151428222656),
        ],
    ],
    [
        [
            (-300.0, 455.0), (-299.7673034667969, 449.0045166015625),
            (-299.53460693359375, 443.009033203125), (-299.3019104003906, 437.0135498046875),
            (-298.8365478515625, 425.0225830078125), (-298.6038513183594, 419.0270690917969),
            (-286.6128845214844, 419.4924621582031), (-287.0782470703125, 431.4834289550781),
            (-287.54364013671875, 443.47442626953125), (-281.54815673828125, 443.70709228515625),
            (-275.55267333984375, 443.9397888183594), (-269.55718994140625, 444.1724853515625),
            (-263.56170654296875, 444.4051818847656), (-263.7944030761719, 450.4006652832031),
            (-264.0270690917969, 456.3961486816406), (-270.0225830078125, 456.1634521484375),
            (-276.01806640625, 455.9307556152344), (-288.009033203125, 455.46539306640625),
            (-294.0045166015625, 455.2326965332031),
        ],
        [
            (-281.8971862792969, 452.7003479003906), (-281.7808532714844, 449.70257568359375),
            (-281.66448974609375, 446.704833984375), (-278.666748046875, 446.8211975097656),
            (-275.66900634765625, 446.9375305175781), (-272.6712646484375, 447.05389404296875),
            (-269.67352294921875, 447.17022705078125), (-269.7898864746094, 450.16796875),
            (-269.9062194824219, 453.16571044921875), (-272.9039611816406, 453.04937744140625),
            (-275.9017028808594, 452.9330139160156), (-278.8994445800781, 452.8166809082031),
        ],
    ],
    [
        [
            (300.0, 305.0), (305.9839172363281, 304.5610046386719), (317.9517517089844, 303.68304443359375),
            (323.9356689453125, 303.2440490722656), (329.9195861816406, 302.8050842285156),
            (335.90350341796875, 302.3660888671875), (336.3424987792969, 308.3500061035156),
            (336.781494140625, 314.33392333984375), (330.7975769042969, 314.7729187011719),
            (324.81365966796875, 315.2119140625), (318.8297424316406, 315.65087890625),
            (312.8457946777344, 316.0898742675781), (313.2847900390625, 322.07379150390625),
            (314.1627502441406, 334.0416259765625), (314.60174560546875, 340.0255432128906),
            (308.6178283691406, 340.46453857421875), (302.6339111328125, 340.90350341796875),
            (302.1949157714844, 334.9195861816406), (301.7559509277344, 328.9356689453125),
            (301.31695556640625, 322.9517517089844), (300.8779602050781, 316.96783447265625),
        ],
        [
            (318.1712646484375, 306.6750183105469), (321.1632080078125, 306.45550537109375),
            (324.1551818847656, 306.23602294921875), (327.1471252441406, 306.01654052734375),
            (330.13909912109375, 305.7970275878906), (330.35858154296875, 308.78900146484375),
            (330.57806396484375, 311.78094482421875), (327.58612060546875, 312.0004577636719),
            (324.5941467285156, 312.2199401855469), (321.6022033691406, 312.4394226074219),
            (318.6102294921875, 312.658935546875), (318.3907470703125, 309.6669616699219),
        ],
    ],
]


def test_collinear_points_and_hole_at_large_coordinates():
    for loops in FLOAT32_LARGE_COORDINATE_CASES:
        rings = [[Vector(p) for p in loop] for loop in loops]
        assert caps_module._collinear_tol(rings) > 50 * caps_module.MIN_COLLINEAR_TOL
        assert_clean_holed_solid(loops)
        assert_clean_holed_solid(loops, flip=True)


TOUCHING_OUTER = [(0, 0), (4, 0), (4, 4), (0, 4)]


def assert_touching_solid(loops, flip=False, max_move=1e-3):
    """Like assert_clean_holed_solid, for loops that touch: the cap is built on points moved
    by about 1e-5 of the extent (at least 12 float32 tolerances, which is about 5e-4 at
    coordinates around 100), so the volume matches the ideal one only approximately."""
    bm, warnings = extrude(loops, height=1.0, flip=flip)
    prep(bm)
    assert warnings == [], warnings
    assert_manifold(bm)
    area = region_area(loops)
    assert abs(signed_volume(bm) - area) < 1e-4 * area, (signed_volume(bm), area)
    assert not [f for f in bm.faces if f.calc_area() < 1e-9], "zero-area face"
    assert components(bm) == 1
    assert_outward(bm, loops, z_top=0.0 if flip else 1.0)
    for v in bm.verts:
        assert min(abs(v.co.x - x) + abs(v.co.y - y) for loop in loops for x, y in loop) < max_move
    bm.free()


def test_hole_sharing_an_outer_corner():
    for flip in (False, True):
        assert_touching_solid([TOUCHING_OUTER, [(0, 0), (2, 1), (1, 2)]], flip)


def test_hole_vertex_on_an_outer_edge():
    assert_touching_solid([TOUCHING_OUTER, [(2, 0), (3, 2), (1, 2)]])


def test_hole_sharing_part_of_an_outer_edge():
    assert_touching_solid([TOUCHING_OUTER, [(1, 0), (3, 0), (3, 2), (1, 2)]])


def test_two_holes_sharing_a_vertex():
    outer = [(0, 0), (8, 0), (8, 4), (0, 4)]
    assert_touching_solid([outer, [(1, 1), (3, 1), (3, 3), (1, 3)], [(3, 3), (5, 3), (5, 1)]])


def test_holes_side_by_side_on_an_outer_edge():
    outer = [(0, 0), (12, 0), (12, 6), (0, 6)]
    assert_touching_solid([outer, [(1, 0), (3, 0), (3, 2), (1, 2)], [(5, 0), (7, 0), (7, 2), (5, 2)]])
    # Neighbouring holes that also share an edge section.
    assert_touching_solid([outer, [(2, 0), (5, 0), (5, 3), (2, 3)], [(5, 0), (7, 0), (7, 2), (5, 2)]])


def test_touching_holes_keep_the_outer_loop_and_move_clear_of_it():
    rings = [[Vector(p) for p in TOUCHING_OUTER], [Vector(p) for p in ((1, 0), (3, 0), (3, 2), (1, 2))]]
    tol = caps_module._collinear_tol(rings)
    moved = caps_module._jitter_touching_holes(rings, tol)
    assert moved[0] is rings[0]
    # Only the two points on the outer edge move into the hole, by one to one and a half steps
    # (the factor keeps neighbouring holes from lining their moved points up).
    assert [p.to_tuple() for p in moved[1][2:]] == [(3.0, 2.0), (1.0, 2.0)]
    step = max(caps_module.JITTER_EXTENT_FRACTION * 4.0, caps_module.JITTER_MIN_TOLS * tol)
    for before, after in zip(rings[1][:2], moved[1][:2]):
        assert step - 1e-6 <= (after - before).length <= 1.5 * step + 1e-6
        assert after.y > 10 * tol, "moved points must be clear of the collinear tolerance"
    assert caps_module._jitter_touching_holes([rings[0], [Vector(p) for p in square(1, 1, 2)]], tol) is None


def test_untouched_holes_are_not_jittered():
    bm, warnings = extrude([square(0, 0, 4), square(1, 1, 2)], height=1.0)
    assert warnings == []
    assert {round(v.co.x, 9) for v in bm.verts} == {0.0, 1.0, 3.0, 4.0}
    bm.free()


REFLEX_OUTER = [(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]


def test_outer_vertex_inside_a_hole_edge():
    # The reflex vertex (5, 4) sits in the middle of the hole's top edge, no hole point near it.
    hole = [(3, 1), (7, 1), (7, 4), (3, 4)]
    for ring in (hole, hole[::-1]):
        for flip in (False, True):
            assert_touching_solid([REFLEX_OUTER, ring], flip)


def test_hole_vertex_inside_another_hole_edge():
    outer = [(0, 0), (12, 0), (12, 8), (0, 8)]
    wide = [(1, 1), (7, 1), (7, 4), (1, 4)]
    # The first hole's top edge holds a corner of the second, whose own points move too.
    for second in ([(4, 4), (6, 6), (2, 6)], [(4, 4), (6, 6), (2, 6)][::-1]):
        for flip in (False, True):
            assert_touching_solid([outer, wide, second], flip)


def test_edge_touched_by_a_foreign_point_moves_both_end_points():
    rings = [[Vector(p) for p in REFLEX_OUTER], [Vector(p) for p in ((3, 1), (7, 1), (7, 4), (3, 4))][::-1]]
    tol = caps_module._collinear_tol(rings)
    moved = caps_module._jitter_touching_holes(rings, tol)
    step = caps_module.JITTER_EXTENT_FRACTION * 10.0
    before, after = rings[1], moved[1]
    # Only the two points of the top edge (y = 4) move, and they move down off the vertex.
    for b, a in zip(before, after):
        if b.y == 4.0:
            assert a.y < 4.0 - 0.5 * step, (b, a)
        else:
            assert a == b


SHARP_OUTER = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_sharp_hole_corner_on_an_outer_corner():
    # About 1.4 degrees: moving along the bisector gains only step * 0.012 from the outer edges.
    for flip in (False, True):
        assert_touching_solid([SHARP_OUTER, [(0, 0), (8, 0.2), (8, 0)]], flip)
    rings = [[Vector(p) for p in SHARP_OUTER], [Vector(p) for p in ((0, 0), (8, 0.2), (8, 0))][::-1]]
    tol = caps_module._collinear_tol(rings)
    moved = caps_module._jitter_touching_holes(rings, tol)
    step = caps_module.JITTER_EXTENT_FRACTION * 10.0
    corner = moved[1][2]
    assert corner.x > 0.5 * step and corner.y > 0.5 * step, corner


def test_sharp_hole_corners_at_other_angles_and_positions():
    for flip in (False, True):
        # Opposite corner, 0.7 degrees.
        assert_touching_solid([SHARP_OUTER, [(10, 10), (2, 9.9), (2, 10)]], flip)
        # In the middle of an outer edge, sharing a section of it, about 3.4 degrees.
        assert_touching_solid([SHARP_OUTER, [(4, 0), (9, 0.3), (9, 0)]], flip)
        # Tip on a slanted edge of a non-square outer loop.
        assert_touching_solid([[(0, 0), (10, 0), (10, 10), (0, 6)], [(0, 6), (7, 8.4), (7, 7.6)]], flip)


def _wedge(apex, start_deg, width_deg, length):
    ax, ay = apex
    return [
        apex,
        (round(ax + length * math.cos(math.radians(start_deg)), 4), round(ay + length * math.sin(math.radians(start_deg)), 4)),
        (round(ax + length * math.cos(math.radians(start_deg + width_deg)), 4),
         round(ay + length * math.sin(math.radians(start_deg + width_deg)), 4)),
    ]


# Two holes meeting at one point of the outer edge; B lies along the edge (26.6 degrees), so the
# normals summed at the apex of A (outer edge up, B's edge across) pointed out of the material.
EDGE_APEX_OUTER = [(0, 0), (10, 0), (10, 10), (0, 10)]
EDGE_APEX_B = [(4.93, 0), (7.93, 0), (7.93, 1.5)]
EDGE_APEX_A = [(4.93, 0), (6.5476, 1.6176), (3.5575, 1.8301)]


def _quantised(loops, dx, dy):
    return [[(_f32(x + dx), _f32(y + dy)) for x, y in loop] for loop in loops]


def test_two_holes_meeting_on_an_outer_edge_stay_inside_the_outer_loop():
    loops = [EDGE_APEX_OUTER, EDGE_APEX_B, EDGE_APEX_A]
    for dx, dy in ((0.0, 0.0), (100.0, -80.0)):
        placed = _quantised(loops, dx, dy)
        for flip in (False, True):
            assert_touching_solid(placed, flip, max_move=3e-3)
        rings = [[Vector(p) for p in loop] for loop in placed]
        jittered = caps_module._jitter_touching_holes(rings, caps_module._collinear_tol(rings))
        for ring in jittered[1:]:
            for p in ring:
                assert p.y >= dy, f"hole point left the outer loop: {p}"


def test_wide_hole_wedge_meeting_another_on_an_outer_edge():
    for width in (100, 120):
        wide = _wedge((4.93, 0), 40, width, 3.0)
        for dx, dy in ((0.0, 0.0), (100.0, -80.0)):
            for flip in (False, True):
                assert_touching_solid(_quantised([EDGE_APEX_OUTER, EDGE_APEX_B, wide], dx, dy), flip, max_move=3e-3)


def test_unchecked_tilt_leaves_the_material_and_the_point_check_falls_back_to_the_bisector():
    rings = [[Vector(p) for p in loop] for loop in (EDGE_APEX_OUTER, EDGE_APEX_B, EDGE_APEX_A)]
    tol = caps_module._collinear_tol(rings)
    point = rings[1][0]
    inward = caps_module._inward_directions(rings[1])[0]
    normals = caps_module._separating_normals(rings, 1, tol)[0]
    neighbours = (rings[1][-1], rings[1][1])
    foreign = [rings[0], rings[2]]
    boxes = [caps_module._padded_box(ring, 0.0) for ring in foreign]
    distance = 20 * tol
    tilted = point + caps_module._separating_direction(inward, normals, 1.0) * distance
    assert tilted.y < 0.0, "scenario no longer pushes the full tilt out of the outer loop"
    assert not caps_module._clear_of_foreign(tilted, neighbours, foreign, boxes, tol)

    strategy = caps_module.JITTER_STRATEGIES[0]
    moved = caps_module._moved_point(point, inward, normals, strategy, distance, neighbours, foreign, boxes, tol)
    bisector = point + caps_module._separating_direction(inward, normals, 0.0) * distance
    assert moved == bisector


def test_failed_cap_retries_the_jitter_with_the_next_candidate():
    # The hole corner touches the outer edge off its bisector, so the strategies differ.
    loops = [TOUCHING_OUTER, [(1, 0), (3, 1), (1, 3)]]
    strategies = caps_module.JITTER_STRATEGIES
    scales = caps_module.JITTER_STEP_SCALES
    used = []
    capped = []
    real_jitter = caps_module._jitter_touching_holes
    real_cap = caps_module._find_valid_cap
    reject = [2]

    def jitter(rings, tol, strategy=strategies[0], pushes=None, step_scale=1.0):
        used.append((strategy, step_scale))
        return real_jitter(rings, tol, strategy, pushes, step_scale)

    def find_cap(rings, tol, attempts):
        # The plain rings keep failing for real; the jittered ones are rejected while reject
        # lasts, and recorded.
        if not used:
            return real_cap(rings, tol, attempts)
        capped.append(rings)
        if reject[0] > 0:
            reject[0] -= 1
            return None, None
        return real_cap(rings, tol, attempts)

    caps_module._jitter_touching_holes = jitter
    caps_module._find_valid_cap = find_cap
    try:
        bm, warnings = extrude(loops, height=1.0)
        assert warnings == [], warnings
        assert len(capped) == 3 and all(capped[a] != capped[b] for a in range(3) for b in range(a))
        assert [strategy for strategy, _scale in used[:2]] == list(strategies[:2])
        bm.free()

        used.clear()
        capped.clear()
        reject[0] = 10 ** 6
        bm, warnings = extrude(loops, height=1.0)
        # Every strategy at every step scale was asked for; repeats were not capped again.
        assert used == [(strategy, scale) for scale in scales for strategy in strategies]
        assert len(capped) > len(strategies) and all(capped[a] != capped[b] for a in range(len(capped)) for b in range(a))
        assert any("Cap triangulation is inconsistent" in w for w in warnings), warnings
        bm.free()
    finally:
        caps_module._jitter_touching_holes = real_jitter
        caps_module._find_valid_cap = real_cap


def run_with_deadline(fn, seconds):
    """Calls fn() and aborts it with an AssertionError once the deadline has passed, so a
    runaway loop fails its test instead of hanging the whole suite."""
    deadline = time.perf_counter() + seconds

    def tracer(_frame, _event, _arg):
        if time.perf_counter() > deadline:
            raise AssertionError(f"still running after {seconds} s")
        return tracer

    sys.settrace(tracer)
    try:
        return fn()
    finally:
        sys.settrace(None)


# Two nearly coincident points (4 and 8) both within tolerance of the line 0-1: the point on
# edge test is not transitive, and the split used to recreate the triangle (4, 0, 1) forever.
NEAR_COINCIDENT_POINTS = [
    (100, -80), (100.7475281, -72.0350037), (92.7825241, -71.2874756), (92.0350037, -79.2524719),
    (100.4670944, -75.022171), (100.1866455, -78.0086212), (97.2000046, -77.7284317),
    (97.4804306, -74.7418823), (100.4671097, -75.0215912), (98.4760666, -74.8347702),
    (98.6628342, -72.8437424), (100.6537704, -73.0307236),
]
NEAR_COINCIDENT_TRIANGLES = [
    (3, 0, 5), (5, 0, 8), (4, 0, 1), (3, 5, 6), (7, 4, 8), (7, 8, 9), (11, 8, 1), (10, 11, 1),
    (10, 1, 2), (3, 6, 7), (7, 9, 10), (7, 10, 2), (3, 7, 2),
]
NEAR_COINCIDENT_LOOP_EDGES = {
    (0, 1), (0, 3), (1, 2), (2, 3), (4, 5), (4, 7), (5, 6), (6, 7), (8, 9), (8, 11), (9, 10), (10, 11),
}


def test_split_terminates_on_nearly_coincident_points():
    pts = [Vector(p) for p in NEAR_COINCIDENT_POINTS]
    result = run_with_deadline(
        lambda: caps_module._split_at_collinear_points(
            NEAR_COINCIDENT_TRIANGLES, pts, NEAR_COINCIDENT_LOOP_EDGES, 4.03e-5), 5.0)
    assert result is None or all(len(set(t)) == 3 for t in result), result


def test_a_failed_split_is_an_invalid_attempt_that_still_leaves_a_triangle_list():
    loops = [TOUCHING_OUTER, [(1, 1), (3, 1), (3, 3), (1, 3)]]
    rings = [c.points for c in _resolved_closed(loops)]
    tol = caps_module._collinear_tol(rings)
    real_split = caps_module._split_at_collinear_points
    caps_module._split_at_collinear_points = lambda *args: None
    try:
        triangles, first = caps_module._find_valid_cap(rings, tol, [caps_module.MAX_CAP_ATTEMPTS])
        assert triangles is None and first, first
        warnings = []
        triangles, used = caps_module.triangulate_holed(rings, warnings)
        assert triangles and used is rings
        assert any("Cap triangulation is inconsistent" in w for w in warnings), warnings
    finally:
        caps_module._split_at_collinear_points = real_split


def test_cap_search_stops_at_the_attempt_budget():
    loops = [TOUCHING_OUTER, [(1, 0), (3, 1), (1, 3)]]
    rings = [c.points for c in _resolved_closed(loops)]
    real_tessellate = caps_module._tessellate
    real_valid = caps_module._cap_is_valid
    real_budget = caps_module.MAX_CAP_ATTEMPTS
    calls = []
    caps_module._tessellate = lambda *args: calls.append(args) or real_tessellate(*args)
    caps_module._cap_is_valid = lambda *args: False
    try:
        # Every attempt fails: the plain rings and then each jittered copy use up the budget,
        # which ends the search with the warning.
        caps_module.MAX_CAP_ATTEMPTS = 11
        warnings = []
        caps_module.triangulate_holed(rings, warnings)
        assert len(calls) == 11, len(calls)
        assert any("Cap triangulation is inconsistent" in w for w in warnings), warnings
    finally:
        caps_module._tessellate = real_tessellate
        caps_module._cap_is_valid = real_valid
        caps_module.MAX_CAP_ATTEMPTS = real_budget


def test_crossing_loops_with_nearly_coincident_points_finish():
    # Seed 425 of the crossing fuzz: two thin triangles from one outer corner that cross.
    outer = [(20.307796478271484, 0), (40.307796478271484, 0), (40.307796478271484, 20), (20.307796478271484, 20)]
    hole_a = [(20.307796478271484, 0), (21.063081741333008, 2.1004204750061035), (21.483278274536133, 1.8974888324737549)]
    hole_b = [(20.307796478271484, 0), (21.354856491088867, 2.7062184810638428), (21.808879852294922, 2.4832847118377686)]
    resolved = resolve_loops([outer, hole_a, hole_b])
    assert any("cross" in w for w in resolved.warnings), resolved.warnings
    # A warning about the cap is acceptable; hanging is not.
    bm, _warnings = run_with_deadline(lambda: build_extrude(resolved, 1.0, 0.0, False, True), 30.0)
    assert len(bm.faces) > 0
    bm.free()


def _circles_on_an_edge(count, segments=32, radius=4.0, pitch=10.0):
    """A rectangle with count circles that each touch its bottom edge."""
    outer = [(0, 0), (count * pitch, 0), (count * pitch, 20), (0, 20)]
    loops = [outer]
    for i in range(count):
        cx = pitch / 2 + pitch * i
        loops.append([
            (cx + radius * math.sin(2 * math.pi * k / segments), radius - radius * math.cos(2 * math.pi * k / segments))
            for k in range(segments)
        ])
    return loops


def test_touch_detection_runs_once_per_hole_not_once_per_strategy():
    real_normals = caps_module._separating_normals
    calls = []
    caps_module._separating_normals = lambda rings, k, tol: calls.append(k) or real_normals(rings, k, tol)
    try:
        bm, warnings = extrude(_circles_on_an_edge(10), height=1.0)
        assert warnings == [], warnings
        bm.free()
    finally:
        caps_module._separating_normals = real_normals
    assert sorted(calls) == list(range(1, 11)), calls


def test_many_touching_holes_on_one_edge_are_fast_and_capped_cleanly():
    loops = _circles_on_an_edge(60)
    started = time.perf_counter()
    bm, warnings = extrude(loops, height=1.0)
    elapsed = time.perf_counter() - started
    # About 0.5 s on the development machine; the quadratic version took 13 s. Not run under
    # the deadline tracer, which would slow it down several times over.
    assert elapsed < 6.0, elapsed
    prep(bm)
    assert warnings == [], warnings
    assert_manifold(bm)
    area = region_area(loops)
    assert abs(signed_volume(bm) - area) < 1e-3 * area, (signed_volume(bm), area)
    assert not [f for f in bm.faces if f.calc_area() < 1e-9], "zero-area face"
    bm.free()
