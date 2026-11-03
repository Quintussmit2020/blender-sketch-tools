import math

from mathutils import Vector

from sketch_tools.core import profiles
from sketch_tools.core.profiles import ProfileRing
from sketch_tools.core.resolve import Chain


def close(a, b, eps=1e-5):
    return abs(a - b) <= eps


def area(points):
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total * 0.5


def bbox(rings):
    pts = [p for r in rings for p in r.points]
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return min(us), min(vs), max(us), max(vs)


def assert_pts(points, expected, eps=1e-5):
    assert len(points) == len(expected), (points, expected)
    for p, e in zip(points, expected):
        assert close(p[0], e[0], eps) and close(p[1], e[1], eps), (points, expected)


def library(kind, width=2.0, height=1.0, radius=1.0, thickness=0.25, segments=16):
    return profiles.library_profile(kind, width, height, radius, thickness, segments)


# Library -----------------------------------------------------------------------------------

def test_square_profile():
    rings, warnings = library("SQUARE", width=4.0, height=2.0)
    assert warnings == [] and len(rings) == 1
    r = rings[0]
    assert r.closed and r.depth == 0 and r.parent is None and len(r.points) == 4
    assert area(r.points) > 0.0 and close(area(r.points), 8.0)
    assert bbox(rings) == (-2.0, -1.0, 2.0, 1.0)


def test_cylinder_profile():
    rings, warnings = library("CYLINDER", radius=1.5, segments=12)
    assert warnings == [] and len(rings) == 1
    r = rings[0]
    assert len(r.points) == 12 and r.closed and r.depth == 0
    assert area(r.points) > 0.0
    assert all(close(math.hypot(*p), 1.5) for p in r.points)


def test_half_round_profile():
    rings, warnings = library("HALF_ROUND", width=2.0, segments=9)
    assert warnings == [] and len(rings) == 1
    r = rings[0]
    assert len(r.points) == 9 and area(r.points) > 0.0
    assert close(r.points[0][0], 1.0) and close(r.points[0][1], 0.0)
    assert close(r.points[-1][0], -1.0) and abs(r.points[-1][1]) < 1e-5
    assert all(p[1] >= -1e-6 for p in r.points)
    assert all(close(math.hypot(*p), 1.0) for p in r.points)
    assert close(bbox(rings)[1], 0.0) and close(bbox(rings)[3], 1.0)
    # Only the two flat-side corners sit on v = 0.
    assert sum(1 for p in r.points if abs(p[1]) < 1e-5) == 2


def test_tube_profile():
    rings, warnings = library("TUBE", radius=1.0, thickness=0.25, segments=10)
    assert warnings == [] and len(rings) == 2
    outer, inner = rings
    assert outer.depth == 0 and outer.parent is None and area(outer.points) > 0.0
    assert inner.depth == 1 and inner.parent == 0 and area(inner.points) < 0.0
    assert len(outer.points) == 10 and len(inner.points) == 10
    assert all(close(math.hypot(*p), 1.0) for p in outer.points)
    assert all(close(math.hypot(*p), 0.75) for p in inner.points)


def test_tube_thickness_clamped():
    rings, warnings = library("TUBE", radius=1.0, thickness=1.5)
    assert len(warnings) == 1 and "clamped" in warnings[0]
    assert len(rings) == 2
    inner_r = max(math.hypot(*p) for p in rings[1].points)
    assert 0.0 < inner_r < 1.0
    assert area(rings[1].points) < 0.0


def test_zero_sizes_give_no_rings():
    cases = [
        ("SQUARE", dict(width=0.0)),
        ("SQUARE", dict(height=-1.0)),
        ("CYLINDER", dict(radius=0.0)),
        ("HALF_ROUND", dict(width=-2.0)),
        ("TUBE", dict(radius=0.0)),
        ("TUBE", dict(thickness=0.0)),
    ]
    for kind, kwargs in cases:
        rings, warnings = library(kind, **kwargs)
        assert rings == [] and len(warnings) == 1, (kind, kwargs)


def test_unknown_kind_raises():
    try:
        library("CUSTOM")
    except ValueError:
        return
    raise AssertionError("CUSTOM must not be handled by library_profile")


# Custom: chains ----------------------------------------------------------------------------

def test_profile_from_chains_square_with_hole():
    outer = Chain([Vector((-2, -2)), Vector((2, -2)), Vector((2, 2)), Vector((-2, 2))],
                  True, 0, None)
    hole = Chain([Vector((-1, -1)), Vector((-1, 1)), Vector((1, 1)), Vector((1, -1))],
                 True, 1, 0)
    line = Chain([Vector((5, 0)), Vector((6, 1))], False)
    rings, warnings = profiles.profile_from_chains([outer, hole, line])
    assert warnings == [] and len(rings) == 3
    assert (rings[0].closed, rings[0].depth, rings[0].parent) == (True, 0, None)
    assert (rings[1].closed, rings[1].depth, rings[1].parent) == (True, 1, 0)
    assert (rings[2].closed, rings[2].depth, rings[2].parent) == (False, -1, None)
    assert area(rings[0].points) > 0.0 and area(rings[1].points) < 0.0
    assert_pts(rings[2].points, [(5, 0), (6, 1)])
    assert isinstance(rings[0].points[0], tuple)


def test_profile_from_chains_empty_warns():
    rings, warnings = profiles.profile_from_chains([])
    assert rings == [] and len(warnings) == 1


# Custom: splines ---------------------------------------------------------------------------

K = 0.5523


def bezier_circle(r, ccw=True):
    pts = [
        ((r, 0, 0), (r, -K * r, 0), (r, K * r, 0)),
        ((0, r, 0), (K * r, r, 0), (-K * r, r, 0)),
        ((-r, 0, 0), (-r, K * r, 0), (-r, -K * r, 0)),
        ((0, -r, 0), (-K * r, -r, 0), (K * r, -r, 0)),
    ]
    if not ccw:
        pts = [(co, hr, hl) for co, hl, hr in pts[::-1]]
    return {"type": "BEZIER", "points": pts, "cyclic": True, "resolution": 4}


def poly(points, cyclic=True):
    return {"type": "POLY", "points": [(x, y, 0.0) for x, y in points], "cyclic": cyclic,
            "resolution": 12}


def test_poly_spline_closed_and_open():
    spline_closed = poly([(0, 0), (2, 0), (2, 1), (0, 1)])
    spline_open = poly([(5, 0), (6, 0), (6, 1)], cyclic=False)
    rings, warnings = profiles.profile_from_splines([spline_closed, spline_open])
    assert warnings == [] and len(rings) == 2
    assert rings[0].closed and rings[0].depth == 0 and area(rings[0].points) > 0.0
    assert len(rings[0].points) == 4
    assert not rings[1].closed and rings[1].depth == -1 and rings[1].parent is None
    assert_pts(rings[1].points, [(5, 0), (6, 0), (6, 1)])


def test_poly_clockwise_is_normalised():
    rings, _ = profiles.profile_from_splines([poly([(0, 0), (0, 1), (2, 1), (2, 0)])])
    assert area(rings[0].points) > 0.0


def test_poly_cyclic_with_two_points_is_open():
    rings, _ = profiles.profile_from_splines([poly([(0, 0), (1, 0)])])
    assert len(rings) == 1 and not rings[0].closed and rings[0].depth == -1


def test_poly_cyclic_repeated_first_point_is_dropped():
    rings, _ = profiles.profile_from_splines([poly([(0, 0), (1, 0), (1, 1), (0, 0)])])
    assert rings[0].closed and len(rings[0].points) == 3


def test_bezier_circle_sampling():
    rings, warnings = profiles.profile_from_splines([bezier_circle(1.0)])
    assert warnings == [] and len(rings) == 1
    r = rings[0]
    # 4 segments x resolution 4, each joint once.
    assert r.closed and len(r.points) == 16
    for i, p in enumerate(r.points):
        q = r.points[(i + 1) % 16]
        assert math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-3
    assert all(0.99 < math.hypot(*p) < 1.01 for p in r.points)
    assert area(r.points) > 0.0
    assert_pts(r.points[:1], [(1, 0)])


def test_bezier_open_sampling():
    pts = [((0, 0, 0), (-1, 0, 0), (1, 0, 0)), ((3, 0, 0), (2, 0, 0), (4, 0, 0)),
           ((6, 0, 0), (5, 0, 0), (7, 0, 0))]
    rings, _ = profiles.profile_from_splines(
        [{"type": "BEZIER", "points": pts, "cyclic": False, "resolution": 3}])
    r = rings[0]
    assert not r.closed and len(r.points) == 2 * 3 + 1
    assert_pts(r.points[:1], [(0, 0)])
    assert_pts(r.points[-1:], [(6, 0)])


def test_nurbs_warns_once_and_uses_control_points():
    nurbs = {"type": "NURBS", "points": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
             "cyclic": True, "resolution": 12}
    rings, warnings = profiles.profile_from_splines([nurbs, nurbs])
    assert warnings == ["NURBS splines are sampled by their control points"]
    assert len(rings) == 2 and rings[0].closed and len(rings[0].points) == 4


def test_non_planar_warns_once_and_projects():
    a = {"type": "POLY", "points": [(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5)], "cyclic": True,
         "resolution": 12}
    rings, warnings = profiles.profile_from_splines([a, a])
    assert len(warnings) == 1 and "planar" in warnings[0]
    assert_pts(rings[0].points, [(0, 0), (1, 0), (1, 1)])


def test_planar_within_tolerance_has_no_warning():
    a = {"type": "POLY", "points": [(0, 0, 1e-6), (1, 0, 0), (1, 1, 0)], "cyclic": True,
         "resolution": 12}
    _, warnings = profiles.profile_from_splines([a])
    assert warnings == []


def test_nested_splines_depth_and_winding():
    outer = bezier_circle(3.0)
    hole = bezier_circle(2.0)  # same direction as outer on purpose
    island = poly([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])  # clockwise
    line = poly([(10, 10), (11, 11)], cyclic=False)
    rings, warnings = profiles.profile_from_splines([outer, hole, island, line])
    assert warnings == [] and len(rings) == 4
    assert [r.depth for r in rings] == [0, 1, 2, -1]
    assert [r.parent for r in rings] == [None, 0, 1, None]
    assert area(rings[0].points) > 0.0
    assert area(rings[1].points) < 0.0
    assert area(rings[2].points) > 0.0


def test_empty_splines_warn():
    rings, warnings = profiles.profile_from_splines([])
    assert rings == [] and len(warnings) == 1


# Transform ---------------------------------------------------------------------------------

def rect(u0, v0, u1, v1):
    return ProfileRing([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], True, 0, None)


def identity(rings, **kw):
    args = dict(pivot="BL", offset_u=0.0, offset_v=0.0, rotation_deg=0.0, mirror_u=False)
    args.update(kw)
    return profiles.transform_profile(rings, **args)


def test_transform_order():
    # Rectangle u 1..3, v 0..1. Mirror: u -3..-1. Rotate 90 CCW: (u, v) -> (-v, u) gives
    # u -1..0, v -3..-1. Pivot TR moves (0, -1) to the origin. Offset adds (0.5, 0.25).
    rings = identity([rect(1, 0, 3, 1)], pivot="TR", offset_u=0.5, offset_v=0.25,
                     rotation_deg=90.0, mirror_u=True)
    b = bbox(rings)
    assert all(close(x, y) for x, y in zip(b, (-0.5, -1.75, 0.5, 0.25))), b
    # Spot-check the first corner: (1, 0) -> mirror (-1, 0) -> rotate (0, -1) -> pivot (0, 0)
    # -> offset (0.5, 0.25).
    assert_pts(rings[0].points[:1], [(0.5, 0.25)])
    assert area(rings[0].points) > 0.0


def test_pivot_grid():
    base = [rect(2, 4, 6, 10)]  # u 2..6 (centre 4), v 4..10 (middle 7)
    expected = {
        "TL": (2, 10), "TC": (4, 10), "TR": (6, 10),
        "ML": (2, 7), "MC": (4, 7), "MR": (6, 7),
        "BL": (2, 4), "BC": (4, 4), "BR": (6, 4),
    }
    for pivot, (pu, pv) in expected.items():
        out = identity(base, pivot=pivot)
        b = bbox(out)
        assert close(b[2] - b[0], 4.0) and close(b[3] - b[1], 6.0)
        assert close(b[0], 2 - pu) and close(b[1], 4 - pv), pivot
        out = identity(base, pivot=pivot, offset_u=1.0, offset_v=-2.0)
        assert close(bbox(out)[0], 2 - pu + 1.0) and close(bbox(out)[1], 4 - pv - 2.0), pivot


def test_pivot_uses_combined_bounding_box():
    rings = identity([rect(0, 0, 1, 1), rect(3, 0, 4, 2)], pivot="MC")
    assert bbox(rings) == (-2.0, -1.0, 2.0, 1.0)


def test_pivot_includes_open_rings():
    open_ring = ProfileRing([(0, 0), (10, 0)], False, -1, None)
    out = identity([rect(0, 0, 1, 1), open_ring], pivot="BR")
    assert bbox(out) == (-10.0, 0.0, 0.0, 1.0)


def test_mirror_keeps_ccw_and_hole_cw():
    outer = ProfileRing([(-2, -2), (2, -2), (2, 2), (-2, 2)], True, 0, None)
    hole = ProfileRing([(-1, -1), (-1, 1), (1, 1), (1, -1)], True, 1, 0)
    out = identity([outer, hole], pivot="MC", mirror_u=True)
    assert area(out[0].points) > 0.0 and area(out[1].points) < 0.0
    assert (out[1].depth, out[1].parent) == (1, 0)
    # An asymmetric shape really mirrors: (1, 0), (3, 0), (1, 1) becomes (2, 0), (2, 1), (0, 0) after the pivot.
    tri = ProfileRing([(1, 0), (3, 0), (1, 1)], True, 0, None)
    mirrored = identity([tri], pivot="BL", mirror_u=True, offset_u=0.0)
    assert_pts(mirrored[0].points, [(2, 0), (2, 1), (0, 0)])
    assert area(mirrored[0].points) > 0.0


def test_rotation_is_ccw_about_origin():
    ring = ProfileRing([(1, 0), (2, 0), (2, 1)], True, 0, None)
    out = profiles.transform_profile([ring], "BL", 0.0, 0.0, 90.0, False)
    # Rotated points are (0, 1), (0, 2), (-1, 2); the pivot then moves the box min (-1, 1)
    # to the origin.
    assert_pts(out[0].points, [(1, 0), (1, 1), (0, 1)])


def test_transform_does_not_mutate_input():
    ring = rect(1, 2, 3, 4)
    original = list(ring.points)
    out = identity([ring], pivot="MC", offset_u=5.0, rotation_deg=33.0, mirror_u=True)
    assert ring.points == original and ring.depth == 0
    assert out[0] is not ring and out[0].points is not ring.points


def test_transform_empty_and_bad_pivot():
    assert identity([]) == []
    try:
        identity([rect(0, 0, 1, 1)], pivot="XX")
    except ValueError:
        return
    raise AssertionError("bad pivot must raise")
