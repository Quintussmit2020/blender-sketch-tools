import math
import struct

from mathutils import Vector

from sketch_tools.core import geometry2d as g2d


def V(x, y):
    return Vector((x, y))


SQUARE = [V(0, 0), V(1, 0), V(1, 1), V(0, 1)]


def close(a, b, eps=1e-6):
    return abs(a - b) <= eps


def test_signed_area():
    assert close(g2d.signed_area(SQUARE), 1.0)
    assert close(g2d.signed_area(SQUARE[::-1]), -1.0)
    assert g2d.signed_area(SQUARE[:2]) == 0.0


def test_point_in_polygon_concave():
    # U shape: the notch between the arms is outside.
    u = [V(0, 0), V(3, 0), V(3, 3), V(2, 3), V(2, 1), V(1, 1), V(1, 3), V(0, 3)]
    assert g2d.point_in_polygon(V(0.5, 2), u)
    assert g2d.point_in_polygon(V(2.5, 2), u)
    assert g2d.point_in_polygon(V(1.5, 0.5), u)
    assert not g2d.point_in_polygon(V(1.5, 2), u)
    assert not g2d.point_in_polygon(V(4, 1), u)
    # Winding must not matter.
    assert g2d.point_in_polygon(V(0.5, 2), u[::-1])


def test_segment_intersection_cross():
    hit = g2d.segment_intersection(V(0, 0), V(2, 2), V(0, 2), V(2, 0))
    assert hit is not None
    assert close(hit[0], 0.5) and close(hit[1], 0.5)
    assert g2d.segments_cross(V(0, 0), V(2, 2), V(0, 2), V(2, 0))


def test_segment_intersection_params():
    hit = g2d.segment_intersection(V(0, 0), V(4, 0), V(1, -1), V(1, 3))
    assert hit is not None
    assert close(hit[0], 0.25) and close(hit[1], 0.25)


def test_segment_intersection_shared_endpoint_excluded():
    assert g2d.segment_intersection(V(0, 0), V(1, 0), V(1, 0), V(1, 1)) is None
    assert g2d.segment_intersection(V(0, 0), V(1, 0), V(0, 0), V(0, 1)) is None


def test_segment_intersection_t_junction():
    # b starts on the interior of a: a real contact (trim needs it), but not a crossing.
    hit = g2d.segment_intersection(V(0, 0), V(2, 0), V(1, 0), V(1, 1))
    assert hit is not None
    assert close(hit[0], 0.5) and close(hit[1], 0.0)
    assert not g2d.segments_cross(V(0, 0), V(2, 0), V(1, 0), V(1, 1))


def test_segment_intersection_misses():
    assert g2d.segment_intersection(V(0, 0), V(1, 0), V(0, 1), V(1, 1)) is None   # parallel
    assert g2d.segment_intersection(V(0, 0), V(2, 0), V(1, 0), V(3, 0)) is None   # collinear
    assert g2d.segment_intersection(V(0, 0), V(1, 0), V(2, -1), V(2, 1)) is None  # short


def test_line_line_intersection():
    p = g2d.line_line_intersection(V(0, 1), V(1, 0), V(3, 0), V(0, 1))
    assert p is not None and close(p.x, 3) and close(p.y, 1)
    assert g2d.line_line_intersection(V(0, 0), V(1, 0), V(0, 1), V(2, 0)) is None


def test_arc_points():
    c = V(0, 0)
    pts = g2d.arc_points(c, V(1, 0), V(0, 1), 4)
    assert len(pts) == 5
    assert pts[0] == V(1, 0) and pts[-1] == V(0, 1)
    for p in pts:
        assert close(p.length, 1.0)
    # Shorter way round: every point stays in the first quadrant.
    assert all(p.x >= -1e-9 and p.y >= -1e-9 for p in pts)
    # Clockwise case.
    cw = g2d.arc_points(c, V(0, 1), V(1, 0), 2)
    assert close(cw[1].x, math.sqrt(0.5)) and close(cw[1].y, math.sqrt(0.5))


def test_fillet_corner_right_angle():
    pts, clamped = g2d.fillet_corner(V(0, 0), V(2, 0), V(2, 2), 0.5, 8)
    assert not clamped
    assert len(pts) == 9
    assert (pts[0] - V(1.5, 0)).length < 1e-6
    assert (pts[-1] - V(2, 0.5)).length < 1e-6
    for p in pts:
        assert close((p - V(1.5, 0.5)).length, 0.5)


def test_fillet_corner_clamp():
    pts, clamped = g2d.fillet_corner(V(0, 0), V(1, 0), V(1, 1), 5.0, 4)
    assert clamped
    assert (pts[0] - V(0.5, 0)).length < 1e-6
    assert (pts[-1] - V(1, 0.5)).length < 1e-6


def test_fillet_corner_skips():
    # Near straight (179.8 degrees).
    a = math.radians(0.2)
    pts, clamped = g2d.fillet_corner(V(-1, 0), V(0, 0), V(math.cos(a), math.sin(a)), 0.1, 8)
    assert len(pts) == 1 and not clamped
    # Degenerate spike (0.5 degrees).
    a = math.radians(179.5)
    pts, _ = g2d.fillet_corner(V(-1, 0), V(0, 0), V(math.cos(a), math.sin(a)), 0.1, 8)
    assert len(pts) == 1
    # Zero radius and zero-length neighbour.
    assert len(g2d.fillet_corner(V(0, 0), V(1, 0), V(1, 1), 0.0, 8)[0]) == 1
    assert len(g2d.fillet_corner(V(1, 0), V(1, 0), V(1, 1), 0.2, 8)[0]) == 1


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _stored_corner(degrees, offset, rot, legs=(1.0, 3.0)):
    """A corner whose three points went through float32 storage, like mesh vertices."""
    theta = math.radians(degrees)
    o = (offset, -0.5 * offset)
    prev = V(o[0] + legs[0] * math.cos(rot), o[1] + legs[0] * math.sin(rot))
    nxt = V(o[0] + legs[1] * math.cos(rot + theta), o[1] + legs[1] * math.sin(rot + theta))
    return prev, V(*o), nxt


def _exact_fit_radius(prev, p, nxt, scale=1.0):
    """The radius whose tangent distance is exactly half the shorter stored segment, worked out
    in doubles from the stored coordinates and stored as float32 like the radius attribute."""
    ax, ay = p.x - prev.x, p.y - prev.y
    bx, by = nxt.x - p.x, nxt.y - p.y
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    theta = math.atan2(abs(ax * by - ay * bx), -(ax * bx + ay * by))
    return _f32(scale * 0.5 * min(la, lb) * math.tan(theta * 0.5))


def test_exact_fit_at_sharp_and_near_flat_corners_is_not_clamped():
    # Float32 Vector maths read these as clamped: acos near 0 and pi amplifies its error.
    degrees = (1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 160.0, 168.0, 175.0, 178.0, 179.0, 179.4)
    checked = 0
    for deg in degrees:
        for offset in (0.0, 10.0, 1000.0, 1e5):
            for rot in (0.0, 0.7, 2.3, 4.1):
                for legs in ((1.0, 3.0), (3.0, 1.0)):
                    prev, p, nxt = _stored_corner(deg, offset, rot, legs)
                    radius = _exact_fit_radius(prev, p, nxt)
                    pts, clamped = g2d.fillet_corner(prev, p, nxt, radius, 8)
                    if len(pts) == 1:
                        # At 1e5 a float32 step is 0.008, so a 179.4 degree corner can store
                        # as straight and is skipped by plan 4.1; nothing to clamp then.
                        assert offset >= 1e5, (deg, offset, rot, legs)
                        continue
                    assert len(pts) == 9 and not clamped, (deg, offset, rot, legs)
                    checked += 1
    assert checked > 0.9 * len(degrees) * 4 * 4 * 2


def test_overshoot_is_clamped_at_large_coordinates():
    # (offset, overshoot) pairs with the short leg 1 long, so the limit is 0.5. With the
    # tolerance 3e-7 times the largest coordinate, 0.1 percent is flagged at 1000, 1 percent at
    # 1e4 and 10 percent at 1e5 (C18).
    for deg in (5.0, 20.0, 45.0, 90.0, 135.0, 160.0, 175.0):
        for offset, over in ((0.0, 1.001), (1000.0, 1.001), (1e4, 1.01), (1e5, 1.1)):
            for rot in (0.0, 0.7, 2.3):
                prev, p, nxt = _stored_corner(deg, offset, rot)
                radius = _exact_fit_radius(prev, p, nxt, over)
                pts, clamped = g2d.fillet_corner(prev, p, nxt, radius, 8)
                assert clamped, (deg, offset, over, rot)
                # Clamped geometry still stays inside the segments.
                assert (pts[0] - p).length <= 0.5 * (p - prev).length * 1.0001 + 1e-3


def test_segment_parameter_epsilon_survives_float32_storage():
    # b starts 5e-7 of a's length short of a's end. Vector coordinates are float32, so the
    # stored t drifts by up to ~4e-8: above the old 1e-9 epsilon, far below the 1e-6 one.
    a0, a1 = V(0, 0), V(100, 0)
    b0, b1 = V(99.99995, -1), V(99.99995, 1)
    hit = g2d.segment_intersection(a0, a1, b0, b1)
    assert hit is not None and close(hit[1], 0.5)
    assert not g2d.segments_cross(a0, a1, b0, b1)
    assert g2d.segments_cross(a0, a1, V(99.0, -1), V(99.0, 1))
