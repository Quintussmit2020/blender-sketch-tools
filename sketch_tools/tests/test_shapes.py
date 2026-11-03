"""Shape point generators (core/shapes.py, C21)."""

import math

from sketch_tools.core import shapes


def close(a, b, eps=1e-9):
    return abs(a - b) <= eps


def pclose(p, q, eps=1e-9):
    return close(p[0], q[0], eps) and close(p[1], q[1], eps)


def signed_area(ring):
    return 0.5 * sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                     for i in range(len(ring)))


def angles_about(points, centre):
    return [math.atan2(y - centre[1], x - centre[0]) for x, y in points]


def unwrapped_sweep(points, centre):
    """Sum of the signed turns between consecutive points about the centre."""
    angles = angles_about(points, centre)
    total = 0.0
    for a, b in zip(angles, angles[1:]):
        total += (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return total


# -- rectangle ---------------------------------------------------------------------------------

def test_rectangle_corners_start_at_a_and_run_counter_clockwise():
    assert shapes.rectangle_points((0, 0), (2, 1)) == [(0, 0), (2, 0), (2, 1), (0, 1)]
    a = (1.0, -2.0)
    for b in ((4.0, 3.0), (-3.0, 3.0), (-3.0, -5.0), (4.0, -5.0)):
        ring = shapes.rectangle_points(a, b)
        assert len(ring) == 4 and ring[0] == a and ring[2] == b, b
        assert signed_area(ring) > 0.0, b
        w, h = abs(b[0] - a[0]), abs(b[1] - a[1])
        assert close(signed_area(ring), w * h), b


def test_rectangle_sides_are_axis_aligned():
    ring = shapes.rectangle_points((0.5, 0.5), (-1.5, 2.5))
    for i in range(4):
        p, q = ring[i], ring[(i + 1) % 4]
        assert close(p[0], q[0]) or close(p[1], q[1])


# -- circle --------------------------------------------------------------------------------------

def test_circle_points_start_at_angle_zero_and_run_counter_clockwise():
    pts = shapes.circle_points((1.0, 2.0), 3.0, 12)
    assert len(pts) == 12 and pclose(pts[0], (4.0, 2.0))
    assert all(close(math.hypot(x - 1.0, y - 2.0), 3.0) for x, y in pts)
    assert pclose(pts[3], (1.0, 5.0))
    assert signed_area(pts) > 0.0
    assert len(shapes.circle_points((0, 0), 1.0, 1)) == 3


# -- segment counts ---------------------------------------------------------------------------------

def test_arc_segment_count_follows_the_circle_density():
    counts = {deg: shapes.arc_segments(32, math.radians(deg)) for deg in (1, 10, 11.25, 11.3, 45, 90, 91, 180, 270, 359)}
    assert counts == {1: 2, 10: 2, 11.25: 2, 11.3: 2, 45: 4, 90: 8, 91: 9, 180: 16, 270: 24, 359: 32}
    assert shapes.arc_segments(32, 2.0 * math.pi) == 32
    assert shapes.arc_segments(8, math.radians(100)) == 3
    assert shapes.arc_segments(3, math.radians(10)) == 2


# -- three-point arc ---------------------------------------------------------------------------------

def test_three_point_arc_runs_counter_clockwise_between_inclusive_end_points():
    p0, p1, p2 = (1.0, 0.0), (math.cos(math.pi / 4), math.sin(math.pi / 4)), (0.0, 1.0)
    arc = shapes.arc_three_points(p0, p1, p2, 32)
    assert len(arc) == 9
    assert arc[0] == p0 and arc[-1] == p2
    assert all(close(math.hypot(x, y), 1.0) for x, y in arc)
    assert close(unwrapped_sweep(arc, (0, 0)), math.pi / 2)
    # p1 is a vertex here because the middle of 8 segments falls on 45 degrees.
    assert any(pclose(p, p1) for p in arc)


def test_three_point_arc_runs_clockwise_when_the_middle_point_says_so():
    p0, p1, p2 = (0.0, 1.0), (math.cos(math.pi / 4), math.sin(math.pi / 4)), (1.0, 0.0)
    arc = shapes.arc_three_points(p0, p1, p2, 32)
    assert arc[0] == p0 and arc[-1] == p2 and len(arc) == 9
    assert close(unwrapped_sweep(arc, (0, 0)), -math.pi / 2)


def test_three_point_arc_larger_than_180_degrees():
    # Through (0, -1): from angle 0 clockwise to angle 90, a sweep of 270 degrees.
    p0, p1, p2 = (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)
    arc = shapes.arc_three_points(p0, p1, p2, 32)
    assert len(arc) == 25 and arc[0] == p0 and arc[-1] == p2
    assert all(close(math.hypot(x, y), 1.0) for x, y in arc)
    assert close(unwrapped_sweep(arc, (0, 0)), -1.5 * math.pi)
    assert any(pclose(p, p1) for p in arc)
    # The mirror image goes the other way round, counter-clockwise through (0, 1).
    arc = shapes.arc_three_points((1.0, 0.0), (0.0, 1.0), (0.0, -1.0), 32)
    assert close(unwrapped_sweep(arc, (0, 0)), 1.5 * math.pi) and len(arc) == 25


def test_three_point_arc_off_origin_and_scaled():
    centre, r = (5.0, -3.0), 4.0
    pts = [(centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)) for a in (0.3, 1.9, 3.5)]
    arc = shapes.arc_three_points(*pts, 24)
    assert arc[0] == pts[0] and arc[-1] == pts[2]
    assert all(close(math.hypot(x - centre[0], y - centre[1]), r, 1e-9) for x, y in arc)
    assert close(unwrapped_sweep(arc, centre), 3.2)
    assert len(arc) == shapes.arc_segments(24, 3.2) + 1


def test_three_point_arc_degenerate_input_is_none():
    assert shapes.arc_three_points((0, 0), (1, 0), (2, 0), 32) is None
    assert shapes.arc_three_points((0, 0), (1, 1), (3, 3), 32) is None
    # Coincident points, any pair.
    assert shapes.arc_three_points((1, 1), (1, 1), (4, 0), 32) is None
    assert shapes.arc_three_points((1, 1), (4, 0), (1, 1), 32) is None
    assert shapes.arc_three_points((4, 0), (1, 1), (1, 1), 32) is None
    assert shapes.arc_three_points((2, 2), (2, 2), (2, 2), 32) is None
    # Almost collinear but a real arc stays an arc.
    assert shapes.arc_three_points((0, 0), (1, 1e-3), (2, 0), 32) is not None


def test_circumcircle():
    centre, radius = shapes.circumcircle((1, 0), (0, 1), (-1, 0))
    assert pclose(centre, (0, 0)) and close(radius, 1.0)
    assert shapes.circumcircle((0, 0), (1, 1), (2, 2)) is None


# -- centre arc ----------------------------------------------------------------------------------------

def test_centre_arc_runs_counter_clockwise_from_start_to_the_projected_end():
    arc = shapes.arc_centre((1.0, 1.0), (3.0, 1.0), (1.0, 4.0), 32)
    assert len(arc) == 9
    assert arc[0] == (3.0, 1.0)
    # The end was 3 away from the centre; the arc has radius 2, so it ends at (1, 3).
    assert pclose(arc[-1], (1.0, 3.0))
    assert all(close(math.hypot(x - 1.0, y - 1.0), 2.0) for x, y in arc)
    assert close(unwrapped_sweep(arc, (1.0, 1.0)), math.pi / 2)


def test_centre_arc_end_inside_the_circle_is_projected_outwards():
    arc = shapes.arc_centre((0, 0), (2, 0), (-0.1, 0.1), 32)
    assert close(math.hypot(*arc[-1]), 2.0)
    assert close(math.atan2(arc[-1][1], arc[-1][0]), math.radians(135.0))


def test_centre_arc_larger_than_180_degrees_goes_the_long_way_round():
    arc = shapes.arc_centre((0, 0), (1, 0), (0, -1), 32)
    assert len(arc) == 25
    assert close(unwrapped_sweep(arc, (0, 0)), 1.5 * math.pi)
    assert pclose(arc[-1], (0, -1))
    # Just short of a full turn.
    arc = shapes.arc_centre((0, 0), (1, 0), (1, -1e-3), 32)
    assert len(arc) == 33 and arc[0] == (1, 0)


def test_centre_arc_exactly_180_degrees():
    arc = shapes.arc_centre((0, 0), (2, 0), (-5, 0), 32)
    assert len(arc) == 17 and pclose(arc[-1], (-2, 0))


def test_centre_arc_degenerate_input_is_none():
    assert shapes.arc_centre((0, 0), (0, 0), (1, 1), 32) is None
    assert shapes.arc_centre((0, 0), (1, 0), (0, 0), 32) is None
    # The end in the start's direction, nearer or further out: a zero angle.
    assert shapes.arc_centre((0, 0), (1, 0), (5, 0), 32) is None
    assert shapes.arc_centre((0, 0), (1, 0), (0.2, 0), 32) is None
    assert shapes.arc_centre((0, 0), (1, 0), (1, 0), 32) is None


def test_centre_arc_start_is_kept_exactly():
    start = (3.123456789, -1.987654321)
    arc = shapes.arc_centre((0.5, 0.25), start, (-4.0, 2.0), 32)
    assert arc[0] == start
