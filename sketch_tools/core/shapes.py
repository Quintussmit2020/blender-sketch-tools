"""Shape point generators of the draw tools (C21): rectangle, circle and arcs.

Pure functions on 2D tuples, no bpy, no mathutils, so the planar tool (draw_logic) and the 3D path
tool (snap3d, which maps its plane to 2D and back) share one definition. Arcs are baked into
straight segments at draw time, with the density of a circle: max(2, ceil(circle_segments * angle
/ (2 * pi))) segments.
"""

import math

TWO_PI = 2.0 * math.pi
# Points closer than this (sketch units) are the same point; shapes through them are degenerate.
MIN_DISTANCE = 1e-9
# sin of the angle at the start point below which three points count as collinear.
COLLINEAR_SIN = 1e-9
# Float slack before ceil, so a quarter circle at 32 segments is 8 segments, not 9 (32 * (pi / 2)
# / (2 * pi) evaluates to 8.000000000000002).
COUNT_SLACK = 1e-9


def rectangle_points(a, b):
    """The four corners counter-clockwise, starting at a, with b the opposite corner. Counter-
    clockwise holds whichever side of a the corner b lies on; a degenerate (zero width or height)
    rectangle is returned as given."""
    w, h = b[0] - a[0], b[1] - a[1]
    first, second = (a[0] + w, a[1]), (a[0], a[1] + h)
    if w * h < 0.0:
        first, second = second, first
    return [tuple(a), first, tuple(b), second]


def circle_points(centre, radius, segments):
    """segments points (at least 3) from angle 0, counter-clockwise."""
    n = max(3, int(segments))
    return [(centre[0] + radius * math.cos(TWO_PI * k / n),
             centre[1] + radius * math.sin(TWO_PI * k / n)) for k in range(n)]


def arc_segments(circle_segments, sweep):
    """Segment count of an arc sweeping `sweep` radians (positive)."""
    return max(2, math.ceil(circle_segments * sweep / TWO_PI - COUNT_SLACK))


def circumcircle(p0, p1, p2):
    """(centre, radius) of the circle through three points, None when two coincide or all three
    lie on a line."""
    ax, ay = p1[0] - p0[0], p1[1] - p0[1]
    bx, by = p2[0] - p0[0], p2[1] - p0[1]
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    if la < MIN_DISTANCE or lb < MIN_DISTANCE or math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < MIN_DISTANCE:
        return None
    cross = ax * by - ay * bx
    if abs(cross) / (la * lb) < COLLINEAR_SIN:
        return None
    d = 2.0 * cross
    a2, b2 = ax * ax + ay * ay, bx * bx + by * by
    ux = (by * a2 - ay * b2) / d
    uy = (ax * b2 - bx * a2) / d
    return (p0[0] + ux, p0[1] + uy), math.hypot(ux, uy)


def _arc(centre, radius, start_angle, sweep, circle_segments, first, last):
    """Points along an arc, with first and last as the end points. sweep is signed."""
    n = arc_segments(circle_segments, abs(sweep))
    points = [tuple(first)]
    for k in range(1, n):
        angle = start_angle + sweep * k / n
        points.append((centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)))
    points.append(tuple(last))
    return points


def arc_three_points(p0, p1, p2, circle_segments):
    """Points from p0 to p2 inclusive along the circular arc that passes through p1, or None when
    the three points are collinear or two of them coincide. Runs counter-clockwise or clockwise,
    whichever visits p1 on the way; may be larger than 180 degrees."""
    circle = circumcircle(p0, p1, p2)
    if circle is None:
        return None
    centre, radius = circle
    a0 = math.atan2(p0[1] - centre[1], p0[0] - centre[0])
    a2 = math.atan2(p2[1] - centre[1], p2[0] - centre[0])
    ccw = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) > 0.0
    sweep = (a2 - a0) % TWO_PI if ccw else -((a0 - a2) % TWO_PI)
    return _arc(centre, radius, a0, sweep, circle_segments, p0, p2)


def arc_centre(centre, start, end, circle_segments):
    """Points counter-clockwise from start to end inclusive, on the circle about centre through
    start. The end point is projected onto that circle (only its direction from the centre
    counts), so the last point is centre + radius * direction. None when start or end sits on the
    centre or the end lies in the start's direction (an arc is never zero)."""
    radius = math.hypot(start[0] - centre[0], start[1] - centre[1])
    reach = math.hypot(end[0] - centre[0], end[1] - centre[1])
    if radius < MIN_DISTANCE or reach < MIN_DISTANCE:
        return None
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    sweep = (a1 - a0) % TWO_PI
    if sweep < COLLINEAR_SIN or TWO_PI - sweep < COLLINEAR_SIN:
        return None
    last = (centre[0] + radius * math.cos(a1), centre[1] + radius * math.sin(a1))
    return _arc(centre, radius, a0, sweep, circle_segments, start, last)
