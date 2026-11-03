"""2D helpers on mathutils.Vector (x, y) points in sketch local space."""

import math

from mathutils import Vector

# Fillet skip thresholds from plan 4.1.
FILLET_MIN_ANGLE = math.radians(1.0)
FILLET_MAX_ANGLE = math.radians(179.5)
# Relative slack for a fillet that fits exactly (C18): 5x the float32 rounding of the radius
# attribute, and still flags 0.1 percent over up to coordinates of about 1e3.
FILLET_FIT_TOLERANCE = 3e-7

# Segment parameters t and u are fractions of a segment; 1e-6 stays well above float32
# rounding (about 6e-8) so endpoints shared by two loops read as endpoints.
SEGMENT_EPS = 1e-6
# Sine of the angle below which two segments count as parallel.
PARALLEL_EPS = 1e-9


def cross2(a, b):
    return a.x * b.y - a.y * b.x


def signed_area(points):
    """Shoelace area; positive for counter-clockwise polygons."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        p = points[i]
        q = points[(i + 1) % n]
        total += p.x * q.y - q.x * p.y
    return total * 0.5


def point_in_polygon(pt, polygon):
    """Even-odd rule. Points exactly on the boundary may land either way; callers that care
    should pick a sample point off the boundary first (see point_segment_distance)."""
    x, y = pt.x, pt.y
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_segment_distance(p, a, b):
    ab = b - a
    denom = ab.length_squared
    if denom == 0.0:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    return (p - (a + ab * t)).length


def nearest_edge_screen(point, segments, max_distance):
    """The segment closest to point, as (key, distance, t), or None when none is within
    max_distance. segments yields (key, p0, p1) with screen (x, y) points; a segment with a
    None end (behind the camera) is skipped. t is the clamped parameter of the closest point on
    that screen segment. The first of equally close segments wins."""
    px, py = point
    best = None
    for key, p0, p1 in segments:
        if p0 is None or p1 is None:
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0.0 else max(0.0, min(1.0, ((px - p0[0]) * dx + (py - p0[1]) * dy) / denom))
        distance = math.hypot(px - (p0[0] + dx * t), py - (p0[1] + dy * t))
        if distance <= max_distance and (best is None or distance < best[1]):
            best = (key, distance, t)
    return best


def segment_intersection(a0, a1, b0, b1, eps=SEGMENT_EPS):
    """Intersection of segments a0-a1 and b0-b1 as parameters (t, u), so the point is
    a0 + (a1 - a0) * t == b0 + (b1 - b0) * u. eps is a tolerance on t and u.

    Returns None for parallel or collinear segments, for segments that do not meet, and when
    the only contact is an endpoint shared by both segments (t and u both at 0 or 1).
    Touching at one segment's endpoint in the other's interior (a T) is returned.

    Computed in Python floats: Vector arithmetic is float32, which can read a shared corner
    as t = 0.99999997 and report a crossing that is not there."""
    ax, ay = a0.x, a0.y
    rx, ry = a1.x - ax, a1.y - ay
    sx, sy = b1.x - b0.x, b1.y - b0.y
    denom = rx * sy - ry * sx
    if denom == 0.0 or abs(denom) <= PARALLEL_EPS * math.hypot(rx, ry) * math.hypot(sx, sy):
        return None
    qx, qy = b0.x - ax, b0.y - ay
    t = (qx * sy - qy * sx) / denom
    u = (qx * ry - qy * rx) / denom
    if t < -eps or t > 1.0 + eps or u < -eps or u > 1.0 + eps:
        return None
    t = min(1.0, max(0.0, t))
    u = min(1.0, max(0.0, u))
    t_end = t <= eps or t >= 1.0 - eps
    u_end = u <= eps or u >= 1.0 - eps
    if t_end and u_end:
        return None
    return t, u


def segments_cross(a0, a1, b0, b1, eps=SEGMENT_EPS):
    """True when the segments cross at a point strictly inside both of them."""
    hit = segment_intersection(a0, a1, b0, b1, eps)
    if hit is None:
        return False
    t, u = hit
    return eps < t < 1.0 - eps and eps < u < 1.0 - eps


def line_line_intersection(p1, d1, p2, d2, eps=1e-12):
    """Intersection of the infinite lines p1 + s * d1 and p2 + s * d2, or None if parallel."""
    denom = cross2(d1, d2)
    if abs(denom) < eps:
        return None
    s = cross2(p2 - p1, d2) / denom
    return p1 + d1 * s


def arc_points(center, start, end, segments):
    """segments + 1 points from start to end around center, taking the shorter way round.
    The radius is |start - center|; start and end are returned exactly."""
    segments = max(1, int(segments))
    v0 = start - center
    v1 = end - center
    radius = v0.length
    a0 = math.atan2(v0.y, v0.x)
    a1 = math.atan2(v1.y, v1.x)
    delta = a1 - a0
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    pts = [start.copy()]
    for k in range(1, segments):
        a = a0 + delta * k / segments
        pts.append(Vector((center.x + radius * math.cos(a), center.y + radius * math.sin(a))))
    pts.append(end.copy())
    return pts


def fillet_corner(prev, p, nxt, radius, segments):
    """Round corner p between prev -> p and p -> nxt (plan 4.1).

    Returns (points, clamped). points replaces p: segments + 1 arc points from the tangent
    point on the incoming segment to the one on the outgoing segment, or [p] when the corner
    is skipped (radius <= 0, zero-length neighbour, near-straight or degenerate spike).
    clamped is True when the radius had to shrink so the tangent distance stays within half
    of the shorter adjacent segment (by more than FILLET_FIT_TOLERANCE, see C18).

    The decision runs in Python floats: Vector maths is float32, and acos near 0 and pi turns
    that into a visible error in the tangent distance, which read exact fits as clamps at
    sharp and near-flat corners."""
    px, py = p.x, p.y
    ax, ay = px - prev.x, py - prev.y
    bx, by = nxt.x - px, nxt.y - py
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    if radius <= 0.0 or la < 1e-9 or lb < 1e-9:
        return [p.copy()], False
    ahx, ahy = ax / la, ay / la
    bhx, bhy = bx / lb, by / lb
    # atan2 of cross and dot stays accurate where acos of the dot does not.
    theta = math.atan2(abs(ahx * bhy - ahy * bhx), -(ahx * bhx + ahy * bhy))
    if theta < FILLET_MIN_ANGLE or theta > FILLET_MAX_ANGLE:
        return [p.copy()], False
    half = theta * 0.5
    t = radius / math.tan(half)
    r = radius
    limit = 0.5 * min(la, lb)
    clamped = False
    if t > limit:
        # A radius that fits exactly lands on the limit up to the float32 storage of the
        # sketch: the stored coordinates are rounded to about 6e-8 of their magnitude, so the
        # slack scales with the coordinates, not with the segment length. Within it the corner
        # is shrunk by that sliver only and not reported (C18).
        scale = max(limit, abs(px), abs(py), abs(prev.x), abs(prev.y), abs(nxt.x), abs(nxt.y))
        clamped = t - limit > FILLET_FIT_TOLERANCE * scale
        t = limit
        r = t * math.tan(half)
    t1 = Vector((px - ahx * t, py - ahy * t))
    t2 = Vector((px + bhx * t, py + bhy * t))
    dx, dy = bhx - ahx, bhy - ahy
    d = math.hypot(dx, dy)
    k = r / math.sin(half) / d
    center = Vector((px + dx * k, py + dy * k))
    return arc_points(center, t1, t2, segments), clamped
