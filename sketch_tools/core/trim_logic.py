"""Decision logic of Trim and Extend (plan 8.5, 8.6, DECISIONS C16).

Plain tuples and floats only, no bpy.context and no bmesh, so everything runs headless. The
modal operator turns the sketch's edges into the segment lists below, asks for a target, and
applies the returned plan to the edit bmesh.

Points are (x, y) in sketch local space. A segment is (key, a, b) where key is whatever the
caller uses to find the edge again (a BMEdge in the operator). The view is any object with
to_screen(xy) -> (x, y) or None and ray_to_local(mouse) -> (x, y) or None."""

import math
from collections import namedtuple

from .draw_logic import Preview
from .geometry2d import SEGMENT_EPS, nearest_edge_screen, segment_intersection

# Intersections closer than this to each other, or to an end of the edge, count as the same
# point. Also the distance DECISIONS C16 welds the split vertices with.
WELD_DISTANCE = 1e-5
# Sine of the angle below which a ray and an edge count as parallel.
PARALLEL_EPS = 1e-9

# hits: ((key, u), ...) of the edges that cross at this point, u the parameter along each.
Bound = namedtuple("Bound", "t hits")
# The part of the edge between t_lo and t_hi goes. bounds holds the Bound at t_lo when
# t_lo > 0 and the one at t_hi when t_hi < 1, in that order: the points the edge is split at.
TrimPlan = namedtuple("TrimPlan", "t_lo t_hi bounds")
TrimTarget = namedtuple("TrimTarget", "key a b plan")
# s is the distance along the ray; u the parameter along the hit edge, exactly 0.0 or 1.0 when
# the ray lands on one of its end vertices.
ExtendHit = namedtuple("ExtendHit", "key s u point")
ExtendTarget = namedtuple("ExtendTarget", "key origin hit")

_P = namedtuple("_P", "x y")


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _snap_end(u, length):
    """u moved onto 0 or 1 when the point is within WELD_DISTANCE of that end of the edge, so
    callers can tell "at a vertex" from "inside the edge" without a length."""
    if u * length <= WELD_DISTANCE:
        return 0.0
    if (1.0 - u) * length <= WELD_DISTANCE:
        return 1.0
    return u


def plan_trim(a, b, segments, t_click, own_key=None):
    """Which part of edge a-b a click at parameter t_click removes.

    The edge is cut at every crossing with the other segments (shared end vertices are not
    crossings) and its own ends. The interval holding t_click goes. Crossings within
    WELD_DISTANCE of each other are one bound, and crossings that close to an end of the edge
    are that end."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    pa, pb = _P(*a), _P(*b)
    found = []
    for key, c, d in segments:
        if key == own_key:
            continue
        hit = segment_intersection(pa, pb, _P(*c), _P(*d))
        if hit is None:
            continue
        t, u = hit
        if t * length <= WELD_DISTANCE or (1.0 - t) * length <= WELD_DISTANCE:
            continue
        found.append((t, key, _snap_end(u, math.hypot(d[0] - c[0], d[1] - c[1]))))
    found.sort(key=lambda item: item[0])

    bounds = []
    for t, key, u in found:
        if bounds and (t - bounds[-1][0]) * length <= WELD_DISTANCE:
            bounds[-1][1].append((key, u))
        else:
            bounds.append((t, [(key, u)]))

    lo = hi = None
    for t, hits in bounds:
        if t <= t_click:
            lo = Bound(t, tuple(hits))
        else:
            hi = Bound(t, tuple(hits))
            break
    split = tuple(bound for bound in (lo, hi) if bound is not None)
    return TrimPlan(lo.t if lo else 0.0, hi.t if hi else 1.0, split)


def plan_extend(origin, direction, segments, own_key=None):
    """The nearest segment a ray from origin along direction meets, or None."""
    length = math.hypot(direction[0], direction[1])
    if length == 0.0:
        return None
    dx, dy = direction[0] / length, direction[1] / length
    best = None
    for key, c, e in segments:
        if key == own_key:
            continue
        ex, ey = e[0] - c[0], e[1] - c[1]
        edge_length = math.hypot(ex, ey)
        denom = dx * ey - dy * ex
        if edge_length == 0.0 or abs(denom) <= PARALLEL_EPS * edge_length:
            continue
        wx, wy = c[0] - origin[0], c[1] - origin[1]
        s = (wx * ey - wy * ex) / denom
        u = (wx * dy - wy * dx) / denom
        if s <= WELD_DISTANCE or u < -SEGMENT_EPS or u > 1.0 + SEGMENT_EPS:
            continue
        u = _snap_end(min(1.0, max(0.0, u)), edge_length)
        if best is None or s < best.s:
            best = ExtendHit(key, s, u, (origin[0] + dx * s, origin[1] + dy * s))
    return best


def _click_parameter(mouse, a, b, view, screen_t):
    """Where along a-b the click is. The cursor ray's hit on the sketch plane is exact in
    perspective too; with the plane out of reach the screen parameter is the best there is."""
    xy = view.ray_to_local(mouse)
    ab = (b[0] - a[0], b[1] - a[1])
    denom = ab[0] * ab[0] + ab[1] * ab[1]
    if xy is None or denom == 0.0:
        return screen_t
    t = ((xy[0] - a[0]) * ab[0] + (xy[1] - a[1]) * ab[1]) / denom
    return min(1.0, max(0.0, t))


def find_trim(mouse, segments, view, threshold):
    """The TrimTarget under the cursor, or None when no edge is within threshold pixels."""
    screen = [(key, view.to_screen(a), view.to_screen(b)) for key, a, b in segments]
    hit = nearest_edge_screen(mouse, screen, threshold)
    if hit is None:
        return None
    key, _distance, screen_t = hit
    a, b = next((a, b) for k, a, b in segments if k == key)
    t = _click_parameter(mouse, a, b, view, screen_t)
    return TrimTarget(key, a, b, plan_trim(a, b, segments, t, own_key=key))


def find_extend(mouse, endpoints, segments, view, threshold):
    """The ExtendTarget for the loose end nearest the cursor, or None when none is within
    threshold pixels. hit is None when the ray along the end's edge meets nothing.

    endpoints holds (key, xy, neighbor_xy, edge_key) for every vertex with one edge."""
    screen = []
    for key, xy, _neighbor, _edge_key in endpoints:
        point = view.to_screen(xy)
        screen.append((key, point, point))
    picked = nearest_edge_screen(mouse, screen, threshold)
    if picked is None:
        return None
    key = picked[0]
    _key, xy, neighbor, edge_key = next(item for item in endpoints if item[0] == key)
    direction = (xy[0] - neighbor[0], xy[1] - neighbor[1])
    return ExtendTarget(key, xy, plan_extend(xy, direction, segments, own_key=edge_key))


def trim_preview(target):
    """The interval that would go as a strip, the whole edge as the guide, bounds as markers."""
    preview = Preview()
    plan = target.plan
    preview.strips.append([lerp(target.a, target.b, plan.t_lo), lerp(target.a, target.b, plan.t_hi)])
    preview.guide = [target.a, target.b]
    preview.markers = [lerp(target.a, target.b, bound.t) for bound in plan.bounds]
    return preview


def extend_preview(target):
    preview = Preview()
    preview.snap_xy = target.origin
    if target.hit is None:
        preview.lines = ["No edge ahead"]
    else:
        preview.strips.append([target.origin, target.hit.point])
        preview.markers = [target.hit.point]
    return preview
