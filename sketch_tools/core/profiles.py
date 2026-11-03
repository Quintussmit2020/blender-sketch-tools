"""Sweep profiles in profile space (plan 6.1, DECISIONS C12, C13, C14).

u lies in the sketch plane, perpendicular to the path, positive to the left of travel; v runs
along the sketch plane normal. Closed solid rings are counter-clockwise in (u, v), holes are
clockwise, so the sweep builder can orient faces from the winding alone (C13)."""

import math
from dataclasses import dataclass

from mathutils import Vector
from mathutils.geometry import interpolate_bezier

from . import geometry2d as g2d

PLANAR_TOL = 1e-4
DEDUPE_DIST = 1e-6
BOUNDARY_DIST = 1e-6
MIN_RING_AREA = 1e-12
# Thickness is clamped to this fraction of the outer radius so the inner ring never collapses.
TUBE_MAX_THICKNESS = 0.99

# Pivot names are row then column: T/M/B (max v, middle, min v) and L/C/R (min u, centre, max u).
PIVOTS = ("TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR")


@dataclass
class ProfileRing:
    points: list  # list[tuple[float, float]] (u, v)
    closed: bool
    depth: int  # nesting depth for closed rings (0 solid, 1 hole, 2 island...), -1 for open
    parent: int | None


def _circle(radius, segments):
    return [
        (radius * math.cos(2.0 * math.pi * k / segments),
         radius * math.sin(2.0 * math.pi * k / segments))
        for k in range(segments)
    ]


def library_profile(kind, width, height, radius, thickness, segments):
    """Profile for a library kind. Returns (rings, warnings); no rings when a size is not
    positive. CUSTOM is built by the callers with profile_from_chains / profile_from_splines."""
    warnings = []
    if kind == "SQUARE":
        if width <= 0.0 or height <= 0.0:
            return [], ["Profile width and height must be greater than zero"]
        hw, hh = width * 0.5, height * 0.5
        pts = [(hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
        return [ProfileRing(pts, True, 0, None)], warnings

    if kind == "CYLINDER":
        if radius <= 0.0:
            return [], ["Profile radius must be greater than zero"]
        return [ProfileRing(_circle(radius, max(3, segments)), True, 0, None)], warnings

    if kind == "HALF_ROUND":
        if width <= 0.0:
            return [], ["Profile width must be greater than zero"]
        r = width * 0.5
        n = max(2, segments)
        # The arc runs from the right end of the flat side over the top to the left end; the
        # closing edge is the flat side on v = 0.
        pts = [
            (r * math.cos(math.pi * k / (n - 1)), r * math.sin(math.pi * k / (n - 1)))
            for k in range(n)
        ]
        return [ProfileRing(pts, True, 0, None)], warnings

    if kind == "TUBE":
        if radius <= 0.0 or thickness <= 0.0:
            return [], ["Profile radius and thickness must be greater than zero"]
        if thickness >= radius:
            thickness = radius * TUBE_MAX_THICKNESS
            warnings.append("Tube thickness clamped below the radius")
        n = max(3, segments)
        outer = _circle(radius, n)
        # Reversed so the hole is clockwise (C13).
        inner = _circle(radius - thickness, n)[::-1]
        return [ProfileRing(outer, True, 0, None), ProfileRing(inner, True, 1, 0)], warnings

    raise ValueError(f"Unknown library profile kind: {kind!r}")


def profile_from_chains(chains):
    """Rings from ResolvedSketch.chains: local x to u, local y to v. The resolver's winding,
    depth and parent indices carry over unchanged (ring index == chain index)."""
    rings = []
    for c in chains:
        pts = [(float(p.x), float(p.y)) for p in c.points]
        if c.closed:
            rings.append(ProfileRing(pts, True, c.depth, c.parent))
        else:
            rings.append(ProfileRing(pts, False, -1, None))
    warnings = [] if rings else ["Custom profile has no usable geometry"]
    return rings, warnings


def _dedupe(points, closed):
    out = []
    for p in points:
        if not out or (p - out[-1]).length > DEDUPE_DIST:
            out.append(p)
    if closed:
        while len(out) > 1 and (out[0] - out[-1]).length <= DEDUPE_DIST:
            out.pop()
    return out


def _sample_bezier(points, cyclic, resolution):
    """Each segment is sampled without its end point (it is the next segment's start), so
    joints appear once."""
    res = max(1, int(resolution))
    n = len(points)
    out = []
    segment_count = n if cyclic else n - 1
    for i in range(segment_count):
        co1, _, hr1 = points[i]
        co2, hl2, _ = points[(i + 1) % n]
        samples = interpolate_bezier(Vector(co1), Vector(hr1), Vector(hl2), Vector(co2), res + 1)
        out.extend(samples[:-1])
    if not cyclic:
        out.append(Vector(points[-1][0]))
    return out


def _sample_point(inner, outer):
    """First point of inner that is not on outer's boundary (C11), or None."""
    n = len(outer)
    for p in inner:
        if all(
            g2d.point_segment_distance(p, outer[k], outer[(k + 1) % n]) > BOUNDARY_DIST
            for k in range(n)
        ):
            return p
    return None


def _nest_and_wind(rings):
    """Even-odd depth and parent for the closed rings, then winding by depth parity."""
    polys = {i: [Vector(p) for p in r.points] for i, r in enumerate(rings) if r.closed}
    containers = {i: [] for i in polys}
    for i in polys:
        for j in polys:
            if i == j:
                continue
            p = _sample_point(polys[i], polys[j])
            if p is not None and g2d.point_in_polygon(p, polys[j]):
                containers[i].append(j)
    for i, ring in enumerate(rings):
        if not ring.closed:
            continue
        ring.depth = len(containers[i])
        if containers[i]:
            ring.parent = max(containers[i], key=lambda j: len(containers[j]))
        ccw = g2d.signed_area(polys[i]) > 0.0
        if ccw != (ring.depth % 2 == 0):
            ring.points = [ring.points[0]] + ring.points[:0:-1]


def profile_from_splines(splines):
    """Rings from plain-data splines (C14). x and y become u and v."""
    warnings = []
    rings = []
    planar_warned = False
    nurbs_warned = False

    for spline in splines:
        kind = spline["type"]
        raw = spline["points"]
        cyclic = bool(spline["cyclic"])
        if kind not in ("POLY", "BEZIER", "NURBS"):
            warnings.append(f"Unsupported spline type {kind}; skipped")
            continue
        if len(raw) < 2:
            continue

        flat = [q for p in raw for q in (p if kind == "BEZIER" else (p,))]
        if not planar_warned and any(abs(q[2]) > PLANAR_TOL for q in flat):
            warnings.append("Profile curve is not planar; projected onto the XY plane")
            planar_warned = True

        if kind == "BEZIER":
            pts3 = _sample_bezier(raw, cyclic, spline.get("resolution", 12))
        else:
            if kind == "NURBS" and not nurbs_warned:
                warnings.append("NURBS splines are sampled by their control points")
                nurbs_warned = True
            pts3 = raw
        pts = _dedupe([Vector((p[0], p[1])) for p in pts3], cyclic)

        # A cyclic spline that collapses to fewer than 3 points or zero area stays open.
        closed = cyclic and len(pts) >= 3 and abs(g2d.signed_area(pts)) >= MIN_RING_AREA
        if len(pts) < 2:
            continue
        out = [(float(p.x), float(p.y)) for p in pts]
        rings.append(ProfileRing(out, closed, 0 if closed else -1, None))

    _nest_and_wind(rings)
    if not rings:
        warnings.append("Custom profile has no usable geometry")
    return rings, warnings


def transform_profile(rings, pivot, offset_u, offset_v, rotation_deg, mirror_u):
    """New rings with, in order: mirror, rotation about the origin (positive is CCW in
    (u, v)), pivot alignment on the combined bounding box, then the offsets."""
    if pivot not in PIVOTS:
        raise ValueError(f"Unknown pivot: {pivot!r}")

    out = []
    for r in rings:
        pts = [(-u, v) for u, v in r.points] if mirror_u else list(r.points)
        if mirror_u and r.closed:
            # Mirroring flips the winding; reversing restores it. Point 0 stays first.
            pts = pts[:1] + pts[:0:-1]
        out.append(ProfileRing(pts, r.closed, r.depth, r.parent))

    if rotation_deg:
        c = math.cos(math.radians(rotation_deg))
        s = math.sin(math.radians(rotation_deg))
        for r in out:
            r.points = [(u * c - v * s, u * s + v * c) for u, v in r.points]

    all_pts = [p for r in out for p in r.points]
    if not all_pts:
        return out
    min_u = min(p[0] for p in all_pts)
    max_u = max(p[0] for p in all_pts)
    min_v = min(p[1] for p in all_pts)
    max_v = max(p[1] for p in all_pts)
    anchor_v = {"T": max_v, "M": (min_v + max_v) * 0.5, "B": min_v}[pivot[0]]
    anchor_u = {"L": min_u, "C": (min_u + max_u) * 0.5, "R": max_u}[pivot[1]]
    du = offset_u - anchor_u
    dv = offset_v - anchor_v
    for r in out:
        r.points = [(u + du, v + dv) for u, v in r.points]
    return out
