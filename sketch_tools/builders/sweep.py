"""Sweep builder (plan 6.2 with DECISIONS C2, C5, C12, C13, C14).

A profile ring is anything with .points [(u, v)], .closed, .depth and .parent (core.profiles
ProfileRing). u is the in-plane offset to the left of travel, v the height above the plane.
Closed solid rings must be counter-clockwise in (u, v) and holes clockwise."""

import math
from dataclasses import dataclass, replace

import bmesh
from mathutils import Vector

from . import caps as cap_module

# Cross product of the unit directions below which two segments count as collinear or reversed.
PARALLEL_EPS = 1e-6
# Corner angle above which a smooth sweep keeps an edge hard (C5).
SHARP_ANGLE = math.radians(30.0)
# Normals that cancel to less than this are a full reversal; the bisector is meaningless there.
BISECTOR_EPS = 1e-9
MIN_SEGMENT = 1e-12


def build_sweep(resolved, rings, mitre_limit: float, smooth: bool, cap_ends: bool):
    """Returns (bm, warnings). bm is a new bmesh in sketch local space; the caller frees it."""
    bm = bmesh.new()
    try:
        warnings = _fill(bm, resolved, rings, mitre_limit, smooth, cap_ends)
    except Exception:
        bm.free()
        raise
    return bm, warnings


@dataclass
class _Ring:
    points: list  # Vector (u, v)
    closed: bool
    depth: int
    parent: int | None  # index into the usable ring list


def _fill(bm, resolved, rings, mitre_limit, smooth, cap_ends):
    warnings = []
    if not rings:
        warnings.append("No profile to sweep")
        return warnings

    usable = _usable_rings(rings)
    chains = [c for c in resolved.chains if len(c.points) >= 2]
    end_rings = usable
    end_caps = None
    if cap_ends and any(not c.closed for c in chains):
        end_caps, end_rings = _plan_end_caps(usable, warnings)

    clamped = 0
    for chain in chains:
        if chain.closed:
            clamped += _sweep_chain(bm, chain, usable, None, mitre_limit, smooth)
        else:
            clamped += _sweep_chain(bm, chain, end_rings, end_caps, mitre_limit, smooth)
    if clamped:
        warnings.append(f"Mitre limit clamped {clamped} corner{'s' if clamped != 1 else ''}")
    return warnings


def _usable_rings(rings):
    """Rings that can form a surface, as plain Vector data with parents re-indexed."""
    keep = [k for k, r in enumerate(rings) if len(r.points) >= (3 if r.closed else 2)]
    new_index = {k: i for i, k in enumerate(keep)}
    return [
        _Ring(
            [Vector((float(u), float(v))) for u, v in rings[k].points],
            rings[k].closed, rings[k].depth, new_index.get(rings[k].parent),
        )
        for k in keep
    ]


def _plan_end_caps(rings, warnings):
    """Returns (caps, rings): caps is a list of (ring indices, triangles or None), the solid
    first; triangles index the concatenated points of those rings. A holed cap may move hole
    points (touching loops); the returned rings carry those moved points, so the walls share
    vertices with the cap (C3)."""
    rings = list(rings)
    caps = []
    for i, ring in enumerate(rings):
        if not ring.closed or ring.depth < 0 or ring.depth % 2:
            continue
        holes = [k for k, h in enumerate(rings) if h.closed and h.parent == i and h.depth % 2 == 1]
        group = [i] + holes
        triangles = None
        if holes:
            triangles, moved = cap_module.triangulate_holed([rings[k].points for k in group], warnings)
            for k, points in zip(group, moved):
                rings[k] = replace(rings[k], points=points)
        caps.append((group, triangles))
    return caps, rings


def _stations(points, closed, mitre_limit):
    """Per path point: (x, y, mx, my, turn). A profile point with offset u sits at
    (x + mx * u, y + my * u). The mitre is linear in u: the intersection of the two offset
    lines is the same vector scaled by u, so the limit (C2, C12) clamps that vector once per
    corner. turn is the signed path turn at the point, None at open ends. Also returns the
    number of corners the limit clamped."""
    n = len(points)
    seg_count = n if closed else n - 1
    dirs = []
    for i in range(seg_count):
        a = points[i]
        b = points[(i + 1) % n]
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length < MIN_SEGMENT:
            # The resolver removes these; keep going with the previous direction if one slips by.
            dirs.append(dirs[-1] if dirs else (1.0, 0.0))
            continue
        dirs.append((dx / length, dy / length))

    stations = []
    clamped = 0
    for i in range(n):
        if closed:
            prev, cur = dirs[i - 1], dirs[i]
        elif i == 0:
            prev = cur = dirs[0]
        elif i == n - 1:
            prev = cur = dirs[-1]
        else:
            prev, cur = dirs[i - 1], dirs[i]
        p = points[i]
        n0 = (-prev[1], prev[0])
        n1 = (-cur[1], cur[0])
        cross = prev[0] * cur[1] - prev[1] * cur[0]
        turn = None if not closed and i in (0, n - 1) else math.atan2(cross, prev[0] * cur[0] + prev[1] * cur[1])
        if abs(cross) < PARALLEL_EPS:
            m = n1
        else:
            # n0 + s * prev == n1 + t * cur, solved for s with a cross product against cur.
            wx, wy = n1[0] - n0[0], n1[1] - n0[1]
            s = (wx * cur[1] - wy * cur[0]) / cross
            m = (n0[0] + prev[0] * s, n0[1] + prev[1] * s)
            if math.hypot(m[0], m[1]) > mitre_limit:
                bx, by = n0[0] + n1[0], n0[1] + n1[1]
                blen = math.hypot(bx, by)
                if blen < BISECTOR_EPS:
                    bx, by, blen = n1[0], n1[1], 1.0
                m = (bx / blen * mitre_limit, by / blen * mitre_limit)
                clamped += 1
        stations.append((p.x, p.y, m[0], m[1], turn))
    return stations, clamped


def _profile_corners(ring):
    """Per ring point: True when the profile's own corner there exceeds the sharp angle.
    Ends of an open ring have only one edge and so no corner."""
    pts = ring.points
    n = len(pts)
    flags = []
    for i in range(n):
        if not ring.closed and i in (0, n - 1):
            flags.append(False)
            continue
        a = pts[i] - pts[i - 1]
        b = pts[(i + 1) % n] - pts[i]
        if a.length < MIN_SEGMENT or b.length < MIN_SEGMENT:
            flags.append(False)
            continue
        flags.append(abs(math.atan2(a.x * b.y - a.y * b.x, a.dot(b))) > SHARP_ANGLE)
    return flags


def _sweep_chain(bm, chain, rings, end_caps, mitre_limit, smooth):
    """Adds the swept surface of one chain; returns the number of clamped corners."""
    points = chain.points
    n = len(points)
    closed = chain.closed
    stations, clamped = _stations(points, closed, mitre_limit)

    verts = [
        [[bm.verts.new((x + mx * q.x, y + my * q.x, q.y)) for q in ring.points] for ring in rings]
        for x, y, mx, my, _turn in stations
    ]

    pairs = n if closed else n - 1
    corners = [_profile_corners(r) for r in rings] if smooth else None
    for k in range(pairs):
        a_station = verts[k]
        b_station = verts[(k + 1) % n]
        for r, ring in enumerate(rings):
            a, b = a_station[r], b_station[r]
            m = len(a)
            for i in range(m if ring.closed else m - 1):
                j = (i + 1) % m
                # Ring CCW in (u, v) means CCW about the travel direction: with the far
                # station last, this winding gives t x d, the outward normal (C13).
                f = bm.faces.new((a[i], a[j], b[j], b[i]))
                f.smooth = smooth
                if smooth:
                    for idx in (i, j):
                        if corners[r][idx]:
                            bm.edges.get((a[idx], b[idx])).smooth = False

    if smooth:
        for k, (_x, _y, _mx, _my, turn) in enumerate(stations):
            if turn is not None and abs(turn) > SHARP_ANGLE:
                _sharpen_ring_edges(bm, verts[k], rings)

    if end_caps:
        _add_end_caps(bm, verts[0], verts[-1], end_caps, smooth)
    return clamped


def _sharpen_ring_edges(bm, station, rings):
    for r, ring in enumerate(rings):
        vs = station[r]
        m = len(vs)
        for i in range(m if ring.closed else m - 1):
            bm.edges.get((vs[i], vs[(i + 1) % m])).smooth = False


def _add_end_caps(bm, first, last, end_caps, smooth):
    """Start cap faces against travel (normal -d), end cap along it (+d). Rings are CCW about
    +d, so the end cap keeps the ring order and the start cap reverses it."""
    for group, triangles in end_caps:
        for station, reverse in ((last, False), (first, True)):
            loops = [station[k] for k in group]
            cap_module.add_cap(bm, loops, triangles, reverse=reverse)
            if smooth:
                # Caps are flat; a shared smooth edge would blend their normals into the walls.
                for vs in loops:
                    for i in range(len(vs)):
                        bm.edges.get((vs[i], vs[(i + 1) % len(vs)])).smooth = False
