"""Sweep along a 3D path (DECISIONS C19, C20; planar algorithm in builders/sweep.py).

Profile space is the planar one: u is the side direction (left of travel when looking down the
up direction), v is up, and up is the path point's surface normal. For a planar path whose
normals are all +Z this builder reproduces sweep.build_sweep exactly.

All vector maths is on plain float tuples, as in the planar builder, so the two agree to
rounding instead of to float32 Vector precision."""

import math

import bmesh

from . import sweep as planar

PARALLEL_EPS = planar.PARALLEL_EPS
SHARP_ANGLE = planar.SHARP_ANGLE
MIN_SEGMENT = planar.MIN_SEGMENT
# An up direction shorter than this after projecting the normal off the tangent has no
# direction (the normal is parallel to the path there).
MIN_UP_LENGTH = 1e-3
# Neighbouring stations whose up directions differ by more than this twist the profile through
# a half turn or more (a single-sided surface drawn from both sides).
REVERSAL_COS = math.cos(math.radians(120.0))


def build_sweep_3d(resolved_path, rings, mitre_limit: float, smooth: bool, cap_ends: bool):
    """Returns (bm, warnings). bm is a new bmesh in object local space; the caller frees it."""
    bm = bmesh.new()
    try:
        warnings = _fill(bm, resolved_path, rings, mitre_limit, smooth, cap_ends)
    except Exception:
        bm.free()
        raise
    return bm, warnings


def _fill(bm, resolved, rings, mitre_limit, smooth, cap_ends):
    warnings = []
    if not rings:
        warnings.append("No profile to sweep")
        return warnings

    usable = planar._usable_rings(rings)
    chains = [c for c in resolved.chains if len(c.points) >= 2]
    end_caps = None
    end_rings = usable
    if cap_ends and any(not c.closed for c in chains):
        end_caps, end_rings = planar._plan_end_caps(usable, warnings)

    clamped = reversals = 0
    for chain in chains:
        if chain.closed:
            c, r = _sweep_chain(bm, chain, usable, None, mitre_limit, smooth)
        else:
            c, r = _sweep_chain(bm, chain, end_rings, end_caps, mitre_limit, smooth)
        clamped += c
        reversals += r
    if clamped:
        warnings.append(f"Mitre limit clamped {clamped} corner{'s' if clamped != 1 else ''}")
    if reversals:
        # The normals are kept as drawn so the profile stays on the side the viewer saw.
        times = "once" if reversals == 1 else f"{reversals} times"
        warnings.append(f"Surface normals reverse {times} along the path; the profile twists there")
    return warnings


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _length(a):
    return math.sqrt(_dot(a, a))


def _unit_perpendicular(vec, axis):
    """vec with its component along the unit axis removed, normalised; None when nothing is left."""
    d = _dot(vec, axis)
    rest = (vec[0] - axis[0] * d, vec[1] - axis[1] * d, vec[2] - axis[2] * d)
    length = _length(rest)
    if length < MIN_UP_LENGTH:
        return None
    return _mul(rest, 1.0 / length)


def _stable_perpendicular(axis):
    reference = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (0.0, 1.0, 0.0)
    return _unit_perpendicular(reference, axis)


def _segment_directions(points, closed):
    n = len(points)
    dirs = []
    for i in range(n if closed else n - 1):
        d = _sub(points[(i + 1) % n], points[i])
        length = _length(d)
        if length < MIN_SEGMENT:
            # The resolver removes these; keep going with the previous direction if one slips by.
            dirs.append(dirs[-1] if dirs else (1.0, 0.0, 0.0))
            continue
        dirs.append(_mul(d, 1.0 / length))
    return dirs


def _fill_ups(ups, tangents, closed):
    """Gives every station without an up direction the nearest earlier station's up, projected
    off the station's own tangent (transport along the path), starting from the first station
    that has one. Stations before it on an open chain take the next one's, a path with none
    starts from any stable perpendicular."""
    n = len(ups)
    have = [i for i, up in enumerate(ups) if up is not None]
    if have:
        start = have[0]
    else:
        start = 0
        ups[0] = _stable_perpendicular(tangents[0])

    def carried(previous, i):
        return _unit_perpendicular(ups[previous], tangents[i]) or _stable_perpendicular(tangents[i])

    order = [(start + k) % n for k in range(1, n)] if closed else range(start + 1, n)
    previous = start
    for i in order:
        if ups[i] is None:
            ups[i] = carried(previous, i)
        previous = i
    if not closed:
        for i in range(start - 1, -1, -1):
            ups[i] = carried(i + 1, i)


class _Station:
    """One path point's frame. The mitre plane has normal `tangent`; a profile point (u, v)
    sits at position + stretch(side * u + up * v), where stretch scales the part of the offset
    along `turn_axis` by `stretch_factor` (None: no scaling)."""

    __slots__ = ("position", "tangent", "up", "side", "turn_axis", "stretch_factor", "turn")

    def place(self, u, v):
        off = _add(_mul(self.side, u), _mul(self.up, v))
        if self.turn_axis is not None:
            along = _dot(off, self.turn_axis) * (self.stretch_factor - 1.0)
            off = _add(off, _mul(self.turn_axis, along))
        return _add(self.position, off)


def _stations(points, normals, closed, mitre_limit):
    """One frame per path point, the number of corners the mitre limit clamped and the number of
    neighbouring station pairs whose up directions reverse (see REVERSAL_COS).

    At a corner the tangent is the bisector of the two segment directions, so the station lies
    in the bisecting plane. The offset of a profile point is stretched along the in-turn axis
    e = normalize(d1 - d0) (perpendicular to the bisector, in the plane of both segments) by
    1 / cos(half turn), which is the planar offset-line intersection; the part of the offset
    along the other perpendicular direction slides along the corner edge and is not scaled.
    The limit (C2, C12) caps that factor, keeping the sign of the offset."""
    n = len(points)
    pts = [(p[0], p[1], p[2]) for p in points]
    dirs = _segment_directions(pts, closed)

    stations = []
    clamped = 0
    ups = []
    for i in range(n):
        if closed:
            prev, cur = dirs[i - 1], dirs[i]
        elif i == 0:
            prev = cur = dirs[0]
        elif i == n - 1:
            prev = cur = dirs[-1]
        else:
            prev, cur = dirs[i - 1], dirs[i]
        st = _Station()
        st.position = pts[i]
        cross_len = _length(_cross(prev, cur))
        st.turn = None if not closed and i in (0, n - 1) else math.atan2(cross_len, _dot(prev, cur))
        if cross_len < PARALLEL_EPS:
            # Collinear or reversed: plain offset along the next segment, as in the planar builder.
            st.tangent = cur
            st.turn_axis = None
            st.stretch_factor = 1.0
        else:
            total = _add(prev, cur)
            total_len = _length(total)
            st.tangent = _mul(total, 1.0 / total_len)
            diff = _sub(cur, prev)
            st.turn_axis = _mul(diff, 1.0 / _length(diff))
            factor = 2.0 / total_len
            if factor > mitre_limit:
                factor = mitre_limit
                clamped += 1
            st.stretch_factor = factor
        normal = (normals[i][0], normals[i][1], normals[i][2])
        ups.append(_unit_perpendicular(normal, st.tangent))
        stations.append(st)

    _fill_ups(ups, [s.tangent for s in stations], closed)
    for st, up in zip(stations, ups):
        st.up = up
        st.side = _cross(up, st.tangent)
    pairs = n if closed else n - 1
    reversals = sum(1 for i in range(pairs) if _dot(ups[i], ups[(i + 1) % n]) < REVERSAL_COS)
    return stations, clamped, reversals


def _sweep_chain(bm, chain, rings, end_caps, mitre_limit, smooth):
    """Adds the swept surface of one chain; returns (clamped corners, reversals)."""
    n = len(chain.points)
    closed = chain.closed
    stations, clamped, reversals = _stations(chain.points, chain.normals, closed, mitre_limit)

    verts = [
        [[bm.verts.new(st.place(q.x, q.y)) for q in ring.points] for ring in rings]
        for st in stations
    ]

    pairs = n if closed else n - 1
    corners = [planar._profile_corners(r) for r in rings] if smooth else None
    for k in range(pairs):
        a_station = verts[k]
        b_station = verts[(k + 1) % n]
        for r, ring in enumerate(rings):
            a, b = a_station[r], b_station[r]
            m = len(a)
            for i in range(m if ring.closed else m - 1):
                j = (i + 1) % m
                # Ring CCW in (u, v) means CCW about the travel direction: (side, up, tangent)
                # is right-handed, so this winding gives the outward normal (C13).
                f = bm.faces.new((a[i], a[j], b[j], b[i]))
                f.smooth = smooth
                if smooth:
                    for idx in (i, j):
                        if corners[r][idx]:
                            bm.edges.get((a[idx], b[idx])).smooth = False

    if smooth:
        for k, st in enumerate(stations):
            if st.turn is not None and abs(st.turn) > SHARP_ANGLE:
                planar._sharpen_ring_edges(bm, verts[k], rings)

    if end_caps:
        planar._add_end_caps(bm, verts[0], verts[-1], end_caps, smooth)
    return clamped, reversals
