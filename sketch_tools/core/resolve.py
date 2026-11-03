"""Sketch wire mesh -> ResolvedSketch (plan section 4, DECISIONS C8, C9, C11)."""

from dataclasses import dataclass, field

import bmesh
from mathutils import Vector

from . import geometry2d as g2d

FILLET_RADIUS_ATTR = "st_fillet_radius"
FILLET_ID_ATTR = "st_fillet_id"

OFF_PLANE_TOL = 1e-4
WELD_DIST = 1e-5
DEDUPE_DIST = 1e-6
BOUNDARY_DIST = 1e-6
MIN_LOOP_AREA = 1e-12


@dataclass
class Chain:
    points: list
    closed: bool
    depth: int = -1
    parent: int | None = None
    children: list = field(default_factory=list)


@dataclass
class ResolvedSketch:
    chains: list
    warnings: list


def resolve_bmesh(bm, fillet_segments: int) -> ResolvedSketch:
    """bm may be the live edit-mode bmesh: never modify it. Work on bm.copy() and free the copy."""
    warnings = []
    work = bm.copy()
    try:
        _prepare(work, warnings)
        raw = _walk_chains(work)
    finally:
        work.free()

    chains = []
    clamped = 0
    for points, radii, closed in raw:
        if closed and len(points) < 3:
            # The dropped closing segment matters for the Phase 2 sweep.
            closed = False
        points, n_clamped = _apply_fillets(points, radii, closed, fillet_segments)
        clamped += n_clamped
        points = _dedupe(points, closed)
        if closed and (len(points) < 3 or abs(g2d.signed_area(points)) < MIN_LOOP_AREA):
            closed = False
        if len(points) < 2:
            continue
        chains.append(Chain(points=points, closed=closed))

    if clamped:
        warnings.append(
            f"Fillet radius too large at {clamped} corner{'s' if clamped != 1 else ''}; clamped to fit"
        )

    _nest(chains)
    _normalise_winding(chains)
    _warn_crossings(chains, warnings)
    return ResolvedSketch(chains=chains, warnings=warnings)


def _prepare(bm, warnings):
    """Project to z = 0, drop faces, weld coincident vertices across chains (C9)."""
    if bm.faces:
        bmesh.ops.delete(bm, geom=list(bm.faces), context='FACES_ONLY')

    off_plane = 0
    for v in bm.verts:
        if abs(v.co.z) > OFF_PLANE_TOL:
            off_plane += 1
        v.co.z = 0.0
    if off_plane:
        warnings.append(
            f"{off_plane} vertex{'es' if off_plane != 1 else ''} off the sketch plane, projected onto it"
        )

    targetmap = bmesh.ops.find_doubles(bm, verts=list(bm.verts), dist=WELD_DIST)["targetmap"]
    if not targetmap:
        return
    layer = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    if layer is not None:
        for v, target in targetmap.items():
            if v[layer] > target[layer]:
                target[layer] = v[layer]
    bmesh.ops.weld_verts(bm, targetmap=targetmap)


def _walk_chains(bm):
    """Split the wire into chains at junctions (valence > 2) and endpoints (valence 1).
    Returns a list of (points, radii, closed)."""
    layer = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    bm.verts.index_update()
    bm.edges.index_update()
    visited = set()
    result = []

    def record(verts, closed):
        points = [Vector((v.co.x, v.co.y)) for v in verts]
        radii = [max(0.0, v[layer]) if layer is not None else 0.0 for v in verts]
        result.append((points, radii, closed))

    def walk(start, edge):
        verts = [start]
        cur = start
        e = edge
        while True:
            visited.add(e.index)
            nxt = e.other_vert(cur)
            if nxt.index == start.index:
                # Back at the start without passing another junction: a loop (C8).
                return verts, True
            verts.append(nxt)
            if len(nxt.link_edges) != 2:
                return verts, False
            e0, e1 = nxt.link_edges
            e = e1 if e0.index == e.index else e0
            cur = nxt

    for v in bm.verts:
        if len(v.link_edges) in (0, 2):
            continue
        for e in v.link_edges:
            if e.index not in visited:
                record(*walk(v, e))

    # Whatever is left consists only of valence-2 vertices: plain loops.
    for e in bm.edges:
        if e.index not in visited:
            record(*walk(e.verts[0], e))

    return result


def _apply_fillets(points, radii, closed, segments):
    n = len(points)
    out = []
    clamped = 0
    for i in range(n):
        interior = closed or 0 < i < n - 1
        if interior and radii[i] > 0.0 and n >= 3:
            arc, was_clamped = g2d.fillet_corner(
                points[i - 1], points[i], points[(i + 1) % n], radii[i], segments
            )
            out.extend(arc)
            clamped += was_clamped
        else:
            out.append(points[i])
    return out, clamped


def _dedupe(points, closed):
    out = []
    for p in points:
        if not out or (p - out[-1]).length > DEDUPE_DIST:
            out.append(p)
    if closed:
        while len(out) > 1 and (out[0] - out[-1]).length <= DEDUPE_DIST:
            out.pop()
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


def _nest(chains):
    """Depth = number of containing loops (even-odd, independent of winding)."""
    loops = [i for i, c in enumerate(chains) if c.closed]
    containers = {i: [] for i in loops}
    for i in loops:
        for j in loops:
            if i == j:
                continue
            p = _sample_point(chains[i].points, chains[j].points)
            if p is not None and g2d.point_in_polygon(p, chains[j].points):
                containers[i].append(j)
    for i in loops:
        chains[i].depth = len(containers[i])
    for i in loops:
        if containers[i]:
            parent = max(containers[i], key=lambda j: chains[j].depth)
            chains[i].parent = parent
            chains[parent].children.append(i)


def _normalise_winding(chains):
    """Even depth counter-clockwise, odd depth clockwise. Point 0 stays first."""
    for c in chains:
        if not c.closed:
            continue
        ccw = g2d.signed_area(c.points) > 0.0
        if ccw != (c.depth % 2 == 0):
            c.points = [c.points[0]] + c.points[:0:-1]


def _bbox(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _loops_cross(a, b):
    na, nb = len(a), len(b)
    for i in range(na):
        a0, a1 = a[i], a[(i + 1) % na]
        for k in range(nb):
            if g2d.segments_cross(a0, a1, b[k], b[(k + 1) % nb]):
                return True
    return False


def _warn_crossings(chains, warnings):
    """Loops are numbered from 1 in the order of the closed chains (C11)."""
    loops = [c for c in chains if c.closed]
    boxes = [_bbox(c.points) for c in loops]
    for i in range(len(loops)):
        for j in range(i + 1, len(loops)):
            bi, bj = boxes[i], boxes[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            if _loops_cross(loops[i].points, loops[j].points):
                warnings.append(f"Loops {i + 1} and {j + 1} cross; extrude result may be wrong")
