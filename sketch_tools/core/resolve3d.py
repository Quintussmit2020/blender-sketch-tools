"""3D path wire mesh -> ResolvedPath (plan section 4 in 3D, DECISIONS C8, C9, C19, C20).

Chain walking, junction splitting, closed loops and the cross-chain weld are the planar
resolver's rules with 3D distances. There are no fillets, no nesting and no winding
normalisation: a path has no inside."""

import math
from dataclasses import dataclass

import bmesh
from mathutils import Vector

PATH_NORMAL_ATTR = "st_normal"

WELD_DIST = 1e-5
# Stored normals shorter than this (zero, or a sum of opposite normals) carry no direction.
MIN_NORMAL_LENGTH = 1e-6
DEFAULT_NORMAL = (0.0, 0.0, 1.0)


@dataclass
class Chain3D:
    points: list  # list[Vector], 3D, object local
    normals: list  # list[Vector], unit, one per point
    closed: bool


@dataclass
class ResolvedPath:
    chains: list
    warnings: list


def resolve_path_bmesh(bm) -> ResolvedPath:
    """bm may be the live edit-mode bmesh: never modify it. Work on bm.copy() and free the copy."""
    warnings = []
    work = bm.copy()
    try:
        layer = work.verts.layers.float_vector.get(PATH_NORMAL_ATTR)
        if layer is None:
            warnings.append("Path has no surface normals; assuming +Z")
        if work.faces:
            bmesh.ops.delete(work, geom=list(work.faces), context='FACES_ONLY')
        _weld(work, layer)
        raw = _walk_chains(work, layer)
    finally:
        work.free()

    chains = []
    replaced = 0
    for points, raw_normals, closed in raw:
        if closed and len(points) < 3:
            closed = False
        points, raw_normals = _dedupe(points, raw_normals, closed)
        if closed and len(points) < 3:
            closed = False
        if len(points) < 2:
            continue
        normals, count = _valid_normals(raw_normals, closed)
        replaced += count
        chains.append(Chain3D(points=points, normals=normals, closed=closed))

    if replaced:
        warnings.append(
            f"{replaced} path point{'s' if replaced != 1 else ''} had no usable surface normal; "
            "used a neighbour's normal"
        )
    return ResolvedPath(chains=chains, warnings=warnings)


def _usable(normal):
    """The normal as a unit Vector, or None when it has no usable direction."""
    if normal is None:
        return None
    length = normal.length
    if not math.isfinite(length) or length < MIN_NORMAL_LENGTH:
        return None
    return normal / length


def _weld(bm, layer):
    """Weld coincident vertices across the whole path (C9). Welded normals are averaged and
    renormalised; a point's normal that has no direction does not take part."""
    targetmap = bmesh.ops.find_doubles(bm, verts=list(bm.verts), dist=WELD_DIST)["targetmap"]
    if not targetmap:
        return
    if layer is not None:
        groups = {}
        for v, target in targetmap.items():
            groups.setdefault(target, [target]).append(v)
        for target, members in groups.items():
            total = Vector((0.0, 0.0, 0.0))
            for member in members:
                unit = _usable(Vector(member[layer]))
                if unit is not None:
                    total += unit
            # Opposite normals cancel to nothing; the chain pass then repairs the point.
            target[layer] = total.normalized() if total.length >= MIN_NORMAL_LENGTH else (0.0, 0.0, 0.0)
    bmesh.ops.weld_verts(bm, targetmap=targetmap)


def _walk_chains(bm, layer):
    """Split the wire into chains at junctions (valence > 2) and endpoints (valence 1).
    Returns a list of (points, raw normals or None entries, closed)."""
    bm.verts.index_update()
    bm.edges.index_update()
    visited = set()
    result = []

    def record(verts, closed):
        points = [v.co.copy() for v in verts]
        normals = [Vector(v[layer]) if layer is not None else Vector(DEFAULT_NORMAL) for v in verts]
        result.append((points, normals, closed))

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


def _dedupe(points, normals, closed):
    """Consecutive points within the weld distance are one point. The weld is not transitive,
    so a run of closely spaced vertices can leave neighbours behind."""
    out_points = []
    out_normals = []
    for p, n in zip(points, normals):
        if not out_points or (p - out_points[-1]).length > WELD_DIST:
            out_points.append(p)
            out_normals.append(n)
    if closed:
        while len(out_points) > 1 and (out_points[0] - out_points[-1]).length <= WELD_DIST:
            out_points.pop()
            out_normals.pop()
    return out_points, out_normals


def _valid_normals(raw, closed):
    """Unit normals for a chain, and how many were replaced. A point without a usable normal
    takes the nearest usable one along the chain (earlier one on a tie), else +Z."""
    n = len(raw)
    units = [_usable(r) for r in raw]
    good = [i for i, u in enumerate(units) if u is not None]
    out = []
    replaced = 0
    for i, u in enumerate(units):
        if u is not None:
            out.append(u)
            continue
        replaced += 1
        best = None
        for j in good:
            gap = abs(i - j)
            if closed:
                gap = min(gap, n - gap)
            if best is None or gap < best[0] or (gap == best[0] and j < best[1]):
                best = (gap, j)
        out.append(units[best[1]].copy() if best else Vector(DEFAULT_NORMAL))
    return out, replaced
