"""Extrude builder (plan 5 with DECISIONS C1 and C3)."""

import bmesh

from . import caps as cap_module


def build_extrude(resolved, height: float, offset: float, flip: bool, caps: bool):
    """Returns (bm, warnings). bm is a new bmesh in sketch local space; the caller frees it."""
    bm = bmesh.new()
    try:
        warnings = _fill(bm, resolved, height, offset, flip, caps)
    except Exception:
        bm.free()
        raise
    return bm, warnings


def _fill(bm, resolved, height, offset, flip, caps):
    warnings = []
    chains = resolved.chains

    n_open = sum(1 for c in chains if not c.closed)
    if n_open:
        warnings.append(f"{n_open} open chain{'s' if n_open != 1 else ''} ignored by Extrude")

    h = -height if flip else height
    if abs(h) < 1e-9:
        warnings.append("Extrude height is zero; nothing to build")
        return warnings

    z0 = offset
    z1 = offset + h
    for chain in chains:
        if not chain.closed or chain.depth < 0 or chain.depth % 2:
            continue
        holes = [
            chains[k] for k in chain.children
            if chains[k].closed and chains[k].depth % 2 == 1
        ]
        _build_group(bm, [chain] + holes, z0, z1, caps, warnings)
    return warnings


def _build_group(bm, loops, z0, z1, caps, warnings):
    """One solid: outer loop (CCW) plus its holes (CW). Walls and caps share vertices (C3)."""
    up = z1 > z0
    rings = [loop.points for loop in loops]
    triangles = None
    if caps and len(rings) > 1:
        # May return jittered rings; walls are built from the same points as the cap.
        triangles, rings = cap_module.triangulate_holed(rings, warnings)

    bottoms = []
    tops = []
    for ring in rings:
        b = [bm.verts.new((p.x, p.y, z0)) for p in ring]
        t = [bm.verts.new((p.x, p.y, z1)) for p in ring]
        n = len(b)
        for i in range(n):
            j = (i + 1) % n
            # CCW outer / CW hole with z rising gives outward normals; reverse when z falls.
            quad = (b[i], b[j], t[j], t[i]) if up else (t[i], t[j], b[j], b[i])
            bm.faces.new(quad)
        bottoms.append(b)
        tops.append(t)

    if caps:
        cap_module.add_cap(bm, tops, triangles, reverse=not up)
        cap_module.add_cap(bm, bottoms, triangles, reverse=up)
