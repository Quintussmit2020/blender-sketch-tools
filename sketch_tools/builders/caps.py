"""Caps for flat (u, v) polygon groups: one ngon, or a validated triangulation when the group
has holes. Shared by the extrude and sweep builders (DECISIONS C1, C3, C14)."""

import math

import bmesh
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

from ..core.geometry2d import cross2, point_in_polygon, point_segment_distance, segments_cross, signed_area

# Cap triangles are exactly coplanar, so a tiny angle merges them without touching anything else.
CAP_DISSOLVE_ANGLE = 1e-3

# Distance from a point to a line below which the point counts as lying on it. One value for
# both "skip this triangle as a sliver" and "this point splits that edge", otherwise a
# near-collinear triple survives as a triangle that float32 storage flattens to zero area.
# Vertex coordinates are stored as float32, whose rounding error grows with the coordinate
# (about 6e-8 per unit), so the tolerance grows with it too; see _collinear_tol.
MIN_COLLINEAR_TOL = 2e-6
COLLINEAR_TOL_PER_UNIT = 4e-7

# Last resort for loops that touch (shared corner, vertex on an edge, shared edge section):
# touching rings cannot be capped without zero-area triangles, so the touching hole points
# are moved this fraction of the group extent into their hole. Never less than
# JITTER_MIN_TOLS collinear tolerances, so the moved points are clearly off the other loop.
JITTER_EXTENT_FRACTION = 1e-5
JITTER_MIN_TOLS = 12.0
GOLDEN_FRACTION = 0.6180339887498949
# A moved point has to end this many tolerances away from every foreign segment.
JITTER_CLEARANCE_TOLS = 3.0
# Weights of the touched edges' normals added to the inward bisector, per strategy. A
# strategy with several weights tries them in order and validates all but the last, which is
# the fallback; the whole cap is retried with the next strategy when these rings fail it.
# Tilting gains distance from a sharp corner's edges but can leave the material side, the
# bisector alone is always inside the hole but gains little at sharp corners.
JITTER_STRATEGIES = ((1.0, 0.5, 0.0), (0.0,), (1.0,))
# Many touching holes along one edge line their moved points up almost collinearly, and the
# resulting slivers cannot be told from real triangles. How near depends on the step, so after
# every strategy failed the whole set is retried with a larger step.
JITTER_STEP_SCALES = (1.0, 2.0, 4.0)

# The scanfill behind tessellate_polygon is orientation sensitive; a failed tessellation is
# retried on the same points rotated by these angles (radians).
RETRY_ANGLES = (0.0, 0.61, 1.27, 2.03, 2.71, 3.67, 4.43, 5.39)
CAP_AREA_TOL = 1e-6
# Upper bound on tessellation attempts for one cap, over the plain rings and every jittered
# copy. The loops below are finite already; this keeps a pathological group from multiplying
# them into a long freeze, and ends it in the "inconsistent" warning instead.
MAX_CAP_ATTEMPTS = 48


def add_cap(bm, verts_per_loop, triangles, reverse):
    """reverse=False builds the cap counter-clockwise seen from +z (normal +z).
    triangles is None for a cap without holes."""
    if triangles is None:
        verts = verts_per_loop[0]
        bm.faces.new(verts[::-1] if reverse else verts)
        return

    flat_verts = [v for verts in verts_per_loop for v in verts]
    faces = []
    for a, b, c in triangles:
        tri = (flat_verts[a], flat_verts[c], flat_verts[b]) if reverse else (flat_verts[a], flat_verts[b], flat_verts[c])
        try:
            faces.append(bm.faces.new(tri))
        except ValueError:
            # Only reachable after an inconsistent triangulation, which already warned.
            pass
    if not faces:
        return

    # faces.new leaves normals at zero, and dissolve_limit compares face normals.
    for f in faces:
        f.normal_update()
    face_set = set(faces)
    interior = {
        e for f in faces for e in f.edges
        if len(e.link_faces) == 2 and all(lf in face_set for lf in e.link_faces)
    }
    if interior:
        bmesh.ops.dissolve_limit(
            bm,
            angle_limit=CAP_DISSOLVE_ANGLE,
            use_dissolve_boundaries=False,
            verts=[],
            edges=list(interior),
            delimit=set(),
        )


def _collinear_tol(rings):
    biggest = max(max(abs(p.x), abs(p.y)) for ring in rings for p in ring)
    return max(MIN_COLLINEAR_TOL, COLLINEAR_TOL_PER_UNIT * biggest)


def triangulate_holed(rings, warnings):
    """Returns (triangles, rings): counter-clockwise triangles (indices into the concatenated
    ring points) that cover the outer ring minus the holes exactly, and the ring points they
    are valid for. Those are the input rings, or a slightly jittered copy when loops touch.
    When nothing validates: the first attempt on the input rings plus a warning.
    tessellate_polygon alone is not trusted: it can return triangles that overlap a hole or
    duplicate a sliver pair."""
    tol = _collinear_tol(rings)
    attempts = [MAX_CAP_ATTEMPTS]
    triangles, first = _find_valid_cap(rings, tol, attempts)
    if triangles is not None:
        return triangles, rings

    # Touch detection does not depend on the strategy, so it runs once.
    pushes = [None] + [_separating_normals(rings, k, tol) for k in range(1, len(rings))]
    if any(normals for hole in pushes[1:] for normals in hole):
        tried = []
        for scale in JITTER_STEP_SCALES:
            for strategy in JITTER_STRATEGIES:
                jittered = _jitter_touching_holes(rings, tol, strategy, pushes, scale)
                # Strategies only differ where a touched point has a tilt to choose.
                if jittered in tried:
                    continue
                tried.append(jittered)
                triangles, _first = _find_valid_cap(jittered, tol, attempts)
                if triangles is not None:
                    return triangles, jittered

    warnings.append("Cap triangulation is inconsistent")
    return first, rings


def _find_valid_cap(rings, tol, attempts):
    """(triangles, first_attempt); triangles is None when no rotation validates or the shared
    attempts budget (a one item list, spent across calls) runs out."""
    flat_pts = [p for ring in rings for p in ring]
    loop_edges = set()
    start = 0
    for ring in rings:
        n = len(ring)
        for i in range(n):
            loop_edges.add(_edge_key(start + i, start + (i + 1) % n))
        start += n

    expected = abs(signed_area(rings[0])) - sum(abs(signed_area(r)) for r in rings[1:])
    first = None
    for angle in RETRY_ANGLES:
        if attempts[0] <= 0:
            break
        attempts[0] -= 1
        cleaned = _clean_triangles(_tessellate(rings, angle), flat_pts, tol)
        triangles = _split_at_collinear_points(cleaned, flat_pts, loop_edges, tol)
        if triangles is not None and _cap_is_valid(triangles, flat_pts, loop_edges, expected, tol):
            return triangles, first
        if first is None:
            first = cleaned if triangles is None else triangles
    return None, first


def _jitter_touching_holes(rings, tol, strategy=JITTER_STRATEGIES[0], pushes=None, step_scale=1.0):
    """Copy of rings in which every hole point that touches another ring is moved into its own
    hole, or None when nothing touches. Touching means a hole point within tol of another
    ring's edge, or another ring's point within tol of the interior of a hole edge (both end
    points of that edge move). The outer ring stays put: at a shared corner both rings would
    otherwise move the same way. strategy: see JITTER_STRATEGIES, step_scale: see
    JITTER_STEP_SCALES. pushes: per ring the result of _separating_normals (index 0 unused),
    computed here when not given."""
    xs = [p.x for p in rings[0]]
    ys = [p.y for p in rings[0]]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    step = step_scale * max(JITTER_EXTENT_FRACTION * extent, JITTER_MIN_TOLS * tol)

    moved = 0
    result = [rings[0]]
    boxes = [_padded_box(ring, 0.0) for ring in rings]
    for k in range(1, len(rings)):
        ring = rings[k]
        # Earlier holes are already moved, later ones not yet: validate against what is there.
        foreign = result + rings[k + 1:]
        foreign_boxes = boxes[:k] + boxes[k + 1:]
        hole_normals = pushes[k] if pushes else _separating_normals(rings, k, tol)
        new_ring = []
        for i, (p, direction, normals) in enumerate(zip(ring, _inward_directions(ring), hole_normals)):
            if normals:
                # Equal steps would line up points of neighbouring holes that sat on one
                # edge, and a point on another triangle's edge is what the cap cannot take.
                moved += 1
                distance = step * (1.0 + 0.5 * ((moved * GOLDEN_FRACTION) % 1.0))
                neighbours = (ring[i - 1], ring[(i + 1) % len(ring)])
                p = _moved_point(p, direction, normals, strategy, distance, neighbours, foreign, foreign_boxes, tol)
            new_ring.append(p)
        result.append(new_ring)
        boxes[k] = _padded_box(new_ring, 0.0)
    return result if moved else None


def _moved_point(p, inward, normals, strategy, distance, neighbours, foreign, boxes, tol):
    """p moved along the first direction of the strategy that keeps it on the material side;
    the last direction is used unchecked."""
    for weight in strategy[:-1]:
        q = p + _separating_direction(inward, normals, weight) * distance
        if _clear_of_foreign(q, neighbours, foreign, boxes, tol):
            return q
    return p + _separating_direction(inward, normals, strategy[-1]) * distance


def _clear_of_foreign(q, neighbours, foreign, boxes, tol):
    """True when q is inside the outer ring and outside every other hole, clear of all their
    segments, and the edges from q to its neighbours cross none of them. boxes: the bounding
    box of every foreign ring, which lets the many rings far from q be skipped."""
    if not point_in_polygon(q, foreign[0]):
        return False
    clearance = JITTER_CLEARANCE_TOLS * tol
    region = _padded_box((q,) + tuple(neighbours), clearance)
    for index, (other, box) in enumerate(zip(foreign, boxes)):
        if not _boxes_overlap(region, box):
            continue
        if index and point_in_polygon(q, other):
            return False
        m = len(other)
        for a in range(m):
            c, d = other[a], other[(a + 1) % m]
            if point_segment_distance(q, c, d) < clearance:
                return False
            if any(segments_cross(q, nb, c, d) for nb in neighbours):
                return False
    return True


def _separating_normals(rings, k, tol):
    """Per point of hole k: unit normals of the touched edges, each pointing to the side the
    point has to move to (the cap material side of another ring's edge, the hole's own inside
    for one of its edges that another ring's point lies in). Empty when the point is free."""
    ring = rings[k]
    n = len(ring)
    own = _inward_edge_normals(ring)
    normals = [[] for _ in ring]
    box = _padded_box(ring, tol)
    for j, other in enumerate(rings):
        # Only rings within tol of this hole can touch it; most pairs are far apart.
        if j == k or not _boxes_overlap(box, _padded_box(other, tol)):
            continue
        m = len(other)
        material = _inward_edge_normals(other)
        if j != 0:
            material = [-v for v in material]
        for a in range(m):
            if material[a].length_squared == 0.0:
                continue
            for i, p in enumerate(ring):
                if point_segment_distance(p, other[a], other[(a + 1) % m]) < tol:
                    normals[i].append(material[a])
        for q in other:
            for i in range(n):
                i2 = (i + 1) % n
                a, b = ring[i], ring[i2]
                # Near an end point q is a touching hole point already, handled above.
                if (q - a).length >= tol and (q - b).length >= tol and point_segment_distance(q, a, b) < tol:
                    normals[i].append(own[i])
                    normals[i2].append(own[i])
    return normals


def _padded_box(ring, pad):
    xs = [p.x for p in ring]
    ys = [p.y for p in ring]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _boxes_overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _separating_direction(inward, normals, weight):
    """The hole's inward direction tilted towards the touched edges' normals by weight. Moving
    along the bisector alone gains only sin(angle / 2) from an edge at a sharp corner, so a
    corner of a few degrees would stay within tolerance of the loop it touches."""
    total = inward.copy()
    for v in normals:
        total += v * weight
    return total.normalized() if total.length > 1e-6 else inward


def _inward_edge_normals(ring):
    """Unit normal of every edge (ring[i] -> ring[i + 1]) pointing into the ring; zero for a
    zero-length edge."""
    side = 1.0 if signed_area(ring) > 0.0 else -1.0
    n = len(ring)
    normals = []
    for i in range(n):
        d = ring[(i + 1) % n] - ring[i]
        length = d.length
        normals.append(Vector((-d.y, d.x)) * (side / length) if length > 1e-12 else Vector((0.0, 0.0)))
    return normals


def _inward_directions(ring):
    """Unit direction into the ring's own interior at every point: the bisector of the two
    adjacent inward edge normals, which also points inward at reflex corners."""
    normals = _inward_edge_normals(ring)
    result = []
    for i in range(len(ring)):
        u = normals[i - 1] + normals[i]
        if u.length < 1e-9:
            # A spike: both edges fold back on each other, so take either edge's normal.
            u = normals[i]
        result.append(u.normalized() if u.length > 0.0 else u)
    return result


def _tessellate(rings, angle):
    outer = rings[0]
    cx = sum(p.x for p in outer) / len(outer)
    cy = sum(p.y for p in outer) / len(outer)
    c, s = math.cos(angle), math.sin(angle)
    coords = [
        [Vector(((p.x - cx) * c - (p.y - cy) * s, (p.x - cx) * s + (p.y - cy) * c, 0.0)) for p in ring]
        for ring in rings
    ]
    return tessellate_polygon(coords)


def _clean_triangles(raw, pts, tol):
    """Orient counter-clockwise, drop slivers and repeated vertex sets."""
    seen = set()
    result = []
    for a, b, c in raw:
        key = frozenset((a, b, c))
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        height = _height(pts, a, b, c)
        if height < tol:
            continue
        area = cross2(pts[b] - pts[a], pts[c] - pts[a])
        result.append((a, b, c) if area > 0.0 else (a, c, b))
    return result


def _height(pts, a, b, c):
    """Height over the longest edge: the same distance _points_on_segment tests against."""
    longest = max((pts[b] - pts[a]).length, (pts[c] - pts[b]).length, (pts[a] - pts[c]).length)
    if longest < 1e-12:
        return 0.0
    return abs(cross2(pts[b] - pts[a], pts[c] - pts[a])) / longest


def _cap_is_valid(triangles, pts, loop_edges, expected, tol):
    if not triangles or any(_height(pts, a, b, c) < tol for a, b, c in triangles):
        return False
    total = sum(cross2(pts[b] - pts[a], pts[c] - pts[a]) for a, b, c in triangles) * 0.5
    if abs(total - expected) > CAP_AREA_TOL * max(abs(expected), 1e-12):
        return False
    count = {}
    for t in triangles:
        for i in range(3):
            key = _edge_key(t[i], t[(i + 1) % 3])
            count[key] = count.get(key, 0) + 1
    return all(n == (1 if key in loop_edges else 2) for key, n in count.items()) and all(
        key in count for key in loop_edges
    )


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _split_at_collinear_points(triangles, pts, loop_edges, tol):
    """tessellate_polygon emits zero-area slivers for collinear boundary points, so a
    triangle edge can span several loop edges. Split that triangle at the skipped points
    so every loop edge is used by the cap and the walls stay crack free."""
    count = {}
    for t in triangles:
        for i in range(3):
            key = _edge_key(t[i], t[(i + 1) % 3])
            count[key] = count.get(key, 0) + 1

    # The collinearity test is tolerance based and so not transitive: with two nearly
    # coincident points both within tol of an edge, the pieces of a fan can be split back into
    # the triangle they came from, forever. An edge that was split or created by a split is
    # final, and the pop budget turns whatever else slips through into a failed attempt.
    spent = set()
    budget = 4 * (len(triangles) + 1) * (len(pts) + 2)
    result = []
    # A triangle can span points on more than one edge, so split repeatedly: each fan
    # leaves the other original edges untouched on one of its pieces.
    stack = list(reversed(triangles))
    while stack:
        budget -= 1
        if budget < 0:
            return None
        t = stack.pop()
        for i in range(3):
            x, y, z = t[i], t[(i + 1) % 3], t[(i + 2) % 3]
            key = _edge_key(x, y)
            if count.get(key, 0) != 1 or key in loop_edges or key in spent:
                continue
            # Excluding t keeps the apex out, which would make a degenerate piece.
            between = _points_on_segment(pts, x, y, exclude=t, tol=tol)
            if not between:
                continue
            chain = [x] + between + [y]
            spent.add(key)
            for k in chain[1:-1]:
                spent.add(_edge_key(z, k))
            for k in range(len(chain) - 1):
                spent.add(_edge_key(chain[k], chain[k + 1]))
            stack.extend((chain[k], chain[k + 1], z) for k in reversed(range(len(chain) - 1)))
            break
        else:
            result.append(t)
    return result


def _points_on_segment(pts, x, y, exclude, tol):
    """Indices of points strictly inside segment x-y, ordered from x to y."""
    d = pts[y] - pts[x]
    length_sq = d.dot(d)
    if length_sq < 1e-18:
        return []
    length = length_sq ** 0.5
    found = []
    for k, p in enumerate(pts):
        if k in exclude:
            continue
        w = p - pts[x]
        t = w.dot(d) / length_sq
        if 1e-9 < t < 1.0 - 1e-9 and abs(cross2(d, w)) / length < tol:
            found.append((t, k))
    found.sort()
    return [k for _t, k in found]
