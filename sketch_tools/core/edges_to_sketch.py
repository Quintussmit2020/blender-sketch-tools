"""Edges of an ordinary mesh -> data for a new sketch object (DECISIONS C25).

Plain data in, plain data out: world space points, edges and normals. No bpy.context, no
operators, so it runs headless. The operator in ops/create.py reads the edit meshes with
`gather_edges` and creates the object from the `EdgesResult`."""

import math
from dataclasses import dataclass, field

from mathutils import Vector
from mathutils.kdtree import KDTree

# Same distance as the resolvers' weld (C9), so edges that meet here also meet there.
WELD_DIST = 1e-5
# Planarity tolerance terms, see planarity_tolerance.
PLANAR_RELATIVE = 1e-5
MIN_TOLERANCE = 1e-9
# float32 coordinates carry about 6e-8 relative error; 10 times that, relative to the largest coordinate.
FLOAT_NOISE = 1e-6
# Summed face normals, view vectors and projected axes shorter than this carry no direction.
MIN_DIRECTION = 1e-3
# Sign decisions: a dot product smaller than this does not decide.
MIN_DOT = 1e-6


class NothingToSketch(Exception):
    """The selection holds no usable edge; the message is shown to the user."""


@dataclass
class EdgeData:
    points: list  # list[Vector], world space
    normals: list  # per point: unit world normal, or None for a vertex without faces
    edges: list  # list[(int, int)]
    face_normals: list = field(default_factory=list)  # unit world normals of faces next to the edges


@dataclass
class EdgesResult:
    planar: bool
    coords: list  # planar: local (x, y, 0); path: world points. list[Vector]
    edges: list  # list[(int, int)] into coords
    normals: list  # path: unit world normal per point; planar: empty
    origin: Vector = None  # planar only
    x: Vector = None
    y: Vector = None
    z: Vector = None
    closed_only: bool = False  # no open end anywhere
    notes: list = field(default_factory=list)


def _unit(v):
    """Unit vector, or None when v is too short to have a direction."""
    length = v.length
    if length < 1e-12:
        return None
    return v / length


def _newell(face):
    """Face normal from the coordinates (Newell's method), unit length. BMFace has no calc_normal
    and normal_update would write the cache of the edit mesh."""
    cos = [v.co for v in face.verts]
    total = Vector()
    for i, a in enumerate(cos):
        b = cos[(i + 1) % len(cos)]
        total.x += (a.y - b.y) * (a.z + b.z)
        total.y += (a.z - b.z) * (a.x + b.x)
        total.z += (a.x - b.x) * (a.y + b.y)
    return total.normalized() if total.length > 0.0 else Vector()


def gather_edges(bm, matrix_world, face_only):
    """Selected edges of an edit-mode bmesh, in world space. Never modifies bm.

    face_only: the mesh select mode is faces only, so the edges are the boundary of the selected
    face region (exactly one selected adjacent face). Vertices are keyed by element, so edges that
    share a vertex in the source share it in the result. Only vertices of the chosen edges are kept."""
    normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
    if face_only:
        edges = [e for e in bm.edges
                 if not e.hide and sum(1 for f in e.link_faces if f.select) == 1]
    else:
        edges = [e for e in bm.edges if e.select and not e.hide]

    # Keyed by the elements themselves: .index is stale or -1 while the indices are dirty. Normals
    # are computed here from the coordinates, because the cached ones can be stale in edit mode.
    face_normal_cache = {}

    def face_normal(face):
        n = face_normal_cache.get(face)
        if n is None:
            n = _newell(face)
            face_normal_cache[face] = n
        return n

    def vertex_normal(vert):
        total = Vector()
        for loop in vert.link_loops:
            total += face_normal(loop.face) * loop.calc_angle()
        return _unit(total)

    index_of = {}
    points, normals = [], []

    def add(vert):
        i = index_of.get(vert)
        if i is None:
            i = len(points)
            index_of[vert] = i
            points.append(matrix_world @ vert.co)
            n = vertex_normal(vert) if vert.link_faces else None
            normals.append(_unit(normal_matrix @ n) if n is not None else None)
        return i

    pairs = [(add(e.verts[0]), add(e.verts[1])) for e in edges]

    seen_faces = set()
    face_normals = []
    for e in edges:
        for f in e.link_faces:
            # Faces next to a boundary edge that are not part of the selection say nothing
            # about the selection's plane.
            if f in seen_faces or (face_only and not f.select):
                continue
            seen_faces.add(f)
            n = _unit(normal_matrix @ face_normal(f))
            if n is not None:
                face_normals.append(n)
    return EdgeData(points, normals, pairs, face_normals)


def merge(parts):
    """Concatenate EdgeData of several objects."""
    points, normals, edges, face_normals = [], [], [], []
    for part in parts:
        base = len(points)
        points += part.points
        normals += part.normals
        edges += [(a + base, b + base) for a, b in part.edges]
        face_normals += part.face_normals
    return EdgeData(points, normals, edges, face_normals)


def weld(data):
    """Points within WELD_DIST become one (so objects whose edges meet are joined), the average
    of their normals is kept, edges that collapse to a point or repeat are dropped, and points no
    edge uses are removed. Raises NothingToSketch when no edge is left."""
    count = len(data.points)
    tree = KDTree(count)
    for i, p in enumerate(data.points):
        tree.insert(p, i)
    tree.balance()
    # Each point joins the lowest-index point within range, so the weld is not transitive,
    # like the resolvers'.
    target = list(range(count))
    for i, p in enumerate(data.points):
        near = [j for _co, j, _d in tree.find_range(p, WELD_DIST) if j < i]
        if near:
            target[i] = target[min(near)]

    edges = []
    seen = set()
    for a, b in data.edges:
        a, b = target[a], target[b]
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            continue
        seen.add(key)
        edges.append((a, b))
    if not edges:
        raise NothingToSketch("The selected edges are too short to make a sketch from")

    used = sorted({i for e in edges for i in e})
    new_index = {old: new for new, old in enumerate(used)}
    sums = {i: Vector() for i in used}
    for i in range(count):
        if data.normals[i] is not None and target[i] in sums:
            sums[target[i]] += data.normals[i]
    points = [data.points[i] for i in used]
    normals = [_unit(sums[i]) for i in used]
    edges = [(new_index[a], new_index[b]) for a, b in edges]
    return EdgeData(points, normals, edges, data.face_normals)


def _eigen_symmetric(m):
    """Eigenvalues and unit eigenvectors (columns of the returned rows list) of a symmetric
    3x3 matrix given as nested lists, by Jacobi rotations. Returns (values, vectors) with
    vectors[k] the eigenvector of values[k]."""
    a = [row[:] for row in m]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _sweep in range(32):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-30:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-300:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    values = [a[i][i] for i in range(3)]
    vectors = [Vector((v[0][i], v[1][i], v[2][i])).normalized() for i in range(3)]
    return values, vectors


def fit_plane(points):
    """(centroid, normal, axis) of the least squares plane: normal is the direction of least
    spread, axis the direction of most spread."""
    n = len(points)
    centre = sum(points, Vector()) / n
    cov = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = p - centre
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
    values, vectors = _eigen_symmetric(cov)
    order = sorted(range(3), key=lambda i: values[i])
    return centre, vectors[order[0]], vectors[order[2]]


def planarity_tolerance(points):
    """1e-5 of the selection size, plus float32 noise of the largest coordinate (1e-6 of it), plus
    an absolute floor of 1e-9. Purely relative in the size, so a 1 cm and a 100 m selection are
    judged alike; the noise term keeps a selection far from the origin from failing on rounding
    alone (at 1e5 it allows 0.1)."""
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    size = (hi - lo).length
    largest = max(max(abs(c) for c in p) for p in points)
    return PLANAR_RELATIVE * size + FLOAT_NOISE * largest + MIN_TOLERANCE, lo, hi


def _perpendicular(vector, axis):
    """vector without its component along the unit axis, as a unit vector, or None when too short."""
    v = vector - axis * vector.dot(axis)
    if v.length < MIN_DIRECTION:
        return None
    return v.normalized()


def _line_plane_normal(axis, face_normals, view):
    """Plane containing a line (single edge or collinear edges). Returns (normal, from_face).
    With adjacent faces it is the plane of one of them (a bisector between two faces would surprise):
    the face that points most towards the viewer, else the one closest to world +Z, ties broken by
    the normal's coordinates. Without a usable face: the plane facing the viewer, else world Z,
    else world X, each with the component along the line removed."""
    best = None
    for fn in face_normals:
        n = _perpendicular(fn, axis)
        if n is None:
            continue
        score = n.dot(view) if view is not None else n.z
        key = (round(score, 6), round(n.z, 6), tuple(round(c, 6) for c in n))
        if best is None or key > best[0]:
            best = (key, n)
    if best is not None:
        return best[1], True
    candidates = [view] if view is not None else []
    candidates += [Vector((0, 0, 1)), Vector((1, 0, 0))]
    for c in candidates:
        n = _perpendicular(c, axis)
        if n is not None:
            return n, False
    # Unreachable: Z and X cannot both be parallel to the axis.
    raise AssertionError("no plane through the line")


def _orient(normal, face_sum, view):
    """Flip the normal to agree with the adjacent faces, else face the viewer, else point to the
    positive world direction (Z, then Y, then X)."""
    for reference in (face_sum, view):
        if reference is not None and abs(normal.dot(reference)) > MIN_DOT:
            return normal if normal.dot(reference) > 0 else -normal
    for i in (2, 1, 0):
        if abs(normal[i]) > MIN_DOT:
            return normal if normal[i] > 0 else -normal
    return normal


def _frame(normal):
    """x = world X projected onto the plane (world Y when that is too short), y = z cross x."""
    x = _perpendicular(Vector((1, 0, 0)), normal)
    if x is None:
        x = _perpendicular(Vector((0, 1, 0)), normal)
    return x, normal.cross(x)


def _open_end_exists(count, edges):
    degree = [0] * count
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return any(d == 1 for d in degree)


def build_result(data, view=None):
    """Decide planar or path and produce the coordinates. data is welded EdgeData; view is the
    unit vector from the scene towards the viewer, or None."""
    points = data.points
    tol, lo, hi = planarity_tolerance(points)
    centre, normal, axis = fit_plane(points)
    notes = []

    face_sum = None
    total = sum(data.face_normals, Vector())
    if total.length > MIN_DIRECTION:
        face_sum = total.normalized()

    spread = max((p - centre).cross(axis).length for p in points)
    from_face = False
    if spread <= tol:
        normal, from_face = _line_plane_normal(axis, data.face_normals, view)
        notes.append("collinear edges: plane chosen from a face, the view or world")
        planar = True
    else:
        planar = max(abs((p - centre).dot(normal)) for p in points) <= tol

    closed_only = not _open_end_exists(len(points), data.edges)
    if not planar:
        return EdgesResult(False, list(points), list(data.edges), _path_normals(data),
                           closed_only=closed_only, notes=notes)

    # A plane taken from a face already faces the way that face does.
    z = normal if from_face else _orient(normal, face_sum, view)
    x, y = _frame(z)
    # The middle of the bounding box, moved onto the plane: stays put when more vertices are
    # added along the outline, unlike the vertex average.
    middle = (lo + hi) / 2.0
    origin = middle - z * (middle - centre).dot(z)
    coords = [Vector(((p - origin).dot(x), (p - origin).dot(y), 0.0)) for p in points]
    return EdgesResult(True, coords, list(data.edges), [], origin, x, y, z, closed_only, notes)


def _path_normals(data):
    """Unit normal per point. Points with face normals keep theirs. The others (wire edges) take
    the mean of the normals of their neighbours, spreading outwards along the edges from points
    that have one; a part of the graph without any normal, or a mean that cancels out, gets the
    perpendicular to the local tangent that is closest to world +Z (closest to +X for a tangent
    along Z)."""
    count = len(data.points)
    neighbours = [[] for _ in range(count)]
    for a, b in data.edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    normals = list(data.normals)

    # Layer by layer from the vertices that have a normal: a vertex takes the mean of the
    # neighbours known before this layer started. Each vertex is looked at when a neighbour
    # becomes known, so the work is linear in the number of edges.
    frontier = [i for i in range(count) if normals[i] is not None]
    while frontier:
        candidates = []
        seen = set()
        for i in frontier:
            for j in neighbours[i]:
                if normals[j] is None and j not in seen:
                    seen.add(j)
                    candidates.append(j)
        found = {}
        for j in candidates:
            total = Vector()
            for k in neighbours[j]:
                if normals[k] is not None:
                    total += normals[k]
            if total.length > MIN_DIRECTION:
                found[j] = total.normalized()
        for j, n in found.items():
            normals[j] = n
        frontier = list(found)

    for i in range(count):
        if normals[i] is None:
            normals[i] = _tangent_normal(data.points, i, neighbours[i])
    return normals


def _tangent_normal(points, i, neighbour_indices):
    p = points[i]
    dirs = [(points[j] - p).normalized() for j in neighbour_indices]
    tangent = dirs[0]
    if len(dirs) > 1 and (dirs[0] - dirs[1]).length > MIN_DIRECTION:
        tangent = (dirs[0] - dirs[1]).normalized()
    for reference in (Vector((0, 0, 1)), Vector((1, 0, 0))):
        n = _perpendicular(reference, tangent)
        if n is not None:
            return n
    raise AssertionError("no perpendicular to the tangent")
