"""Snapping, placement and run logic of the 3D path draw tool (DECISIONS C19, plan 8.3 in 3D).

Pure mathutils and plain Python: no bpy, so every rule runs headless. All positions are world
space; the operator converts to the path's local space when it writes.

The view is injected as an object with
    key                  hashable, changes whenever the projection does (caches the projected scene)
    project(co)          (x, y, w) in region pixels with w > 0 the depth factor, or None behind the camera
    ray(mouse)           (origin, direction) of the cursor ray
    sight(co)            (origin, direction, distance) of a ray from the eye to co
The scene is a SceneSource: one BVH per object (the objects' evaluated meshes) plus the world
space vertices and edges used for point snapping.

Events handed to PathSession.handle (placement is a Placement or None):
    ("MOVE", placement)
    ("CLICK", placement)
    ("KEY", name)                        ENTER RMB ESC C R O P A
A CLICK may carry a third item, the view's right vector (world), which fixes the x axis of a shape
plane; without it a stable axis is used.

Commands returned (applied in order):
    ("ADD_VERT", ref, co, normal)        new vertex at world co with the surface normal
    ("ADD_EDGE", ref_a, ref_b)           edge, skipped by the applier if it already exists
    ("SET_NORMAL", ref, normal)          new normal for a vertex this session added (ref < 0)
    ("END",)                             chain finished, keep everything added
    ("CANCEL",)                          drop everything added this session

Refs follow draw_logic: >= 0 is a vertex that already exists, < 0 one this session creates.
"""

import math
from collections import namedtuple

from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

from . import shapes
from .draw_logic import DEFAULT_CIRCLE_SEGMENTS

VERTEX = 'VERTEX'
MIDPOINT = 'MIDPOINT'
EDGE = 'EDGE'
FACE = 'FACE'
PATH_VERTEX = 'PATH_VERTEX'
PLANE = 'PLANE'

# The shapes of C21; the names are the items of the WindowManager enum sketch_tools_shape.
POLYLINE = 'POLYLINE'
RECTANGLE = 'RECTANGLE'
CIRCLE = 'CIRCLE'
ARC_3PT = 'ARC_3PT'
ARC_CENTER = 'ARC_CENTER'
SHAPES = (POLYLINE, RECTANGLE, CIRCLE, ARC_3PT, ARC_CENTER)
ARCS = (ARC_3PT, ARC_CENTER)
# Shapes whose clicks after the first are taken in the plane of the first point; a three-point arc
# places all its points on the surface like a polyline and lies in the plane they span.
PLANE_SHAPES = (RECTANGLE, CIRCLE, ARC_CENTER)
CLOSED_SHAPES = (RECTANGLE, CIRCLE)
SHAPE_KEYS = {"R": RECTANGLE, "O": CIRCLE, "P": POLYLINE}

KIND_LABELS = {
    VERTEX: "Vertex",
    MIDPOINT: "Edge midpoint",
    EDGE: "On edge",
    FACE: "On face",
    PATH_VERTEX: "Path point",
    PLANE: "In shape plane",
}

ANGLE_STEP = math.radians(45.0)
# A cursor ray within this cosine of parallel to a shape plane has no usable intersection.
PLANE_GRAZING = 1e-3
# Three points whose spanned parallelogram has less than this sine of the angle between the two
# sides are in a line: no arc.
COLLINEAR_SINE = 1e-6
# Shorter segments are not placed (the cursor has not left the previous point).
MIN_LENGTH = 1e-6
# A snapped point this close to a vertex of the path is that vertex, relative to the larger of 1,
# the point's coordinates and the path object's offset from the origin (the local coordinates the
# vertex went through are as large as that): about 10 float32 ulp, enough for the round trip
# through the path's local space without merging clicks that are merely close at large coordinates.
REUSE_DISTANCE = 1e-6
# A vertex of the path wins over a scene vertex within this many pixels of it on screen, so
# clicking a path vertex that was snapped to a scene vertex never depends on the round trip. It
# must also lie close in the world (relative to the larger of the point's coordinates and the path
# object's offset, at least OWN_MIN_DISTANCE): a vertex far behind the scene vertex on the same
# sight line is another point.
OWN_PIXELS = 0.5
OWN_DISTANCE = 1e-4
OWN_MIN_DISTANCE = 1e-3
# How far a point may sit behind the first surface on its sight line and still count as visible,
# relative to the sight line length (the BVH stores float32), plus a part relative to the point's
# coordinates: the ray and the point are float32 too, so close-up work far from the origin has an
# error that the sight line length does not show.
SIGHT_TOLERANCE = 1e-4
SIGHT_COORD_TOLERANCE = 1e-6
# Candidates per kind that get an occlusion ray, nearest first. Dense scenes would otherwise
# spend a ray cast on every hidden vertex behind the cursor.
MAX_OCCLUSION_TESTS = 8
# How far above the locked point the re-projection ray starts, as a fraction of the segment length
# (plus a small absolute part for tiny segments).
LOCK_LIFT = 0.25
# Fallback for snap points that carry no adjacency (path vertices, loose geometry): how far from
# the point a face still counts as touching it, relative to the point's coordinates (the BVH and
# the world point are float32, so this is about 10 ulp), and the most normals kept.
FACE_RANGE = 1e-6
MAX_FALLBACK_NORMALS = 64
# A face contains the previous path point when the point is this close to the face's plane,
# relative to the distance between the two points (plus a float32 sized part of the coordinates).
CONTAIN_ANGLE = 1e-4
CONTAIN_COORD = 1e-6
# Faces whose angle to the eye differs by less than this are a tie, settled by continuity with
# the previous path point's normal.
TIE_ANGLE = math.radians(3.0)
# A new point re-picks the normals of the run's earlier points this close behind it (it can be the
# evidence for a point up to this many places back), and each re-pick looks at this many
# neighbours.
REPICK_REACH = 2
REPICK_EVIDENCE = 4
# Whether a face is visible near a snap point is tested with a ray to a probe point just above the
# face: from the point it moves this fraction of the way to a target inside the face next to the
# point (the centroid of the face's triangle there) and as far up along the normal. The step is
# never more than PROBE_SIGHT of the distance to the eye (in an orthographic view that distance is
# the size of the scene), PROBE_EDGE of the shortest edge of the face at the point, nor so far that
# it lands more than PROBE_PIXELS from the point on screen, so a huge face is probed close to the
# point and unrelated geometry (a box floating above a big block) is not reached.
PROBE_FACE = 0.05
PROBE_SIGHT = 0.01
PROBE_EDGE = 0.1
PROBE_PIXELS = 10.0
# A normal shared by several coplanar faces has a probe per face and counts as visible when any is
# (a neighbour may bury one of them): at most this many probes per normal, chosen by a canonical
# order of the probes, and this many rays in all, spent on the most frontal normals first.
MAX_PROBES_PER_NORMAL = 4
MAX_PROBE_RAYS = 16
# A triangle of a face counts as degenerate (a vertex of the face lies on an edge, so the loop
# triangulation has a flat one) when its area is below DEGENERATE_AREA times its longest side
# squared plus DEGENERATE_COORD times its longest side and largest coordinate (the float32 error
# of the corner positions). Its centroid lies on the face boundary, so it is no place to look at
# the face from.
DEGENERATE_AREA = 1e-6
DEGENERATE_COORD = 2e-7
# A snapped or hit point. co and normal are world space; ref is the path vertex it reuses (None
# creates a new one); locked marks a point moved by the angle lock. faces holds the candidate
# normals of a snap that touches several faces (empty otherwise), to_eye the direction to the eye
# at that moment and front the faces among them that face the eye and are visible there (empty
# when not tested), so the pick can be redone once the next point is known.
Placement = namedtuple("Placement", "co normal kind ref locked faces to_eye front",
                       defaults=(None, False, (), None, ()))
# index is the scene vertex (VERTEX) or scene edge (MIDPOINT, EDGE) the snap lies on, else None.
Snap = namedtuple("Snap", "co kind ref dist index", defaults=(None,))
# Where to look at a face from a snap point: a target inside the face next to the point (not the
# face centre, which lies outside a concave face) and the length of the shortest edge of the face at
# the point.
Probe = namedtuple("Probe", "target size")


class ObjectBVH:
    """A BVHTree in an object's local space, queried with world space rays and points."""

    def __init__(self, tree, matrix):
        self.tree = tree
        self.matrix = matrix.copy()
        self.inverse = matrix.inverted_safe()
        # Normals move with the inverse transpose; a mirrored object keeps its outward normals.
        self._normal_matrix = self.inverse.to_3x3().transposed()

    def _world_normal(self, normal):
        out = self._normal_matrix @ normal
        out.normalize()
        return out

    def ray_cast(self, origin, direction):
        """(world co, world normal, distance) of the first hit, or None. The normal is flipped to
        face the ray's origin (C19), so a back face reports the side the viewer sees."""
        local_dir = self.inverse.to_3x3() @ direction
        if local_dir.length_squared == 0.0:
            return None
        local_dir.normalize()
        co, normal, _, _ = self.tree.ray_cast(self.inverse @ origin, local_dir)
        if co is None:
            return None
        world = self.matrix @ co
        world_normal = self._world_normal(normal)
        if world_normal.dot(direction) > 0.0:
            world_normal.negate()
        return world, world_normal, (world - origin).length

    def nearest(self, point):
        """(world co, world normal, distance) of the surface point nearest to a world point."""
        co, normal, _, _ = self.tree.find_nearest(self.inverse @ point)
        if co is None:
            return None
        world = self.matrix @ co
        return world, self._world_normal(normal), (world - point).length

    def normals_at(self, point):
        """World normals of the faces touching a world point, nearest first, without duplicates
        and at most MAX_FALLBACK_NORMALS. Only the fallback for points without exact adjacency:
        the reach is about 10 float32 ulp of the larger of the point's local and world coordinates
        (the world point is float32 too), seen from the object's space, so it is a few thousandths
        of a face at 1e4 from the origin where a fixed reach would span hundreds of faces."""
        local = self.inverse @ point
        reach = FACE_RANGE * max(local.length, point.length * self.inverse.median_scale, 1e-3)
        hits = sorted(self.tree.find_nearest_range(local, reach), key=lambda hit: hit[3])
        seen, out = set(), []
        for _, normal, _, _ in hits:
            world = self._world_normal(normal)
            key = tuple(round(c, 5) for c in world)
            if key not in seen:
                seen.add(key)
                out.append(world)
                if len(out) >= MAX_FALLBACK_NORMALS:
                    break
        return out


def _newell_normal(points):
    """Unit normal of a polygon from its corner points, counter-clockwise outward."""
    normal = Vector((0.0, 0.0, 0.0))
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        normal.x += (a.y - b.y) * (a.z + b.z)
        normal.y += (a.z - b.z) * (a.x + b.x)
        normal.z += (a.x - b.x) * (a.y + b.y)
    normal.normalize()
    return normal


def face_probes(points, triangles):
    """(corner probes, side probes) of a polygon: a Probe for each corner and for each side (corner
    k to k + 1), from the corner positions and a triangulation of the polygon (index triples into
    points, for example a mesh's loop triangles).

    The target of a corner is the centroid of the triangle at it with the largest angle there, the
    target of a side the centroid of the triangle that has the side. A triangle lies inside the
    polygon, so a target is inside the face even when the polygon is concave and its centre is not.
    Degenerate triangles (see DEGENERATE_AREA) are ignored: a corner or side left without one takes
    the centroid of the nearest other triangle of the face, and a face with none its vertex mean.
    The size is the shortest edge of the polygon at the corner, the length of the side."""
    count = len(points)
    centre = sum(points, Vector((0.0, 0.0, 0.0))) / count

    def solid(triangle):
        a, b, c = (points[i] for i in triangle)
        longest = max((b - a).length_squared, (c - b).length_squared, (a - c).length_squared)
        coordinate = max(abs(x) for point in (a, b, c) for x in point)
        return 0.5 * (b - a).cross(c - a).length > (DEGENERATE_AREA * longest
                                                    + DEGENERATE_COORD * coordinate * math.sqrt(longest))

    def centroid(triangle):
        return (points[triangle[0]] + points[triangle[1]] + points[triangle[2]]) / 3.0

    def angle_at(corner, triangle):
        a, b = [points[c] - points[corner] for c in triangle if c != corner][:2]
        return a.angle(b, 0.0)

    triangles = [t for t in triangles if solid(t)]
    centroids = [centroid(t) for t in triangles]
    at_corner = [[] for _ in range(count)]
    for triangle in triangles:
        for corner in triangle:
            at_corner[corner].append(triangle)

    def nearest(point):
        return min(centroids, key=lambda c: (c - point).length_squared, default=centre)

    sides = [(points[(k + 1) % count] - points[k]).length for k in range(count)]
    corners, edges = [], []
    for k in range(count):
        size = min(sides[k - 1], sides[k])
        best = max(at_corner[k], key=lambda t: angle_at(k, t), default=None)
        corners.append(Probe(centroid(best) if best is not None else nearest(points[k]), size))
        other = (k + 1) % count
        owner = next((t for t in at_corner[k] if other in t), None)
        edges.append(Probe(centroid(owner) if owner is not None
                           else nearest((points[k] + points[other]) * 0.5), sides[k]))
    return corners, edges


def adjacent_normals(vertex_count, edge_count, faces):
    """(per vertex, per edge, vertex probes, edge probes): lists of the distinct normals of the
    faces around each, and parallel lists of a tuple with the Probe of every face that contributed
    that normal. faces is an iterable of (vertex indices, edge indices, unit normal, corner probes,
    side probes) in world space, the probes parallel to the indices (see face_probes). Equal normals
    (rounded to 5 places) share one Vector, so a vertex of a flat patch lists it once, with the
    probes of all its faces there: one of them may be buried while the others are in view."""
    per_vertex = [[] for _ in range(vertex_count)]
    per_edge = [[] for _ in range(edge_count)]
    vertex_probes = [[] for _ in range(vertex_count)]
    edge_probes = [[] for _ in range(edge_count)]
    interned = {}
    for verts, edges, normal, corner_probes, side_probes in faces:
        shared = interned.setdefault(tuple(round(c, 5) for c in normal), normal)
        for lst, probes, indices, own in ((per_vertex, vertex_probes, verts, corner_probes),
                                          (per_edge, edge_probes, edges, side_probes)):
            for i, probe in zip(indices, own):
                at = next((j for j, n in enumerate(lst[i]) if n is shared), None)
                if at is None:
                    lst[i].append(shared)
                    probes[i].append([probe])
                else:
                    probes[i][at].append(probe)
    return (per_vertex, per_edge, [[tuple(p) for p in each] for each in vertex_probes],
            [[tuple(p) for p in each] for each in edge_probes])


def triangulate(points):
    """Index triples into the corner positions of a polygon that triangulate it."""
    if len(points) == 3:
        return [(0, 1, 2)]
    triangles = [tuple(t) for t in tessellate_polygon([list(points)])]
    # A degenerate polygon gives no triangles; a fan still gives every corner one.
    return triangles or [(0, k, k + 1) for k in range(1, len(points) - 1)]


class SceneSource:
    """What the draw tool snaps to: the BVHs for surface hits and normals, and the world space
    vertices and edges for point snapping."""

    def __init__(self):
        self.trees = []
        self.vertices = []
        self.edges = []
        # The normals of the faces around each snap vertex and each snap edge (world space, parallel
        # to vertices and edges), taken from the same mesh the points come from.
        self.vertex_normals = []
        self.edge_normals = []
        # A tuple of Probes for each of those normals (parallel lists), one per face that has it,
        # to test whether the face is visible.
        self.vertex_probes = []
        self.edge_probes = []
        # An upper bound of the distance from the world origin to any scene point, used to start
        # orthographic sight lines outside the scene.
        self.reach = 0.0
        # A note for the header, for example that point snapping was limited.
        self.notice = ""
        self._projected = None
        self._projected_key = None

    def add_tree(self, tree, matrix, reach=0.0):
        self.trees.append(ObjectBVH(tree, matrix))
        self.reach = max(self.reach, reach)

    def add_points(self, vertices, edges, vertex_normals=None, edge_normals=None,
                   vertex_probes=None, edge_probes=None):
        """Snap vertices (world) and edges (index pairs into vertices), with the face normals around
        each when known (and a Probe of each face); points without them fall back to a distance
        search."""
        base = len(self.vertices)
        self.vertices.extend(vertices)
        self.edges.extend((a + base, b + base) for a, b in edges)
        self.vertex_normals.extend(vertex_normals if vertex_normals is not None else [()] * len(vertices))
        self.edge_normals.extend(edge_normals if edge_normals is not None else [()] * len(edges))
        self.vertex_probes.extend(vertex_probes if vertex_probes is not None else [()] * len(vertices))
        self.edge_probes.extend(edge_probes if edge_probes is not None else [()] * len(edges))

    def add_polygons(self, vertices, polygons, matrix):
        """Everything from plain mesh data in the object's local space: the tree and the points."""
        tree = BVHTree.FromPolygons([tuple(v) for v in vertices], [tuple(p) for p in polygons])
        world = [matrix @ Vector(v) for v in vertices]
        edges = sorted({tuple(sorted((poly[i], poly[(i + 1) % len(poly)]))) for poly in polygons
                        for i in range(len(poly))})
        edge_index = {pair: i for i, pair in enumerate(edges)}
        self.add_tree(tree, matrix, max((v.length for v in world), default=0.0))
        bvh = self.trees[-1]
        faces = []
        for poly in polygons:
            normal = bvh._world_normal(_newell_normal([Vector(vertices[i]) for i in poly]))
            sides = [edge_index[tuple(sorted((poly[i], poly[(i + 1) % len(poly)])))] for i in range(len(poly))]
            corners = [world[i] for i in poly]
            faces.append((poly, sides, normal, *face_probes(corners, triangulate(corners))))
        self.add_points(world, edges, *adjacent_normals(len(vertices), len(edges), faces))

    @property
    def empty(self):
        return not self.trees

    def clear(self):
        """Drop every reference; the trees are Python objects, there is nothing else to free."""
        self.trees = []
        self.vertices = []
        self.edges = []
        self.vertex_normals = []
        self.edge_normals = []
        self.vertex_probes = []
        self.edge_probes = []
        self._projected = None

    def ray_cast(self, origin, direction):
        best = None
        for tree in self.trees:
            hit = tree.ray_cast(origin, direction)
            if hit is not None and (best is None or hit[2] < best[2]):
                best = hit
        return best

    def nearest(self, point):
        best = None
        for tree in self.trees:
            hit = tree.nearest(point)
            if hit is not None and (best is None or hit[2] < best[2]):
                best = hit
        return best

    def normals_at(self, point):
        return [n for tree in self.trees for n in tree.normals_at(point)]

    def snap_normals(self, snap):
        """The exact normals of the faces around a scene vertex or edge snap, empty for snaps that
        are not on one (path vertices) or on loose geometry."""
        if snap.index is None:
            return ()
        if snap.kind == VERTEX:
            return self.vertex_normals[snap.index]
        if snap.kind in (MIDPOINT, EDGE):
            return self.edge_normals[snap.index]
        return ()

    def snap_probes(self, snap):
        """Probes parallel to snap_normals, or () when the scene has none for the snap."""
        if snap.index is None:
            return ()
        if snap.kind == VERTEX:
            return self.vertex_probes[snap.index]
        if snap.kind in (MIDPOINT, EDGE):
            return self.edge_probes[snap.index]
        return ()

    def projected(self, view):
        """The scene vertices projected by the view, kept while the view does not change."""
        if self._projected is None or self._projected_key != view.key:
            self._projected = [view.project(v) for v in self.vertices]
            self._projected_key = view.key
        return self._projected


def is_visible(scene, view, co):
    """True when nothing in the scene lies between the eye and co. A point on a surface hits that
    surface at its own distance, which the tolerance allows; a point behind a surface hits the
    surface first."""
    origin, direction, distance = view.sight(co)
    hit = scene.ray_cast(origin, direction)
    tolerance = SIGHT_TOLERANCE * max(1.0, distance) + SIGHT_COORD_TOLERANCE * max(1.0, co.length)
    return hit is None or hit[2] >= distance - tolerance


def _first_visible(scene, view, candidates):
    """The nearest of (dist, co, ref, kind, index) tuples that is not occluded, testing a few."""
    for dist, co, ref, kind, index in sorted(candidates, key=lambda c: c[0])[:MAX_OCCLUSION_TESTS]:
        if is_visible(scene, view, co):
            return Snap(co, kind, ref, dist, index)
    return None


def _scene_candidates(scene, view, mouse, threshold):
    """(vertex, midpoint, edge) candidate lists of (dist, co, ref, kind, index) within threshold
    pixels; index is the scene vertex or edge the candidate lies on."""
    mx, my = mouse
    proj = scene.projected(view)
    verts = scene.vertices
    vertex_c, mid_c, edge_c = [], [], []
    for i, p in enumerate(proj):
        if p is None or abs(p[0] - mx) > threshold or abs(p[1] - my) > threshold:
            continue
        dist = math.hypot(p[0] - mx, p[1] - my)
        if dist <= threshold:
            vertex_c.append((dist, verts[i], None, VERTEX, i))
    for e, (a, b) in enumerate(scene.edges):
        pa, pb = proj[a], proj[b]
        if pa is None or pb is None:
            continue
        if ((pa[0] < mx - threshold and pb[0] < mx - threshold)
                or (pa[0] > mx + threshold and pb[0] > mx + threshold)
                or (pa[1] < my - threshold and pb[1] < my - threshold)
                or (pa[1] > my + threshold and pb[1] > my + threshold)):
            continue
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-12:
            continue
        # Nearest point of the screen segment; the ends are vertex candidates already.
        t2 = ((mx - pa[0]) * dx + (my - pa[1]) * dy) / length2
        if 0.0 < t2 < 1.0:
            dist = math.hypot(pa[0] + t2 * dx - mx, pa[1] + t2 * dy - my)
            if dist <= threshold:
                # Screen interpolation is perspective-divided: walk back to the edge parameter.
                t3 = t2 * pa[2] / (t2 * pa[2] + (1.0 - t2) * pb[2])
                edge_c.append((dist, verts[a].lerp(verts[b], t3), None, EDGE, e))
        mid = (verts[a] + verts[b]) * 0.5
        pm = view.project(mid)
        if pm is not None:
            dist = math.hypot(pm[0] - mx, pm[1] - my)
            if dist <= threshold:
                mid_c.append((dist, mid, None, MIDPOINT, e))
    return vertex_c, mid_c, edge_c


def _reuse(co, path_points, scale=0.0):
    """(ref, co) of the path vertex coinciding with co, else (None, co). scale is the offset of the
    path object from the origin: the vertices went through local coordinates that large."""
    limit = REUSE_DISTANCE * max(1.0, co.length, scale)
    for ref, point in path_points:
        if (point - co).length <= limit:
            return ref, point
    return None, co


def _own_beside(scene, view, own, snap, scale=0.0):
    """The visible own path vertex candidate within OWN_PIXELS of a scene snap on screen and close
    to it in the world (see OWN_DISTANCE; scale is the path object's offset), or None."""
    p = view.project(snap.co)
    if p is None:
        return None
    limit = max(OWN_MIN_DISTANCE, OWN_DISTANCE * max(snap.co.length, scale))
    near = []
    for candidate in own:
        q = view.project(candidate[1])
        if (q is not None and math.hypot(q[0] - p[0], q[1] - p[1]) <= OWN_PIXELS
                and (candidate[1] - snap.co).length <= limit):
            near.append(candidate)
    return _first_visible(scene, view, near)


def find_snap(scene, view, mouse, threshold, path_points=(), reuse_scale=0.0):
    """The snap target for the cursor, or None.

    Priority: vertex (a vertex of the path or of the scene), then edge midpoint, then the nearest
    point on an edge; inside a kind the nearest on screen wins. Only targets within threshold
    pixels count, and a target behind a surface is skipped. path_points is [(ref, world co)] of
    the path's own vertices: a snap onto one of them reuses it. A path vertex within OWN_PIXELS of
    the chosen scene vertex on screen (and near it in the world) is taken instead of it, whatever
    the float error of the round trip through the path's space; otherwise a scene point within the reuse distance (scaled by
    reuse_scale, see _reuse) of a path vertex is that vertex."""
    own = []
    for ref, co in path_points:
        p = view.project(co)
        if p is None:
            continue
        dist = math.hypot(p[0] - mouse[0], p[1] - mouse[1])
        if dist <= threshold:
            own.append((dist, co, ref, PATH_VERTEX, None))
    vertex_c, mid_c, edge_c = _scene_candidates(scene, view, mouse, threshold)
    for candidates in (own + vertex_c, mid_c, edge_c):
        snap = _first_visible(scene, view, candidates)
        if snap is not None:
            if snap.ref is None:
                beside = _own_beside(scene, view, own, snap, reuse_scale)
                if beside is not None:
                    return snap._replace(co=beside.co, ref=beside.ref)
                ref, co = _reuse(snap.co, path_points, reuse_scale)
                return snap._replace(co=co, ref=ref)
            return snap
    return None


def _containing(normals, point, other):
    """The normals of the faces through point that also contain other (the segment between them
    lies on the face), within a tolerance relative to the segment and to float32 coordinates."""
    offset = other - point
    limit = CONTAIN_ANGLE * offset.length + CONTAIN_COORD * max(1.0, point.length)
    return [n for n in normals if n.length_squared > 0.0 and abs(n.dot(offset)) <= limit]


def _pixel_limit(view, co, direction, step):
    """The factor (at most 1) that keeps a probe step from co along direction within PROBE_PIXELS
    on screen. Linear, which is exact in an orthographic view and close in a perspective one at
    these small steps."""
    start, end = view.project(co), view.project(co + direction * step)
    if start is None or end is None:
        return 1.0
    shown = math.hypot(end[0] - start[0], end[1] - start[1])
    return PROBE_PIXELS / shown if shown > PROBE_PIXELS else 1.0


def _probes_to_test(probes, co):
    """[(reach, Probe)] of at most MAX_PROBES_PER_NORMAL probes of one normal, chosen without
    looking at the order the faces came in: the testable probes (some extent, a target away from
    co) are sorted by the distance of the target from co, then by its coordinates, and a spread
    across that order is taken (the nearest, the farthest and evenly spaced ones between), so
    that a flat patch with many buried faces gives the same answer whichever face is listed
    first."""
    ranked = sorted((((p.target - co).length, p.target.x, p.target.y, p.target.z, p.size, p)
                     for p in probes if p.size > 0.0 and p.target != co), key=lambda r: r[:5])
    if len(ranked) > MAX_PROBES_PER_NORMAL:
        gaps = max(MAX_PROBES_PER_NORMAL - 1, 1)
        ranked = [ranked[round(j * (len(ranked) - 1) / gaps)] for j in range(MAX_PROBES_PER_NORMAL)]
    return [(r[0], r[5]) for r in ranked]


def front_faces(scene, view, co, normals, probes, to_eye):
    """The normals among a snap's faces that face the eye and are visible, or () when there is
    nothing to decide (fewer than two face the eye, or the scene has no probes). They come in the
    order the tests ran: the most frontal first.

    A face buried by an abutting object faces the eye as well, but the first thing the eye sees
    above it is the other object. So a ray goes to a probe point just above the face, next to the
    snap point on the face's side (straight above the point it would graze the neighbour's edge at
    a shared corner), and the face counts when nothing is in the way. The probe point moves from
    the snap point towards the probe's target, inside the face, and up along the normal; the step
    is bounded by PROBE_FACE, PROBE_SIGHT, PROBE_EDGE and PROBE_PIXELS. A normal has a probe for
    each coplanar face that shares it and is visible when any of them is, tried in turn (see
    _probes_to_test); a normal with no probe to test counts as visible. The normals are taken
    most frontal first (the largest dot product with the direction to the eye, ties by their
    coordinates), so the budget of MAX_PROBE_RAYS rays goes to the faces the pick would choose,
    whatever order the scene listed them in; once it is spent the rest count as visible."""
    facing = sorted((i for i, n in enumerate(normals) if n.dot(to_eye) > 1e-6),
                    key=lambda i: (-normals[i].dot(to_eye), normals[i].x, normals[i].y, normals[i].z))
    if len(facing) < 2 or len(probes) != len(normals):
        return ()
    _, _, distance = view.sight(co)
    rays = MAX_PROBE_RAYS
    out = []
    for i in facing:
        visible = True
        for reach, probe in _probes_to_test(probes[i], co):
            if rays <= 0:
                visible = True
                break
            rays -= 1
            step = min(PROBE_FACE * reach, PROBE_SIGHT * distance, PROBE_EDGE * probe.size)
            direction = (probe.target - co) / reach + normals[i]
            step *= _pixel_limit(view, co, direction, step)
            visible = is_visible(scene, view, co + direction * step)
            if visible:
                break
        if visible:
            out.append(normals[i])
    return tuple(out)


def _pick_normal(normals, to_eye, previous=None, point=None, previous_point=None, earlier_points=(),
                 front=()):
    """The normal for a snapped point that touches several faces.

    Faces that face the eye and are visible (front, when it was tested) come first, then those that
    merely face it; when only back faces are left (the inside of a room) they are all kept and the
    most frontal one is flipped (C19). With a previous path point, the faces left that contain it
    (the segment lies on them) are preferred: one is taken, several are narrowed to those. If
    several are left, the points before it (nearest first) narrow them the same way; a point that
    lies on none of the faces left is skipped, later ones may still narrow them. Then the eye
    decides among what is left: the smallest angle to the eye, where faces within TIE_ANGLE of the
    best are a tie settled by the closest match to the previous normal.

    The visibility test is a heuristic and the path is evidence: when it removed exactly one face
    that faces the eye, and that is the only face through the point that contains the previous
    point, the face stays in the pool."""
    facing = [n for n in normals if n.dot(to_eye) > 1e-6]
    pool = list(front) or facing or list(normals)
    if front and point is not None and previous_point is not None:
        removed = [n for n in facing if not any(n is f for f in front)]
        if len(removed) == 1 and _containing(facing, point, previous_point) == removed:
            pool.append(removed[0])
    if previous_point is not None and point is not None:
        for other in (previous_point, *earlier_points):
            if len(pool) <= 1:
                break
            pool = _containing(pool, point, other) or pool
    oriented = [(n if n.dot(to_eye) >= 0.0 else -n) for n in pool]
    scored = sorted(((math.acos(min(1.0, o.dot(to_eye))), o) for o in oriented), key=lambda s: s[0])
    best = scored[0][0]
    tied = [o for angle, o in scored if angle - best <= TIE_ANGLE]
    if previous is not None and len(tied) > 1:
        return max(tied, key=lambda o: o.dot(previous))
    return scored[0][1]


def snap_placement(scene, view, snap, previous_normal=None, previous_point=None, earlier_points=()):
    """A Placement for a snap. A vertex, midpoint or edge touches several faces, the exact ones the
    scene recorded for it: faces that face the eye and are visible come first (a face buried by an
    abutting object does not count); the normal is the face that contains the previous path point
    (and the earlier ones) if one does, else the one that most faces the eye, with ties settled by
    continuity with previous_normal. A normal always faces the viewer (C19)."""
    _, direction, _ = view.sight(snap.co)
    to_eye = -direction.normalized()
    normals = list(scene.snap_normals(snap)) or scene.normals_at(snap.co)
    if not normals:
        hit = scene.nearest(snap.co)
        normals = [hit[1]] if hit is not None else [Vector((0.0, 0.0, 1.0))]
    front = front_faces(scene, view, snap.co, normals, scene.snap_probes(snap), to_eye)
    normal = _pick_normal(normals, to_eye, previous_normal, snap.co, previous_point, earlier_points, front)
    kind = PATH_VERTEX if snap.ref is not None and snap.kind != PATH_VERTEX else snap.kind
    faces = tuple(normals) if len(normals) > 1 else ()
    return Placement(snap.co, normal, kind, snap.ref, False, faces, to_eye, front)


def lock_in_plane(anchor, target, normal, previous_dir=None, step=ANGLE_STEP):
    """target moved to the nearest multiple of 45 degrees from the previous segment, measured in
    the plane through anchor with the given normal, with its distance from anchor kept.

    The reference direction is previous_dir projected into the plane; without one (first
    segment, or a previous segment along the normal) it is world X projected, or world Y when
    that is along the normal too. None when target has no distance from anchor in the plane."""
    n = normal.normalized()
    offset = target - anchor
    offset -= n * offset.dot(n)
    length = offset.length
    if length < MIN_LENGTH:
        return None
    reference = None
    for candidate in ([previous_dir] if previous_dir is not None else []) + [Vector((1, 0, 0)), Vector((0, 1, 0))]:
        projected = candidate - n * candidate.dot(n)
        if projected.length > 1e-6:
            reference = projected.normalized()
            break
    side = n.cross(reference)
    angle = math.atan2(offset.dot(side), offset.dot(reference))
    locked = round(angle / step) * step
    return anchor + (reference * math.cos(locked) + side * math.sin(locked)) * length


def lock_to_surface(scene, anchor, previous_dir, hit_co, hit_normal):
    """Placement for the angle locked version of a cursor hit, or None when the locked point has
    no surface under it (then the lock is skipped and the hit is used as it is).

    The lock happens in the plane through the anchor with the surface normal at the hit; the
    locked point is cast back onto the surface along that normal, from above."""
    locked = lock_in_plane(anchor, hit_co, hit_normal, previous_dir)
    if locked is None:
        return None
    n = hit_normal.normalized()
    lift = LOCK_LIFT * (locked - anchor).length + 1e-3 * max(1.0, locked.length)
    hit = scene.ray_cast(locked + n * lift, -n)
    if hit is None:
        return None
    return Placement(hit[0], hit[1], FACE, None, True)


def place(scene, view, mouse, threshold, path_points=(), shift=False, anchor=None, previous_dir=None,
          ctrl=False, previous_normal=None, previous_point=None, earlier_points=(), reuse_scale=0.0):
    """The Placement for the cursor, or None when it is over no surface.

    Without Shift or Ctrl a snap target within threshold pixels wins; otherwise the point is where
    the cursor ray meets the scene. With Shift the point is angle locked against the anchor (the
    last point of the run) and snapping is off, so the lock is not pulled towards nearby vertices.
    Ctrl alone turns snapping off and places on the surface under the cursor, without the lock.
    reuse_scale is the path object's offset from the origin (see find_snap)."""
    if not (shift or ctrl):
        snap = find_snap(scene, view, mouse, threshold, path_points, reuse_scale)
        if snap is not None:
            return snap_placement(scene, view, snap, previous_normal, previous_point, earlier_points)
    origin, direction = view.ray(mouse)
    hit = scene.ray_cast(origin, direction)
    if hit is None:
        return None
    if shift and anchor is not None:
        locked = lock_to_surface(scene, anchor, previous_dir, hit[0], hit[1])
        if locked is not None:
            return locked
    return Placement(hit[0], hit[1], FACE, None)


def plane_basis(normal, right=None):
    """(x, y) unit axes of a plane with the given normal, with x, y, normal right handed (so a
    counter-clockwise turn in (x, y) is counter-clockwise seen against the normal). x is the
    view's right vector projected into the plane; when that has no length (the plane is seen edge
    on from the side) or right is None it is world X projected, or world Y when that is along the
    normal too."""
    n = normal.normalized()
    for candidate in ([right] if right is not None else []) + [Vector((1, 0, 0)), Vector((0, 1, 0))]:
        x = candidate - n * candidate.dot(n)
        if x.length > 1e-3 * candidate.length:
            x.normalize()
            return x, n.cross(x)
    raise ValueError("no axis in the plane")


def ray_plane_point(origin, direction, plane_co, normal):
    """Where a ray meets a plane, or None when it is (nearly) parallel to it or the plane lies
    behind the origin."""
    direction = direction.normalized()
    denominator = normal.dot(direction)
    if abs(denominator) < PLANE_GRAZING:
        return None
    t = normal.dot(plane_co - origin) / denominator
    return origin + direction * t if t > 0.0 else None


def place_on_plane(scene, view, mouse, threshold, plane, path_points=(), snapping=True, reuse_scale=0.0):
    """The Placement for the cursor inside a shape plane ((origin, unit normal) of the first point
    of the shape), or None when the cursor ray does not meet it.

    The point is where the cursor ray meets the plane, so the shape never bends over the faces
    the ray passes. With snapping a snap target within threshold pixels wins, moved onto the
    plane along its normal. The placement carries the plane's normal, and never a path vertex to
    reuse: a shape point is a position of the shape."""
    origin, normal = plane
    if snapping:
        snap = find_snap(scene, view, mouse, threshold, path_points, reuse_scale)
        if snap is not None:
            return Placement(snap.co - normal * normal.dot(snap.co - origin), normal, snap.kind, None)
    ray_origin, direction = view.ray(mouse)
    co = ray_plane_point(ray_origin, direction, origin, normal)
    return None if co is None else Placement(co, normal, PLANE, None)


class Pt:
    """A point of the run. emitted: the vertex already exists in the path."""

    __slots__ = ("ref", "co", "normal", "emitted", "faces", "to_eye", "own", "front")

    def __init__(self, ref, co, normal, emitted, faces=(), to_eye=None, own=False, front=()):
        self.ref = ref
        self.co = co
        self.normal = normal
        self.emitted = emitted
        # The snap's candidate faces and the eye direction, for picking again once more of the run
        # is known. own: the run created this vertex, so it may rewrite its normal; a vertex that
        # existed before (or a second visit to one of the run's own) is never rewritten.
        self.faces = faces
        self.to_eye = to_eye
        self.own = own
        self.front = front


class Preview3D:
    """What the overlay draws, with 3D world points where the planar Preview has 2D ones (the
    projector handed to the overlay takes them): strips (the chain, the rubber band), markers
    (the run's points), snap_xy (the candidate), guide (unused) and the text lines."""

    def __init__(self):
        self.strips = []
        self.markers = []
        self.snap_xy = None
        self.guide = None
        self.lines = []


_MODE_HELP = {
    POLYLINE: "3D path: LMB places a point on a surface or snap point",
    RECTANGLE: "3D rectangle: LMB sets the opposite corner in the plane of the first point",
    CIRCLE: "3D circle: LMB sets the radius in the plane of the first point",
    ARC_3PT: "3D arc: LMB places the middle point, then the end",
    ARC_CENTER: "3D centre arc: LMB sets the start, then the end (CCW)",
}
_DEGENERATE_NOTES = {
    ARC_3PT: "The three points are in a line, no arc",
    ARC_CENTER: "The arc has no angle, move the end point",
}


class PathSession:
    """One chain, rectangle, circle or arc, from the first point to END or CANCEL (one undo step,
    as in C15 and C21).

    The first point is not written until a second one exists, so ending or cancelling after one
    click leaves the mesh untouched. Rectangle, circle and arc write everything when the click
    that completes the shape arrives, as closed rings (rectangle, circle) or open chains (arcs),
    every point with the normal of the first point."""

    def __init__(self, shape=POLYLINE, circle_segments=DEFAULT_CIRCLE_SEGMENTS):
        self.shape = shape
        self.circle_segments = max(3, int(circle_segments))
        self.points = []
        self.done = False
        self.cancelled = False
        self.cursor = None
        # Why the last click did nothing (a collinear arc), shown in the header until the next click.
        self.note = ""
        # The view's right vector at the first click: the x axis of the shape plane stays put while
        # the view is orbited.
        self._right = None
        self._next_ref = -1

    def _alloc(self):
        ref = self._next_ref
        self._next_ref -= 1
        return ref

    def handle(self, event):
        if self.done:
            return []
        kind = event[0]
        if kind == "MOVE":
            self.cursor = event[1]
            return []
        if kind == "CLICK":
            self.cursor = event[1]
            return self._click(event[1], event[2] if len(event) > 2 else None)
        if kind == "KEY":
            return self._key(event[1])
        raise ValueError(f"unknown path event {kind!r}")

    def anchor(self):
        """(last point, direction of the previous segment or None) for the angle lock, which only
        a polyline has."""
        if not self.points or self.shape != POLYLINE:
            return None, None
        last = self.points[-1].co
        previous = (last - self.points[-2].co) if len(self.points) > 1 else None
        return last, previous

    def last_normal(self):
        """The normal of the last point of the run, or None before the first."""
        return self.points[-1].normal if self.points else None

    def last_point(self):
        """The world position of the last point of the run, or None before the first."""
        return self.points[-1].co if self.points else None

    def earlier_points(self, count=4):
        """World positions of the points before the last, nearest first, at most count."""
        return [p.co for p in reversed(self.points[:-1])][:count]

    def plane(self):
        """(origin, unit normal) of the plane the next click is taken in, or None when it is placed
        on the surface like a path point (polyline, three-point arc, any first click)."""
        if self.shape in PLANE_SHAPES and self.points:
            return self.points[0].co, self.points[0].normal
        return None

    def _key(self, name):
        if name == "ESC":
            return self._cancel()
        if name in ("RMB", "ENTER"):
            return self._end()
        if name == "C":
            return self._close()
        if name in SHAPE_KEYS:
            self._set_shape(SHAPE_KEYS[name])
            return []
        if name == "A":
            self._set_shape(ARC_CENTER if self.shape == ARC_3PT else ARC_3PT)
            return []
        raise ValueError(f"unknown path key {name!r}")

    def _set_shape(self, shape):
        """Only before a second point exists, so nothing has been decided yet."""
        if len(self.points) <= 1:
            self.shape = shape
            self.note = ""

    def _finish(self, commands):
        self.done = True
        return commands

    def _cancel(self):
        self.cancelled = True
        return self._finish([("CANCEL",)])

    def _end(self):
        """RMB or Enter keep a polyline of two points or more; a shape that is not complete, and a
        lone point, have nothing to keep."""
        if self.shape == POLYLINE and len(self.points) >= 2:
            return self._finish([("END",)])
        return self._cancel()

    def _close(self):
        if self.shape != POLYLINE or len(self.points) < 3 or self.points[-1].ref == self.points[0].ref:
            return []
        return self._finish([("ADD_EDGE", self.points[-1].ref, self.points[0].ref), ("END",)])

    def _click(self, placement, right=None):
        if placement is None:
            return []
        if not self.points:
            ref = placement.ref if placement.ref is not None else self._alloc()
            self.points.append(Pt(ref, placement.co, placement.normal, placement.ref is not None,
                                  placement.faces, placement.to_eye, placement.ref is None,
                                  placement.front))
            self._right = right
            return []
        if self.shape != POLYLINE:
            return self._shape_click(placement)
        anchor, first = self.points[-1], self.points[0]
        if placement.ref is not None:
            if placement.ref == anchor.ref:
                return []
            # Closing a loop needs three points; two would make an edge twice.
            if placement.ref == first.ref and len(self.points) < 3:
                return []
        elif (placement.co - anchor.co).length < MIN_LENGTH:
            return []
        closing = placement.ref is not None and placement.ref == first.ref
        if placement.ref is not None:
            point = Pt(placement.ref, placement.co, placement.normal, True, placement.faces, placement.to_eye,
                       front=placement.front)
        else:
            point = Pt(self._alloc(), placement.co, placement.normal, True, placement.faces, placement.to_eye,
                       True, placement.front)
        # Closing adds no position, so nothing new is known about the earlier points.
        rewritten = [] if closing else self._repick(point)
        commands = []
        if not anchor.emitted:
            commands.append(("ADD_VERT", anchor.ref, anchor.co, anchor.normal))
            anchor.emitted = True
        commands.extend(("SET_NORMAL", pt.ref, pt.normal) for pt in rewritten)
        if placement.ref is None:
            commands.append(("ADD_VERT", point.ref, point.co, point.normal))
        commands.append(("ADD_EDGE", anchor.ref, point.ref))
        if closing:
            return self._finish(commands + [("END",)])
        self.points.append(point)
        return commands

    # -- rectangle, circle and arcs ----------------------------------------

    def _shape_click(self, placement):
        points = self.points
        if self.shape in ARCS and len(points) == 1:
            if (placement.co - points[0].co).length < MIN_LENGTH:
                return []
            points.append(Pt(self._alloc(), placement.co, placement.normal, False))
            self.note = ""
            return []
        generated = self._generated(placement.co)
        if generated is None:
            self.note = _DEGENERATE_NOTES.get(self.shape, "")
            return []
        self.note = ""
        if self.shape == ARC_3PT:
            return self._commit_shape(generated, self._arc_normal(placement.co), placement.ref)
        return self._commit_shape(generated, self.points[0].normal, None)

    def _arc_normal(self, end):
        """The normal of a three-point arc: its first point's, picked again with the other two
        points as evidence when the run created it on several faces (a corner), so an arc along
        the top of a box takes the top as a polyline along it does. A path vertex that existed
        before keeps its normal."""
        first, middle = self.points[0], self.points[1]
        if not first.own or len(first.faces) < 2:
            return first.normal
        return _pick_normal(first.faces, first.to_eye, None, first.co, middle.co, [end], first.front)

    def _generated(self, co):
        """World points of the shape that the points so far and the cursor position co make, from
        start to end (a ring is not closed: the last point is not the first again), or None while
        there is not enough of it. Rectangle, circle and centre arc are built by core/shapes in the
        plane of the first point (plane_basis, centre at the origin); a three-point arc in the
        plane its three points span."""
        points = self.points
        if self.shape == ARC_3PT:
            return self._three_point_arc(co) if len(points) >= 2 else None
        if not points or (self.shape == ARC_CENTER and len(points) < 2):
            return None
        origin = points[0].co
        x, y = plane_basis(points[0].normal, self._right)

        def plane_xy(point):
            offset = point - origin
            return offset.dot(x), offset.dot(y)

        u, v = plane_xy(co)
        if self.shape == RECTANGLE:
            if abs(u) < MIN_LENGTH or abs(v) < MIN_LENGTH:
                return None
            flat = shapes.rectangle_points((0.0, 0.0), (u, v))
        elif self.shape == CIRCLE:
            radius = math.hypot(u, v)
            if radius < MIN_LENGTH:
                return None
            flat = shapes.circle_points((0.0, 0.0), radius, self.circle_segments)
        else:
            flat = shapes.arc_centre((0.0, 0.0), plane_xy(points[1].co), (u, v), self.circle_segments)
        return [origin + x * a + y * b for a, b in flat] if flat else None

    def _three_point_arc(self, end):
        start, middle = self.points[0].co, self.points[1].co
        a, b = middle - start, end - start
        spanned = a.cross(b)
        if spanned.length <= COLLINEAR_SINE * a.length * b.length:
            return None
        x = a.normalized()
        y = spanned.normalized().cross(x)
        flat = shapes.arc_three_points((0.0, 0.0), (a.length, 0.0), (b.dot(x), b.dot(y)), self.circle_segments)
        if not flat:
            return None
        world = [start + x * p + y * q for p, q in flat]
        # The ends are the clicked points, whatever the plane round trip did to them.
        world[0], world[-1] = start.copy(), end.copy()
        return world

    def _commit_shape(self, generated, normal, end_ref):
        """Vertices and edges of the finished shape, every new vertex with the given normal. The
        first point is reused when it is an existing path vertex (the rectangle's first corner, the
        start of an arc), and so is an arc's end when the click that ended it was on one."""
        first = self.points[0]
        reuse = {}
        if first.ref >= 0:
            if self.shape == RECTANGLE:
                corner = min(range(len(generated)), key=lambda k: (generated[k] - first.co).length_squared)
                reuse[corner] = first.ref
            elif self.shape == ARC_3PT:
                reuse[0] = first.ref
        if end_ref is not None and end_ref != first.ref:
            reuse[len(generated) - 1] = end_ref
        commands, refs = [], []
        for k, co in enumerate(generated):
            ref = reuse.get(k)
            if ref is None:
                ref = self._alloc()
                commands.append(("ADD_VERT", ref, co, normal))
            refs.append(ref)
        last = len(refs) if self.shape in CLOSED_SHAPES else len(refs) - 1
        commands.extend(("ADD_EDGE", refs[k], refs[(k + 1) % len(refs)]) for k in range(last))
        commands.append(("END",))
        return self._finish(commands)

    def _repick(self, new):
        """Picks again the normal of the run's last few snapped points that touch several faces,
        now that the new point is known: the segments next to a point lie on the face that contains
        both neighbours. Only vertices the run created are touched. Returns the points whose
        normal changed and that are already written (the unwritten first point just carries the
        new normal into its ADD_VERT)."""
        run = self.points + [new]
        changed = []
        for i in range(max(0, len(run) - 1 - REPICK_REACH), len(run) - 1):
            pt = run[i]
            if not pt.own or len(pt.faces) < 2:
                continue
            around = [run[j] for k in range(1, len(run)) for j in (i - k, i + k) if 0 <= j < len(run)]
            around = [q.co for q in around][:REPICK_EVIDENCE]
            previous = run[i - 1].normal if i > 0 else None
            normal = _pick_normal(pt.faces, pt.to_eye, previous, pt.co, around[0], around[1:], pt.front)
            if (normal - pt.normal).length > 1e-6:
                pt.normal = normal
                if pt.emitted:
                    changed.append(pt)
        return changed

    def preview(self):
        out = Preview3D()
        points = [p.co for p in self.points]
        out.markers = points
        cursor = self.cursor.co if self.cursor is not None else None
        if self.shape == POLYLINE:
            if len(points) > 1:
                out.strips.append(points)
            if cursor is not None and points:
                out.strips.append([points[-1], cursor])
        elif cursor is not None and points:
            self._preview_shape(out, points, cursor)
        if self.cursor is not None:
            out.snap_xy = cursor
            label = KIND_LABELS[self.cursor.kind]
            out.lines = [label + (" (angle locked)" if self.cursor.locked else "")]
        return out

    def _preview_shape(self, out, points, cursor):
        """The outline the next click would write, with the lines that define it."""
        generated = self._generated(cursor)
        if self.shape in CLOSED_SHAPES:
            if generated:
                out.strips.append(generated + [generated[0]])
            if self.shape == CIRCLE:
                out.strips.append([points[0], cursor])
        elif self.shape == ARC_CENTER:
            out.strips.append([points[0], points[1] if len(points) > 1 else cursor])
            if len(points) > 1:
                out.strips.append([points[0], cursor])
            if generated:
                out.strips.append(generated)
        elif len(points) == 1:
            out.strips.append([points[0], cursor])
        else:
            # Where the three points do not make an arc the polyline through them shows why.
            out.strips.append(generated or [points[0], points[1], cursor])

    def header_text(self):
        """The instructions of the header. The controller puts the snapping state, the hint and the
        scene notice in front (C24); Blender clips an overflowing header at both ends, so this stays
        short."""
        parts = [_MODE_HELP[self.shape]]
        if self.note:
            parts.insert(0, self.note)
        if len(self.points) >= 2 and self.shape == POLYLINE:
            parts.append("RMB/Enter end, C close")
        if len(self.points) <= 1:
            # In an arc shape A goes to the other arc, which is not obvious from "A arc".
            parts.append("R rect, O circle, P poly, " + ("A other arc" if self.shape in ARCS else "A arc"))
        if self.shape == POLYLINE:
            parts.append("hold Shift or Ctrl: snapping off, Shift locks 45 degrees")
        else:
            parts.append("hold Shift or Ctrl to turn snapping off")
        parts.append("no typed values")
        parts.append("Esc cancels, Ctrl+Z off")
        return " | ".join(parts)
