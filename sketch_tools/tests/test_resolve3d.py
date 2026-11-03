import math

import bmesh
from mathutils import Vector

from sketch_tools.core.resolve3d import PATH_NORMAL_ATTR, resolve_path_bmesh

S2 = math.sqrt(0.5)


class Path3D:
    """Builds a 3D wire bmesh with a normal layer the way the draw tool will."""

    def __init__(self, with_layer=True):
        self.bm = bmesh.new()
        self.layer = self.bm.verts.layers.float_vector.new(PATH_NORMAL_ATTR) if with_layer else None

    def vert(self, co, normal=(0.0, 0.0, 1.0)):
        v = self.bm.verts.new(co)
        if self.layer is not None:
            v[self.layer] = normal
        return v

    def edge(self, a, b):
        return self.bm.edges.new((a, b))

    def poly(self, items, closed=False):
        """items are points or (point, normal) pairs."""
        verts = [self.vert(*item) if isinstance(item[0], (tuple, list)) else self.vert(item) for item in items]
        for i in range(len(verts) - 1):
            self.edge(verts[i], verts[i + 1])
        if closed:
            self.edge(verts[-1], verts[0])
        return verts

    def resolve(self):
        return resolve_path_bmesh(self.bm)

    def free(self):
        self.bm.free()


def close(a, b, tol=1e-6):
    return (Vector(a) - Vector(b)).length < tol


def test_open_path_over_two_cube_faces():
    p = Path3D()
    items = [
        ((0, -1, 0.2), (0, -1, 0)),
        ((0, -1, 1.0), (0, -1, 0)),
        ((0, 0, 1.0), (0, 0, 1)),
        ((0, 0.8, 1.0), (0, 0, 1)),
    ]
    p.poly(items)
    r = p.resolve()
    p.free()
    assert r.warnings == []
    assert len(r.chains) == 1
    c = r.chains[0]
    assert not c.closed and len(c.points) == 4 and len(c.normals) == 4
    # The walk may start at either end; match by the first point.
    order = items if close(c.points[0], items[0][0]) else items[::-1]
    for (co, normal), q, n in zip(order, c.points, c.normals):
        assert close(q, co) and close(n, normal)
        assert abs(n.length - 1.0) < 1e-6


def test_closed_loop_around_cube_equator():
    p = Path3D()
    items = [
        ((1, 0, 0), (1, 0, 0)), ((1, 1, 0), (S2, S2, 0)), ((0, 1, 0), (0, 1, 0)), ((-1, 1, 0), (-S2, S2, 0)),
        ((-1, 0, 0), (-1, 0, 0)), ((-1, -1, 0), (-S2, -S2, 0)), ((0, -1, 0), (0, -1, 0)), ((1, -1, 0), (S2, -S2, 0)),
    ]
    p.poly(items, closed=True)
    r = p.resolve()
    p.free()
    assert r.warnings == []
    assert len(r.chains) == 1 and r.chains[0].closed
    c = r.chains[0]
    assert len(c.points) == 8
    by_point = {tuple(round(x, 5) for x in co): normal for co, normal in items}
    for q, n in zip(c.points, c.normals):
        assert close(n, by_point[tuple(round(x, 5) for x in q)], 1e-5)


def test_chains_split_at_junctions():
    p = Path3D()
    centre = p.vert((0, 0, 0))
    for end in ((2, 0, 0), (0, 2, 1), (-2, -1, 3)):
        p.edge(centre, p.vert(end))
    r = p.resolve()
    p.free()
    assert len(r.chains) == 3
    assert all(not c.closed and len(c.points) == 2 for c in r.chains)
    assert all(close(c.points[0], (0, 0, 0)) for c in r.chains)


def test_junction_loop_is_closed_with_open_tail():
    p = Path3D()
    loop = p.poly([(0, 0, 0), (2, 0, 1), (2, 2, 1), (0, 2, 0)], closed=True)
    tail = p.vert((-3, 0, 0))
    p.edge(loop[0], tail)
    r = p.resolve()
    p.free()
    closed = [c for c in r.chains if c.closed]
    opened = [c for c in r.chains if not c.closed]
    assert len(closed) == 1 and len(opened) == 1
    assert len(closed[0].points) == 4 and len(opened[0].points) == 2


def test_weld_closes_loose_ends_and_averages_normals():
    p = Path3D()
    p.poly([((0, 0, 0), (1, 0, 0)), ((2, 0, 0), (1, 0, 0)), ((2, 2, 0), (0, 0, 1))])
    p.poly([((2, 2, 0), (0, 0, 1)), ((0, 2, 0), (0, 0, 1)), ((0, 0, 0.000001), (0, 1, 0))])
    r = p.resolve()
    p.free()
    assert len(r.chains) == 1 and r.chains[0].closed and len(r.chains[0].points) == 4
    c = r.chains[0]
    start = [i for i, q in enumerate(c.points) if close(q, (0, 0, 0), 1e-5)]
    assert len(start) == 1
    n = c.normals[start[0]]
    assert close(n, (S2, S2, 0), 1e-5) and abs(n.length - 1.0) < 1e-6


def test_weld_never_modifies_the_input():
    p = Path3D()
    p.poly([((0, 0, 0), (1, 0, 0)), ((2, 0, 0), (1, 0, 0))])
    p.poly([((0, 0, 0), (0, 1, 0)), ((0, 2, 0), (0, 1, 0))])
    p.bm.faces.new(p.poly([(5, 5, 5), (6, 5, 5), (6, 6, 5)], closed=True))
    before = [(tuple(v.co), tuple(v[p.layer])) for v in p.bm.verts]
    counts = (len(p.bm.verts), len(p.bm.edges), len(p.bm.faces))
    r = p.resolve()
    assert (len(p.bm.verts), len(p.bm.edges), len(p.bm.faces)) == counts
    assert [(tuple(v.co), tuple(v[p.layer])) for v in p.bm.verts] == before
    # The face is ignored: its edges are still a closed loop.
    assert sum(1 for c in r.chains if c.closed) == 1
    p.free()


def test_missing_normal_layer():
    p = Path3D(with_layer=False)
    p.poly([(0, 0, 0), (1, 1, 1), (2, 0, 3)])
    r = p.resolve()
    p.free()
    assert len(r.warnings) == 1 and "normals" in r.warnings[0]
    assert all(close(n, (0, 0, 1)) for n in r.chains[0].normals)


def test_invalid_normals_use_the_nearest_valid_neighbour():
    p = Path3D()
    nan = float("nan")
    p.poly([
        ((0, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((2, 0, 0), (0, 0, 0)),
        ((3, 0, 0), (nan, 0, 0)),
        ((4, 0, 0), (1e-9, 0, 0)),
        ((5, 0, 0), (1, 0, 0)),
    ])
    r = p.resolve()
    p.free()
    assert len(r.warnings) == 1 and "4 path points" in r.warnings[0]
    c = r.chains[0]
    xs = [round(q.x) for q in c.points]
    normals = {x: n for x, n in zip(xs, c.normals)}
    assert close(normals[1], (0, 1, 0))
    # Point 0 and point 2 are next to the valid point 1; the tie at 2 and 3 goes to the earlier.
    assert close(normals[0], (0, 1, 0)) and close(normals[2], (0, 1, 0))
    assert close(normals[5], (1, 0, 0)) and close(normals[4], (1, 0, 0))
    assert all(abs(n.length - 1.0) < 1e-6 for n in c.normals)


def test_all_normals_invalid_falls_back_to_plus_z():
    p = Path3D()
    p.poly([((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 0, 0))])
    r = p.resolve()
    p.free()
    assert len(r.warnings) == 1
    assert all(close(n, (0, 0, 1)) for n in r.chains[0].normals)


def test_non_unit_normals_are_normalised_without_warning():
    p = Path3D()
    p.poly([((0, 0, 0), (0, 3, 0)), ((1, 0, 0), (0, 0, 2))])
    r = p.resolve()
    p.free()
    assert r.warnings == []
    assert all(abs(n.length - 1.0) < 1e-6 for n in r.chains[0].normals)


def test_duplicates_and_degenerate_loops():
    p = Path3D()
    # A run of vertices 8e-6 apart: the weld is not transitive, so whatever it leaves behind
    # must not survive as consecutive points closer than the weld distance.
    p = Path3D()
    run = p.poly([(0, 0, 0), (8e-6, 0, 0), (1.6e-5, 0, 0), (2.4e-5, 0, 0), (1, 0, 0)])
    assert len(run) == 5
    r = p.resolve()
    p.free()
    assert len(r.chains) == 1
    pts = r.chains[0].points
    assert len(pts) == len(r.chains[0].normals) >= 2
    assert all((a - b).length > 1e-5 for a, b in zip(pts, pts[1:]))
    assert any(abs(q.x - 1.0) < 1e-6 for q in pts)

    # A closed loop that collapses to two distinct points is not a loop.
    p = Path3D()
    p.poly([(0, 0, 0), (1, 0, 0), (1.000005, 0, 0)], closed=True)
    r = p.resolve()
    p.free()
    assert all(not ch.closed for ch in r.chains)
