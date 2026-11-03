"""sketch.trim and sketch.extend: pure logic in core/trim_logic.py, then the editor, controller and
operators driven with fake events and a fake view, as the draw tool tests do."""

import io
import os
import types
from contextlib import contextmanager, redirect_stdout

import bmesh
import bpy

from sketch_tools.core import resolve, trim_logic
from sketch_tools.core.geometry2d import nearest_edge_screen, signed_area
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.core.trim_logic import find_extend, find_trim, plan_extend, plan_trim
from sketch_tools.ops import draw_tool, fillet, modal_common, trim
from sketch_tools.ops.trim import EXTEND, TRIM, TrimController, TrimEditor
from sketch_tools.tests.test_addon import (
    _TopView, _edit_sketch, fake_area, install_fake_area, _event, _mesh_state, _outside_event, _overlay_event,
    addon, assert_closed_and_consistent, output_bm, signed_volume,
)
from sketch_tools.ui import overlay

# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def _seg(key, a, b):
    return (key, a, b)


def test_nearest_edge_screen_picks_the_closest_within_the_threshold():
    segments = [("a", (0.0, 0.0), (100.0, 0.0)), ("b", (0.0, 20.0), (100.0, 20.0))]
    key, distance, t = nearest_edge_screen((25.0, 4.0), segments, 12.0)
    assert key == "a" and abs(distance - 4.0) < 1e-9 and abs(t - 0.25) < 1e-9
    assert nearest_edge_screen((25.0, 12.0), segments, 5.0) is None
    # Beyond an end the parameter clamps and the distance is to the end point.
    key, distance, t = nearest_edge_screen((110.0, 0.0), segments, 12.0)
    assert key == "a" and t == 1.0 and abs(distance - 10.0) < 1e-9
    # Equally close: the first wins. Segments with a None end and degenerate ones are handled.
    tie = [("x", (0.0, 5.0), (10.0, 5.0)), ("y", (0.0, -5.0), (10.0, -5.0)), ("z", None, (1.0, 1.0))]
    assert nearest_edge_screen((5.0, 0.0), tie, 12.0)[0] == "x"
    assert nearest_edge_screen((3.0, 4.0), [("p", (0.0, 0.0), (0.0, 0.0))], 5.0) == ("p", 5.0, 0.0)
    assert nearest_edge_screen((0.0, 0.0), [], 12.0) is None


# A long edge crossed by two verticals at x = 2 and x = 4.
LONG = ("e", (0.0, 0.0), (6.0, 0.0))
V2 = ("v2", (2.0, -1.0), (2.0, 1.0))
V4 = ("v4", (4.0, -1.0), (4.0, 1.0))


def test_plan_trim_picks_the_interval_holding_the_click():
    segments = [LONG, V2, V4]
    first = plan_trim(LONG[1], LONG[2], segments, 0.1, own_key="e")
    assert (first.t_lo, round(first.t_hi, 6)) == (0.0, round(2 / 6, 6))
    assert [round(b.t, 6) for b in first.bounds] == [round(2 / 6, 6)]
    middle = plan_trim(LONG[1], LONG[2], segments, 0.5, own_key="e")
    assert (round(middle.t_lo, 6), round(middle.t_hi, 6)) == (round(2 / 6, 6), round(4 / 6, 6))
    assert [b.hits[0][0] for b in middle.bounds] == ["v2", "v4"]
    assert all(abs(b.hits[0][1] - 0.5) < 1e-9 for b in middle.bounds)
    last = plan_trim(LONG[1], LONG[2], segments, 0.9, own_key="e")
    assert (round(last.t_lo, 6), last.t_hi) == (round(4 / 6, 6), 1.0)
    assert [b.hits[0][0] for b in last.bounds] == ["v4"]


def test_plan_trim_without_crossings_removes_the_whole_edge():
    plan = plan_trim(LONG[1], LONG[2], [LONG], 0.5, own_key="e")
    assert plan == trim_logic.TrimPlan(0.0, 1.0, ())


def test_plan_trim_ignores_shared_end_vertices_and_parallel_edges():
    corner = ("c", (6.0, 0.0), (6.0, 3.0))
    start = ("s", (0.0, 0.0), (0.0, 3.0))
    parallel = ("p", (1.0, 0.0), (5.0, 0.0))
    other_side = ("o", (6.0, 0.0), (9.0, 0.0))
    plan = plan_trim(LONG[1], LONG[2], [LONG, corner, start, parallel, other_side], 0.5, own_key="e")
    assert plan == trim_logic.TrimPlan(0.0, 1.0, ())


def test_plan_trim_t_junctions():
    # Another edge ending on this one's interior is a bound, with u at its end vertex.
    ends_on = ("f", (3.0, 2.0), (3.0, 0.0))
    plan = plan_trim(LONG[1], LONG[2], [LONG, ends_on], 0.9, own_key="e")
    assert [(b.t, b.hits) for b in plan.bounds] == [(0.5, (("f", 1.0),))]
    # This edge ending on another one's interior is just the end of the edge: nothing to split.
    through = ("g", (6.0, -2.0), (6.0, 2.0))
    assert plan_trim(LONG[1], LONG[2], [LONG, through], 0.5, own_key="e").bounds == ()


def test_plan_trim_merges_crossings_at_one_point():
    tip_a = ("a", (1.0, 2.0), (3.0, 0.0))
    tip_b = ("b", (3.0, 0.0), (5.0, 2.0))
    plan = plan_trim(LONG[1], LONG[2], [LONG, tip_a, tip_b], 0.9, own_key="e")
    assert len(plan.bounds) == 1 and plan.bounds[0].t == 0.5
    assert sorted(plan.bounds[0].hits) == [("a", 1.0), ("b", 0.0)]
    # Two edges crossing the same point from different sides.
    cross_a = ("a", (2.0, -1.0), (4.0, 1.0))
    cross_b = ("b", (4.0, -1.0), (2.0, 1.0))
    plan = plan_trim(LONG[1], LONG[2], [LONG, cross_a, cross_b], 0.1, own_key="e")
    assert len(plan.bounds) == 1 and [h[0] for h in plan.bounds[0].hits] == ["a", "b"]
    # Crossings a hair apart (under the weld distance) are one bound too.
    near = ("n", (3.0 + 4e-6, -1.0), (3.0 + 4e-6, 1.0))
    mid = ("m", (3.0, -1.0), (3.0, 1.0))
    plan = plan_trim(LONG[1], LONG[2], [LONG, mid, near], 0.1, own_key="e")
    assert len(plan.bounds) == 1 and len(plan.bounds[0].hits) == 2


def test_plan_trim_works_for_a_reversed_and_a_diagonal_edge():
    diagonal = ("d", (4.0, 4.0), (0.0, 0.0))
    cross = ("x", (0.0, 4.0), (4.0, 0.0))
    plan = plan_trim(diagonal[1], diagonal[2], [diagonal, cross], 0.2, own_key="d")
    assert (plan.t_lo, plan.t_hi) == (0.0, 0.5)
    plan = plan_trim(diagonal[1], diagonal[2], [diagonal, cross], 0.8, own_key="d")
    assert (plan.t_lo, plan.t_hi) == (0.5, 1.0)


def test_plan_extend_hits_the_nearest_edge_in_front():
    far = ("far", (5.0, -1.0), (5.0, 1.0))
    near = ("near", (3.0, -1.0), (3.0, 2.0))
    own = ("own", (0.0, 0.0), (1.0, 0.0))
    hit = plan_extend((1.0, 0.0), (1.0, 0.0), [own, far, near], own_key="own")
    assert hit.key == "near" and abs(hit.s - 2.0) < 1e-9 and abs(hit.u - 1.0 / 3.0) < 1e-9
    assert abs(hit.point[0] - 3.0) < 1e-9 and abs(hit.point[1]) < 1e-9
    # The direction need not be unit length.
    assert plan_extend((1.0, 0.0), (10.0, 0.0), [far, near]).key == "near"


def test_plan_extend_finds_nothing_behind_beside_or_parallel():
    segments = [("behind", (-3.0, -1.0), (-3.0, 1.0)), ("beside", (3.0, 1.0), (3.0, 4.0)),
                ("parallel", (2.0, 1.0), (6.0, 1.0)), ("short", (3.0, -0.5), (3.0, -0.1))]
    assert plan_extend((1.0, 0.0), (1.0, 0.0), segments) is None
    assert plan_extend((1.0, 0.0), (0.0, 0.0), segments) is None
    assert plan_extend((1.0, 0.0), (1.0, 0.0), []) is None
    # An edge the end already lies on is not "in front".
    assert plan_extend((1.0, 0.0), (1.0, 0.0), [("on", (1.0, -1.0), (1.0, 1.0))]) is None


def test_plan_extend_snaps_to_an_end_vertex():
    assert plan_extend((0.0, 0.0), (1.0, 0.0), [("v", (3.0, 0.0), (3.0, 2.0))]).u == 0.0
    assert plan_extend((0.0, 0.0), (1.0, 0.0), [("v", (3.0, -2.0), (3.0, 0.0))]).u == 1.0
    # Two edges meeting at the hit point: the first of them, either way an end vertex.
    tip = [("a", (3.0, 0.0), (4.0, 2.0)), ("b", (2.0, 2.0), (3.0, 0.0))]
    hit = plan_extend((0.0, 0.0), (1.0, 0.0), tip)
    assert hit.key == "a" and hit.u == 0.0


class _ConvergingView:
    """Orthographic top view with the x axis foreshortened, so screen and local parameters differ:
    screen x = 500 + 300 x / (1 + 0.5 x), screen y = 400 + 100 y."""

    def to_screen(self, xy):
        return (500.0 + 300.0 * xy[0] / (1.0 + 0.5 * xy[0]), 400.0 + 100.0 * xy[1])

    def ray_to_local(self, mouse):
        sx = mouse[0] - 500.0
        return (sx / (300.0 - 0.5 * sx), (mouse[1] - 400.0) / 100.0)


class _BlindView(_ConvergingView):
    def ray_to_local(self, mouse):
        return None


def test_find_trim_reads_the_click_from_the_sketch_plane_not_the_screen():
    # The crossing at x = 3 is at local t = 0.75 but screen t = 0.9. A click at local x = 2.8
    # (t = 0.7) is left of the crossing; the screen parameter 0.875 would say right.
    segments = [("e", (0.0, 0.0), (4.0, 0.0)), ("v", (3.0, -1.0), (3.0, 1.0))]
    view = _ConvergingView()
    click = view.to_screen((2.8, 0.0))
    target = find_trim(click, segments, view, 12.0)
    assert target.key == "e" and (target.plan.t_lo, target.plan.t_hi) == (0.0, 0.75)
    # Without a plane hit (edge-on view) the screen parameter stands in.
    blind = find_trim(click, segments, _BlindView(), 12.0)
    assert (blind.plan.t_lo, blind.plan.t_hi) == (0.75, 1.0)


def test_find_trim_needs_an_edge_within_the_threshold():
    view = _TopView()
    segments = [("e", (0.0, 0.0), (4.0, 0.0))]
    assert find_trim(view.to_screen((2.0, 0.11)), segments, view, 12.0).key == "e"
    assert find_trim(view.to_screen((2.0, 0.13)), segments, view, 12.0) is None
    assert find_trim(view.to_screen((2.0, 0.0)), [], view, 12.0) is None


def test_find_extend_picks_the_nearest_loose_end_with_its_ray():
    view = _TopView()
    endpoints = [(1, (1.0, 0.0), (0.0, 0.0), "e1"), (2, (0.0, 3.0), (0.0, 2.0), "e2")]
    segments = [("e1", (0.0, 0.0), (1.0, 0.0)), ("e2", (0.0, 2.0), (0.0, 3.0)),
                ("wall", (3.0, -1.0), (3.0, 1.0))]
    target = find_extend(view.to_screen((1.05, 0.05)), endpoints, segments, view, 12.0)
    assert target.key == 1 and target.hit.key == "wall"
    # The end at (0, 3) points up and meets nothing.
    target = find_extend(view.to_screen((0.0, 3.0)), endpoints, segments, view, 12.0)
    assert target.key == 2 and target.hit is None
    assert find_extend(view.to_screen((2.0, 2.0)), endpoints, segments, view, 12.0) is None


def test_previews_describe_the_removal_and_the_extension():
    segments = [LONG, V2, V4]
    view = _TopView()
    target = find_trim(view.to_screen((3.0, 0.0)), segments, view, 12.0)
    preview = trim_logic.trim_preview(target)
    assert [tuple(round(c, 6) for c in p) for p in preview.strips[0]] == [(2.0, 0.0), (4.0, 0.0)]
    assert preview.guide == [LONG[1], LONG[2]] and len(preview.markers) == 2
    ext = trim_logic.ExtendTarget(1, (1.0, 0.0), trim_logic.ExtendHit("w", 2.0, 0.5, (3.0, 0.0)))
    preview = trim_logic.extend_preview(ext)
    assert preview.strips == [[(1.0, 0.0), (3.0, 0.0)]] and preview.snap_xy == (1.0, 0.0)
    assert trim_logic.extend_preview(trim_logic.ExtendTarget(1, (1.0, 0.0), None)).lines == ["No edge ahead"]


# ---------------------------------------------------------------------------
# Editor, controller and operators
# ---------------------------------------------------------------------------

A_RECT = [(0, 0), (2, 0), (2, 2), (0, 2)]
B_RECT = [(1, 1), (3, 1), (3, 3), (1, 3)]


def _edit_edges(segments, radii=None):
    """New sketch in edit mode holding the given (a, b) segments, vertices shared by position.
    radii maps a position to (radius, id)."""
    from sketch_tools.core import attributes

    obj = _edit_sketch()
    bm = bmesh.from_edit_mesh(obj.data)
    rad, ids = attributes.ensure_bm_layers(bm)
    verts = {}

    def vert(xy):
        if xy not in verts:
            verts[xy] = bm.verts.new((xy[0], xy[1], 0.0))
        return verts[xy]

    for a, b in segments:
        bm.edges.new((vert(a), vert(b)))
    for xy, (radius, vid) in (radii or {}).items():
        verts[xy][rad] = radius
        verts[xy][ids] = vid
    bmesh.update_edit_mesh(obj.data)
    return obj


def _geometry(obj):
    """Positions, fillet attributes and edges by position: independent of element order."""
    bm = bmesh.from_edit_mesh(obj.data)
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)

    def pos(v):
        return (round(v.co.x, 5) + 0.0, round(v.co.y, 5) + 0.0)

    verts = sorted((*pos(v), round(v[rad], 5), v[ids]) for v in bm.verts)
    edges = sorted(tuple(sorted(pos(v) for v in e.verts)) for e in bm.edges)
    return verts, edges


def _run(obj, kind, view=None, threshold=12):
    headers, reports = [], []
    controller = TrimController(
        kind, TrimEditor(obj), lambda: view or _TopView(), threshold,
        header=headers.append, redraw=lambda: None, report=lambda level, text: reports.append((level, text)))
    return controller, headers, reports


def _lmb(c, x, y, value='PRESS'):
    return c.handle(_event('LEFTMOUSE', value, (x, y)))


def _move(c, x, y):
    return c.handle(_event('MOUSEMOVE', 'NOTHING', (x, y)))


def _leave_edit_mode(obj):
    bpy.ops.object.mode_set(mode='OBJECT')


def _trim_all(obj, clicks):
    c, headers, reports = _run(obj, TRIM)
    for x, y in clicks:
        assert _lmb(c, x, y) == 'RUNNING_MODAL'
    assert c.count == len(clicks), (c.count, clicks)
    return c


def _check_union_outline(obj, clicks):
    c = _trim_all(obj, clicks)
    assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
    closed, open_ = _chains_of(obj)
    assert len(closed) == 1 and not open_
    chain = closed[0]
    assert len(chain.points) == 8 and abs(signed_area(chain.points) - 7.0) < 1e-5
    # The chain read back is exactly the outline of the union.
    expected = {(0, 0), (2, 0), (2, 1), (3, 1), (3, 3), (1, 3), (1, 2), (0, 2)}
    assert {(round(p.x), round(p.y)) for p in chain.points} == expected
    obj.sketch_tools.output_mode = 'EXTRUDE'
    out = output_bm(obj)
    try:
        assert_closed_and_consistent(out)
        assert abs(abs(signed_volume(out)) - 7.0 * obj.sketch_tools.ex_height) < 1e-4
    finally:
        out.free()


def _chains_of(obj):
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        resolved = resolve.resolve_bmesh(bm, 8)
    finally:
        bm.free()
    return [c for c in resolved.chains if c.closed], [c for c in resolved.chains if not c.closed]


def test_two_overlapping_rectangles_trim_to_one_closed_outline_a_first():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        _check_union_outline(obj, [(2, 1.5), (1.5, 2), (1.5, 1), (1, 1.5)])


def test_two_overlapping_rectangles_trim_to_one_closed_outline_b_first():
    # The crossing edges are split from the other side here: B's edges carry the crossings.
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        _check_union_outline(obj, [(1.5, 1), (1, 1.5), (2, 1.5), (1.5, 2)])


def test_two_overlapping_rectangles_trim_to_one_closed_outline_mixed_order():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        _check_union_outline(obj, [(1.5, 1), (2, 1.5), (1, 1.5), (1.5, 2)])


def test_trimming_the_middle_interval_splits_the_crossing_edges_and_welds():
    segments = [((0, 0), (6, 0)), ((2, -1), (2, 1)), ((4, -1), (4, 1))]
    with addon():
        obj = _edit_edges(segments)
        c, headers, reports = _run(obj, TRIM)
        assert _lmb(c, 3, 0) == 'RUNNING_MODAL' and c.count == 1
        verts, edges = _geometry(obj)
        assert len(verts) == 8 and len(edges) == 6
        assert edges == sorted([((0, 0), (2, 0)), ((4, 0), (6, 0)), ((2, -1), (2, 0)), ((2, 0), (2, 1)),
                                ((4, -1), (4, 0)), ((4, 0), (4, 1))])
        bm = bmesh.from_edit_mesh(obj.data)
        by_pos = {(round(v.co.x), round(v.co.y)): v for v in bm.verts}
        # The welded vertex joins the horizontal piece and both vertical pieces.
        assert len(by_pos[(2, 0)].link_edges) == 3 and len(by_pos[(4, 0)].link_edges) == 3
        c.abort()


def test_trimming_an_end_interval_leaves_the_rest_and_drops_the_orphan_vertex():
    segments = [((0, 0), (6, 0)), ((2, -1), (2, 1))]
    with addon():
        obj = _edit_edges(segments)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 5, 0) == 'RUNNING_MODAL'
        verts, edges = _geometry(obj)
        # (6, 0) had no other edge and goes with its edge.
        positions = sorted((v[0], v[1]) for v in verts)
        assert positions == [(0, 0), (2, -1), (2, 0), (2, 1)]
        assert edges == sorted([((0, 0), (2, 0)), ((2, -1), (2, 0)), ((2, 0), (2, 1))])
        c.abort()


def test_trimming_an_edge_without_crossings_deletes_it_and_only_its_orphans():
    segments = [((0, 0), (2, 0)), ((2, 0), (2, 2)), ((5, 5), (6, 5))]
    with addon():
        obj = _edit_edges(segments)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL' and c.count == 1
        verts, edges = _geometry(obj)
        # (0, 0) is gone, (2, 0) still ends the other edge.
        assert sorted((v[0], v[1]) for v in verts) == [(2, 0), (2, 2), (5, 5), (6, 5)]
        assert edges == sorted([((2, 0), (2, 2)), ((5, 5), (6, 5))])
        # Trimming the last edge of its chain removes both ends.
        assert _lmb(c, 5.5, 5) == 'RUNNING_MODAL'
        assert sorted((v[0], v[1]) for v in _geometry(obj)[0]) == [(2, 0), (2, 2)]
        c.abort()


def test_trim_at_a_crossing_that_is_a_vertex_keeps_that_vertex():
    # A V whose tip lies on the edge: the crossing is the V's own vertex, with its fillet.
    segments = [((0, 0), (4, 0)), ((1, 2), (2, 0)), ((2, 0), (3, 2))]
    with addon():
        obj = _edit_edges(segments, radii={(2, 0): (0.7, 91)})
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 3, 0) == 'RUNNING_MODAL'
        verts, edges = _geometry(obj)
        assert verts == sorted([(0, 0, 0, 0), (1, 2, 0, 0), (2, 0, 0.7, 91), (3, 2, 0, 0)])
        assert edges == sorted([((0, 0), (2, 0)), ((1, 2), (2, 0)), ((2, 0), (3, 2))])
        # The left part of the edge now ends on the tip, so trimming it takes the whole piece.
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL'
        assert len(_geometry(obj)[1]) == 2
        c.abort()


def test_trim_when_the_edge_ends_on_another_edge_does_not_split_that_edge():
    segments = [((0, 0), (2, 0)), ((2, -1), (2, 1))]
    with addon():
        obj = _edit_edges(segments)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL'
        verts, edges = _geometry(obj)
        assert edges == [((2, -1), (2, 1))] and len(verts) == 2
        c.abort()


def test_trim_leaves_new_vertices_sharp_and_existing_radii_alone():
    segments = [((0, 0), (6, 0)), ((2, -1), (2, 1)), ((4, -1), (4, 1))]
    radii = {(0, 0): (0.5, 11), (6, 0): (0.25, 22), (2, -1): (0.4, 33), (2, 1): (0.8, 44), (4, 1): (0.9, 55)}
    with addon():
        obj = _edit_edges(segments, radii=radii)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 3, 0) == 'RUNNING_MODAL'
        verts = {(v[0], v[1]): (v[2], v[3]) for v in _geometry(obj)[0]}
        assert verts[(2, 0)] == (0, 0) and verts[(4, 0)] == (0, 0)
        assert verts[(0, 0)] == (0.5, 11) and verts[(6, 0)] == (0.25, 22)
        assert verts[(2, -1)] == (0.4, 33) and verts[(2, 1)] == (0.8, 44) and verts[(4, 1)] == (0.9, 55)
        c.abort()


def test_trim_keeps_a_filleted_corner_after_leaving_edit_mode():
    # The fillet id cleanup (DECISIONS C10) must not take the radius of a corner that was kept.
    with addon():
        obj = _edit_edges([((0, 0), (4, 0)), ((4, 0), (4, 4)), ((2, -1), (2, 1))], radii={(4, 0): (0.5, 7)})
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 3, 0) == 'RUNNING_MODAL'
        assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh = obj.data
        radii = {(round(v.co.x), round(v.co.y)): mesh.attributes[FILLET_RADIUS_ATTR].data[v.index].value
                 for v in mesh.vertices}
        assert abs(radii[(4, 0)] - 0.5) < 1e-6 and radii[(2, 0)] == 0.0


def test_trim_clicks_away_from_every_edge_do_nothing():
    with addon():
        obj = _edit_edges([((0, 0), (4, 0))])
        before = _geometry(obj)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 2, 0.5) == 'RUNNING_MODAL' and c.count == 0
        assert _geometry(obj) == before and not c.editor.dirty
        assert c.handle(_event('RIGHTMOUSE')) == 'CANCELLED'


def test_trim_ignores_hidden_edges_when_picking_and_crossing():
    with addon():
        obj = _edit_edges([((0, 0), (6, 0)), ((3, -1), (3, 1))])
        bm = bmesh.from_edit_mesh(obj.data)
        for e in bm.edges:
            if abs(e.verts[0].co.x - 3) < 1e-6 and abs(e.verts[1].co.x - 3) < 1e-6:
                e.hide_set(True)
        bmesh.update_edit_mesh(obj.data)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL'
        assert len(bmesh.from_edit_mesh(obj.data).edges) == 1
        c.abort()


def test_hover_shows_the_interval_that_a_click_would_remove():
    with addon():
        obj = _edit_edges([((0, 0), (6, 0)), ((2, -1), (2, 1)), ((4, -1), (4, 1))])
        c, *_ = _run(obj, TRIM, view=_TopView(size=(1000.0, 840.0)))
        assert c.snapshot() is None
        assert _move(c, 3, 0) == 'RUNNING_MODAL'
        preview, mouse, to_screen = c.snapshot()
        ends = [tuple(round(v, 5) for v in p) for p in preview.strips[0]]
        assert ends == [(2.0, 0.0), (4.0, 0.0)] and mouse == (800.0, 400.0)
        assert to_screen((2.0, 0.0)) == (700.0, 400.0)
        _move(c, 2.0, 3.0)
        assert c.snapshot() is None
        # Leaving the region clears the highlight.
        _move(c, 3, 0)
        assert c.handle(_outside_event('MOUSEMOVE', 'NOTHING')) == 'PASS_THROUGH'
        assert c.snapshot() is None
        # The highlight follows the edit: after the click it points at what is left.
        _lmb(c, 3, 0)
        _move(c, 1, 0)
        assert [tuple(round(v, 5) for v in p) for p in c.snapshot()[0].strips[0]] == [(0.0, 0.0), (2.0, 0.0)]
        c.abort()


# -- extend --------------------------------------------------------------------------------

def test_extend_joins_the_loose_end_to_the_first_edge_and_splits_it():
    segments = [((0, 0), (1, 0)), ((3, -1), (3, 1)), ((5, -1), (5, 1))]
    radii = {(3, -1): (0.4, 77), (3, 1): (0.8, 88), (0, 0): (0.2, 5)}
    with addon():
        obj = _edit_edges(segments, radii=radii)
        c, headers, reports = _run(obj, EXTEND)
        assert _lmb(c, 1.0, 0.0) == 'RUNNING_MODAL' and c.count == 1 and not reports
        verts, edges = _geometry(obj)
        by_pos = {(v[0], v[1]): (v[2], v[3]) for v in verts}
        assert (3, 0) in by_pos and by_pos[(3, 0)] == (0, 0)
        assert by_pos[(3, -1)] == (0.4, 77) and by_pos[(3, 1)] == (0.8, 88) and by_pos[(0, 0)] == (0.2, 5)
        assert edges == sorted([((0, 0), (1, 0)), ((1, 0), (3, 0)), ((3, -1), (3, 0)), ((3, 0), (3, 1)),
                                ((5, -1), (5, 1))])
        assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'


def test_extend_with_no_hit_reports_and_leaves_the_mesh_unchanged():
    segments = [((0, 0), (1, 0)), ((3, 2), (3, 4))]
    with addon():
        obj = _edit_edges(segments)
        before = (_geometry(obj), _mesh_state(obj))
        c, headers, reports = _run(obj, EXTEND)
        assert _lmb(c, 1.0, 0.0) == 'RUNNING_MODAL' and c.count == 0
        assert reports == [('WARNING', "Extend: no edge in that direction")]
        assert (_geometry(obj), _mesh_state(obj)) == before and not c.editor.dirty
        assert c.handle(_event('RIGHTMOUSE')) == 'CANCELLED'


def test_extend_onto_a_vertex_reuses_it_without_splitting():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((3, 0), (3, 2)), ((3, 0), (5, 1))], radii={(3, 0): (0.6, 9)})
        c, *_ = _run(obj, EXTEND)
        assert _lmb(c, 1.0, 0.0) == 'RUNNING_MODAL'
        verts, edges = _geometry(obj)
        assert len(verts) == 5 and ((1, 0), (3, 0)) in edges and len(edges) == 4
        assert {(v[0], v[1]): (v[2], v[3]) for v in verts}[(3, 0)] == (0.6, 9)
        c.abort()


def test_extend_follows_a_diagonal_edge_to_the_wall():
    with addon():
        obj = _edit_edges([((2, 2), (1, 1)), ((-3, -5), (-3, 5))])
        c, *_ = _run(obj, EXTEND)
        assert _lmb(c, 1.0, 1.0) == 'RUNNING_MODAL' and c.count == 1
        verts, edges = _geometry(obj)
        assert any(abs(v[0] + 3) < 1e-4 and abs(v[1] + 3) < 1e-4 and v[2:] == (0, 0) for v in verts)
        assert len(edges) == 4
        c.abort()


def test_extend_ignores_clicks_that_are_not_on_a_loose_end():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((1, 0), (1, 1)), ((3, -1), (3, 2))])
        c, headers, reports = _run(obj, EXTEND)
        # (1, 0) is a corner (two edges); (0, 0) is the loose end but the click is far from it.
        assert _lmb(c, 1.0, 0.0) == 'RUNNING_MODAL' and _lmb(c, 0.5, 0.0) == 'RUNNING_MODAL'
        assert c.count == 0 and not reports
        c.abort()


def test_extend_hover_previews_the_extension_and_the_missing_hit():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((3, -1), (3, 1)), ((0, 4), (0, 5))])
        c, *_ = _run(obj, EXTEND)
        _move(c, 1.0, 0.0)
        preview = c.snapshot()[0]
        assert [tuple(round(v, 5) for v in p) for p in preview.strips[0]] == [(1.0, 0.0), (3.0, 0.0)]
        assert preview.snap_xy == (1.0, 0.0)
        _move(c, 0.0, 5.0)
        assert c.snapshot()[0].lines == ["No edge ahead"]
        c.abort()


# -- session, undo and exit paths ------------------------------------------------------------

def _two_trims(obj):
    c, headers, reports = _run(obj, TRIM)
    assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL' and _lmb(c, 1, 1.5) == 'RUNNING_MODAL'
    return c, headers


def test_esc_cancels_the_whole_session_and_restores_the_mesh_exactly():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        bm = bmesh.from_edit_mesh(obj.data)
        rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
        ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
        for i, v in enumerate(bm.verts):
            v[rad], v[ids] = (0.25 * i, 100 + i) if i % 3 == 0 else (0.0, 0)
            v.select_set(i % 2 == 0)
        bmesh.update_edit_mesh(obj.data)
        order = [(round(v.co.x), round(v.co.y), v[rad], v[ids], v.select) for v in bmesh.from_edit_mesh(obj.data).verts]
        before = (_geometry(obj), _mesh_state(obj))
        c, headers = _two_trims(obj)
        assert _geometry(obj) != before[0] and c.count == 2
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert c.closed and headers[-1] is None
        assert (_geometry(obj), _mesh_state(obj)) == before
        bm = bmesh.from_edit_mesh(obj.data)
        rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
        ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
        # Same vertex order, which the fillet id cleanup relies on.
        assert [(round(v.co.x), round(v.co.y), v[rad], v[ids], v.select) for v in bm.verts] == order


def test_rmb_and_enter_keep_the_session_and_report_one_finished_step():
    for key in ('RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'):
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            before = _geometry(obj)
            c, headers = _two_trims(obj)
            assert c.handle(_event(key)) == 'FINISHED', key
            assert c.closed and headers[-1] is None and _geometry(obj) != before, key
            assert not c.editor.dirty


def test_finishing_without_an_edit_is_not_an_undo_step():
    with addon():
        obj = _edit_sketch([A_RECT])
        c, *_ = _run(obj, TRIM)
        assert c.handle(_event('RET')) == 'CANCELLED'
        c, *_ = _run(obj, EXTEND)
        assert c.handle(_event('RIGHTMOUSE')) == 'CANCELLED'


def test_the_header_names_the_keys_and_counts_the_edits():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        c, headers = _two_trims(obj)
        assert "Esc cancels all" in headers[0] and "(2 done)" in headers[-1]
        c.abort()
        c, headers, _ = _run(obj, EXTEND)
        assert headers[0].startswith("Extend:")
        c.abort()


def test_operators_register_with_an_undo_step_and_unregister():
    with addon():
        for name in ("trim", "extend", "fillet"):
            assert hasattr(bpy.ops.sketch, name)
        for cls in (trim.SKETCH_OT_trim, trim.SKETCH_OT_extend):
            assert 'UNDO' in cls.bl_options
    assert not hasattr(bpy.types, "SKETCH_OT_trim") and not hasattr(bpy.types, "SKETCH_OT_extend")
    assert not hasattr(bpy.types, "SKETCH_OT_fillet")


def test_poll_refuses_everything_but_a_sketch_in_a_3d_view_edit_mode():
    with addon():
        sketch = _edit_sketch([A_RECT])
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.collection.objects.link(plain)

        def ctx(mode, area_type, active):
            area = None if area_type is None else types.SimpleNamespace(type=area_type)
            return types.SimpleNamespace(mode=mode, area=area, active_object=active)

        for cls in (trim.SKETCH_OT_trim, trim.SKETCH_OT_extend):
            assert cls.poll(ctx('EDIT_MESH', 'VIEW_3D', sketch))
            assert not cls.poll(ctx('EDIT_MESH', 'VIEW_3D', plain))
            assert not cls.poll(ctx('EDIT_MESH', 'VIEW_3D', None))
            assert not cls.poll(ctx('OBJECT', 'VIEW_3D', sketch))
            assert not cls.poll(ctx('EDIT_MESH', 'PROPERTIES', sketch))
            assert not cls.poll(ctx('EDIT_MESH', None, sketch))
        _leave_edit_mode(sketch)
        bpy.data.objects.remove(plain)


def _shell(obj, view=None, region_type='WINDOW'):
    """A stand-in `self` and context for calling the operator methods directly."""
    headers, handlers = [], []
    area = fake_area(type='VIEW_3D', header_text_set=headers.append, tag_redraw=lambda: None)
    wm = types.SimpleNamespace(modal_handler_add=handlers.append)
    context = types.SimpleNamespace(
        active_object=obj, area=area, window_manager=wm, scene=bpy.context.scene,
        preferences=bpy.context.preferences,
        region=types.SimpleNamespace(type=region_type, as_pointer=lambda: 4242))
    shell = types.SimpleNamespace(reports=[])
    shell.report = lambda kind, text: shell.reports.append((kind, text))
    restore = install_fake_area(area, lambda o: (lambda: view or _TopView()))
    return shell, context, headers, handlers, restore


def test_operator_paths_clean_up_after_themselves():
    # (operator, clicks that edit, a click that reaches the editor after those)
    cases = ((trim.SKETCH_OT_trim, [(1.5, 1), (1, 1.5)], (2, 0.5)), (trim.SKETCH_OT_extend, [(1, 0)], (1, 3)))
    for op_cls, clicks, error_click in cases:
        with addon():
            for how in ('finish', 'esc', 'error', 'cancel_callback'):
                if op_cls is trim.SKETCH_OT_trim:
                    obj = _edit_sketch([A_RECT, B_RECT])
                else:
                    obj = _edit_edges([((0, 0), (1, 0)), ((0, 3), (1, 3)), ((3, -1), (3, 4))])
                before = _geometry(obj)
                shell, ctx, headers, handlers, restore = _shell(obj)
                try:
                    assert op_cls.invoke(shell, ctx, _event('LEFTMOUSE', at=(9, 9))) == {'RUNNING_MODAL'}
                    assert handlers == [shell] and len(overlay._live) == 1 and headers[-1] is not None
                    # The invoking event is not a click: nothing changed yet.
                    assert _geometry(obj) == before
                    for x, y in clicks:
                        assert op_cls.modal(shell, ctx, _event('LEFTMOUSE', at=(x, y))) == {'RUNNING_MODAL'}
                    if how == 'finish':
                        assert op_cls.modal(shell, ctx, _event('RIGHTMOUSE')) == {'FINISHED'}
                    elif how == 'esc':
                        assert op_cls.modal(shell, ctx, _event('ESC')) == {'CANCELLED'}
                    elif how == 'error':
                        def broken(target):
                            raise ZeroDivisionError("boom")
                        shell._controller.editor.trim = shell._controller.editor.extend = broken
                        with redirect_stdout(io.StringIO()):
                            assert op_cls.modal(shell, ctx, _event('LEFTMOUSE', at=error_click)) == {'CANCELLED'}
                        assert shell.reports and shell.reports[0][0] == {'ERROR'}
                    else:
                        op_cls.cancel(shell, ctx)
                finally:
                    restore()
                assert not trim._live and not overlay._live and headers[-1] is None, how
                if how == 'finish':
                    assert _geometry(obj) != before
                else:
                    assert _geometry(obj) == before, how
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.data.objects.remove(obj)


def _start_is_refused(op_cls, obj):
    """invoke() of op_cls must neither start nor touch anything while another tool is live."""
    shell, ctx, headers, handlers, restore = _shell(obj)
    overlays = len(overlay._live)
    try:
        result = op_cls.invoke(shell, ctx, _event('LEFTMOUSE'))
    finally:
        restore()
    assert not handlers and len(overlay._live) == overlays and not headers, op_cls.bl_idname
    if op_cls is draw_tool.SKETCH_OT_draw:
        # Its keymap re-enters on clicks in another viewport or a gap of the sidebar. Blender
        # discards the reports of an invoke that passes through, and those clicks stay silent.
        assert result == {'PASS_THROUGH'} and shell.reports == [], result
    else:
        # Started from the panel or the search; CANCELLED is what lets the warning show.
        assert result == {'CANCELLED'}, (op_cls.bl_idname, result)
        assert shell.reports == [({'WARNING'}, "Finish the running sketch tool first")], op_cls.bl_idname


def test_a_second_start_is_refused_in_every_direction_while_any_sketch_modal_is_live():
    from sketch_tools.ops import draw_tool as draw_module

    operators = (draw_module.SKETCH_OT_draw, fillet.SKETCH_OT_fillet, trim.SKETCH_OT_trim, trim.SKETCH_OT_extend)
    with addon():
        obj = _edit_sketch([A_RECT])
        assert not modal_common.any_sketch_modal_live()
        # A running draw or fillet run, stood in for by a token in its registry.
        for module in (draw_tool, fillet):
            token = object()
            module._live.add(token)
            try:
                assert modal_common.any_sketch_modal_live()
                for op_cls in operators:
                    _start_is_refused(op_cls, obj)
            finally:
                module._live.discard(token)
            assert not modal_common.any_sketch_modal_live()
        # Real Trim and Extend runs.
        for starter in (trim.SKETCH_OT_trim, trim.SKETCH_OT_extend):
            shell, ctx, headers, handlers, restore = _shell(obj)
            try:
                assert starter.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
                assert modal_common.any_sketch_modal_live() and len(trim._live) == 1
                for op_cls in operators:
                    _start_is_refused(op_cls, obj)
                assert len(trim._live) == 1 and len(overlay._live) == 1
                starter.cancel(shell, ctx)
            finally:
                restore()
            assert not modal_common.any_sketch_modal_live() and not overlay._live
        # Once everything is closed a new run starts again.
        shell, ctx, headers, handlers, restore = _shell(obj)
        try:
            assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
            trim.SKETCH_OT_trim.cancel(shell, ctx)
        finally:
            restore()
        assert not trim._live and not overlay._live


def test_start_failures_clean_up_and_leave_the_mesh_alone():
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    with addon():
        for failing in ('overlay', 'handler'):
            obj = _edit_sketch([A_RECT])
            before = _geometry(obj)
            shell, ctx, headers, handlers, restore = _shell(obj)
            real_add = overlay.DrawOverlay.add
            if failing == 'overlay':
                overlay.DrawOverlay.add = boom
            else:
                ctx.window_manager.modal_handler_add = boom
            try:
                with redirect_stdout(io.StringIO()):
                    assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'CANCELLED'}
            finally:
                overlay.DrawOverlay.add = real_add
                restore()
            assert shell.reports[0][0] == {'ERROR'} and not trim._live and not overlay._live, failing
            assert headers[-1] is None and shell._controller.closed and _geometry(obj) == before, failing
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


@contextmanager
def _fake_ops(result):
    """Replace trim.bpy so the relaunch can be observed without a window."""
    calls = []
    ops = types.SimpleNamespace(sketch=types.SimpleNamespace(
        trim=lambda mode: calls.append(("trim", mode)) or result,
        extend=lambda mode: calls.append(("extend", mode)) or result))
    real = trim.bpy
    trim.bpy = types.SimpleNamespace(ops=ops, types=real.types)
    try:
        yield calls
    finally:
        trim.bpy = real


def test_started_from_a_panel_button_the_modal_is_relaunched_in_the_viewport_region():
    with addon():
        obj = _edit_sketch([A_RECT])
        for op_cls, name in ((trim.SKETCH_OT_trim, "trim"), (trim.SKETCH_OT_extend, "extend")):
            shell, ctx, headers, handlers, restore = _shell(obj, region_type='UI')
            window_region = types.SimpleNamespace(type='WINDOW')
            ctx.area.regions = [ctx.region, window_region]
            ctx.window = object()
            overrides = []

            @contextmanager
            def override(**kwargs):
                overrides.append(kwargs)
                yield

            ctx.temp_override = override
            try:
                with _fake_ops({'RUNNING_MODAL'}) as calls:
                    assert op_cls.invoke(shell, ctx, _event('LEFTMOUSE')) == {'CANCELLED'}
                assert calls == [(name, 'INVOKE_DEFAULT')]
                assert overrides == [{"window": ctx.window, "area": ctx.area, "region": window_region}]
                assert not shell.reports and not handlers and not trim._live
                # A nested start that does not run is reported.
                with _fake_ops({'CANCELLED'}):
                    assert op_cls.invoke(shell, ctx, _event('LEFTMOUSE')) == {'CANCELLED'}
                assert shell.reports[-1][0] == {'WARNING'}
                # No viewport window region: reported as an error.
                ctx.area.regions = [ctx.region]
                with _fake_ops({'RUNNING_MODAL'}) as calls:
                    assert op_cls.invoke(shell, ctx, _event('LEFTMOUSE')) == {'CANCELLED'}
                assert not calls and shell.reports[-1][0] == {'ERROR'}
            finally:
                restore()
        bpy.ops.object.mode_set(mode='OBJECT')


def test_close_all_winds_a_running_session_down_and_restores_the_mesh():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        before = _geometry(obj)
        shell, ctx, headers, handlers, restore = _shell(obj)
        try:
            assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
            trim.SKETCH_OT_trim.modal(shell, ctx, _event('LEFTMOUSE', at=(1.5, 1)))
            assert _geometry(obj) != before
            # While the area is still there, so the header can be cleared.
            trim.close_all()
        finally:
            restore()
        assert shell._controller.closed and not trim._live and not overlay._live
        assert headers[-1] is None and _geometry(obj) == before


def test_the_controller_stops_when_edit_mode_is_left():
    # DECISIONS C16: the edits made so far stay, nothing is restored, and the modal closes cleanly.
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        meshes = set(bpy.data.meshes)
        before = _geometry(obj)
        c, headers = _two_trims(obj)
        edited = _geometry(obj)
        assert edited != before and len(bpy.data.meshes) == len(meshes) + 1
        bpy.ops.object.mode_set(mode='OBJECT')
        assert c.handle(_event('MOUSEMOVE', 'NOTHING')) == 'CANCELLED'
        assert c.closed and headers[-1] is None and not trim._live and not overlay._live
        assert set(bpy.data.meshes) == meshes
        mesh = obj.data
        kept_verts = sorted((round(v.co.x, 5) + 0.0, round(v.co.y, 5) + 0.0) for v in mesh.vertices)
        kept_edges = sorted(tuple(sorted((round(mesh.vertices[i].co.x, 5) + 0.0, round(mesh.vertices[i].co.y, 5) + 0.0)
                                         for i in e.vertices)) for e in mesh.edges)
        assert kept_verts == [(x, y) for x, y, *_ in edited[0]] and kept_edges == edited[1]
        # Closing again, and unregistering, stay harmless.
        c.abort()
        trim.close_all()
        assert set(bpy.data.meshes) == meshes


# -- pass-through ------------------------------------------------------------------------------

def test_events_outside_the_region_or_over_overlays_pass_through_and_never_edit():
    header = (30.0, 800.0, 970.0, 38.0)
    sidebar = (860.0, 120.0, 170.0, 680.0)
    with addon():
        obj = _edit_edges([((0, 0), (6, 0)), ((2, -1), (2, 1))])
        view = _TopView(size=(1000.0, 840.0), overlays=(header, sidebar))
        c, headers, reports = _run(obj, TRIM, view=view)
        before = _geometry(obj)
        assert c.handle(_outside_event('LEFTMOUSE', 'PRESS')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFTMOUSE', 'RELEASE')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('LEFTMOUSE', 'CLICK')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('RIGHTMOUSE', 'PRESS')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('RIGHTMOUSE', 'RELEASE')) == 'PASS_THROUGH'
        assert c.handle(_outside_event('MOUSEMOVE', 'NOTHING')) == 'PASS_THROUGH'
        for spot in ((500.0, 820.0), (900.0, 400.0)):
            assert c.handle(_overlay_event('LEFTMOUSE', 'PRESS', spot)) == 'PASS_THROUGH'
            assert c.handle(_overlay_event('RIGHTMOUSE', 'PRESS', spot)) == 'PASS_THROUGH'
            assert c.handle(_overlay_event('LEFTMOUSE', 'RELEASE', spot)) == 'PASS_THROUGH'
        assert _geometry(obj) == before and not c.closed and c.count == 0
        # A press inside ends the record of passed presses: its release is swallowed again.
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL' and c.count == 1
        assert _lmb(c, 1, 0, 'RELEASE') == 'RUNNING_MODAL'
        c.abort()


def test_a_passed_alt_lmb_press_passes_its_release_and_clicks_through():
    with addon():
        obj = _edit_edges([((0, 0), (6, 0))])
        c, *_ = _run(obj, TRIM)
        before = _geometry(obj)
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (3, 0), alt=True)) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (4, 3))) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'CLICK', (4, 3))) == 'PASS_THROUGH'
        assert _geometry(obj) == before and c.count == 0
        c.abort()


def test_navigation_and_system_events_pass_through_and_keys_are_swallowed():
    with addon():
        obj = _edit_edges([((0, 0), (6, 0))])
        c, *_ = _run(obj, TRIM)
        for etype in ('MIDDLEMOUSE', 'WHEELUPMOUSE', 'TRACKPADPAN', 'NDOF_MOTION', 'TIMER', 'WINDOW_DEACTIVATE',
                      'NUMPAD_4', 'NUMPAD_PERIOD'):
            assert c.handle(_event(etype)) == 'PASS_THROUGH', etype
        for etype in ('A', 'G', 'TAB', 'Z'):
            assert c.handle(_event(etype)) == 'RUNNING_MODAL', etype
        assert c.handle(_event('A', 'RELEASE')) == 'RUNNING_MODAL'
        assert not c.closed and c.count == 0
        c.abort()


# -- panel ---------------------------------------------------------------------------------------

def test_the_sketch_panel_offers_fillet_trim_and_extend_in_edit_mode_only():
    from sketch_tools.tests.test_addon import _FakeLayout
    from sketch_tools.ui import panel

    class Recording(_FakeLayout):
        def __init__(self, seen=None):
            super().__init__()
            self.seen = seen if seen is not None else []

        def _child(self, *_args, **_kwargs):
            return Recording(self.seen)

        column = row = box = _child

        def operator(self, idname, **kwargs):
            super().operator(idname, **kwargs)
            self.seen.append(idname)
            return types.SimpleNamespace()

    with addon():
        sketch = _edit_sketch([A_RECT])
        for mode, expected in (('EDIT_MESH', ["sketch.fillet", "sketch.trim", "sketch.extend", "sketch.subdivide"]), ('OBJECT', [])):
            ctx = types.SimpleNamespace(mode=mode, active_object=sketch, window_manager=bpy.context.window_manager)
            layout = Recording()
            panel.SKETCH_PT_sketch.draw(types.SimpleNamespace(layout=layout), ctx)
            assert layout.seen == expected, mode
        _leave_edit_mode(sketch)


# -- weld target (QA round 1: B1, W4) ------------------------------------------------------------

def _verts_near(obj, x, y, tol=1e-3):
    bm = bmesh.from_edit_mesh(obj.data)
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    return [(tuple(v.co), round(v[rad], 5), v[ids]) for v in bm.verts
            if abs(v.co.x - x) < tol and abs(v.co.y - y) < tol]


def test_a_weld_keeps_an_existing_filleted_vertex_that_lies_on_the_trimmed_diagonal():
    # P ends a second edge and lies on the diagonal to float32 precision, exactly or 1e-7 off to
    # either side. The new split vertex lands within 1e-7 of it; the existing vertex has to win
    # the weld whatever the coordinate order, and the click can be on either side of P.
    for offset in (0.0, 1e-7, -1e-7):
        for click in ((2.0, 1.0), (-2.0, -1.0)):
            with addon():
                obj = _edit_edges([((-4, -2), (4, 2)), ((0, offset), (0.5, 3))], radii={(0, offset): (0.5, 77)})
                p_before = _verts_near(obj, 0, 0)
                assert len(p_before) == 1
                c, *_ = _run(obj, TRIM)
                assert _lmb(c, *click) == 'RUNNING_MODAL' and c.count == 1, (offset, click)
                near = _verts_near(obj, 0, 0)
                # One vertex, the very same position, radius and id.
                assert near == p_before, (offset, click, near, p_before)
                assert near[0][1:] == (0.5, 77)
                c.abort()


def test_a_weld_keeps_existing_vertices_on_random_diagonals():
    import random

    from sketch_tools.core import attributes

    rnd = random.Random(5)
    trimmed = 0
    with addon():
        for _ in range(120):
            obj = _edit_sketch()
            bm = bmesh.from_edit_mesh(obj.data)
            rad, ids = attributes.ensure_bm_layers(bm)
            slope = rnd.uniform(-1.0, 1.0)
            x = rnd.uniform(1.0, 5.0)
            v0 = bm.verts.new((0, 0, 0))
            v1 = bm.verts.new((8, 8 * slope, 0))
            p = bm.verts.new((x, x * slope + rnd.uniform(-3e-6, 3e-6), 0))
            p[rad], p[ids] = 0.5, 77
            q = bm.verts.new((x + 1, x * slope + 3, 0))
            bm.edges.new((v0, v1))
            bm.edges.new((p, q))
            bmesh.update_edit_mesh(obj.data)
            where = (p.co.x, p.co.y)
            editor = TrimEditor(obj)
            segments = editor.segments()
            key, a, b = segments[0]
            plan = trim_logic.plan_trim(a, b, segments, 0.95, own_key=key)
            # P a few micro-units off the diagonal can miss it; the rest must hit.
            if plan.bounds:
                trimmed += 1
                editor.trim(trim_logic.TrimTarget(key, a, b, plan))
                near = _verts_near(obj, *where)
                assert len(near) == 1 and near[0][1:] == (0.5, 77) and near[0][0][:2] == where, near
            editor.restore()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)
    assert trimmed > 40, trimmed


def test_a_weld_target_keeps_its_data_when_several_new_vertices_join_it():
    # Two crossing edges cross the trimmed one at the vertex T of a third edge: new vertices
    # from the split and the crossing edges, and the existing T, which keeps its fillet.
    segments = [((0, 0), (6, 0)), ((3, -1), (3, 1)), ((3, 0), (4, 2)), ((3, 0), (2, 3))]
    with addon():
        obj = _edit_edges(segments, radii={(3, 0): (0.6, 5)})
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 5, 0) == 'RUNNING_MODAL'
        near = _verts_near(obj, 3, 0)
        assert len(near) == 1 and near[0][1:] == (0.6, 5)
        c.abort()


def test_an_existing_vertex_at_a_bound_that_is_not_a_hit_is_welded():
    # D is collinear with the trimmed edge and ends exactly where X crosses it. Parallel edges are
    # not hits, so only the point's neighbourhood tells the weld that V sits there (W4).
    segments = [((0, 0), (8, 0)), ((2, -1), (2, 1)), ((-5, 0), (2, 0))]
    with addon():
        obj = _edit_edges(segments, radii={(2, 0): (0.6, 5)})
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 5, 0) == 'RUNNING_MODAL'
        near = _verts_near(obj, 2, 0, tol=1e-6)
        assert len(near) == 1 and near[0][1:] == (0.6, 5)
        verts, edges = _geometry(obj)
        assert edges == sorted([((2, -1), (2, 0)), ((2, 0), (2, 1)), ((0, 0), (2, 0)), ((-5, 0), (2, 0))])
        c.abort()


def test_a_bound_with_no_existing_vertex_still_welds_the_new_ones_to_one():
    segments = [((0, 0), (6, 0)), ((2, -1), (2, 1)), ((4, -1), (4, 1))]
    with addon():
        obj = _edit_edges(segments)
        c, *_ = _run(obj, TRIM)
        assert _lmb(c, 3, 0) == 'RUNNING_MODAL'
        for x in (2, 4):
            assert len(_verts_near(obj, x, 0, tol=1e-6)) == 1
        assert len(_geometry(obj)[0]) == 8
        c.abort()


# -- exact restore (QA round 1: W3) --------------------------------------------------------------

def _rich_sketch():
    """Two overlapping rectangles carrying everything a hand-rolled restore would lose: an extra
    vertex attribute, edge attributes (crease, a custom one), seams, a vertex group with weights,
    a hidden edge next to a selected vertex and select history."""
    obj = _edit_sketch([A_RECT, B_RECT])
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.layers.float.new("my_vertex_attr")
    bm.edges.layers.float.new("crease_edge")
    bm.edges.layers.int.new("my_edge_attr")
    bm.verts.layers.deform.verify()
    bmesh.update_edit_mesh(obj.data)
    # Layers invalidate element references: fetch everything again.
    bm = bmesh.from_edit_mesh(obj.data)
    obj.vertex_groups.new(name="Group")
    attr = bm.verts.layers.float["my_vertex_attr"]
    crease = bm.edges.layers.float["crease_edge"]
    custom = bm.edges.layers.int["my_edge_attr"]
    deform = bm.verts.layers.deform.active
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    for i, v in enumerate(bm.verts):
        v[attr] = 1.5 * i
        v[deform][0] = 0.1 * i
        v[rad], v[ids] = (0.25, 100 + i) if i % 3 == 0 else (0.0, 0)
    for i, e in enumerate(bm.edges):
        e[crease] = 0.125 * i
        e[custom] = 40 + i
        e.seam = i % 3 == 1
        e.smooth = i % 2 == 0
    # The corner (0, 2) of A is next to the hidden edge but stays selected.
    corner = next(v for v in bm.verts if (round(v.co.x), round(v.co.y)) == (0, 2))
    hidden = next(e for e in corner.link_edges if e.other_vert(corner).co.y < 1.0)
    hidden.hide_set(True)
    corner.select_set(True)
    bm.select_history.add(corner)
    bmesh.update_edit_mesh(obj.data)
    return obj


def _rich_state(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.index_update()
    layers = {
        "vf": sorted(bm.verts.layers.float.keys()), "vi": sorted(bm.verts.layers.int.keys()),
        "ef": sorted(bm.edges.layers.float.keys()), "ei": sorted(bm.edges.layers.int.keys()),
        "deform": bm.verts.layers.deform.active is not None,
    }
    attr = bm.verts.layers.float["my_vertex_attr"]
    deform = bm.verts.layers.deform.active
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    crease = bm.edges.layers.float["crease_edge"]
    custom = bm.edges.layers.int["my_edge_attr"]
    verts = [(tuple(v.co), v[attr], dict(v[deform]), v[rad], v[ids], v.select, v.hide) for v in bm.verts]
    edges = [(e.verts[0].index, e.verts[1].index, e[crease], e[custom], e.seam, e.smooth, e.select, e.hide)
             for e in bm.edges]
    history = [(type(h).__name__, h.index) for h in bm.select_history]
    return layers, verts, edges, history


def test_esc_restores_attributes_seams_vertex_groups_selection_and_hidden_edges_exactly():
    with addon():
        obj = _rich_sketch()
        before = _rich_state(obj)
        # The fixture must actually hold what the test claims to preserve.
        assert any(v[5] and not v[6] for v in before[1]) and any(e[7] for e in before[2]) and before[3]
        assert any(e[4] for e in before[2]) and any(v[2] for v in before[1])
        c, headers, reports = _run(obj, TRIM)
        for x, y in ((1.5, 1), (1, 1.5)):
            assert _lmb(c, x, y) == 'RUNNING_MODAL'
        assert c.count == 2 and _rich_state(obj) != before
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert _rich_state(obj) == before
        assert not trim._live


def test_extend_esc_restores_edge_attributes_exactly_too():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((0, 3), (1, 3)), ((3, -1), (3, 4))])
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.layers.float.new("crease_edge")
        bmesh.update_edit_mesh(obj.data)
        bm = bmesh.from_edit_mesh(obj.data)
        crease = bm.edges.layers.float["crease_edge"]
        for i, e in enumerate(bm.edges):
            e[crease] = 0.25 * (i + 1)
            e.seam = i == 1
        bmesh.update_edit_mesh(obj.data)

        def state():
            b = bmesh.from_edit_mesh(obj.data)
            b.verts.index_update()
            cr = b.edges.layers.float["crease_edge"]
            return ([tuple(v.co) for v in b.verts],
                    [(e.verts[0].index, e.verts[1].index, e[cr], e.seam) for e in b.edges])

        before = state()
        c, *_ = _run(obj, EXTEND)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL' and c.count == 1 and state() != before
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert state() == before


def test_esc_restores_shape_keys_too_and_leaves_no_key_behind():
    for active in (0, 1):
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.shape_key_add(name="Basis")
            key = obj.shape_key_add(name="Moved")
            for i, point in enumerate(key.data):
                point.co.z += 0.5 * i
            obj.active_shape_key_index = active
            bpy.ops.object.mode_set(mode='EDIT')
            keys = set(bpy.data.shape_keys)

            def state():
                b = bmesh.from_edit_mesh(obj.data)
                layers = [b.verts.layers.shape[name] for name in ("Basis", "Moved")]
                return [(tuple(v.co), *(tuple(v[layer]) for layer in layers)) for v in b.verts]

            before = state()
            c, *_ = _run(obj, TRIM)
            assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL' and state() != before, active
            assert c.handle(_event('ESC')) == 'CANCELLED'
            assert state() == before, active
            assert set(bpy.data.shape_keys) == keys, active


def test_no_snapshot_mesh_is_left_behind_on_any_exit_path():
    with addon():
        for how in ('finish', 'enter', 'esc', 'edit_mode_left', 'close_all', 'abort', 'no_edit'):
            obj = _edit_sketch([A_RECT, B_RECT])
            meshes = set(bpy.data.meshes)
            c, headers, reports = _run(obj, TRIM)
            if how != 'no_edit':
                assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL'
                # The snapshot is a temporary mesh datablock, exactly one while the run lives.
                assert len(set(bpy.data.meshes) - meshes) == 1, how
            if how == 'finish':
                assert c.handle(_event('RIGHTMOUSE')) == 'FINISHED'
            elif how == 'enter':
                assert c.handle(_event('RET')) == 'FINISHED'
            elif how == 'esc':
                assert c.handle(_event('ESC')) == 'CANCELLED'
            elif how == 'edit_mode_left':
                bpy.ops.object.mode_set(mode='OBJECT')
                assert c.handle(_event('MOUSEMOVE', 'NOTHING')) == 'CANCELLED'
            elif how == 'close_all':
                trim.close_all()
            elif how == 'abort':
                c.abort()
            else:
                assert c.handle(_event('RET')) == 'CANCELLED'
            assert c.closed and set(bpy.data.meshes) == meshes, how
            if how != 'edit_mode_left':
                bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_operator_error_and_unregister_paths_leave_no_snapshot_mesh():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        meshes = set(bpy.data.meshes)
        shell, ctx, headers, handlers, restore = _shell(obj)
        try:
            assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
            assert trim.SKETCH_OT_trim.modal(shell, ctx, _event('LEFTMOUSE', at=(1.5, 1))) == {'RUNNING_MODAL'}
            assert len(set(bpy.data.meshes) - meshes) == 1

            def broken(target):
                raise ZeroDivisionError("boom")

            shell._controller.editor.trim = broken
            with redirect_stdout(io.StringIO()):
                assert trim.SKETCH_OT_trim.modal(shell, ctx, _event('LEFTMOUSE', at=(1, 1.5))) == {'CANCELLED'}
            assert set(bpy.data.meshes) == meshes
            assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
            trim.SKETCH_OT_trim.modal(shell, ctx, _event('LEFTMOUSE', at=(1.5, 1)))
            assert len(set(bpy.data.meshes) - meshes) == 1
        finally:
            restore()
        # Unregistering winds the run down.
        trim.close_all()
        assert set(bpy.data.meshes) == meshes


# -- QA round 2 ------------------------------------------------------------------------------------

def _invert_probe(obj, mode):
    """Select vertex 0 only, run Select > Invert in the given select mode, return what is selected."""
    bpy.context.tool_settings.mesh_select_mode = (mode == 'VERT', mode == 'EDGE', mode == 'FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.verts[0].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='INVERT')
    bm = bmesh.from_edit_mesh(obj.data)
    return sum(v.select for v in bm.verts), sum(e.select for e in bm.edges)


def test_esc_keeps_the_edit_bmesh_select_mode_and_select_invert_still_works():
    for mode in ({'VERT'}, {'EDGE'}, {'FACE'}, {'VERT', 'EDGE'}, {'VERT', 'FACE'}, {'EDGE', 'FACE'},
                 {'VERT', 'EDGE', 'FACE'}):
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            bm = bmesh.from_edit_mesh(obj.data)
            bm.select_mode = mode
            bmesh.update_edit_mesh(obj.data)
            c, headers, reports = _run(obj, TRIM)
            assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL' and c.editor.dirty
            assert c.handle(_event('ESC')) == 'CANCELLED'
            assert bmesh.from_edit_mesh(obj.data).select_mode == mode, mode
            if len(mode) == 1:
                kind = next(iter(mode))
                restored = _invert_probe(obj, kind)
                control = _edit_sketch([A_RECT, B_RECT])
                assert _invert_probe(control, kind) == restored, mode


def test_extend_esc_keeps_the_select_mode_too():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((0, 3), (1, 3)), ((3, -1), (3, 4))])
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {'EDGE'}
        bmesh.update_edit_mesh(obj.data)
        c, *_ = _run(obj, EXTEND)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL' and c.count == 1
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert bmesh.from_edit_mesh(obj.data).select_mode == {'EDGE'}


def test_an_outliner_purge_during_the_run_keeps_the_snapshot_and_esc_still_restores():
    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        meshes = set(bpy.data.meshes)
        before = _geometry(obj)
        c, headers, reports = _run(obj, TRIM)
        assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL' and _geometry(obj) != before
        assert len(set(bpy.data.meshes) - meshes) == 1
        bpy.data.orphans_purge(do_recursive=True)
        assert len(set(bpy.data.meshes) - meshes) == 1
        assert c.handle(_event('ESC')) == 'CANCELLED'
        assert _geometry(obj) == before
        assert set(bpy.data.meshes) == meshes


def test_abort_and_finish_survive_an_area_that_is_already_gone():
    # A closed area does not raise ReferenceError, its struct just dangles: the modal must not call
    # into it at all, so the stand-in counts every call made after the area was closed.
    for how in ('esc', 'enter', 'error'):
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            meshes = set(bpy.data.meshes)
            before = _geometry(obj)
            shell, ctx, headers, handlers, restore = _shell(obj)
            closed, late = [], []
            area = ctx.area
            area.header_text_set = lambda text: late.append(text) if closed else headers.append(text)
            area.tag_redraw = lambda: late.append("redraw") if closed else None
            try:
                assert trim.SKETCH_OT_trim.invoke(shell, ctx, _event('LEFTMOUSE')) == {'RUNNING_MODAL'}
                trim.SKETCH_OT_trim.modal(shell, ctx, _event('LEFTMOUSE', at=(1.5, 1)))
                closed.append(True)
                modal_common.find_area = lambda pointer: None
                if how == 'esc':
                    result = trim.SKETCH_OT_trim.modal(shell, ctx, _event('ESC'))
                    assert result == {'CANCELLED'} and _geometry(obj) == before
                elif how == 'enter':
                    assert trim.SKETCH_OT_trim.modal(shell, ctx, _event('RET')) == {'FINISHED'}
                else:
                    shell._controller._projector_factory = lambda: 1 / 0
                    with redirect_stdout(io.StringIO()):
                        result = trim.SKETCH_OT_trim.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING'))
                    assert result == {'CANCELLED'} and shell.reports[-1][0] == {'ERROR'}
            finally:
                restore()
            assert shell._controller.closed and set(bpy.data.meshes) == meshes and not trim._live, how
            assert not late, (how, late)


# -- QA round 3 fixes ----------------------------------------------------------------------------

def _saved_mesh_names(path):
    with bpy.data.libraries.load(path) as (source, _target):
        return list(source.meshes)


def _snapshot_meshes():
    return [m for m in bpy.data.meshes if m.name.startswith(trim.SNAPSHOT_NAME)]


def test_a_save_mid_run_does_not_write_the_snapshot_and_esc_still_restores():
    import shutil
    import tempfile

    from sketch_tools import handlers

    folder = tempfile.mkdtemp(prefix="sketch_tools_save_")
    try:
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            before = _geometry(obj)
            c, headers, reports = _run(obj, TRIM)
            assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL' and c.editor.dirty
            backup = c.editor._backup
            assert backup.use_fake_user and backup.name == trim.SNAPSHOT_NAME

            # Control: without the handler the snapshot goes into the file.
            bpy.app.handlers.save_pre.remove(handlers._on_save_pre)
            try:
                bare = os.path.join(folder, "bare.blend")
                assert bpy.ops.wm.save_as_mainfile(filepath=bare, copy=True) == {'FINISHED'}
            finally:
                bpy.app.handlers.save_pre.append(handlers._on_save_pre)
            assert trim.SNAPSHOT_NAME in _saved_mesh_names(bare)
            assert backup.use_fake_user

            saved = os.path.join(folder, "mid_run.blend")
            assert bpy.ops.wm.save_as_mainfile(filepath=saved, copy=True) == {'FINISHED'}
            assert trim.SNAPSHOT_NAME not in _saved_mesh_names(saved)
            # The live run is unharmed: the fake user is back and Esc restores.
            assert c.editor._backup is backup and backup.use_fake_user
            assert c.handle(_event('ESC')) == 'CANCELLED'
            assert _geometry(obj) == before
            assert not _snapshot_meshes()
            # With no run live the save handlers have nothing to do.
            assert bpy.ops.wm.save_as_mainfile(filepath=saved, copy=True) == {'FINISHED'}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_a_failed_save_gives_the_snapshot_its_fake_user_back():
    from sketch_tools import handlers

    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        c, headers, reports = _run(obj, TRIM)
        assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL'
        backup = c.editor._backup
        handlers._on_save_pre()
        assert not backup.use_fake_user
        handlers._on_save_done()
        assert backup.use_fake_user
        c.abort()
        # Both are harmless once the snapshot is gone.
        handlers._on_save_pre()
        handlers._on_save_done()


def _orphan_snapshot(name=trim.SNAPSHOT_NAME):
    mesh = bpy.data.meshes.new(name)
    mesh[trim.SNAPSHOT_TAG] = True
    mesh.use_fake_user = True
    return mesh


def test_loading_a_file_with_an_orphan_snapshot_removes_it():
    import shutil
    import tempfile

    folder = tempfile.mkdtemp(prefix="sketch_tools_load_")
    try:
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            bpy.ops.object.mode_set(mode='OBJECT')
            name = obj.name
            _orphan_snapshot()
            path = os.path.join(folder, "orphan.blend")
            assert bpy.ops.wm.save_as_mainfile(filepath=path, copy=True) == {'FINISHED'}
            assert trim.SNAPSHOT_NAME in _saved_mesh_names(path)
            assert bpy.ops.wm.open_mainfile(filepath=path) == {'FINISHED'}
            assert not _snapshot_meshes()
            assert bpy.data.objects.get(name) is not None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_undo_and_redo_remove_an_orphan_snapshot_that_came_back_with_the_step():
    with addon():
        _orphan_snapshot()
        bpy.ops.ed.undo_push(message="with orphan")
        bpy.data.objects.new("Marker", None)
        bpy.ops.ed.undo_push(message="marker")
        assert bpy.ops.ed.undo() == {'FINISHED'}
        assert not _snapshot_meshes()
        _orphan_snapshot()
        assert bpy.ops.ed.redo() == {'FINISHED'}
        assert not _snapshot_meshes()


def test_orphan_snapshot_cleanup_only_touches_unused_snapshot_meshes_and_never_a_live_run():
    from sketch_tools import handlers

    with addon():
        obj = _edit_sketch([A_RECT, B_RECT])
        bpy.ops.object.mode_set(mode='OBJECT')
        orphan = _orphan_snapshot()
        suffixed = _orphan_snapshot()
        assert suffixed.name == trim.SNAPSHOT_NAME + ".001"
        # None of these may go: other names and a snapshot-named mesh an object uses.
        other = _orphan_snapshot("my_" + trim.SNAPSHOT_NAME)
        similar = _orphan_snapshot(trim.SNAPSHOT_NAME + "_keep")
        used = _orphan_snapshot()
        user = bpy.data.objects.new("UsesBackup", used)
        bpy.context.collection.objects.link(user)
        # Nothing holds this one at all, so it is as unused as an orphan.
        bare = bpy.data.meshes.new(trim.SNAPSHOT_NAME)
        bare[trim.SNAPSHOT_TAG] = True
        kept = {other.name, similar.name, used.name}
        unused = {orphan.name, suffixed.name, bare.name}

        token = object()
        draw_tool._live.add(token)
        try:
            handlers._on_load_post()
            assert unused | kept <= {m.name for m in bpy.data.meshes}
        finally:
            draw_tool._live.discard(token)
        handlers._on_load_post()
        left = {m.name for m in bpy.data.meshes}
        assert kept <= left
        assert not unused & left
        assert obj.name in bpy.data.objects and user.data is used


def test_a_user_mesh_with_the_snapshot_name_and_a_fake_user_is_left_alone():
    from sketch_tools import handlers

    with addon():
        mine = bpy.data.meshes.new(trim.SNAPSHOT_NAME)
        mine.use_fake_user = True
        mine_suffixed = bpy.data.meshes.new(trim.SNAPSHOT_NAME)
        mine_suffixed.use_fake_user = True
        ours = _orphan_snapshot()
        assert not mine.get(trim.SNAPSHOT_TAG) and ours.get(trim.SNAPSHOT_TAG)
        names = {mine.name, mine_suffixed.name}
        ours_name = ours.name
        handlers._on_load_post()
        handlers._on_undo_redo_post()
        left = {m.name for m in bpy.data.meshes}
        assert names <= left and ours_name not in left
        assert mine.use_fake_user and mine_suffixed.use_fake_user


def test_a_real_run_tags_its_snapshot_and_the_tag_is_written_to_disk():
    import shutil
    import tempfile

    folder = tempfile.mkdtemp(prefix="sketch_tools_tag_")
    try:
        with addon():
            obj = _edit_sketch([A_RECT, B_RECT])
            c, headers, reports = _run(obj, TRIM)
            assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL'
            backup = c.editor._backup
            assert backup.get(trim.SNAPSHOT_TAG)
            # What an autosave or quit.blend would write, which skips the save handlers.
            path = os.path.join(folder, "tagged.blend")
            bpy.data.libraries.write(path, {backup}, fake_user=True)
            with bpy.data.libraries.load(path) as (source, target):
                target.meshes = list(source.meshes)
            loaded = target.meshes[0]
            assert loaded.get(trim.SNAPSHOT_TAG), list(loaded.keys())
            bpy.data.meshes.remove(loaded)
            c.abort()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_the_handlers_register_without_save_post_fail_on_a_blender_that_lacks_it():
    from sketch_tools import handlers

    names = ("depsgraph_update_post", "load_post", "undo_post", "redo_post", "save_pre", "save_post")
    fake = types.SimpleNamespace(app=types.SimpleNamespace(handlers=types.SimpleNamespace(**{n: [] for n in names})))
    real = handlers.bpy
    handlers.bpy = fake
    try:
        assert len(handlers._handler_pairs()) == 6
        handlers.register()
        handlers.register()
        lists = vars(fake.app.handlers)
        assert all(len(lists[n]) == 1 for n in names), lists
        handlers.unregister()
        assert not any(lists.values())
    finally:
        handlers.bpy = real
    assert len(handlers._handler_pairs()) == 7


def test_the_save_and_cleanup_handlers_register_and_unregister_symmetrically():
    from sketch_tools import handlers

    names = ("save_pre", "save_post", "save_post_fail", "load_post", "undo_post", "redo_post")

    def ours():
        return {name: [h for h in getattr(bpy.app.handlers, name) if h.__module__ == handlers.__name__]
                for name in names}

    with addon():
        assert {name: len(found) for name, found in ours().items()} == dict.fromkeys(names, 1)
        handlers.register()
        assert {name: len(found) for name, found in ours().items()} == dict.fromkeys(names, 1)
        assert all(hasattr(h, "_bpy_persistent") for found in ours().values() for h in found)
    assert not any(ours().values())


def test_losing_the_view_ends_a_trim_run_and_keeps_its_edits():
    # The area was retyped, so the projector is gone. Like leaving edit mode (C16) the edits stay.
    with addon():
        for etype, value in (('MOUSEMOVE', 'NOTHING'), ('A', 'PRESS'), ('TIMER', 'NOTHING')):
            obj = _edit_sketch([A_RECT, B_RECT])
            meshes = set(bpy.data.meshes)
            before = _geometry(obj)
            c, headers, reports = _run(obj, TRIM)
            assert _lmb(c, 1.5, 1) == 'RUNNING_MODAL'
            edited = _geometry(obj)
            assert edited != before and len(set(bpy.data.meshes) - meshes) == 1
            c._projector_factory = lambda: None
            assert c.handle(_event(etype, value)) == 'FINISHED', etype
            assert c.closed and headers[-1] is None and not trim._live, etype
            assert not modal_common.any_sketch_modal_live()
            assert _geometry(obj) == edited and set(bpy.data.meshes) == meshes
            assert c.snapshot() is None
            # Another tool can start again.
            again, *_ = _run(obj, TRIM)
            again.abort()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_losing_the_view_ends_an_extend_run_and_a_run_without_edits_cancels():
    with addon():
        obj = _edit_edges([((0, 0), (1, 0)), ((0, 3), (1, 3)), ((3, -1), (3, 4))])
        meshes = set(bpy.data.meshes)
        c, headers, reports = _run(obj, EXTEND)
        assert _lmb(c, 1, 0) == 'RUNNING_MODAL' and c.count == 1
        edited = _geometry(obj)
        c._projector_factory = lambda: None
        assert c.handle(_event('MOUSEMOVE', 'NOTHING')) == 'FINISHED'
        assert c.closed and headers[-1] is None and not trim._live
        assert _geometry(obj) == edited and set(bpy.data.meshes) == meshes
        # Nothing was edited: no undo step.
        idle, headers, reports = _run(obj, EXTEND)
        idle._projector_factory = lambda: None
        assert idle.handle(_event('MOUSEMOVE', 'NOTHING')) == 'CANCELLED'
        assert idle.closed and headers[-1] is None and not trim._live and _geometry(obj) == edited
