"""The modal fillet (sketch.fillet): pure logic in core/fillet_logic.py, then the controller and
operator driven with fake events and a fake view, as the draw tool tests do."""

import io
import types
from contextlib import redirect_stdout

import bmesh
import bpy
from mathutils import Matrix, Vector

from sketch_tools.core import fillet_logic as fl
from sketch_tools.core import resolve
from sketch_tools.core.fillet_logic import FilletSession, find_corners
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.ops import fillet, modal_common
from sketch_tools.tests.test_addon import (
    _WINDOW_OFFSET, _TopView, _edit_sketch, _event, addon, fake_area, install_fake_area, _REAL_FIND_AREA,
)

_REAL_VIEW_FACTORY = fillet._view_factory

RECT = [(0, 0), (4, 0), (4, 1), (0, 1)]
SQUARE4 = [(0, 0), (4, 0), (4, 4), (0, 4)]


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def _corners(coords, edges, selected, radii=None):
    return find_corners(coords, radii or [0.0] * len(coords), edges, selected)


def _ring(n):
    return [(i, (i + 1) % n) for i in range(n)]


def test_every_selected_corner_of_a_loop_is_filletable():
    corners, skipped = _corners(SQUARE4, _ring(4), [0, 1, 2, 3])
    assert [c.index for c in corners] == [0, 1, 2, 3] and skipped == 0
    c = corners[1]
    assert tuple(c.prev) == (0.0, 0.0) and tuple(c.p) == (4.0, 0.0) and tuple(c.next) == (4.0, 4.0)


def test_unselected_vertices_are_not_corners_and_not_counted():
    corners, skipped = _corners(SQUARE4, _ring(4), [2])
    assert [c.index for c in corners] == [2] and skipped == 0


def test_endpoints_junctions_and_straight_corners_are_skipped_and_counted():
    # 0-1-2-3 open chain with a straight middle (1 is collinear), 2 is a junction with a branch 2-4.
    coords = [(0, 0), (1, 0), (2, 0), (2, 2), (3, -1)]
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    corners, skipped = _corners(coords, edges, [0, 1, 2, 3, 4])
    assert corners == [] and skipped == 5
    # An open chain's interior corner is fine.
    corners, skipped = _corners([(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 2)], [0, 1, 2])
    assert [c.index for c in corners] == [1] and skipped == 2


def test_degenerate_and_coincident_vertices_are_skipped():
    # A spike that doubles back on itself (theta below 1 degree).
    coords = [(0, 0), (4, 0), (0, 0.0001)]
    corners, skipped = _corners(coords, [(0, 1), (1, 2)], [1])
    assert corners == [] and skipped == 1
    # Two loose endpoints at the same spot: they are chain ends, so they are skipped, and the
    # corners between them stay fine.
    coords = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    corners, skipped = _corners(coords, edges, [0, 1, 2, 3, 4])
    assert [c.index for c in corners] == [1, 2, 3] and skipped == 2
    # Two squares that share a corner position: the resolver welds the coincident corners into a
    # junction (C9), so neither is a corner of its own.
    coords = SQUARE4 + [(4, 0), (8, 0), (8, 4), (4, 4)]
    edges = _ring(4) + [(4 + a, 4 + b) for a, b in _ring(4)]
    corners, skipped = _corners(coords, edges, [1, 4, 5])
    assert [c.index for c in corners] == [5] and skipped == 2


def test_the_start_radius_is_the_shared_radius_else_zero():
    same, _ = _corners(SQUARE4, _ring(4), [0, 1], radii=[0.5, 0.5, 0.0, 0.0])
    assert fl.initial_radius(same) == 0.5
    mixed, _ = _corners(SQUARE4, _ring(4), [0, 1], radii=[0.5, 0.25, 0.0, 0.0])
    assert fl.initial_radius(mixed) == 0.0
    assert fl.initial_radius([]) == 0.0


def _session(unit_scale=1.0, grid=0.1, radii=None, selected=(0, 1, 2, 3), coords=SQUARE4):
    corners, skipped = _corners(coords, _ring(len(coords)), list(selected), radii)
    return FilletSession(corners, skipped, 8, grid, unit_scale)


def _drag(session, *xs, upp=0.01, ctrl=False):
    for x in xs:
        session.handle(("MOVE", x, upp, ctrl))


def test_dragging_changes_the_radius_by_the_pixels_moved_times_the_view_scale():
    s = _session()
    _drag(s, 100.0, 150.0)
    assert abs(s.radius - 0.5) < 1e-9
    _drag(s, 250.0)
    assert abs(s.radius - 1.5) < 1e-9
    # Moving left shrinks it again.
    _drag(s, 200.0)
    assert abs(s.radius - 1.0) < 1e-9


def test_the_same_mouse_move_scales_with_the_view_distance():
    near, far = _session(), _session()
    _drag(near, 0.0, 100.0, upp=0.01)
    _drag(far, 0.0, 100.0, upp=0.04)
    assert abs(near.radius - 1.0) < 1e-9 and abs(far.radius - 4.0) < 1e-9


def test_the_radius_never_goes_negative_and_has_no_dead_zone():
    s = _session()
    _drag(s, 100.0, 0.0, -200.0)
    assert s.radius == 0.0
    # Reversing at once grows it again instead of first paying back the overshoot.
    _drag(s, -100.0)
    assert abs(s.radius - 1.0) < 1e-9


def test_a_drag_starts_from_the_existing_shared_radius():
    s = _session(radii=[0.5] * 4)
    assert s.radius == 0.5
    _drag(s, 0.0, 10.0)
    assert abs(s.radius - 0.6) < 1e-9


def test_leaving_the_viewport_does_not_turn_the_gap_into_a_drag():
    s = _session()
    _drag(s, 100.0)
    s.handle(("LEAVE",))
    _drag(s, 900.0, 910.0)
    assert abs(s.radius - 0.1) < 1e-9


def test_a_view_without_a_scale_ignores_the_move_but_tracks_the_pointer():
    s = _session()
    s.handle(("MOVE", 100.0, None, False))
    _drag(s, 200.0)
    assert abs(s.radius - 1.0) < 1e-9


def test_ctrl_rounds_the_radius_to_the_grid_increment():
    s = _session(grid=0.25)
    _drag(s, 0.0, 37.0, ctrl=True)
    assert abs(s.radius - 0.25) < 1e-9
    _drag(s, 50.0, ctrl=True)
    assert abs(s.radius - 0.5) < 1e-9
    # Releasing Ctrl shows the unrounded drag again.
    s.handle(("MOVE", 50.0, 0.01, False))
    assert abs(s.radius - 0.5) < 1e-9
    s.handle(("MOVE", 54.0, 0.01, False))
    assert abs(s.radius - 0.54) < 1e-9
    s.handle(("MOVE", 54.0, 0.01, True))
    assert abs(s.radius - 0.5) < 1e-9


def test_typed_digits_set_the_radius_directly():
    s = _session()
    _drag(s, 0.0, 300.0)
    for ch in "1.5":
        s.handle(("TYPE", ch))
    assert s.radius == 1.5 and s.numeric.is_typing()
    # A minus is not allowed for a radius.
    s.handle(("TYPE", "-"))
    assert s.radius == 1.5
    s.handle(("KEY", "BACKSPACE"))
    s.handle(("KEY", "BACKSPACE"))
    assert s.radius == 1.0
    s.handle(("KEY", "BACKSPACE"))
    # Nothing typed: the drag value is back.
    assert abs(s.radius - 3.0) < 1e-9 and not s.numeric.is_typing()


def test_a_lone_decimal_point_falls_back_to_the_drag():
    s = _session()
    _drag(s, 0.0, 100.0)
    s.handle(("TYPE", "."))
    assert abs(s.radius - 1.0) < 1e-9


def test_typed_radius_is_in_the_scene_base_unit_with_the_unit_scale_applied():
    s = _session(unit_scale=0.01)
    for ch in "2":
        s.handle(("TYPE", ch))
    assert abs(s.radius - 200.0) < 1e-6
    assert s.preview().lines[0] == "Radius: 2_"
    s.handle(("KEY", "BACKSPACE"))
    s.handle(("MOVE", 0.0, 1.0, False))
    s.handle(("MOVE", 3.0, 1.0, False))
    assert abs(s.radius - 3.0) < 1e-9
    # The readout shows local lengths times the unit scale.
    assert s.preview().lines[0] == "Radius: 0.030"


def test_typing_ignores_ctrl_rounding():
    s = _session(grid=0.5)
    for ch in "0.3":
        s.handle(("TYPE", ch))
    s.handle(("MOVE", 0.0, 0.01, True))
    assert abs(s.radius - 0.3) < 1e-9


def test_enter_and_lmb_commit_while_esc_and_rmb_cancel():
    for key, cancelled in (("ENTER", False), ("LMB", False), ("ESC", True), ("RMB", True)):
        s = _session()
        _drag(s, 0.0, 100.0)
        s.handle(("KEY", key))
        assert s.done and s.cancelled == cancelled, key
        before = s.radius
        _drag(s, 500.0)
        s.handle(("TYPE", "9"))
        assert s.radius == before, key


def _resolved_with(coords, radius, segments=8):
    bm = bmesh.new()
    rad = bm.verts.layers.float.new(FILLET_RADIUS_ATTR)
    verts = [bm.verts.new((x, y, 0.0)) for x, y in coords]
    for v in verts:
        v[rad] = radius
    for i in range(len(verts)):
        bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
    try:
        return resolve.resolve_bmesh(bm, segments)
    finally:
        bm.free()


def test_preview_clamping_matches_the_resolver():
    for radius in (0.3, 0.5, 0.6, 2.0):
        resolved = _resolved_with(RECT, radius)
        corners, _ = _corners(RECT, _ring(4), [0, 1, 2, 3])
        arcs = fl.arc_preview(corners, radius, 8)
        clamped = sum(1 for a in arcs if a.clamped)
        warned = any("clamped" in w for w in resolved.warnings)
        assert (clamped > 0) == warned, radius
        if warned:
            assert f"{clamped} corner" in resolved.warnings[0]
        # The same arc points. The resolver may reverse the winding and drops the duplicate where
        # two arcs meet on a short side, so compare as sets of rounded points.
        key = lambda p: (round(p.x, 5), round(p.y, 5))
        assert {key(p) for a in arcs for p in a.points} == {key(p) for p in resolved.chains[0].points}, radius


def _agreement(coords, corner, radius):
    """(resolver warned, preview clamped count) for one filleted corner of a closed loop."""
    bm = bmesh.new()
    rad = bm.verts.layers.float.new(FILLET_RADIUS_ATTR)
    verts = [bm.verts.new((x, y, 0.0)) for x, y in coords]
    verts[corner][rad] = radius
    for i in range(len(verts)):
        bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
    try:
        warned = any("clamped" in w for w in resolve.resolve_bmesh(bm, 8).warnings)
        # The modal reads the same float32 coordinates and radius back from the mesh.
        stored = [(v.co.x, v.co.y) for v in verts]
        radius = verts[corner][rad]
    finally:
        bm.free()
    corners, _ = _corners(stored, _ring(len(stored)), [corner])
    return warned, sum(1 for a in fl.arc_preview(corners, radius, 8) if a.clamped)


def test_the_modal_count_follows_the_resolver_at_the_exact_fit_and_over_it():
    top_cut = [(0, 0), (4, 0), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]
    assert _agreement(top_cut, 2, 0.5)== (False, 0)
    assert _agreement(top_cut, 2, 0.505)== (True, 1)
    # Far from the origin the float32 error grows; the rule still agrees.
    far = [(x + 1000.3, y + 1000.7) for x, y in top_cut]
    assert _agreement(far, 2, 0.5)== (False, 0)
    assert _agreement(far, 2, 0.505)== (True, 1)


def test_the_readout_has_no_clamped_line_at_the_exact_fit():
    top_cut = [(0, 0), (4, 0), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]
    corners, _ = _corners(top_cut, _ring(7), [2])
    s = FilletSession(corners, 0, 8, 0.1)
    s.handle(("TYPE", "0"))
    s.handle(("TYPE", "."))
    s.handle(("TYPE", "5"))
    assert not any("Clamped" in line for line in s.preview().lines)
    s.handle(("KEY", "BACKSPACE"))
    s.handle(("TYPE", "6"))
    assert "Clamped at 1 of 1 corners" in s.preview().lines


def test_clamped_corners_are_flagged_individually():
    # Only the short side of a 4 x 1 rectangle limits a radius of 0.6.
    corners, _ = _corners(RECT, _ring(4), [0, 1, 2, 3])
    arcs = fl.arc_preview(corners, 0.6, 8)
    assert [a.clamped for a in arcs] == [True] * 4
    # A 4 x 4 square can take it.
    corners, _ = _corners(SQUARE4, _ring(4), [0, 1, 2, 3])
    assert not any(a.clamped for a in fl.arc_preview(corners, 0.6, 8))


def test_preview_arcs_use_the_session_segment_count_and_zero_radius_draws_nothing():
    s = _session()
    arcs = s.preview().arcs
    assert all(len(a.points) == 1 for a in arcs)
    _drag(s, 0.0, 100.0)
    assert all(len(a.points) == 9 for a in s.preview().arcs)
    assert "Clamped" not in " ".join(s.preview().lines)
    _drag(s, 800.0)
    assert any("Clamped at 4 of 4 corners" == line for line in s.preview().lines)


def test_header_text_reports_radius_corners_skipped_and_keys():
    s = _session(selected=(0, 1))
    s.skipped = 2
    text = s.header_text()
    assert "Fillet radius: 0.000" in text and "2 corners" in text and "2 skipped" in text
    assert "Enter" in text and "Esc" in text and "Ctrl" in text
    assert text.isascii()


# ---------------------------------------------------------------------------
# Controller and operator
# ---------------------------------------------------------------------------

class _View(_TopView):
    """The draw tests' orthographic top view (100 px per unit) with a window offset and a pixel
    size of 0.01 units, so one unit of `at` moves the radius by one unit."""

    def __init__(self, upp=0.01, **kwargs):
        super().__init__(**kwargs)
        self.upp = upp

    def to_region(self, window):
        return (window[0] - _WINDOW_OFFSET[0], window[1] - _WINDOW_OFFSET[1])

    def units_per_pixel(self, xy):
        return self.upp


def _select(obj, indices):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.index_update()
    for v in bm.verts:
        v.select_set(v.index in indices)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def _add_chain(obj, coords, closed=False):
    """Extra loose chain in the edit mesh; returns the new vertex indices."""
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [bm.verts.new((x, y, 0.0)) for x, y in coords]
    for a, b in zip(verts, verts[1:] + ([verts[0]] if closed else [])):
        bm.edges.new((a, b))
    bm.verts.index_update()
    bmesh.update_edit_mesh(obj.data)
    return [v.index for v in verts]


def _attributes(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    bm.verts.index_update()
    return [(v[rad], v[ids]) for v in bm.verts]


def _set_attributes(obj, values):
    bm = bmesh.from_edit_mesh(obj.data)
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    for index, (r, i) in values.items():
        bm.verts[index][rad] = r
        bm.verts[index][ids] = i
    bmesh.update_edit_mesh(obj.data)


def _controller(obj, view=None, grid=0.1, unit_scale=1.0, segments=8):
    from sketch_tools.ops.fillet import FilletController, FilletEditor

    view = view or _View()
    headers, redraws, reports = [], [0], []
    editor = FilletEditor(obj)
    corners, skipped = editor.collect()
    session = FilletSession(corners, skipped, segments, grid, unit_scale)
    controller = FilletController(
        editor, session, lambda: view, headers.append,
        lambda: redraws.__setitem__(0, redraws[0] + 1), lambda level, text: reports.append((level, text)),
        region_pointer=7)
    return controller, headers, redraws, reports


def _move(controller, x, y=0.0, **mods):
    return controller.handle(_event('MOUSEMOVE', 'NOTHING', (x, y), **mods))


def _press(controller, etype, x=0.0, y=0.0, **mods):
    return controller.handle(_event(etype, 'PRESS', (x, y), **mods))


def _drag_to(controller, radius):
    """Pointer path that leaves the radius at `radius` (1 unit of `at` = 1 radius unit)."""
    assert _move(controller, 0.0) == 'RUNNING_MODAL'
    assert _move(controller, radius) == 'RUNNING_MODAL'


def test_commit_writes_radius_and_fresh_ids_to_the_selected_filletable_corners_only():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [0, 2])
        c, headers, *_ = _controller(obj)
        _drag_to(c, 0.5)
        assert _press(c, 'LEFTMOUSE', 0.5) == 'FINISHED'
        values = _attributes(obj)
        assert abs(values[0][0] - 0.5) < 1e-6 and abs(values[2][0] - 0.5) < 1e-6
        assert values[0][1] != 0 and values[2][1] != 0 and values[0][1] != values[2][1]
        assert values[1] == (0.0, 0) and values[3] == (0.0, 0)
        assert c.closed and headers[-1] is None


def test_a_vertex_deleted_during_the_run_cancels_the_commit_without_any_write():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1, 3])
        before = _attributes(obj)
        c, headers, _, reports = _controller(obj)
        _drag_to(c, 0.5)
        # A panel operator removes one of the corners while the modal runs.
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.remove(bm.verts[3])
        bmesh.update_edit_mesh(obj.data)
        assert _press(c, 'LEFTMOUSE', 0.5) == 'CANCELLED'
        assert c.closed and headers[-1] is None
        assert reports == [('WARNING', "Fillet cancelled: a selected vertex was deleted during the run")]
        # The surviving corner was not written either.
        assert _attributes(obj) == before[:3]


def test_a_radius_of_zero_clears_the_id_too():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _set_attributes(obj, {0: (0.5, 123), 1: (0.5, 456)})
        _select(obj, [0, 1])
        c, *_ = _controller(obj)
        # Both share 0.5, so the drag starts there; drag back to nothing.
        _drag_to(c, 0.0)
        _move(c, -1.0)
        assert _press(c, 'RET') == 'FINISHED'
        values = _attributes(obj)
        assert values[0] == (0.0, 0) and values[1] == (0.0, 0)


def test_endpoints_junctions_and_straight_corners_keep_their_values():
    with addon():
        obj = _edit_sketch([SQUARE4])
        # A line hanging off corner 0 makes it a junction; the far end is an endpoint.
        tail = _add_chain(obj, [(-3, 0), (-6, 0)])
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.new((bm.verts[0], bm.verts[tail[0]]))
        # A straight run: 3 collinear points, the middle one is not a corner.
        straight = _add_chain(obj, [(10, 0), (11, 0), (12, 0)])
        bm.verts.index_update()
        bmesh.update_edit_mesh(obj.data)
        marks = {0: (0.25, 777), tail[1]: (0.25, 778), straight[1]: (0.25, 779)}
        _set_attributes(obj, marks)
        everything = [v.index for v in bm.verts]
        _select(obj, everything)
        c, headers, *_ = _controller(obj)
        corners = c.session.corners
        assert sorted(x.index for x in corners) == [1, 2, 3]
        assert c.session.skipped == len(everything) - 3
        assert "skipped" in headers[-1]
        _drag_to(c, 0.5)
        _press(c, 'LEFTMOUSE', 0.5)
        values = _attributes(obj)
        for index, mark in marks.items():
            assert values[index] == mark, index
        for index in (1, 2, 3):
            assert abs(values[index][0] - 0.5) < 1e-6 and values[index][1] > 0
        assert all(values[i] == (0.0, 0) for i in tail[:1] + [straight[0], straight[2]])


def test_esc_and_rmb_restore_radii_and_ids_exactly():
    for key in ('ESC', 'RIGHTMOUSE'):
        with addon():
            obj = _edit_sketch([SQUARE4])
            _set_attributes(obj, {1: (0.3, 4242), 3: (0.1, 99)})
            _select(obj, [0, 1, 2, 3])
            before = _attributes(obj)
            c, headers, *_ = _controller(obj)
            _drag_to(c, 1.2)
            assert _press(c, 'ONE') == 'RUNNING_MODAL'
            assert _press(c, key) == 'CANCELLED', key
            assert _attributes(obj) == before and c.closed and headers[-1] is None


def test_the_output_rebuilds_after_commit_with_the_expected_arc_points():
    with addon():
        obj = _edit_sketch([SQUARE4])
        obj.sketch_tools.output_mode = 'EXTRUDE'
        _select(obj, [0])
        c, *_ = _controller(obj)
        out = obj.sketch_tools.output_object
        flat = len(out.data.vertices)
        assert flat == 8
        _drag_to(c, 0.5)
        # Still previewing: the output is not touched before the commit.
        assert len(out.data.vertices) == 8
        assert _press(c, 'LEFTMOUSE', 0.5) == 'FINISHED'
        # One corner became nine arc points: 4 - 1 + 9 points, top and bottom.
        assert len(out.data.vertices) == 2 * 12
        assert not obj.sketch_tools.warnings


def test_a_clamped_commit_shows_the_resolver_warning_for_the_same_corners():
    with addon():
        obj = _edit_sketch([RECT])
        obj.sketch_tools.output_mode = 'EXTRUDE'
        _select(obj, [0, 1, 2, 3])
        c, *_ = _controller(obj)
        _drag_to(c, 0.6)
        assert all(a.clamped for a in c.session.preview().arcs)
        _press(c, 'LEFTMOUSE', 0.6)
        assert "at 4 corners" in obj.sketch_tools.warnings and "clamped" in obj.sketch_tools.warnings


def test_typed_radius_commits_in_the_scene_base_unit():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj, unit_scale=0.5)
        _press(c, 'ONE')
        _press(c, 'PERIOD')
        _press(c, 'FIVE')
        assert _press(c, 'RET') == 'FINISHED'
        assert abs(_attributes(obj)[1][0] - 3.0) < 1e-6


def test_enter_commits_the_dragged_radius_and_ctrl_rounds_it():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj, grid=0.5)
        assert _move(c, 0.0) == 'RUNNING_MODAL'
        assert _move(c, 0.8) == 'RUNNING_MODAL'
        assert _press(c, 'RET', 0.8) == 'FINISHED'
        assert abs(_attributes(obj)[1][0] - 0.8) < 1e-6
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj, grid=0.5)
        _move(c, 0.0)
        _move(c, 0.8, ctrl=True)
        assert abs(c.session.radius - 1.0) < 1e-9
        assert _press(c, 'LEFTMOUSE', 0.8, ctrl=True) == 'FINISHED'
        assert abs(_attributes(obj)[1][0] - 1.0) < 1e-6


def test_one_commit_only_and_later_events_do_nothing():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj)
        _drag_to(c, 0.5)
        assert _press(c, 'LEFTMOUSE', 0.5) == 'FINISHED'
        written = _attributes(obj)
        # A late event on a closed controller: the edit bmesh is alive, the session is done.
        c.session.handle(("TYPE", "7"))
        assert c.session.radius == 0.5 and _attributes(obj) == written


def test_the_controller_passes_events_outside_the_region_through_and_keeps_going():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        view = _View(size=(1000.0, 800.0), overlays=((_WINDOW_OFFSET[0] + 900.0, 0.0, 100.0, 2000.0),))
        c, *_ = _controller(obj, view=view)
        before = _attributes(obj)
        # Pointer over the sidebar rectangle (px x 950), then beyond the region (px x 1100).
        assert _move(c, 4.5) == 'PASS_THROUGH'
        assert _move(c, 6.0) == 'PASS_THROUGH'
        assert c.mouse is None
        # A press in the sidebar goes on to Blender with its release; nothing is committed.
        assert _press(c, 'LEFTMOUSE', 4.5) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (4.5, 0))) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'CLICK', (4.5, 0))) == 'PASS_THROUGH'
        assert _press(c, 'RIGHTMOUSE', 6.0) == 'PASS_THROUGH'
        assert c.handle(_event('RIGHTMOUSE', 'RELEASE', (6.0, 0))) == 'PASS_THROUGH'
        assert not c.closed and _attributes(obj) == before
        # Alt+LMB navigates.
        assert c.handle(_event('LEFTMOUSE', 'PRESS', (1, 0), alt=True)) == 'PASS_THROUGH'
        assert c.handle(_event('LEFTMOUSE', 'RELEASE', (1, 0))) == 'PASS_THROUGH'
        # Back inside: the next press commits.
        _move(c, 0.0)
        assert _press(c, 'LEFTMOUSE', 1.0) == 'FINISHED'


def test_navigation_numpad_and_system_events_pass_through_but_numpad_digits_type_once_typing():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj)
        for etype in ('WHEELUPMOUSE', 'MIDDLEMOUSE', 'NUMPAD_4', 'TIMER', 'NDOF_MOTION', 'WINDOW_DEACTIVATE'):
            assert c.handle(_event(etype)) == 'PASS_THROUGH', etype
        assert _press(c, 'NUMPAD_5') == 'PASS_THROUGH'
        assert _press(c, 'TWO') == 'RUNNING_MODAL'
        assert _press(c, 'NUMPAD_5') == 'RUNNING_MODAL'
        assert c.session.numeric.texts[0] == "25"
        assert _press(c, 'MINUS') == 'RUNNING_MODAL' and c.session.numeric.texts[0] == "25"
        assert _press(c, 'BACK_SPACE') == 'RUNNING_MODAL' and c.session.numeric.texts[0] == "2"
        # Unrelated keys are swallowed so they cannot edit the sketch mid-run.
        assert _press(c, 'X') == 'RUNNING_MODAL'
        c.abort()


def test_leaving_edit_mode_or_losing_the_view_cancels_without_writing():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, headers, *_ = _controller(obj)
        _drag_to(c, 0.5)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert _move(c, 0.7) == 'CANCELLED' and c.closed and headers[-1] is None
        assert all(v.value == 0.0 for v in obj.data.attributes[FILLET_RADIUS_ATTR].data)
        bpy.ops.object.mode_set(mode='EDIT')
        c2, headers2, *_ = _controller(obj)
        c2._view_factory = lambda: None
        assert _move(c2, 0.7) == 'CANCELLED' and c2.closed and headers2[-1] is None


def test_the_view_factory_looks_the_area_up_live_and_ends_the_run_when_it_is_gone():
    from sketch_tools.ops import fillet

    with addon():
        obj = _edit_sketch([SQUARE4])
        region = types.SimpleNamespace(type='WINDOW', data=object(), x=0, y=0, width=1200, height=900)
        space = types.SimpleNamespace(show_gizmo=False, show_gizmo_navigate=False, region_quadviews=[])
        area = types.SimpleNamespace(type='VIEW_3D', regions=[region], spaces=types.SimpleNamespace(active=space))
        found = {7001: area}
        modal_common.find_area = found.get
        try:
            factory = fillet._view_factory(obj, 7001)
            assert factory() is not None
            # The area is closed: the lookup no longer finds it, whatever the old struct says.
            found.clear()
            assert factory() is None
            # A closing area still listed but with no active space left.
            found[7001] = area
            area.spaces.active = None
            assert factory() is None
            area.spaces.active = space
            assert factory() is not None
            # Retyped, or without a WINDOW region.
            area.type = 'PROPERTIES'
            assert factory() is None
            area.type = 'VIEW_3D'
            area.regions = [types.SimpleNamespace(type='HEADER', data=None, x=0, y=0, width=10, height=10)]
            assert factory() is None
            area.regions = [region]
            region.data = None
            assert factory() is None
        finally:
            modal_common.find_area = _REAL_FIND_AREA


def test_closing_the_area_ends_a_fillet_run_through_the_normal_abort_not_the_error_path():
    from sketch_tools.ops import fillet
    from sketch_tools.ops.fillet import SKETCH_OT_fillet
    from sketch_tools.ui import overlay

    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        before = _attributes(obj)
        real_factory = fillet._view_factory
        shell, ctx, headers, handlers, restore = _fillet_shell(obj)
        try:
            # The real factory on the fake area; its active space has no gizmo switches to read.
            fillet._view_factory = real_factory
            space = ctx.area.spaces.active
            space.show_gizmo = space.show_gizmo_navigate = False
            outside = _event('LEFTMOUSE', 'PRESS', (5000, 5000))
            assert SKETCH_OT_fillet.invoke(shell, ctx, outside) == {'RUNNING_MODAL'}
            assert len(fillet._live) == 1 and len(overlay._live) == 1 and headers[-1] is not None
            shown = len(headers)
            # The area is closed: nothing may call into it any more, header and redraw included.
            modal_common.find_area = lambda pointer: None
            assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (0, 0))) == {'CANCELLED'}
            assert not shell.reports and len(headers) == shown
            assert not fillet._live and not overlay._live
        finally:
            restore()
        assert _attributes(obj) == before
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.objects.remove(obj)


def test_the_snapshot_feeds_the_overlay_and_is_empty_once_closed():
    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        c, *_ = _controller(obj)
        assert c.snapshot() is not None
        _drag_to(c, 0.5)
        preview, mouse, to_screen = c.snapshot()
        assert len(preview.arcs[0].points) == 9 and mouse == (550.0, 400.0)
        assert to_screen((1.0, 0.0)) == (600.0, 400.0)
        c.abort()
        assert c.snapshot() is None


# -- operator ---------------------------------------------------------------

def _fillet_shell(obj, view=None, quad=False):
    from sketch_tools.ops import fillet

    headers, modal_handlers = [], []
    region = types.SimpleNamespace(
        type='WINDOW', data=object(), as_pointer=lambda: 4242,
        x=int(_WINDOW_OFFSET[0]), y=int(_WINDOW_OFFSET[1]), width=1200, height=900)
    space = types.SimpleNamespace(region_quadviews=[object()] * (4 if quad else 0))
    area = fake_area(
        type='VIEW_3D', regions=[region], spaces=types.SimpleNamespace(active=space),
        header_text_set=headers.append, tag_redraw=lambda: None)
    context = types.SimpleNamespace(
        active_object=obj, area=area, window_manager=types.SimpleNamespace(modal_handler_add=modal_handlers.append),
        scene=bpy.context.scene, preferences=bpy.context.preferences)
    shell = types.SimpleNamespace(reports=[])
    shell.report = lambda kind, text: shell.reports.append((kind, text))
    fillet._view_factory = lambda o, pointer: (lambda: view or _View())
    restore_area = install_fake_area(area)

    def restore():
        fillet._view_factory = _REAL_VIEW_FACTORY
        restore_area()
    return shell, context, headers, modal_handlers, restore


def test_operator_paths_clean_up_after_themselves():
    from sketch_tools.ops import fillet
    from sketch_tools.ops.fillet import SKETCH_OT_fillet
    from sketch_tools.ui import overlay

    with addon():
        for how in ('finish', 'esc', 'rmb', 'error', 'cancel_callback'):
            obj = _edit_sketch([SQUARE4])
            _select(obj, [1, 3])
            before = _attributes(obj)
            shell, ctx, headers, handlers, restore = _fillet_shell(obj)
            try:
                invoke_event = _event('LEFTMOUSE', 'PRESS', (8, 8))
                assert SKETCH_OT_fillet.invoke(shell, ctx, invoke_event) == {'RUNNING_MODAL'}
                assert handlers == [shell] and len(overlay._live) == 1 and headers[-1] is not None
                assert len(fillet._live) == 1
                assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (0, 0))) == {'RUNNING_MODAL'}
                assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (0.5, 0))) == {'RUNNING_MODAL'}
                if how == 'finish':
                    assert SKETCH_OT_fillet.modal(shell, ctx, _event('RET')) == {'FINISHED'}
                elif how == 'esc':
                    assert SKETCH_OT_fillet.modal(shell, ctx, _event('ESC')) == {'CANCELLED'}
                elif how == 'rmb':
                    assert SKETCH_OT_fillet.modal(shell, ctx, _event('RIGHTMOUSE')) == {'CANCELLED'}
                elif how == 'error':
                    def broken(event):
                        raise ZeroDivisionError("boom")
                    shell._controller.session.handle = broken
                    with redirect_stdout(io.StringIO()):
                        assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (1, 0))) == {'CANCELLED'}
                    assert shell.reports and shell.reports[0][0] == {'ERROR'}
                else:
                    SKETCH_OT_fillet.cancel(shell, ctx)
            finally:
                restore()
            assert not overlay._live and not fillet._live and headers[-1] is None, how
            after = _attributes(obj)
            if how == 'finish':
                assert after[1][0] > 0 and after[3][0] > 0 and after[0] == (0.0, 0)
            else:
                assert after == before, how
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)


def test_operator_starts_from_a_press_outside_the_viewport_and_drags_after_entering_it():
    from sketch_tools.ops.fillet import SKETCH_OT_fillet

    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        view = _View(size=(1000.0, 800.0))
        shell, ctx, headers, handlers, restore = _fillet_shell(obj, view=view)
        try:
            # The panel button press is over the sidebar, outside the region.
            assert SKETCH_OT_fillet.invoke(shell, ctx, _event('LEFTMOUSE', 'PRESS', (7, 0))) == {'RUNNING_MODAL'}
            assert SKETCH_OT_fillet.modal(shell, ctx, _event('LEFTMOUSE', 'RELEASE', (7, 0))) == {'RUNNING_MODAL'}
            assert shell._controller.session.radius == 0.0
            assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (2, 0))) == {'RUNNING_MODAL'}
            assert SKETCH_OT_fillet.modal(shell, ctx, _event('MOUSEMOVE', 'NOTHING', (2.5, 0))) == {'RUNNING_MODAL'}
            assert abs(shell._controller.session.radius - 0.5) < 1e-9
            assert SKETCH_OT_fillet.modal(shell, ctx, _event('LEFTMOUSE', 'PRESS', (2.5, 0))) == {'FINISHED'}
        finally:
            restore()
        assert abs(_attributes(obj)[1][0] - 0.5) < 1e-6


def test_operator_with_nothing_to_fillet_or_no_usable_view_does_not_start():
    from sketch_tools.ops import fillet
    from sketch_tools.ops.fillet import SKETCH_OT_fillet
    from sketch_tools.ui import overlay

    with addon():
        obj = _edit_sketch([SQUARE4])
        for selection, quad in (([], False), ([1], True)):
            _select(obj, selection)
            before = _attributes(obj)
            shell, ctx, headers, handlers, restore = _fillet_shell(obj, quad=quad)
            try:
                assert SKETCH_OT_fillet.invoke(shell, ctx, _event('LEFTMOUSE', 'PRESS', (0, 0))) == {'CANCELLED'}
            finally:
                restore()
            assert shell.reports and not handlers and not headers
            assert not overlay._live and not fillet._live and _attributes(obj) == before
        # Only a chain endpoint selected: still nothing to round.
        ends = _add_chain(obj, [(10, 0), (12, 0)])
        _select(obj, ends)
        shell, ctx, headers, handlers, restore = _fillet_shell(obj)
        try:
            assert SKETCH_OT_fillet.invoke(shell, ctx, _event('LEFTMOUSE', 'PRESS', (0, 0))) == {'CANCELLED'}
        finally:
            restore()
        assert shell.reports[0][0] == {'WARNING'} and not handlers


def test_a_second_invoke_is_cancelled_with_a_warning_while_a_run_is_live_and_unregister_winds_it_down():
    from sketch_tools.ops import fillet
    from sketch_tools.ops.fillet import SKETCH_OT_fillet
    from sketch_tools.ui import overlay

    with addon():
        obj = _edit_sketch([SQUARE4])
        _select(obj, [1])
        first, ctx, headers, handlers, restore = _fillet_shell(obj)
        second, ctx2, headers2, handlers2, restore2 = _fillet_shell(obj)
        try:
            assert SKETCH_OT_fillet.invoke(first, ctx, _event('LEFTMOUSE', 'PRESS', (0, 0))) == {'RUNNING_MODAL'}
            # CANCELLED, not PASS_THROUGH: Blender discards the reports of a passed-through invoke.
            assert SKETCH_OT_fillet.invoke(second, ctx2, _event('LEFTMOUSE', 'PRESS', (0, 0))) == {'CANCELLED'}
            assert second.reports == [({'WARNING'}, "Finish the running sketch tool first")]
            assert not hasattr(second, "_controller") and not handlers2 and not headers2
            assert len(fillet._live) == 1 and len(overlay._live) == 1
            fillet.close_all()
            assert not fillet._live and not overlay._live and headers[-1] is None
        finally:
            restore()
            restore2()


def test_operator_poll_needs_edit_mode_in_a_3d_view_on_a_sketch():
    from sketch_tools.ops.fillet import SKETCH_OT_fillet

    with addon():
        obj = _edit_sketch([SQUARE4])
        plain = bpy.data.objects.new("Plain", bpy.data.meshes.new("Plain"))
        bpy.context.collection.objects.link(plain)

        def ctx(mode, area_type, active):
            area = None if area_type is None else types.SimpleNamespace(type=area_type)
            return types.SimpleNamespace(mode=mode, area=area, active_object=active)

        poll = SKETCH_OT_fillet.poll
        assert poll(ctx('EDIT_MESH', 'VIEW_3D', obj))
        assert not poll(ctx('EDIT_MESH', 'VIEW_3D', plain))
        assert not poll(ctx('EDIT_MESH', 'VIEW_3D', None))
        assert not poll(ctx('OBJECT', 'VIEW_3D', obj))
        assert not poll(ctx('EDIT_MESH', 'PROPERTIES', obj))
        assert not poll(ctx('EDIT_MESH', None, obj))
        assert SKETCH_OT_fillet.bl_idname == "sketch.fillet" and 'UNDO' in SKETCH_OT_fillet.bl_options


def test_operator_class_registers_and_unregisters():
    from sketch_tools.ops.fillet import SKETCH_OT_fillet

    with addon():
        already = hasattr(bpy.ops.sketch, "fillet")
        if not already:
            bpy.utils.register_class(SKETCH_OT_fillet)
        try:
            assert hasattr(bpy.ops.sketch, "fillet")
        finally:
            if not already:
                bpy.utils.unregister_class(SKETCH_OT_fillet)


def test_the_view_scale_per_pixel_follows_the_distance_to_the_sketch():
    from sketch_tools.ops import fillet

    # Perspective stand-in: the camera sits at z = 10 and a pixel spans 0.01 units at z = 0,
    # proportionally less nearer the camera.
    def to_region(region, rv3d, world):
        return Vector((400.0, 300.0))

    def to_location(region, rv3d, coord, depth):
        scale = 0.01 * (10.0 - depth.z) / 10.0
        return Vector((coord[0] * scale, coord[1] * scale, depth.z))

    real = fillet.view3d_utils
    fillet.view3d_utils = types.SimpleNamespace(
        location_3d_to_region_2d=to_region, region_2d_to_location_3d=to_location)
    try:
        region = types.SimpleNamespace(x=30, y=20, width=1200, height=900)
        far = fillet.FilletView(region, object(), Matrix.Identity(4), ())
        near = fillet.FilletView(region, object(), Matrix.Translation((0.0, 0.0, 5.0)), ())
        assert abs(far.units_per_pixel((1.0, 2.0)) - 0.01) < 1e-6
        assert abs(near.units_per_pixel((1.0, 2.0)) - 0.005) < 1e-6
        # The size is in sketch local units: a scaled sketch object spans more of them per pixel.
        scaled = fillet.FilletView(region, object(), Matrix.Scale(0.5, 4), ())
        assert abs(scaled.units_per_pixel((0.0, 0.0)) - 0.02) < 1e-6
        assert far.to_region((130.0, 220.0)) == (100.0, 200.0)
        # Behind the camera.
        fillet.view3d_utils = types.SimpleNamespace(
            location_3d_to_region_2d=lambda *a: None, region_2d_to_location_3d=to_location)
        assert far.units_per_pixel((0.0, 0.0)) is None
    finally:
        fillet.view3d_utils = real


# -- overlay ----------------------------------------------------------------

def test_overlay_geometry_separates_clamped_arcs_and_marks_every_corner():
    from sketch_tools.ui import fillet_overlay

    corners, _ = _corners(RECT, _ring(4), [0, 1, 2, 3])
    to_screen = _View().to_screen
    free = fillet_overlay.screen_geometry(fl.FilletPreview(fl.arc_preview(corners, 0.3, 8), []), to_screen)
    assert len(free["arcs"]) == 4 and not free["clamped"] and len(free["corners"]) == 4
    assert all(len(strip) == 9 for strip in free["arcs"])
    tight = fillet_overlay.screen_geometry(fl.FilletPreview(fl.arc_preview(corners, 0.6, 8), []), to_screen)
    assert not tight["arcs"] and len(tight["clamped"]) == 4
    flat = fillet_overlay.screen_geometry(fl.FilletPreview(fl.arc_preview(corners, 0.0, 8), []), to_screen)
    assert not flat["arcs"] and not flat["clamped"] and len(flat["corners"]) == 4
    # A corner behind the camera drops its strip.
    hidden = fillet_overlay.screen_geometry(
        fl.FilletPreview(fl.arc_preview(corners, 0.3, 8), []), lambda xy: None)
    assert hidden == {"arcs": [], "clamped": [], "corners": []}


def test_fillet_overlay_draws_only_in_its_region_and_restores_the_blend_state():
    from sketch_tools.ui import fillet_overlay, overlay

    s = _session(coords=RECT)
    _drag(s, 0.0, 60.0)
    state = (s.preview(), (10.0, 10.0), _View().to_screen)
    calls = {"state": 0, "blend": [], "colors": [], "text": 0}

    class Shader:
        def uniform_float(self, name, value):
            if name == "color":
                calls["colors"].append(tuple(value))

    fake_gpu = types.SimpleNamespace(
        shader=types.SimpleNamespace(from_builtin=lambda name: Shader()),
        state=types.SimpleNamespace(
            blend_get=lambda: 'ADDITIVE', blend_set=lambda mode: calls["blend"].append(mode)))
    fake_blf = types.SimpleNamespace(
        size=lambda *a: None, position=lambda *a: None, color=lambda *a: None,
        draw=lambda *a: calls.__setitem__("text", calls["text"] + 1))
    batch = types.SimpleNamespace(draw=lambda shader: None)
    current = types.SimpleNamespace(as_pointer=lambda: 7, width=800, height=600)
    real = (fillet_overlay.bpy, fillet_overlay.gpu, overlay.blf, overlay.batch_for_shader)

    def get_state():
        calls["state"] += 1
        return state

    fillet_overlay.bpy = types.SimpleNamespace(context=types.SimpleNamespace(region=current))
    fillet_overlay.gpu, overlay.blf = fake_gpu, fake_blf
    overlay.batch_for_shader = lambda shader, kind, data: batch
    try:
        other = fillet_overlay.FilletOverlay(get_state, region_pointer=8)
        other._draw()
        assert calls["state"] == 0 and not calls["blend"] and not calls["text"]
        mine = fillet_overlay.FilletOverlay(get_state, region_pointer=7)
        mine._draw()
        assert calls["state"] == 1 and not mine._failed and calls["text"] > 0
        assert calls["blend"] == ['ALPHA', 'ADDITIVE']
        # Corners, then the clamped arcs (all four clamp at 0.6 on a 4 x 1 rectangle) in their own colour.
        assert fillet_overlay.COLOR_CLAMPED in calls["colors"]
        assert fillet_overlay.COLOR_ARC not in calls["colors"] or calls["colors"].index(
            fillet_overlay.COLOR_ARC) < calls["colors"].index(fillet_overlay.COLOR_CLAMPED)
        # Pointer outside the viewport: no readout.
        calls["text"] = 0
        state = (s.preview(), None, _View().to_screen)
        mine._draw()
        assert calls["text"] == 0
        # Handler add and remove go through the shared registry.
        mine.add()
        assert mine in overlay._live
        mine.remove()
        assert mine not in overlay._live
    finally:
        fillet_overlay.bpy, fillet_overlay.gpu, overlay.blf, overlay.batch_for_shader = real
