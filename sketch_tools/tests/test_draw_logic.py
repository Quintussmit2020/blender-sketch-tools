"""Sketch draw tool decision logic (core/draw_logic.py), driven with abstract events."""

import math

from mathutils import Matrix, Vector

from sketch_tools.core import draw_logic as dl
from sketch_tools.core.draw_logic import (
    ARC_3PT, ARC_CENTER, CIRCLE, POLYLINE, RECTANGLE, DrawSession, Mods, Snap,
)


def close(a, b, eps=1e-6):
    return abs(a - b) <= eps


def pclose(p, q, eps=1e-6):
    return close(p[0], q[0], eps) and close(p[1], q[1], eps)


def run(session, *events):
    out = []
    for event in events:
        out.extend(session.handle(event))
    return out


def new_session(mode=POLYLINE, **kwargs):
    return DrawSession(mode, grid=kwargs.pop("grid", 0.1), **kwargs)


def verts(commands):
    return [(c[1], c[2]) for c in commands if c[0] == "ADD_VERT"]


def edges(commands):
    return [(c[1], c[2]) for c in commands if c[0] == "ADD_EDGE"]


def typed(text):
    return [("TYPE", ch) for ch in text]


def poly_with(*points):
    """Session with the given points clicked in order."""
    s = new_session()
    cmds = run(s, *[("CLICK", p) for p in points])
    return s, cmds


# -- helpers ------------------------------------------------------------------

def test_effective_unit_scale_ignores_scale_length_without_a_unit_system():
    assert dl.effective_unit_scale('METRIC', 0.01) == 0.01
    assert dl.effective_unit_scale('IMPERIAL', 2.0) == 2.0
    assert dl.effective_unit_scale('NONE', 0.01) == 1.0
    assert dl.effective_unit_scale('NONE', 2.0) == 1.0
    assert dl.effective_unit_scale('METRIC', 1.0) == 1.0
    assert dl.effective_unit_scale('METRIC', 0.0) == 1.0


def test_lock_angle_snaps_to_45_degrees_from_the_reference():
    ref = math.radians(30.0)
    for raw_deg, want_deg in ((40, 30), (60, 75), (100, 120), (-10, -15), (200, 210), (118, 120)):
        got = dl.lock_angle(math.radians(raw_deg), ref)
        assert close(got, math.radians(want_deg), 1e-9), (raw_deg, math.degrees(got))


def test_turn_angle_is_signed_and_wrapped():
    assert close(dl.turn_angle_deg(math.radians(90), 0.0), 90.0, 1e-9)
    assert close(dl.turn_angle_deg(math.radians(-90), 0.0), -90.0, 1e-9)
    assert close(dl.turn_angle_deg(math.radians(350), 0.0), -10.0, 1e-9)
    assert close(dl.turn_angle_deg(math.pi, 0.0), 180.0, 1e-9)


def test_nearest_snap_respects_the_pixel_threshold():
    cands = [(3, (1.0, 1.0), (100.0, 100.0)), (4, (2.0, 2.0), (130.0, 100.0)), (5, (9.0, 9.0), None)]
    hit = dl.nearest_snap((105.0, 100.0), cands, 12)
    assert hit == Snap(3, (1.0, 1.0))
    # Within 12 px of vertex 4 only.
    assert dl.nearest_snap((120.0, 100.0), cands, 12).ref == 4
    assert dl.nearest_snap((115.0, 140.0), cands, 12) is None
    # Exactly on the threshold counts.
    assert dl.nearest_snap((112.0, 100.0), cands, 12).ref == 3
    assert dl.nearest_snap((113.0, 100.0), cands[:1], 12) is None
    # The nearest wins when two are in range.
    assert dl.nearest_snap((118.0, 100.0), cands, 20).ref == 4


def test_circle_points_are_evenly_spaced_on_the_radius():
    pts = dl.circle_points((1.0, 2.0), 3.0, 12)
    assert len(pts) == 12
    assert pclose(pts[0], (4.0, 2.0))
    assert all(close(math.hypot(x - 1.0, y - 2.0), 3.0) for x, y in pts)
    assert len(dl.circle_points((0, 0), 1.0, 1)) == 3


def test_ray_to_plane_local():
    inv = Matrix.Identity(4)
    assert pclose(dl.ray_to_plane_local((1, 2, 5), (0, 0, -1), inv), (1, 2))
    # Edge-on and grazing rays do not place points.
    assert dl.ray_to_plane_local((0, 0, 5), (1, 0, 0), inv) is None
    assert dl.ray_to_plane_local((0, 0, 5), (1, 0, -0.005), inv) is None
    # A plane behind a perspective camera is rejected, an orthographic origin may sit past it.
    assert dl.ray_to_plane_local((0, 0, -5), (0, 0, -1), inv) is None
    assert pclose(dl.ray_to_plane_local((3, 4, -5), (0, 0, -1), inv, allow_behind=True), (3, 4))
    # A tilted, moved plane: local z = 0 is world x + y = 2.
    world = Matrix.Translation((1, 1, 0)) @ Matrix.Rotation(math.radians(90), 4, 'Y')
    p = dl.ray_to_plane_local((5, 3, 0), (-1, 0, 0), world.inverted())
    assert p is not None
    hit_world = world @ Vector((p[0], p[1], 0.0))
    assert close(hit_world.x, 1.0, 1e-5) and close(hit_world.y, 3.0, 1e-5)


# -- numeric input ----------------------------------------------------------------

def test_numeric_input_digits_point_minus_and_backspace():
    n = dl.NumericInput(("A", "B"))
    for ch in "1.2.5":
        n.type(ch)
    assert n.texts[0] == "1.25" and close(n.value(0), 1.25)
    n.type('-')
    assert n.texts[0] == "-1.25"
    n.type('-')
    assert n.texts[0] == "1.25"
    n.backspace()
    assert n.texts[0] == "1.2"
    n.next_field()
    assert n.active == 1 and n.value(1) is None
    n.type('9')
    n.backspace()
    n.backspace()
    assert n.active == 0
    n.next_field()
    n.next_field()
    assert n.active == 0
    n.clear()
    assert not n.is_typing() and not n.has_value()


def test_numeric_input_incomplete_numbers_have_no_value():
    n = dl.NumericInput(("A",))
    n.type('-')
    assert n.is_typing() and n.value(0) is None and not n.has_value()
    n.type('.')
    assert n.value(0) is None
    n.type('5')
    assert close(n.value(0), -0.5)


# -- polyline -------------------------------------------------------------------------

def test_polyline_first_click_adds_nothing_then_chain_grows():
    s = new_session()
    assert run(s, ("CLICK", (0, 0))) == []
    cmds = run(s, ("CLICK", (1, 0)))
    assert verts(cmds) == [(-1, (0, 0)), (-2, (1, 0))]
    assert edges(cmds) == [(-1, -2)]
    cmds = run(s, ("CLICK", (1, 1)))
    assert verts(cmds) == [(-3, (1, 1))] and edges(cmds) == [(-2, -3)]
    assert not s.done


def test_polyline_ignores_zero_length_clicks():
    s, _ = poly_with((0, 0), (1, 0))
    assert run(s, ("CLICK", (1, 0))) == []
    assert run(s, ("CLICK", (1.0000000001, 0))) == []
    assert len(s.points) == 2


def test_rmb_and_enter_end_a_chain_of_two_or_more_points():
    for key in ("RMB", "ENTER", "N"):
        s, _ = poly_with((0, 0), (1, 0), (1, 1))
        assert run(s, ("KEY", key)) == [("END",)], key
        assert s.done and not s.cancelled
        assert run(s, ("CLICK", (5, 5)), ("KEY", "ESC")) == []


def test_ending_after_a_single_click_leaves_nothing():
    for key in ("RMB", "ENTER", "N", "ESC"):
        s = new_session()
        run(s, ("CLICK", (0, 0)))
        assert run(s, ("KEY", key)) == [("CANCEL",)], key
        assert s.cancelled


def test_esc_cancels_the_whole_chain():
    s, cmds = poly_with((0, 0), (1, 0), (1, 1), (0, 1))
    assert len(verts(cmds)) == 4
    assert run(s, ("KEY", "ESC")) == [("CANCEL",)]
    assert s.done and s.cancelled


def test_c_closes_to_the_first_vertex_and_ends():
    s, _ = poly_with((0, 0), (1, 0), (1, 1))
    cmds = run(s, ("KEY", "C"))
    assert cmds == [("ADD_EDGE", -3, -1), ("END",)]
    assert s.done


def test_c_needs_three_points():
    s, _ = poly_with((0, 0), (1, 0))
    assert run(s, ("KEY", "C")) == []
    assert not s.done
    s = new_session()
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("KEY", "C")) == []


def test_n_ends_the_chain_like_enter_and_a_new_session_starts_a_new_chain():
    s, _ = poly_with((0, 0), (1, 0))
    assert run(s, ("KEY", "N")) == [("END",)]
    s2 = new_session()
    assert run(s2, ("CLICK", (5, 5))) == []
    assert verts(run(s2, ("CLICK", (6, 5)))) == [(-1, (5, 5)), (-2, (6, 5))]


# -- angle lock ---------------------------------------------------------------------------

def test_shift_locks_the_first_segment_relative_to_sketch_x():
    s = new_session()
    run(s, ("CLICK", (0, 0)))
    # Cursor at about 40 degrees locks to 45.
    cursor = (math.cos(math.radians(40)) * 2.0, math.sin(math.radians(40)) * 2.0)
    cmds = run(s, ("MOVE", cursor, Mods(shift=True)), ("CLICK", cursor, Mods(shift=True)))
    (_, p0), (_, p1) = verts(cmds)
    assert pclose(p1, (math.sqrt(2.0), math.sqrt(2.0)))
    # Roughly vertical locks to 90.
    s = new_session()
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("CLICK", (0.2, 3.0), Mods(shift=True)))
    assert pclose(verts(cmds)[1][1], (0.0, math.hypot(0.2, 3.0)))


def test_shift_locks_relative_to_the_previous_segment():
    # Previous segment points up (90 degrees): lock directions are 90 + k * 45.
    s, _ = poly_with((0, 0), (0, 1))
    # Cursor right and slightly up from (0, 1): raw angle ~ 8 degrees, nearest lock is 0 (turn -90).
    cmds = run(s, ("CLICK", (2.0, 1.28), Mods(shift=True)))
    (_, p), = verts(cmds)
    length = math.hypot(2.0, 0.28)
    assert pclose(p, (length, 1.0), 1e-5)
    # Raw angle ~ 60 degrees locks to 45 (turn -45 from the previous 90).
    s, _ = poly_with((0, 0), (0, 1))
    cur = (1.0 * math.cos(math.radians(60)), 1.0 + 1.0 * math.sin(math.radians(60)))
    cmds = run(s, ("CLICK", cur, Mods(shift=True)))
    (_, p), = verts(cmds)
    assert pclose(p, (math.cos(math.radians(45)), 1.0 + math.sin(math.radians(45))), 1e-5)


def test_shift_shows_a_guide_line_through_the_anchor():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (2.0, 0.3), Mods(shift=True)))
    guide = s.preview().guide
    assert guide is not None
    (ax, ay), (bx, by) = guide
    assert close(ay, 0.0) and close(by, 0.0) and ax < 1.0 < bx
    run(s, ("MOVE", (2.0, 0.3), Mods()))
    assert s.preview().guide is None


# -- typed input -----------------------------------------------------------------------------

def test_typed_length_sets_the_segment_length_along_the_cursor_direction():
    s = new_session()
    run(s, ("CLICK", (0, 0)), ("MOVE", (0.0, 5.0)), *typed("2.5"))
    cmds = run(s, ("KEY", "ENTER"))
    assert verts(cmds) == [(-1, (0, 0)), (-2, (0.0, 2.5))] or \
        (len(verts(cmds)) == 2 and pclose(verts(cmds)[1][1], (0.0, 2.5)))
    assert not s.done
    assert not s.numeric.is_typing()


def test_typed_length_then_tab_then_angle_is_relative_to_the_previous_segment():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (3.0, 0.0)), *typed("2"), ("KEY", "TAB"), *typed("90"))
    cmds = run(s, ("KEY", "ENTER"))
    (_, p), = verts(cmds)
    assert pclose(p, (1.0, 2.0))
    # The next angle is relative to the new (upward) segment.
    run(s, *typed("1"), ("KEY", "TAB"), *typed("-90"))
    (_, p), = verts(run(s, ("KEY", "ENTER")))
    assert pclose(p, (2.0, 2.0))


def test_typed_first_segment_angle_is_relative_to_sketch_x():
    s = new_session()
    run(s, ("CLICK", (1, 1)), ("MOVE", (5.0, 1.0)), *typed("2"), ("KEY", "TAB"), *typed("90"))
    cmds = run(s, ("KEY", "ENTER"))
    assert pclose(verts(cmds)[1][1], (1.0, 3.0))


def test_typed_angle_alone_keeps_the_cursor_distance():
    s = new_session()
    run(s, ("CLICK", (0, 0)), ("MOVE", (3.0, 4.0)), ("KEY", "TAB"), *typed("90"))
    cmds = run(s, ("KEY", "ENTER"))
    assert pclose(verts(cmds)[1][1], (0.0, 5.0))


def test_negative_typed_length_runs_backwards():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (3.0, 0.0)), *typed("-2"))
    (_, p), = verts(run(s, ("KEY", "ENTER")))
    assert pclose(p, (-1.0, 0.0))
    # '-' pressed after the digits flips the sign too.
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (3.0, 0.0)), *typed("2-"))
    (_, p), = verts(run(s, ("KEY", "ENTER")))
    assert pclose(p, (-1.0, 0.0))


def test_backspace_edits_the_typed_length():
    s = new_session()
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 0.0)), *typed("25"), ("KEY", "BACKSPACE"))
    cmds = run(s, ("KEY", "ENTER"))
    assert pclose(verts(cmds)[1][1], (2.0, 0.0))
    s = new_session()
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 0.0)), *typed("5"), ("KEY", "BACKSPACE"))
    # Nothing typed any more: Enter ends the chain, which has a single point, so it cancels.
    assert run(s, ("KEY", "ENTER")) == [("CANCEL",)]


def test_enter_commits_when_a_value_is_typed_and_ends_when_not():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (1.0, 1.0)), *typed("1"))
    cmds = run(s, ("KEY", "ENTER"))
    assert len(verts(cmds)) == 1 and not s.done
    assert run(s, ("KEY", "ENTER")) == [("END",)]


def test_enter_with_only_a_minus_typed_ends_the_chain():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (2.0, 0.0)), *typed("-"))
    assert run(s, ("KEY", "ENTER")) == [("END",)]


def test_typed_zero_length_commits_nothing():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (2.0, 0.0)), *typed("0"))
    assert run(s, ("KEY", "ENTER")) == []
    assert not s.done


def test_a_click_commits_the_typed_value_too():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (5.0, 0.0)), *typed("2"))
    (_, p), = verts(run(s, ("CLICK", (5.0, 0.0))))
    assert pclose(p, (3.0, 0.0))


def test_typing_before_the_first_point_is_ignored():
    s = new_session()
    run(s, *typed("12"))
    assert not s.numeric.is_typing()


def test_typing_feedback_shows_the_buffer_and_marks_the_active_field():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (2.0, 0.0)), *typed("12"))
    lines = s.preview().lines
    assert lines[0] == "> Length: 12_" and lines[1].startswith("  Angle: ")
    run(s, ("KEY", "TAB"), *typed("4"))
    lines = s.preview().lines
    assert lines[0].startswith("  Length: 12") and lines[1] == "> Angle: 4_"
    run(s, ("KEY", "BACKSPACE"), ("KEY", "BACKSPACE"))
    assert s.numeric.active == 0


def test_idle_feedback_shows_length_and_turn_angle():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("MOVE", (1.0, 2.0)))
    lines = s.preview().lines
    assert lines == ["Length: 2.000", "Angle: 90.000 deg"]
    (strip,) = s.preview().strips
    assert pclose(strip[0], (1, 0)) and pclose(strip[1], (1.0, 2.0))


# -- snapping ----------------------------------------------------------------------------------------

def test_ctrl_snaps_to_the_grid_with_a_fractional_step():
    s = DrawSession(POLYLINE, grid=0.25)
    run(s, ("CLICK", (0.0, 0.0), Mods(ctrl=True)))
    cmds = run(s, ("CLICK", (0.93, 0.51), Mods(ctrl=True)))
    assert pclose(verts(cmds)[1][1], (1.0, 0.5), 1e-9)


def test_ctrl_with_shift_rounds_the_locked_length_to_the_grid():
    s = new_session(grid=0.5)
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("CLICK", (1.3, 0.1), Mods(shift=True, ctrl=True)))
    assert pclose(verts(cmds)[1][1], (1.5, 0.0), 1e-9)


def test_ctrl_snaps_to_an_existing_vertex_and_reuses_it():
    s = new_session()
    run(s, ("CLICK", (0, 0)))
    snap = Snap(7, (4.0, 3.0))
    cmds = run(s, ("CLICK", (4.05, 3.02), Mods(ctrl=True, snap=snap)))
    assert verts(cmds) == [(-1, (0, 0))]
    assert edges(cmds) == [(-1, 7)]
    assert not s.done
    # Chain continues from the existing vertex.
    cmds = run(s, ("CLICK", (4.0, 6.0)))
    assert edges(cmds) == [(7, -2)]


def test_snap_is_ignored_without_ctrl():
    s = new_session()
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("CLICK", (4.05, 3.02), Mods(ctrl=False, snap=Snap(7, (4.0, 3.0)))))
    assert pclose(verts(cmds)[1][1], (4.05, 3.02))


def test_first_point_can_start_on_an_existing_vertex():
    s = new_session()
    run(s, ("CLICK", (1.0, 1.0), Mods(ctrl=True, snap=Snap(2, (1.0, 1.0)))))
    cmds = run(s, ("CLICK", (3.0, 1.0)))
    assert verts(cmds) == [(-1, (3.0, 1.0))]
    assert edges(cmds) == [(2, -1)]


def test_snapping_back_onto_the_first_vertex_closes_the_loop_without_a_duplicate():
    s, _ = poly_with((0, 0), (2, 0), (2, 2))
    cmds = run(s, ("CLICK", (0.02, 0.01), Mods(ctrl=True, snap=Snap(-1, (0.0, 0.0)))))
    assert verts(cmds) == []
    assert cmds == [("ADD_EDGE", -3, -1), ("END",)]
    assert s.done and not s.cancelled


def test_snapping_back_onto_the_first_vertex_of_a_two_point_chain_does_nothing():
    s, _ = poly_with((0, 0), (2, 0))
    assert run(s, ("CLICK", (0, 0), Mods(ctrl=True, snap=Snap(-1, (0.0, 0.0))))) == []
    assert not s.done


def test_snap_to_the_previous_point_is_ignored():
    s, _ = poly_with((0, 0), (2, 0))
    assert run(s, ("CLICK", (2, 0), Mods(ctrl=True, snap=Snap(-2, (2.0, 0.0))))) == []


def test_preview_reports_the_snap_target():
    s, _ = poly_with((0, 0), (2, 0))
    run(s, ("MOVE", (2.0, 2.0), Mods(ctrl=True, snap=Snap(3, (2.0, 2.0)))))
    assert s.preview().snap_xy == (2.0, 2.0)


# -- rectangle -------------------------------------------------------------------------------------

def rect_ring(commands):
    return [xy for _, xy in verts(commands)]


def test_rectangle_by_two_clicks():
    s = new_session(RECTANGLE)
    assert run(s, ("CLICK", (1, 1))) == []
    cmds = run(s, ("CLICK", (4, 3)))
    assert rect_ring(cmds) == [(1, 1), (4, 1), (4, 3), (1, 3)]
    assert edges(cmds) == [(-1, -2), (-2, -3), (-3, -4), (-4, -1)]
    assert cmds[-1] == ("END",) and s.done


def test_rectangle_by_click_drag_commits_on_release_only_when_dragged():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("RELEASE", (0.01, 0.01), None, False)) == []
    assert not s.done
    cmds = run(s, ("RELEASE", (2, 1), None, True))
    assert rect_ring(cmds) == [(0, 0), (2, 0), (2, 1), (0, 1)] and s.done


def test_rectangle_typed_width_tab_height():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 1.0)), *typed("3"), ("KEY", "TAB"), *typed("2"))
    assert s.preview().lines[:2] == ["  Width: 3", "> Height: 2_"]
    cmds = run(s, ("KEY", "ENTER"))
    assert rect_ring(cmds) == [(0, 0), (3, 0), (3, 2), (0, 2)]
    assert s.done


def test_rectangle_typed_sizes_follow_the_cursor_side_and_minus_flips():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)), ("MOVE", (-1.0, 1.0)), *typed("3"), ("KEY", "TAB"), *typed("2"))
    assert rect_ring(run(s, ("KEY", "ENTER"))) == [(0, 0), (-3, 0), (-3, 2), (0, 2)]
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 1.0)), *typed("-3"))
    ring = rect_ring(run(s, ("KEY", "ENTER")))
    assert ring == [(0, 0), (-3, 0), (-3, 1), (0, 1)]


def test_rectangle_typed_width_keeps_the_cursor_height():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 1.5)), *typed("4"))
    assert rect_ring(run(s, ("CLICK", (1.0, 1.5)))) == [(0, 0), (4, 0), (4, 1.5), (0, 1.5)]


def test_rectangle_degenerate_is_not_committed():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("CLICK", (3, 0))) == []
    assert not s.done


def test_rectangle_enter_without_a_value_cancels():
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("KEY", "ENTER")) == [("CANCEL",)]


def test_rectangle_ctrl_uses_grid_and_vertex_snap_for_the_opposite_corner():
    s = new_session(RECTANGLE, grid=0.5)
    run(s, ("CLICK", (0, 0)))
    assert rect_ring(run(s, ("CLICK", (1.2, 0.8), Mods(ctrl=True)))) == [(0, 0), (1.0, 0), (1.0, 1.0), (0, 1.0)]
    s = new_session(RECTANGLE)
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("CLICK", (2.01, 1.0), Mods(ctrl=True, snap=Snap(9, (2.0, 1.0)))))
    # The snapped corner reuses vertex 9; the other three are new.
    assert verts(cmds) == [(-1, (0, 0)), (-2, (2, 0)), (-3, (0, 1))]
    assert edges(cmds) == [(-1, -2), (-2, 9), (9, -3), (-3, -1)]


# -- circle -------------------------------------------------------------------------------------------

def test_circle_by_click_and_radius_has_the_configured_point_count():
    s = new_session(CIRCLE, circle_segments=8)
    assert run(s, ("CLICK", (1, 1))) == []
    cmds = run(s, ("CLICK", (3, 1)))
    ring = rect_ring(cmds)
    assert len(ring) == 8 and len(edges(cmds)) == 8
    assert all(close(math.hypot(x - 1, y - 1), 2.0) for x, y in ring)
    assert pclose(ring[0], (3, 1))
    refs = [r for r, _ in verts(cmds)]
    assert edges(cmds) == [(refs[i], refs[(i + 1) % 8]) for i in range(8)]
    assert cmds[-1] == ("END",) and s.done


def test_circle_default_segments_and_typed_radius():
    s = new_session(CIRCLE)
    run(s, ("CLICK", (0, 0)), ("MOVE", (5.0, 5.0)), *typed("1.5"))
    assert s.preview().lines == ["> Radius: 1.5_"]
    cmds = run(s, ("KEY", "ENTER"))
    ring = rect_ring(cmds)
    assert len(ring) == dl.DEFAULT_CIRCLE_SEGMENTS == 32
    assert all(close(math.hypot(x, y), 1.5, 1e-9) for x, y in ring)


def test_circle_negative_typed_radius_uses_the_magnitude():
    s = new_session(CIRCLE, circle_segments=4)
    run(s, ("CLICK", (0, 0)), *typed("-2"))
    ring = rect_ring(run(s, ("KEY", "ENTER")))
    assert all(close(math.hypot(x, y), 2.0, 1e-9) for x, y in ring)


def test_circle_zero_radius_and_enter_without_a_value():
    s = new_session(CIRCLE)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("CLICK", (0, 0))) == []
    assert run(s, ("KEY", "ENTER")) == [("CANCEL",)]


def test_circle_ctrl_rounds_the_radius_to_the_grid():
    s = new_session(CIRCLE, grid=0.5, circle_segments=4)
    run(s, ("CLICK", (0, 0)))
    ring = rect_ring(run(s, ("CLICK", (1.3, 0.0), Mods(ctrl=True))))
    assert pclose(ring[0], (1.5, 0.0))


def test_circle_drag_commits_on_release():
    s = new_session(CIRCLE, circle_segments=6)
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("RELEASE", (2, 0), None, True))
    assert len(rect_ring(cmds)) == 6 and s.done


# -- modes -----------------------------------------------------------------------------------------------

def test_mode_keys_switch_before_geometry_exists():
    s = new_session()
    run(s, ("KEY", "R"))
    assert s.mode == RECTANGLE
    run(s, ("CLICK", (0, 0)), ("KEY", "O"))
    assert s.mode == CIRCLE and len(s.points) == 1
    run(s, ("KEY", "P"))
    assert s.mode == POLYLINE
    cmds = run(s, ("CLICK", (1, 0)))
    assert edges(cmds) == [(-1, -2)]
    # Two points exist now: the mode is locked in.
    run(s, ("KEY", "R"))
    assert s.mode == POLYLINE


def test_switching_mode_drops_typed_input():
    s = new_session()
    run(s, ("CLICK", (0, 0)), *typed("12"), ("KEY", "R"))
    assert s.mode == RECTANGLE and not s.numeric.is_typing()


def test_events_after_the_end_do_nothing():
    s, _ = poly_with((0, 0), (1, 0))
    run(s, ("KEY", "RMB"))
    assert run(s, ("CLICK", (3, 3)), ("KEY", "ESC"), ("MOVE", (1, 1)), ("TYPE", "1")) == []


def test_ray_plane_hit_tells_edge_on_from_behind():
    inv = Matrix.Identity(4)
    assert dl.ray_plane_hit((0, 0, 5), (1, 0, 0), inv) == (None, dl.MISS_EDGE_ON)
    assert dl.ray_plane_hit((0, 0, 5), (1, 0, -0.005), inv) == (None, dl.MISS_EDGE_ON)
    assert dl.ray_plane_hit((0, 0, -5), (0, 0, -1), inv) == (None, dl.MISS_BEHIND)
    xy, miss = dl.ray_plane_hit((1, 2, 5), (0, 0, -1), inv)
    assert pclose(xy, (1, 2)) and miss is None


def test_unknown_events_are_errors():
    s = new_session()
    for event in (("BOGUS",), ("KEY", "Q")):
        try:
            s.handle(event)
        except ValueError:
            continue
        raise AssertionError(event)


def test_header_always_tells_how_to_back_out():
    for mode in (POLYLINE, RECTANGLE, CIRCLE):
        s = new_session(mode)
        assert "Esc cancels" in s.header_text()
        run(s, ("CLICK", (0, 0)))
        assert "Esc cancels" in s.header_text()


def test_header_text_is_plain_ascii():
    for mode in (POLYLINE, RECTANGLE, CIRCLE):
        s = new_session(mode)
        text = s.header_text()
        assert text.isascii()
        run(s, ("CLICK", (0, 0)))
        assert s.header_text().isascii()


# -- units: typed lengths and the readout are scene base units ----------------------------------

def unit_session(mode=POLYLINE, scale=0.01):
    return DrawSession(mode, grid=0.1, unit_scale=scale)


def test_typed_polyline_length_is_divided_by_the_unit_scale():
    s = unit_session()
    run(s, ("CLICK", (0, 0)), ("MOVE", (5.0, 0.0)))
    cmds = run(s, *typed("2"), ("KEY", "ENTER"))
    assert pclose(verts(cmds)[1][1], (200.0, 0.0))


def test_typed_rectangle_and_circle_sizes_are_divided_by_the_unit_scale():
    s = unit_session(RECTANGLE, 2.0)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 1.0)))
    cmds = run(s, *typed("4"), ("KEY", "TAB"), *typed("2"), ("KEY", "ENTER"))
    corners = [xy for _, xy in verts(cmds)]
    assert len(corners) == 4
    assert all(any(pclose(xy, want) for xy in corners) for want in ((0, 0), (2, 0), (2, 1), (0, 1)))
    s = unit_session(CIRCLE, 0.5)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 0.0)))
    cmds = run(s, *typed("3"), ("KEY", "ENTER"))
    assert all(close(math.hypot(*xy), 6.0, 1e-6) for _, xy in verts(cmds))


def test_typed_angle_is_not_scaled():
    s = unit_session(POLYLINE, 0.01)
    run(s, ("CLICK", (0, 0)), ("CLICK", (100.0, 0.0)), ("MOVE", (100.0, 50.0)))
    cmds = run(s, *typed("1"), ("KEY", "TAB"), *typed("90"), ("KEY", "ENTER"))
    assert pclose(verts(cmds)[-1][1], (100.0, 100.0), 1e-6)


def test_readout_shows_local_lengths_times_the_unit_scale():
    s = unit_session(POLYLINE, 0.01)
    run(s, ("CLICK", (0, 0)), ("MOVE", (300.0, 0.0)))
    assert s.preview().lines[0] == "Length: 3.000"
    s = unit_session(RECTANGLE, 2.0)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 3.0)))
    assert s.preview().lines == ["Width: 2.000", "Height: 6.000"]
    s = unit_session(CIRCLE, 0.5)
    run(s, ("CLICK", (0, 0)), ("MOVE", (4.0, 0.0)))
    assert s.preview().lines == ["Radius: 2.000"]


def test_typed_text_in_the_readout_stays_as_typed():
    s = unit_session(POLYLINE, 0.01)
    run(s, ("CLICK", (0, 0)), ("MOVE", (300.0, 0.0)), *typed("12"))
    assert s.preview().lines[0] == "> Length: 12_"


def test_ctrl_grid_is_in_local_units_whatever_the_unit_scale():
    for scale in (1.0, 0.01, 2.0):
        s = unit_session(POLYLINE, scale)
        run(s, ("CLICK", (0, 0), Mods(ctrl=True)))
        cmds = run(s, ("CLICK", (0.32, 0.0), Mods(ctrl=True)))
        assert close(verts(cmds)[1][1][0], 0.3, 1e-9), scale


def test_circle_radius_under_ctrl_vertex_snap_is_the_distance_to_the_vertex():
    s = new_session(CIRCLE, grid=0.5)
    run(s, ("CLICK", (0, 0)))
    snap = Snap(3, (1.37, 0.0))
    cmds = run(s, ("CLICK", (1.4, 0.2), Mods(ctrl=True, snap=snap)))
    pts = [xy for _, xy in verts(cmds)]
    assert all(close(math.hypot(*xy), 1.37, 1e-9) for xy in pts)
    # Without a vertex snap the radius still rounds to the grid.
    s = new_session(CIRCLE, grid=0.5)
    run(s, ("CLICK", (0, 0)))
    cmds = run(s, ("CLICK", (1.4, 0.0), Mods(ctrl=True)))
    assert all(close(math.hypot(*xy), 1.5, 1e-9) for _, xy in verts(cmds))


# -- arcs (C21) ------------------------------------------------------------------------------------------

def arc_ring(commands):
    return [xy for _, xy in verts(commands)]


def assert_open_chain(commands):
    """The commands make one open chain: consecutive edges, every new vertex in it, then END."""
    chain = edges(commands)
    assert commands[-1] == ("END",)
    for (a, b), (c, d) in zip(chain, chain[1:]):
        assert b == c
    assert chain[0][0] != chain[-1][1]
    assert {r for r, _ in verts(commands)} <= {r for e in chain for r in e}


def three_point_arc(*points, **kwargs):
    s = new_session(ARC_3PT, **kwargs)
    cmds = run(s, *[("CLICK", p) for p in points])
    return s, cmds


def test_three_point_arc_by_three_clicks_is_an_open_chain():
    s = new_session(ARC_3PT, circle_segments=32)
    assert run(s, ("CLICK", (1, 0))) == []
    assert run(s, ("CLICK", (math.cos(math.pi / 4), math.sin(math.pi / 4)))) == []
    assert not s.done
    cmds = run(s, ("CLICK", (0, 1)))
    ring = arc_ring(cmds)
    assert len(ring) == 9 and pclose(ring[0], (1, 0)) and pclose(ring[-1], (0, 1))
    assert all(close(math.hypot(x, y), 1.0) for x, y in ring)
    assert len(edges(cmds)) == 8
    assert_open_chain(cmds)
    assert s.done and not s.cancelled


def test_three_point_arc_preview_ends_at_the_cursor_once_two_points_exist():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (1, 0)), ("MOVE", (0.0, 0.9)))
    # One point: a straight line to the cursor and a marker for the start.
    assert s.preview().strips == [[(1, 0), (0.0, 0.9)]] and s.preview().markers == [(1, 0)]
    run(s, ("CLICK", (0.7, 0.7)), ("MOVE", (0.0, 1.0)))
    pv = s.preview()
    assert len(pv.strips) == 1 and len(pv.strips[0]) > 3
    assert pclose(pv.strips[0][0], (1, 0)) and pclose(pv.strips[0][-1], (0.0, 1.0))
    assert pv.markers == [(1, 0), (0.7, 0.7)]
    assert pv.lines[0].startswith("Radius: ")


def test_three_point_arc_preview_of_collinear_points_shows_the_corner():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, 0)), ("MOVE", (2, 0)))
    assert s.preview().strips == [[(0, 0), (1, 0), (2, 0)]]


def test_three_point_arc_over_180_degrees_and_clockwise():
    _, cmds = three_point_arc((1, 0), (0, -1), (0, 1))
    ring = arc_ring(cmds)
    assert len(ring) == 25 and pclose(ring[0], (1, 0)) and pclose(ring[-1], (0, 1))
    assert any(pclose(p, (0, -1)) for p in ring)
    _, cmds = three_point_arc((0, 1), (0.7, 0.7), (1, 0))
    assert pclose(arc_ring(cmds)[0], (0, 1)) and pclose(arc_ring(cmds)[-1], (1, 0))


def test_three_point_arc_collinear_or_coincident_third_click_does_nothing():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, 0)))
    assert run(s, ("CLICK", (2, 0))) == [] and not s.done
    assert run(s, ("CLICK", (0, 0))) == [] and not s.done
    assert len(verts(run(s, ("CLICK", (1, 1))))) > 3 and s.done


def test_three_point_arc_second_click_on_the_start_places_nothing():
    s = new_session(ARC_3PT, grid=0.5)
    ctrl = Mods(ctrl=True)
    run(s, ("CLICK", (1.02, 1.01), ctrl))
    # Both clicks round to the same grid point: the second is refused and the run is still usable.
    assert run(s, ("CLICK", (1.1, 0.9), ctrl)) == [] and len(s.points) == 1
    assert run(s, ("CLICK", (1.0, 1.0))) == [] and len(s.points) == 1
    assert run(s, ("CLICK", (1.5, 1.5), ctrl)) == [] and len(s.points) == 2
    cmds = run(s, ("CLICK", (2.0, 1.0), ctrl))
    assert s.done and not s.cancelled
    ring = arc_ring(cmds)
    assert len(ring) > 3 and pclose(ring[0], (1.0, 1.0)) and pclose(ring[-1], (2.0, 1.0))


def test_three_point_arc_uses_the_segment_density_of_the_preference():
    _, cmds = three_point_arc((1, 0), (0.7, 0.7), (0, 1), circle_segments=64)
    assert len(arc_ring(cmds)) == 17
    _, cmds = three_point_arc((1, 0), (0.7, 0.7), (0, 1), circle_segments=4)
    assert len(arc_ring(cmds)) == 3


def test_three_point_arc_escape_cancels_at_every_stage():
    for clicks in ([], [(0, 0)], [(0, 0), (1, 1)]):
        s = new_session(ARC_3PT)
        run(s, *[("CLICK", p) for p in clicks])
        assert run(s, ("KEY", "ESC")) == [("CANCEL",)] and s.cancelled
    # RMB and Enter keep nothing either: an arc has no usable intermediate state.
    for key in ("RMB", "ENTER", "N"):
        s = new_session(ARC_3PT)
        run(s, ("CLICK", (0, 0)), ("CLICK", (1, 1)))
        assert run(s, ("KEY", key)) == [("CANCEL",)], key


def test_three_point_arc_takes_no_typed_values_and_ignores_the_keys_for_them():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)), *typed("12.5-"), ("KEY", "TAB"), ("KEY", "BACKSPACE"), ("CLICK", (1, 1)))
    assert not s.numeric.is_typing() and not s.numeric.has_value()
    assert len(s.points) == 2
    assert "no typed values" in s.header_text()
    # Enter with nothing typed cancels, as for the other shapes.
    assert run(s, ("KEY", "ENTER")) == [("CANCEL",)]


def test_three_point_arc_ctrl_snaps_each_point_and_reuses_the_end_vertices():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (1.02, 0.01), Mods(ctrl=True, snap=Snap(5, (1.0, 0.0)))),
        ("CLICK", (0.71, 0.69), Mods(ctrl=True, snap=Snap(6, (0.7, 0.7)))))
    cmds = run(s, ("CLICK", (0.0, 1.03), Mods(ctrl=True, snap=Snap(7, (0.0, 1.0)))))
    chain = edges(cmds)
    # Start and end are the existing vertices; the middle snap only places a point of the arc.
    assert chain[0][0] == 5 and chain[-1][1] == 7
    assert 5 not in [r for r, _ in verts(cmds)] and 7 not in [r for r, _ in verts(cmds)]
    assert 6 not in [r for e in chain for r in e]
    assert_open_chain(cmds)
    # (1, 0), (0.7, 0.7), (0, 1) lie on a circle of radius ~1 about a centre near the origin.
    assert all(abs(math.hypot(x, y) - 1.0) < 0.02 for x, y in arc_ring(cmds))


def test_three_point_arc_ctrl_grid_snaps_the_points():
    s = new_session(ARC_3PT, grid=0.5)
    cmds = run(s, ("CLICK", (0.1, 0.1), Mods(ctrl=True)), ("CLICK", (1.1, 0.9), Mods(ctrl=True)),
               ("CLICK", (2.1, -0.1), Mods(ctrl=True)))
    ring = arc_ring(cmds)
    assert pclose(ring[0], (0, 0)) and pclose(ring[-1], (2, 0))
    # The middle point (1, 1) is on the circle about (1, 0).
    assert all(close(math.hypot(x - 1.0, y), 1.0, 1e-6) for x, y in ring)


def test_three_point_arc_shift_locks_the_end_direction():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, -1)))
    run(s, ("MOVE", (3.0, 0.4), Mods(shift=True)))
    assert pclose(s.preview().strips[0][-1], (math.hypot(3.0, 0.4), 0.0))
    ring = arc_ring(run(s, ("CLICK", (3.0, 0.4), Mods(shift=True))))
    assert pclose(ring[-1], (math.hypot(3.0, 0.4), 0.0))
    # With Ctrl the locked length goes to the grid.
    s = new_session(ARC_3PT, grid=0.5)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, -1)))
    ring = arc_ring(run(s, ("CLICK", (3.1, 0.9), Mods(shift=True, ctrl=True))))
    assert pclose(ring[-1], (3.0, 0.0))


def test_three_point_arc_has_no_drag_commit():
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("RELEASE", (1, 1), None, True)) == [] and not s.done and len(s.points) == 1


def centre_arc(*points, **kwargs):
    s = new_session(ARC_CENTER, **kwargs)
    cmds = run(s, *[("CLICK", p) for p in points])
    return s, cmds


def test_centre_arc_by_three_clicks_runs_counter_clockwise_from_the_start():
    s = new_session(ARC_CENTER, circle_segments=32)
    assert run(s, ("CLICK", (1, 1))) == []
    assert run(s, ("CLICK", (3, 1))) == []
    cmds = run(s, ("CLICK", (1, 4)))
    ring = arc_ring(cmds)
    assert len(ring) == 9 and ring[0] == (3, 1) and pclose(ring[-1], (1, 3))
    assert all(close(math.hypot(x - 1, y - 1), 2.0) for x, y in ring)
    # The centre is not a vertex.
    assert not any(pclose(p, (1, 1)) for p in ring)
    assert_open_chain(cmds)
    assert s.done and not s.cancelled


def test_centre_arc_second_click_sets_the_radius_in_the_cursor_direction():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("MOVE", (3, 4)))
    pv = s.preview()
    assert len(pv.strips) == 1 and pv.strips[0][0] == (0, 0) and pclose(pv.strips[0][1], (3, 4))
    assert pv.lines == ["Radius: 5.000"]
    run(s, ("CLICK", (3, 4)))
    assert len(s.points) == 2 and pclose(s.points[1].xy, (3, 4))
    # Radius 5 whatever distance the next click has.
    ring = arc_ring(run(s, ("CLICK", (-30, 0))))
    assert all(close(math.hypot(x, y), 5.0) for x, y in ring)
    assert pclose(ring[-1], (-5, 0))


def test_centre_arc_preview_shows_the_arc_radii_and_angle():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2, 0)), ("MOVE", (0, 5)))
    pv = s.preview()
    assert len(pv.strips) == 3
    arc, r_start, r_end = pv.strips
    assert pclose(arc[0], (2, 0)) and pclose(arc[-1], (0, 2))
    assert r_start == [(0, 0), (2, 0)] and pclose(r_end[1], (0, 2))
    assert pv.lines == ["Angle: 90.000 deg", "Radius: 2.000"]
    assert pv.markers == [(0, 0), (2, 0)]


def test_centre_arc_over_180_degrees():
    _, cmds = centre_arc((0, 0), (1, 0), (0, -1))
    ring = arc_ring(cmds)
    assert len(ring) == 25 and pclose(ring[-1], (0, -1))


def test_centre_arc_end_in_the_start_direction_is_not_committed():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2, 0)))
    assert run(s, ("CLICK", (5, 0))) == [] and not s.done
    assert run(s, ("CLICK", (0, 0))) == [] and not s.done
    assert run(s, ("CLICK", (0, 1))) != []


def test_centre_arc_zero_radius_is_not_placed():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (1, 1)))
    assert run(s, ("CLICK", (1, 1))) == [] and len(s.points) == 1


def test_centre_arc_typed_radius_for_the_second_click_and_typed_angle_for_the_third():
    s = new_session(ARC_CENTER, circle_segments=32)
    run(s, ("CLICK", (0, 0)), ("MOVE", (5.0, 0.0)), *typed("1.5"))
    assert s.preview().lines == ["> Radius: 1.5_"]
    # Enter places the start at the typed radius, towards the cursor.
    assert run(s, ("KEY", "ENTER")) == [] and not s.done
    assert s.points[1].xy == (1.5, 0.0)
    assert s.numeric.labels == ("Angle",) and not s.numeric.is_typing()
    run(s, ("MOVE", (-4.0, 4.0)), *typed("90"))
    assert s.preview().lines == ["> Angle: 90_", "Radius: 1.500"]
    cmds = run(s, ("KEY", "ENTER"))
    ring = arc_ring(cmds)
    assert len(ring) == 9 and pclose(ring[0], (1.5, 0.0)) and pclose(ring[-1], (0.0, 1.5))
    assert s.done


def test_centre_arc_typed_radius_can_be_committed_by_a_click_and_negative_is_the_magnitude():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("MOVE", (0.0, 9.0)), *typed("-2"))
    run(s, ("CLICK", (0.0, 9.0)))
    assert pclose(s.points[1].xy, (0.0, 2.0))


def test_centre_arc_typed_angle_follows_the_start_and_wraps():
    for text, want_deg in (("45", 45.0), ("270", 270.0), ("-90", 270.0), ("450", 90.0)):
        s = new_session(ARC_CENTER, circle_segments=32)
        run(s, ("CLICK", (0, 0)), ("CLICK", (0, 2)), *typed(text))
        ring = arc_ring(run(s, ("KEY", "ENTER")))
        end_angle = math.degrees(math.atan2(ring[-1][1], ring[-1][0]))
        assert close((end_angle - 90.0 - want_deg + 180.0) % 360.0 - 180.0, 0.0, 1e-6), (text, end_angle)
        assert len(ring) == max(2, math.ceil(32 * want_deg / 360.0 - 1e-9)) + 1, text


def test_centre_arc_typed_angle_of_zero_or_a_full_turn_is_not_committed():
    for text in ("0", "360", "-720"):
        s = new_session(ARC_CENTER)
        run(s, ("CLICK", (0, 0)), ("CLICK", (2, 0)), *typed(text))
        assert run(s, ("KEY", "ENTER")) == [] and not s.done, text


def test_centre_arc_tab_has_one_field_per_stage():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("KEY", "TAB"))
    assert s.numeric.active == 0
    run(s, ("CLICK", (2, 0)), ("KEY", "TAB"), *typed("30"))
    assert s.numeric.texts == ["30"]


def test_centre_arc_typed_radius_is_divided_by_the_unit_scale():
    s = new_session(ARC_CENTER, unit_scale=0.01)
    run(s, ("CLICK", (0, 0)), ("MOVE", (1.0, 0.0)), *typed("3"), ("KEY", "ENTER"))
    assert pclose(s.points[1].xy, (300.0, 0.0))
    assert s.preview().lines == ["Angle: 0.000 deg", "Radius: 3.000"]


def test_centre_arc_shift_locks_the_start_and_the_sweep_to_45_degrees():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2.0, 0.3), Mods(shift=True)))
    # The start direction snapped to sketch X, the radius is the cursor distance.
    r = math.hypot(2.0, 0.3)
    assert pclose(s.points[1].xy, (r, 0.0))
    ring = arc_ring(run(s, ("CLICK", (0.5, 1.8), Mods(shift=True))))
    assert pclose(ring[-1], (0.0, r))


def test_centre_arc_shift_locks_the_sweep_from_the_start_direction():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (0.0, 2.0)))
    ring = arc_ring(run(s, ("CLICK", (0.1, -1.0), Mods(shift=True))))
    # About 186 degrees from the start direction, locked to 180.
    assert pclose(ring[-1], (0.0, -2.0))


def test_centre_arc_ctrl_grid_rounds_the_radius_and_snaps_the_end_direction():
    s = new_session(ARC_CENTER, grid=0.5)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2.1, 0.0), Mods(ctrl=True)))
    assert pclose(s.points[1].xy, (2.0, 0.0))
    ring = arc_ring(run(s, ("CLICK", (0.1, 3.0), Mods(ctrl=True))))
    # The cursor snaps to the grid point (0, 3); the arc ends in that direction at radius 2.
    assert pclose(ring[-1], (0.0, 2.0))


def test_centre_arc_ctrl_vertex_snap_for_the_start_and_end():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2.05, 0.02), Mods(ctrl=True, snap=Snap(4, (2.0, 0.0)))))
    assert s.points[1].ref == 4 and s.points[1].emitted
    # An end snap that lies on the circle is reused ...
    cmds = run(s, ("CLICK", (0.0, 2.05), Mods(ctrl=True, snap=Snap(8, (0.0, 2.0)))))
    assert edges(cmds)[0][0] == 4 and edges(cmds)[-1][1] == 8
    assert all(r not in (4, 8) for r, _ in verts(cmds))
    assert_open_chain(cmds)
    # ... one off the circle only gives the direction.
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (2, 0)))
    cmds = run(s, ("CLICK", (0.0, 3.0), Mods(ctrl=True, snap=Snap(8, (0.0, 3.0)))))
    assert pclose(arc_ring(cmds)[-1], (0.0, 2.0)) and 8 not in [r for e in edges(cmds) for r in e]


def test_centre_arc_escape_and_enter_cancel_at_every_stage():
    for clicks in ([], [(0, 0)], [(0, 0), (1, 0)]):
        s = new_session(ARC_CENTER)
        run(s, *[("CLICK", p) for p in clicks])
        assert run(s, ("KEY", "ESC")) == [("CANCEL",)]
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, 0)))
    assert run(s, ("KEY", "ENTER")) == [("CANCEL",)]


def test_centre_arc_drag_does_not_commit_and_close_does_nothing():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)))
    assert run(s, ("RELEASE", (2, 0), None, True)) == [] and len(s.points) == 1
    assert run(s, ("KEY", "C")) == [] and not s.done


def test_arc_new_vertices_have_distinct_refs_below_zero():
    _, cmds = centre_arc((0, 0), (2, 0), (0, 2))
    assert all(r < 0 for r, _ in verts(cmds))
    assert len({r for r, _ in verts(cmds)}) == len(verts(cmds))


# -- choosing the shape --------------------------------------------------------------------------------------

def test_a_key_alternates_the_arcs_before_two_points_exist():
    s = new_session()
    run(s, ("KEY", "A"))
    assert s.mode == ARC_3PT
    run(s, ("KEY", "A"))
    assert s.mode == ARC_CENTER
    run(s, ("KEY", "A"))
    assert s.mode == ARC_3PT
    run(s, ("CLICK", (0, 0)), ("KEY", "A"))
    assert s.mode == ARC_CENTER and len(s.points) == 1
    run(s, ("KEY", "A"))
    assert s.mode == ARC_3PT
    # Two points are down: the shape is locked.
    run(s, ("CLICK", (1, 1)), ("KEY", "A"), ("KEY", "R"), ("KEY", "O"), ("KEY", "P"))
    assert s.mode == ARC_3PT and len(s.points) == 2


def test_a_key_from_every_other_shape_starts_with_the_three_point_arc():
    for mode in (POLYLINE, RECTANGLE, CIRCLE):
        s = new_session(mode)
        run(s, ("KEY", "A"))
        assert s.mode == ARC_3PT, mode


def test_a_key_is_locked_once_a_centre_arc_has_its_start():
    s = new_session(ARC_CENTER)
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, 0)), ("KEY", "A"))
    assert s.mode == ARC_CENTER


def test_other_shape_keys_switch_away_from_an_arc_before_two_points_exist():
    for key, mode in (("R", RECTANGLE), ("O", CIRCLE), ("P", POLYLINE)):
        s = new_session(ARC_3PT)
        run(s, ("CLICK", (0, 0)), ("KEY", key))
        assert s.mode == mode and len(s.points) == 1, key
    s = new_session(ARC_3PT)
    run(s, ("KEY", "R"))
    assert s.mode == RECTANGLE


def test_switching_between_a_polyline_start_and_an_arc_keeps_the_first_point():
    s = new_session()
    run(s, ("CLICK", (1, 1)), ("KEY", "A"), ("CLICK", (2, 1)), ("CLICK", (2, 2)))
    assert s.done and not s.cancelled
    s = new_session(ARC_3PT)
    run(s, ("CLICK", (0, 0)), ("KEY", "R"))
    assert len(rect_ring(run(s, ("CLICK", (2, 1))))) == 4


def test_a_session_starts_in_the_given_arc_shape():
    for mode in (ARC_3PT, ARC_CENTER):
        s = new_session(mode)
        assert s.mode == mode
        assert run(s, ("CLICK", (0, 0))) == [] and len(s.points) == 1


def test_header_names_the_arc_and_the_key_for_the_other_one():
    s = new_session(ARC_3PT)
    assert "Three-point arc" in s.header_text() and "A other arc" in s.header_text()
    s = new_session(ARC_CENTER)
    assert "Centre arc" in s.header_text() and "P poly, A other arc" in s.header_text()
    for mode in (POLYLINE, RECTANGLE, CIRCLE):
        assert "P poly, A arc" in new_session(mode).header_text() and "other arc" not in new_session(mode).header_text()
    run(s, ("CLICK", (0, 0)), ("CLICK", (1, 0)))
    # Past two points the shape keys are no longer offered.
    assert "A other arc" not in s.header_text() and "A arc" not in s.header_text() and "angle (deg)" in s.header_text()
    for mode in (POLYLINE, RECTANGLE, CIRCLE, ARC_3PT, ARC_CENTER):
        s = new_session(mode)
        for _ in range(2):
            text = s.header_text()
            assert text.isascii() and "Esc cancels" in text
            run(s, ("CLICK", (0, 0)))
