"""The 3D path header (DECISIONS C24): the snapping state first, then the hint and the scene notice,
then the instructions, all of it short enough that Blender does not clip it. Mirrors the planar
tests in test_snap_toggle."""

import bpy
from mathutils import Vector

from sketch_tools.core import snap3d
from sketch_tools.core.snap3d import PathSession, Placement
from sketch_tools.ops import draw3d
from sketch_tools.tests.test_addon import addon
from sketch_tools.tests.test_draw3d import TOP, at, edit_path, ev, make_controller

# Blender clips an overflowing header at both ends; about 305 characters fit in the default 1574 px
# wide 3D area. 260 leaves room for a smaller viewport, and the notice and the hint together still
# have to fit in 300 (the same bounds as the planar tool, C23).
BOUND_PLAIN = 260
BOUND_WITH_NOTICE_AND_HINT = 300
UP = Vector((0, 0, 1))


def pl(x, y, z=1.0):
    return Placement(Vector((x, y, z)), UP.copy(), snap3d.FACE, None)


def header_states():
    """(shape, point count, session) for every state the header text depends on: each shape before
    and after its clicks, and the arcs that could not be made (the note is on the line)."""
    clicks = [pl(float(n + 1), 0.5 * (n % 2)) for n in range(6)]
    for shape in snap3d.SHAPES:
        for count in range(7):
            session = PathSession(shape)
            for click in clicks[:count]:
                session.handle(("CLICK", click))
            if session.done or len(session.points) != count:
                break
            yield shape, count, session
    for shape in (snap3d.ARC_3PT, snap3d.ARC_CENTER):
        session = PathSession(shape)
        for click in (pl(0.0, 0.0), pl(1.0, 0.0), pl(2.0, 0.0) if shape == snap3d.ARC_3PT else pl(1.0, 0.0)):
            session.handle(("CLICK", click))
        assert session.note, shape
        yield shape, len(session.points), session


def state_of(head):
    return head.split(" | ")[0]


def test_the_3d_header_covers_every_shape_and_point_count():
    seen = {(shape, count) for shape, count, _ in header_states()}
    for shape in snap3d.SHAPES:
        assert (shape, 0) in seen and (shape, 1) in seen
    assert (snap3d.POLYLINE, 6) in seen and (snap3d.ARC_3PT, 2) in seen and (snap3d.ARC_CENTER, 2) in seen
    assert any(session.note for _, _, session in header_states())


def test_every_3d_header_wording_says_shift_and_ctrl_turn_snapping_off_while_held():
    for shape, count, session in header_states():
        text = session.header_text()
        assert text.isascii() and chr(0x2014) not in text
        assert text.endswith("Esc cancels, Ctrl+Z off"), (shape, count)
        assert "hold Shift or Ctrl" in text, (shape, count, text)
        assert ("hold Shift or Ctrl: snapping off, Shift locks 45 degrees" in text) == (shape == snap3d.POLYLINE)
        assert ("hold Shift or Ctrl to turn snapping off" in text) == (shape != snap3d.POLYLINE)
        assert "no typed values" in text, (shape, count)
        # The arcs say A goes to the other arc; the other shapes name the plain key.
        tail = "A other arc" if shape in snap3d.ARCS else "A arc"
        assert (f"R rect, O circle, P poly, {tail}" in text) == (count <= 1), (shape, count)
        assert ("A other arc" in text) == (shape in snap3d.ARCS and count <= 1), (shape, count)
        assert ("RMB/Enter end, C close" in text) == (shape == snap3d.POLYLINE and count >= 2)
        if session.note:
            assert text.startswith(session.note)


def test_the_3d_header_stays_within_the_width_that_fits_with_the_state_first():
    with addon():
        obj = edit_path()
        real = draw3d.MAX_SNAP_VERTICES
        draw3d.MAX_SNAP_VERTICES = 0
        try:
            capped = draw3d.build_scene_source(bpy.context, obj)
        finally:
            draw3d.MAX_SNAP_VERTICES = real
        assert capped.notice == "Snap limit: 1 object(s) surface only"
        # Room for a three digit count.
        notice = capped.notice.replace("1 object", "999 object")
        hint = max(draw3d.NO_SURFACE_HINT, draw3d.NO_MESH_HINT, draw3d.NO_PLANE_HINT, key=len)
        c, headers, _, _ = make_controller(obj, source=capped)
        for shape, count, session in header_states():
            c.session = session
            for snap_off in (False, True):
                state = "Snapping off" if snap_off else "Snapping on"
                c._snap_off = snap_off
                c.source.notice = ""
                c._hint = None
                c._update_header()
                head = headers[-1]
                assert state_of(head) == state, head
                assert len(head) <= BOUND_PLAIN, (shape, count, len(head), head)
                c.source.notice = notice
                for the_hint in (None, hint):
                    c._hint = the_hint
                    c._update_header()
                    head = headers[-1]
                    assert state_of(head) == state, head
                    if the_hint:
                        assert head.split(" | ")[1] == the_hint
                    assert notice in head and head.isascii()
                    assert len(head) <= BOUND_WITH_NOTICE_AND_HINT, (shape, count, len(head), head)
        c.abort()


def test_the_3d_header_order_is_state_hint_notice_instructions():
    with addon():
        obj = edit_path()
        c, headers, _, _ = make_controller(obj)
        c.source.notice = "Snap limit: 2 object(s) surface only"
        c._hint = draw3d.NO_SURFACE_HINT
        c._update_header()
        parts = headers[-1].split(" | ")
        assert parts[:3] == ["Snapping on", draw3d.NO_SURFACE_HINT, c.source.notice]
        assert parts[3] == snap3d._MODE_HELP[snap3d.POLYLINE]
        c._hint = None
        c.source.notice = ""
        c._update_header()
        assert headers[-1].split(" | ")[:2] == ["Snapping on", snap3d._MODE_HELP[snap3d.POLYLINE]]
        c.abort()


def test_the_3d_header_state_follows_shift_and_ctrl_during_a_run():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        px = view.screen(TOP)
        assert state_of(headers[-1]) == "Snapping on"
        # Pressing and releasing a modifier are events of their own.
        for key, mods in (('LEFT_SHIFT', {"shift": True}), ('LEFT_CTRL', {"ctrl": True}),
                          ('LEFT_SHIFT', {"shift": True, "ctrl": True})):
            c.handle(ev(key, 'PRESS', px=px, **mods))
            assert state_of(headers[-1]) == "Snapping off", (key, headers[-1])
            c.handle(ev(key, 'RELEASE', px=px))
            assert state_of(headers[-1]) == "Snapping on", (key, headers[-1])
        # A click placed with a modifier held shows the state it was placed with.
        c.handle(at(view, TOP, ctrl=True))
        assert state_of(headers[-1]) == "Snapping off"
        c.handle(ev('MOUSEMOVE', px=px))
        assert state_of(headers[-1]) == "Snapping on"
        c.abort()


def test_the_3d_header_state_is_correct_for_every_modifier_combination():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        for shift in (False, True):
            for ctrl in (False, True):
                c.handle(ev('MOUSEMOVE', px=view.screen(TOP), shift=shift, ctrl=ctrl))
                state = "Snapping off" if shift or ctrl else "Snapping on"
                other = "Snapping on" if shift or ctrl else "Snapping off"
                assert state_of(headers[-1]) == state, (shift, ctrl, headers[-1])
                assert other not in headers[-1]
        c.abort()


def test_a_3d_run_started_with_a_modifier_held_shows_the_state_at_once():
    for shift, ctrl in ((False, False), (True, False), (False, True), (True, True)):
        with addon():
            obj = edit_path()
            c, headers, _, view = make_controller(obj)
            assert c.start(at(view, TOP, shift=shift, ctrl=ctrl)) == 'RUNNING_MODAL'
            state = "Snapping off" if shift or ctrl else "Snapping on"
            assert state_of(headers[-1]) == state, (shift, ctrl, headers[-1])
            c.abort()


def test_a_modifier_pressed_outside_the_region_flips_the_3d_header_at_once():
    with addon():
        obj = edit_path()
        c, headers, _, view = make_controller(obj)
        c.handle(at(view, TOP))
        c.handle(ev('LEFTMOUSE', 'RELEASE', px=view.screen(TOP)))
        outside = (900.0, 300.0)
        for key, mods in (('LEFT_SHIFT', {"shift": True}), ('LEFT_CTRL', {"ctrl": True})):
            assert state_of(headers[-1]) == "Snapping on"
            assert c.handle(ev(key, 'PRESS', px=outside, **mods)) == 'PASS_THROUGH'
            assert state_of(headers[-1]) == "Snapping off", (key, headers[-1])
            assert c.handle(ev(key, 'RELEASE', px=outside)) == 'PASS_THROUGH'
            assert state_of(headers[-1]) == "Snapping on", (key, headers[-1])
        # Any other pass-through event carries the state too, a navigation event included.
        assert c.handle(ev('WHEELUPMOUSE', px=outside, ctrl=True)) == 'PASS_THROUGH'
        assert state_of(headers[-1]) == "Snapping off"
        assert c.handle(ev('MIDDLEMOUSE', px=outside)) == 'PASS_THROUGH'
        assert state_of(headers[-1]) == "Snapping on"
        c.abort()


def test_the_arc_tails_say_a_goes_to_the_other_arc_and_stay_within_the_bound():
    for shape, count, session in header_states():
        text = session.header_text()
        if count <= 1:
            assert ("A other arc" in text) == (shape in snap3d.ARCS), (shape, text)
            assert text.count("A arc") == (0 if shape in snap3d.ARCS else 1)
