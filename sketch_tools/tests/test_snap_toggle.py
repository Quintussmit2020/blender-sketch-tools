"""C23: the planar draw tool follows Blender's snap toggle (the magnet) and Ctrl flips it.

Snapping is on when scene.tool_settings.use_snap differs from event.ctrl. The controller folds that
into Mods.ctrl, so core/draw_logic is untouched; the same fake camera and cube as
test_planar_scene_snap.py drive the controller and the operator shell."""

import bmesh
import bpy

from sketch_tools.core import draw_logic
from sketch_tools.ops import draw3d, draw_tool
from sketch_tools.ops.draw_tool import effective_snap
from sketch_tools.tests.test_addon import addon
from sketch_tools.tests.test_draw3d import ev
from sketch_tools.tests.test_planar_scene_snap import (
    A, D, edge_count, local_verts, near, points_near, setup, snap_operator,
)

# Far from every cube corner and edge, so only the grid can catch them.
FREE_A = (60.0, 540.0)
FREE_B = (200.0, 420.0)
FREE_C = (330.0, 500.0)


def test_the_effective_snap_flag_is_the_xor_of_the_magnet_and_ctrl():
    assert effective_snap(False, False) is False
    assert effective_snap(False, True) is True
    assert effective_snap(True, False) is True
    assert effective_snap(True, True) is False


def test_the_pointer_passes_the_effective_flag_as_mods_ctrl_and_keeps_shift():
    with addon():
        s = setup()
        for magnet in (False, True):
            for ctrl in (False, True):
                c = s.controller(snap_toggle=lambda m=magnet: m)
                s.move(c, px=FREE_A, ctrl=ctrl, shift=True)
                assert c.session.mods.ctrl is (magnet != ctrl), (magnet, ctrl)
                assert c.session.mods.shift is True
                c.abort()


def test_magnet_on_snaps_to_a_scene_vertex_without_ctrl():
    with addon():
        s = setup()
        c = s.controller(snap_toggle=lambda: True)
        assert s.click(c, A, first=True) == 'RUNNING_MODAL'
        assert s.click(c, D) == 'RUNNING_MODAL'
        assert c.handle(ev('RIGHTMOUSE')) == 'FINISHED'
        assert points_near(local_verts(s.obj), [(1.0, -1.0), (-1.0, 1.0)])
        assert edge_count(s.obj) == 1


def test_magnet_on_snaps_to_a_sketch_vertex_without_ctrl_and_reuses_it():
    with addon():
        s = setup()
        under = s.view.ray_to_local(s.px(A))
        bm = bmesh.from_edit_mesh(s.obj.data)
        bm.verts.new((under[0], under[1], 0.0))
        bmesh.update_edit_mesh(s.obj.data)
        c = s.controller(snap_toggle=lambda: True)
        s.click(c, A, first=True)
        s.click(c, D)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        # The start reused the existing vertex instead of adding one at the scene corner.
        assert len(verts) == 2 and near(verts[0][:2], under, 1e-4)
        assert not any(near(v[:2], (1.0, -1.0), 0.05) for v in verts)


def test_magnet_on_snaps_a_free_point_to_the_grid_without_ctrl():
    with addon():
        s = setup()
        c = s.controller(snap_toggle=lambda: True)
        s.click(c, px=FREE_A, first=True)
        s.click(c, px=FREE_B)
        c.handle(ev('RIGHTMOUSE'))
        expected = [draw_logic.grid_snap_point(s.view.ray_to_local(p), 0.1) for p in (FREE_A, FREE_B)]
        assert points_near(local_verts(s.obj), expected, 1e-4)
        free = [s.view.ray_to_local(p) for p in (FREE_A, FREE_B)]
        assert not points_near(local_verts(s.obj), free, 1e-4)


def test_magnet_on_with_ctrl_held_places_free_points():
    with addon():
        s = setup()
        c = s.controller(snap_toggle=lambda: True)
        s.click(c, A, first=True, ctrl=True)
        s.click(c, px=FREE_B, ctrl=True)
        assert c.session.mods.snap is None
        c.handle(ev('RIGHTMOUSE'))
        expected = [s.view.ray_to_local(p) for p in (s.px(A), FREE_B)]
        assert points_near(local_verts(s.obj), expected, 1e-4)
        assert not near(local_verts(s.obj)[0][:2], (1.0, -1.0), 0.05)


def test_magnet_off_keeps_the_old_behaviour():
    with addon():
        # Without Ctrl: free.
        s = setup()
        c = s.controller(snap_toggle=lambda: False)
        s.click(c, A, first=True)
        s.click(c, px=FREE_B)
        c.handle(ev('RIGHTMOUSE'))
        expected = [s.view.ray_to_local(p) for p in (s.px(A), FREE_B)]
        assert points_near(local_verts(s.obj), expected, 1e-4)
    with addon():
        # With Ctrl: the scene vertex, then the grid.
        s = setup()
        c = s.controller(snap_toggle=lambda: False)
        s.click(c, A, first=True, ctrl=True)
        s.click(c, px=FREE_B, ctrl=True)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)
        assert near(verts[0][:2], (1.0, -1.0), 1e-3)
        assert near(verts[1][:2], draw_logic.grid_snap_point(s.view.ray_to_local(FREE_B), 0.1), 1e-4)


def test_a_controller_without_a_toggle_reads_it_as_off():
    with addon():
        s = setup()
        c = s.controller()
        s.move(c, px=FREE_A)
        assert c.session.mods.ctrl is False
        s.move(c, px=FREE_A, ctrl=True)
        assert c.session.mods.ctrl is True
        c.abort()


def test_flipping_the_magnet_mid_run_takes_effect_on_the_next_event():
    with addon():
        s = setup()
        state = {"on": False}
        c = s.controller(snap_toggle=lambda: state["on"])
        s.click(c, px=FREE_A, first=True)
        s.move(c, px=FREE_B)
        assert c.session.mods.ctrl is False
        state["on"] = True
        s.move(c, px=FREE_B)
        assert c.session.mods.ctrl is True
        s.click(c, px=FREE_B)
        state["on"] = False
        s.click(c, px=FREE_C)
        c.handle(ev('RIGHTMOUSE'))
        verts = local_verts(s.obj)

        def free(px):
            return s.view.ray_to_local(px)
        assert near(verts[0][:2], free(FREE_A), 1e-4)
        assert near(verts[1][:2], draw_logic.grid_snap_point(free(FREE_B), 0.1), 1e-4)
        assert near(verts[2][:2], free(FREE_C), 1e-4)


def sized_shape(mode, magnet, ctrl, clicks):
    """The vertices of a finished shape drawn with the given magnet and Ctrl state. Scene snapping
    is off so the grid is the only thing that can move a point."""
    with addon():
        s = setup()
        c = s.controller(mode, scene=False, snap_toggle=lambda: magnet)
        for i, px in enumerate(clicks):
            s.click(c, px=px, first=(i == 0), ctrl=ctrl)
        c.abort()
        return local_verts(s.obj)


def same(a, b):
    return len(a) == len(b) and all(near(x, y, 1e-4) for x, y in zip(a, b))


def test_rectangle_circle_and_arc_sizes_round_to_the_grid_by_the_effective_flag():
    for mode, clicks in (('RECTANGLE', (FREE_A, FREE_B)), ('CIRCLE', (FREE_A, FREE_B)),
                         ('ARC_3PT', (FREE_A, FREE_B, FREE_C))):
        snapped = sized_shape(mode, False, True, clicks)
        free = sized_shape(mode, False, False, clicks)
        assert snapped and free and not same(snapped, free), mode
        assert same(sized_shape(mode, True, False, clicks), snapped), mode
        assert same(sized_shape(mode, True, True, clicks), free), mode


def test_shift_tab_passes_through_and_plain_tab_stays_the_next_field_key():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, px=FREE_A, first=True)
        assert c.handle(ev('TAB', shift=True)) == 'PASS_THROUGH'
        assert c.handle(ev('TAB', 'RELEASE', shift=True)) == 'PASS_THROUGH'
        assert c.handle(ev('TAB')) == 'RUNNING_MODAL'
        c.abort()


def test_the_operator_reads_the_scene_toggle_live():
    with addon():
        s = setup()
        settings = bpy.context.scene.tool_settings
        saved = settings.use_snap
        shell, ctx, headers, handlers, sources, undo = snap_operator(s)
        try:
            op = draw_tool.SKETCH_OT_draw
            settings.use_snap = True
            assert op.invoke(shell, ctx, ev('LEFTMOUSE', px=s.px(A))) == {'RUNNING_MODAL'}
            settings.use_snap = False
            assert op.modal(shell, ctx, ev('LEFTMOUSE', px=FREE_B)) == {'RUNNING_MODAL'}
            settings.use_snap = True
            assert op.modal(shell, ctx, ev('LEFTMOUSE', px=s.px(D), ctrl=True)) == {'RUNNING_MODAL'}
            assert op.modal(shell, ctx, ev('RIGHTMOUSE')) == {'FINISHED'}
        finally:
            settings.use_snap = saved
            undo()
        verts = local_verts(s.obj)
        # Magnet on: the cube corner. Magnet off, no Ctrl: free. Magnet on with Ctrl: free.
        assert near(verts[0][:2], (1.0, -1.0), 1e-3)
        assert near(verts[1][:2], s.view.ray_to_local(FREE_B), 1e-4)
        assert near(verts[2][:2], s.view.ray_to_local(s.px(D)), 1e-4)
        assert not draw_tool._live


def test_the_header_hint_is_true_in_both_states():
    with addon():
        s = setup()
        c = s.controller()
        s.click(c, px=FREE_A, first=True)
        assert "hold Ctrl to flip snapping" in s.headers[-1] and "Ctrl snap," not in s.headers[-1]
        assert "toggles snapping" not in s.headers[-1]
        c.abort()


ALL_MODES = (draw_logic.POLYLINE, draw_logic.RECTANGLE, draw_logic.CIRCLE,
             draw_logic.ARC_3PT, draw_logic.ARC_CENTER)
# Blender clips an overflowing header at both ends. Measured at the default layout (a 1574 px wide
# 3D area, about 5.15 px per character) about 305 characters fit, so a line over that loses the
# snapping state. 260 leaves room for a smaller viewport; the notice and the hint are each about 40
# characters, and with both on the line the whole must still fit in 300.
BOUND_PLAIN = 260
BOUND_WITH_NOTICE_AND_HINT = 300


def header_states():
    """(mode, point count, typed, session) for every state the planar header text depends on,
    each with a session of its own."""
    for mode in ALL_MODES:
        for typed_text in ("", "12.5"):
            for count in range(5):
                session = draw_logic.DrawSession(mode)
                for n in range(count):
                    session.handle(("CLICK", (float(n + 1), 0.5 * n), draw_logic.Mods()))
                if session.done or len(session.points) != count:
                    break
                for ch in typed_text:
                    session.handle(("TYPE", ch))
                yield mode, count, bool(typed_text), session


def test_every_header_wording_says_ctrl_flips_snapping_only_while_held():
    seen = set()
    for mode, count, typed_text, session in header_states():
        text = session.header_text()
        seen.add((mode, count))
        assert "toggles snapping" not in text, (mode, count)
        # Both tools say Esc drops the run and Ctrl+Z is off, not that nothing can be undone.
        assert text.endswith("Esc cancels, Ctrl+Z off") and "no undo" not in text, (mode, count)
        assert "hold Ctrl to flip snapping" in text, (mode, count, typed_text)
    # Every shape was looked at with no point and with a later point.
    assert all((mode, 0) in seen and (mode, 1) in seen for mode in ALL_MODES)


def test_the_header_stays_within_the_width_that_fits_with_the_state_first():
    with addon():
        s = setup()
        real = draw3d.MAX_SNAP_VERTICES
        draw3d.MAX_SNAP_VERTICES = 0
        try:
            capped = draw3d.build_scene_source(bpy.context, s.obj, normals=False)
        finally:
            draw3d.MAX_SNAP_VERTICES = real
        assert capped.notice
        # Room for a three digit count.
        notice = capped.notice.replace("1 object", "999 object")
        hint = max(draw_tool.EDGE_ON_HINT, draw_tool.BEHIND_HINT, key=len)
        states = list(header_states())
        for magnet in (False, True):
            c = s.controller(snap_toggle=lambda m=magnet: m)
            for mode, count, typed_text, session in states:
                c.session = session
                c._scene.notice = ""
                c._hint = None
                c._update_header()
                head = s.headers[-1]
                assert head.startswith(("Snapping on | ", "Snapping off | ")), head
                assert len(head) <= BOUND_PLAIN, (mode, count, typed_text, len(head), head)
                c._scene.notice = notice
                for the_hint in (None, hint):
                    c._hint = the_hint
                    c._update_header()
                    head = s.headers[-1]
                    assert head.startswith(("Snapping on | ", "Snapping off | ")), head
                    if the_hint:
                        assert head.split(" | ")[1] == hint and notice in head
                    assert len(head) <= BOUND_WITH_NOTICE_AND_HINT, (mode, count, len(head), head)
                    assert head.isascii()
            c.abort()


def test_the_header_shows_the_effective_snapping_state_for_all_four_combinations():
    with addon():
        s = setup()
        for magnet in (False, True):
            for ctrl in (False, True):
                c = s.controller(snap_toggle=lambda m=magnet: m)
                s.move(c, px=FREE_A, ctrl=ctrl)
                head = s.headers[-1]
                state = "Snapping on" if magnet != ctrl else "Snapping off"
                other = "Snapping off" if magnet != ctrl else "Snapping on"
                assert head.startswith(state + " | "), (magnet, ctrl, head)
                assert other not in head
                c.abort()


def test_the_header_state_follows_ctrl_and_the_magnet_during_a_run():
    with addon():
        s = setup()
        state = {"on": False}
        c = s.controller(snap_toggle=lambda: state["on"])
        assert s.headers[-1].startswith("Snapping off | ")
        # Pressing and releasing Ctrl are modifier key events of their own.
        c.handle(ev('LEFT_CTRL', 'PRESS', px=FREE_A, ctrl=True))
        assert s.headers[-1].startswith("Snapping on | ")
        c.handle(ev('LEFT_CTRL', 'RELEASE', px=FREE_A))
        assert s.headers[-1].startswith("Snapping off | ")
        # The magnet flips between events and shows on the next one.
        state["on"] = True
        s.move(c, px=FREE_B)
        assert s.headers[-1].startswith("Snapping on | ")
        s.click(c, px=FREE_B, first=True)
        assert s.headers[-1].startswith("Snapping on | ")
        s.move(c, px=FREE_C, ctrl=True)
        assert s.headers[-1].startswith("Snapping off | ")
        state["on"] = False
        s.move(c, px=FREE_C, ctrl=True)
        assert s.headers[-1].startswith("Snapping on | ")
        assert len(s.headers[-1].split(" | ")[0]) <= 12
        c.abort()


def test_a_missing_scene_or_context_reads_the_magnet_as_off():
    import types
    on = types.SimpleNamespace(scene=types.SimpleNamespace(tool_settings=types.SimpleNamespace(use_snap=True)))
    assert draw_tool.magnet_on(on) is True
    assert draw_tool.magnet_on(types.SimpleNamespace(scene=None)) is False
    assert draw_tool.magnet_on(types.SimpleNamespace()) is False
    assert draw_tool.magnet_on(None) is False
    assert draw_tool.magnet_on(types.SimpleNamespace(scene=types.SimpleNamespace(tool_settings=None))) is False
    with addon():
        assert draw_tool.magnet_on(bpy.context) is False
        bpy.context.scene.tool_settings.use_snap = True
        assert draw_tool.magnet_on(bpy.context) is True
        bpy.context.scene.tool_settings.use_snap = False


def test_a_run_goes_on_when_the_live_reader_finds_no_scene():
    import types
    with addon():
        s = setup()
        c = s.controller(snap_toggle=lambda: draw_tool.magnet_on(types.SimpleNamespace(scene=None)))
        assert s.click(c, px=FREE_A, first=True) == 'RUNNING_MODAL'
        assert s.move(c, px=FREE_B) == 'RUNNING_MODAL'
        assert s.headers[-1].startswith("Snapping off | ")
        c.abort()


def test_a_run_started_with_ctrl_held_shows_the_effective_state_at_once():
    with addon():
        s = setup()
        for magnet in (False, True):
            for ctrl in (False, True):
                c = s.controller(snap_toggle=lambda m=magnet: m)
                s.click(c, px=FREE_A, first=True, ctrl=ctrl)
                state = "Snapping on" if magnet != ctrl else "Snapping off"
                assert s.headers[-1].startswith(state + " | "), (magnet, ctrl, s.headers[-1])
                c.abort()


def test_ctrl_pressed_outside_the_region_flips_the_planar_header_at_once():
    with addon():
        s = setup()
        c = s.controller(snap_toggle=lambda: False)
        s.click(c, px=FREE_A, first=True)
        outside = (-4.0, 10.0)
        assert s.headers[-1].startswith("Snapping off | ")
        assert c.handle(ev('LEFT_CTRL', 'PRESS', px=outside, ctrl=True)) == 'PASS_THROUGH'
        assert s.headers[-1].startswith("Snapping on | "), s.headers[-1]
        assert c.handle(ev('LEFT_CTRL', 'RELEASE', px=outside)) == 'PASS_THROUGH'
        assert s.headers[-1].startswith("Snapping off | "), s.headers[-1]
        # Any other pass-through event carries the state too, a navigation event included.
        assert c.handle(ev('WHEELUPMOUSE', px=outside, ctrl=True)) == 'PASS_THROUGH'
        assert s.headers[-1].startswith("Snapping on | "), s.headers[-1]
        assert c.handle(ev('MIDDLEMOUSE', px=outside)) == 'PASS_THROUGH'
        assert s.headers[-1].startswith("Snapping off | "), s.headers[-1]
        c.abort()


def test_the_planar_arc_tails_say_a_goes_to_the_other_arc():
    for mode in ALL_MODES:
        text = draw_logic.DrawSession(mode).header_text()
        tail = "A other arc" if mode in draw_logic.ARCS else "A arc"
        assert f"R rect, O circle, P poly, {tail}" in text, (mode, text)
