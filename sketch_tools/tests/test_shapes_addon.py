"""Shape options of the planar Sketch tool inside Blender (C21): the WindowManager enum, the tool
header, the operator start and the controller applying arcs to a real edit mesh."""

import math
import types

import bmesh
import bpy

from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.tests.test_addon import (
    _FakeLayout, _click_at, _edit_sketch, _event, _make_controller, _mesh_state, _operator_shell, addon,
)

SHAPES = ('POLYLINE', 'RECTANGLE', 'CIRCLE', 'ARC_3PT', 'ARC_CENTER')


def verts_of(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return [(v.co.x, v.co.y, v.co.z) for v in bm.verts]


def chain_ends(obj):
    """Vertices with one edge (an open chain has exactly two)."""
    bm = bmesh.from_edit_mesh(obj.data)
    return [v for v in bm.verts if len(v.link_edges) == 1]


def test_shape_enum_is_registered_with_the_five_shapes_and_removed_again():
    assert not hasattr(bpy.types.WindowManager, "sketch_tools_shape")
    with addon():
        prop = bpy.types.WindowManager.bl_rna.properties["sketch_tools_shape"]
        assert [item.identifier for item in prop.enum_items] == list(SHAPES)
        assert prop.default == 'POLYLINE'
        assert bpy.context.window_manager.sketch_tools_shape == 'POLYLINE'
        bpy.context.window_manager.sketch_tools_shape = 'ARC_CENTER'
        assert bpy.context.window_manager.sketch_tools_shape == 'ARC_CENTER'
    assert not hasattr(bpy.types.WindowManager, "sketch_tools_shape")


def test_draw_operator_accepts_every_shape_and_the_header_choice():
    with addon():
        items = [item.identifier for item in bpy.ops.sketch.draw.get_rna_type().properties["mode"].enum_items]
        assert items == [*SHAPES, 'SETTING']


def test_tool_header_shows_the_shape_enum_and_the_lmb_item_follows_it():
    from sketch_tools.ui import tool

    calls, labels = [], []
    layout = types.SimpleNamespace(prop=lambda *a, **k: calls.append((a, k)), label=lambda text: labels.append(text))
    wm = types.SimpleNamespace()
    tool.SKETCH_TOOL_sketch.draw_settings(types.SimpleNamespace(window_manager=wm), layout, None)
    assert len(calls) == 1 and calls[0][0] == (wm, "sketch_tools_shape")
    assert calls[0][1].get("expand") is True
    assert len(labels) == 1 and "hold Ctrl to flip snapping to points" in labels[0]
    # Only R and O start a shape when idle (idle A is select all, C21); P and A work inside a run.
    assert labels[0].startswith("R rectangle, O circle; while drawing also P polyline, A arc;")
    lmb = tool.SKETCH_TOOL_sketch.bl_keymap[0]
    assert lmb[0] == "sketch.draw" and dict(lmb[2]["properties"]) == {"mode": 'SETTING'}
    # R and O name their shapes and stay modifier-free.
    assert [dict(i[2]["properties"])["mode"] for i in tool.SKETCH_TOOL_sketch.bl_keymap if i[1]["type"] in 'RO'] == [
        'RECTANGLE', 'CIRCLE']
    # No idle A item: Blender's A (select all) keeps working while no run is active.
    assert all(item[1]["type"] != 'A' for item in tool.SKETCH_TOOL_sketch.bl_keymap)


class _PickerLayout(_FakeLayout):
    """Records the shape picker's prop calls and every label."""

    def __init__(self, picks=None, labels=None, kind="root"):
        super().__init__()
        self.picks = [] if picks is None else picks
        self.labels = [] if labels is None else labels
        self.kind = kind

    def column(self, *_args, **_kwargs):
        return _PickerLayout(self.picks, self.labels, "column")

    def row(self, *_args, **_kwargs):
        return _PickerLayout(self.picks, self.labels, "row")

    def box(self, *_args, **_kwargs):
        return _PickerLayout(self.picks, self.labels, "box")

    def prop(self, data, name, **kwargs):
        super().prop(data, name, **kwargs)
        if name == "sketch_tools_shape":
            # The layout kind matters: expand in a column stacks the items vertically.
            self.picks.append({**kwargs, "layout_kind": self.kind})

    def label(self, **kwargs):
        self.labels.append(kwargs.get("text", ""))


def draw_sketch_panel(obj, mode):
    from sketch_tools.ui import panel

    layout = _PickerLayout()
    ctx = types.SimpleNamespace(mode=mode, active_object=obj, window_manager=bpy.context.window_manager)
    panel.SKETCH_PT_sketch.draw(types.SimpleNamespace(layout=layout), ctx)
    return layout


def test_the_sketch_panel_shows_the_shape_picker_in_edit_mode_for_both_kinds():
    from sketch_tools.core import build
    from sketch_tools.tests.test_path3d_addon import make_path

    with addon():
        path = make_path()
        planar = _edit_sketch()
        for obj in (planar, path):
            layout = draw_sketch_panel(obj, 'EDIT_MESH')
            assert len(layout.picks) == 1, (obj.name, layout.picks)
            # The same icon row as the tool header.
            assert layout.picks[0].get("expand") is True and layout.picks[0].get("icon_only") is True
            assert layout.picks[0].get("text") == ""
            assert layout.picks[0]["layout_kind"] == "row", layout.picks
            assert "Shape" in layout.labels and "Click in the viewport with the Sketch tool" in layout.labels
            assert not any(chr(0x2014) in text for text in layout.labels)
            # Object mode cannot draw, so the enum is not offered.
            assert draw_sketch_panel(obj, 'OBJECT').picks == []
        assert build.is_sketch(planar) and build.is_path_3d(path)


def test_changing_the_picker_in_the_panel_changes_the_shape_the_next_run_starts_with():
    with addon():
        wm = bpy.context.window_manager
        obj = _edit_sketch()
        layout = draw_sketch_panel(obj, 'EDIT_MESH')
        assert len(layout.picks) == 1
        # The picker is a plain prop on the WindowManager enum, so setting it is what a click does.
        for shape in SHAPES:
            wm.sketch_tools_shape = shape
            assert start_operator(obj, shape) == shape
        wm.sketch_tools_shape = 'POLYLINE'


def start_operator(obj, shape, mode='SETTING'):
    from sketch_tools.ops.draw_tool import SKETCH_OT_draw

    shell, ctx, headers, handlers, restore = _operator_shell(obj, mode=mode)
    ctx.window_manager.sketch_tools_shape = shape
    try:
        assert SKETCH_OT_draw.invoke(shell, ctx, _event('LEFTMOUSE', at=(0, 0))) == {'RUNNING_MODAL'}
        return shell._controller.session.mode
    finally:
        SKETCH_OT_draw.cancel(shell, ctx)
        restore()


def test_the_lmb_run_starts_with_the_header_shape_and_explicit_modes_ignore_it():
    with addon():
        for shape in SHAPES:
            obj = _edit_sketch()
            assert start_operator(obj, shape) == shape
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(obj)
        obj = _edit_sketch()
        assert start_operator(obj, 'ARC_CENTER', mode='RECTANGLE') == 'RECTANGLE'
        assert start_operator(obj, 'CIRCLE', mode='POLYLINE') == 'POLYLINE'


def test_a_key_in_a_run_switches_the_arc_and_ctrl_a_is_left_alone():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, 'POLYLINE')
        assert c.handle(_event('A', at=(0, 0))) == 'RUNNING_MODAL'
        assert c.session.mode == 'ARC_3PT'
        assert c.handle(_event('A', at=(0, 0))) == 'RUNNING_MODAL'
        assert c.session.mode == 'ARC_CENTER'
        c.handle(_event('A', at=(0, 0), ctrl=True))
        assert c.session.mode == 'ARC_CENTER'
        c.abort()


def test_three_point_arc_is_written_as_one_sharp_open_chain():
    with addon():
        obj = _edit_sketch()
        c, editor, headers, _ = _make_controller(obj, 'ARC_3PT', segments=32)
        assert c.start(_event('LEFTMOUSE', at=(1, 0))) == 'RUNNING_MODAL'
        assert _click_at(c, 0.7071, 0.7071) == 'RUNNING_MODAL'
        # Nothing is written until the end exists.
        assert len(bmesh.from_edit_mesh(obj.data).verts) == 0
        assert _click_at(c, 0, 1) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 9 and len(bm.edges) == 8
        assert len(chain_ends(obj)) == 2
        rad = bm.verts.layers.float[FILLET_RADIUS_ATTR]
        ids = bm.verts.layers.int[FILLET_ID_ATTR]
        assert all(v[rad] == 0.0 and v[ids] == 0 and v.co.z == 0.0 for v in bm.verts)
        assert all(abs(math.hypot(v.co.x, v.co.y) - 1.0) < 1e-3 for v in bm.verts)
        assert headers[-1] is None and c.closed


def test_centre_arc_is_written_from_the_start_counter_clockwise():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, 'ARC_CENTER', segments=32)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        _click_at(c, 2, 0)
        assert _click_at(c, 0, 3) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 9 and len(bm.edges) == 8
        ends = {(round(v.co.x, 4), round(v.co.y, 4)) for v in chain_ends(obj)}
        assert ends == {(2.0, 0.0), (0.0, 2.0)}
        assert all(abs(math.hypot(v.co.x, v.co.y) - 2.0) < 1e-4 for v in bm.verts)
        assert all(v.co.x > -1e-6 and v.co.y > -1e-6 for v in bm.verts)


def test_arcs_join_existing_vertices_under_ctrl_snap():
    with addon():
        obj = _edit_sketch([[(0, 0), (1, 0), (1, 1), (0, 1)]])
        before = len(bmesh.from_edit_mesh(obj.data).verts)
        c, *_ = _make_controller(obj, 'ARC_3PT')
        c.start(_event('LEFTMOUSE', at=(1, 0), ctrl=True))
        _click_at(c, 1.5, 0.5)
        assert _click_at(c, 1, 1, ctrl=True) == 'FINISHED'
        bm = bmesh.from_edit_mesh(obj.data)
        # The arc starts and ends on the square's corners, so only its inner vertices are new.
        assert len(bm.verts) - before == 15
        corners = [v for v in bm.verts if (round(v.co.x, 5), round(v.co.y, 5)) in ((1.0, 0.0), (1.0, 1.0))]
        assert len(corners) == 2 and all(len(v.link_edges) == 3 for v in corners)


def test_escape_cancels_an_arc_at_every_stage_and_leaves_the_mesh_alone():
    with addon():
        for shape in ('ARC_3PT', 'ARC_CENTER'):
            for clicks in (1, 2):
                obj = _edit_sketch([[(5, 5), (6, 5), (6, 6)]])
                before = _mesh_state(obj)
                c, *_ = _make_controller(obj, shape)
                c.start(_event('LEFTMOUSE', at=(0, 0)))
                if clicks == 2:
                    _click_at(c, 1, 1)
                assert c.handle(_event('ESC')) == 'CANCELLED', (shape, clicks)
                assert _mesh_state(obj) == before and c.closed
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.data.objects.remove(obj)


def test_centre_arc_typed_radius_and_angle_reach_the_mesh():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, 'ARC_CENTER', segments=32)
        c.start(_event('LEFTMOUSE', at=(0, 0)))
        c.handle(_event('MOUSEMOVE', at=(3, 0)))
        for key in ('ONE', 'PERIOD', 'FIVE'):
            c.handle(_event(key))
        assert c.handle(_event('RET')) == 'RUNNING_MODAL'
        for key in ('NINE', 'ZERO'):
            c.handle(_event(key))
        assert c.handle(_event('RET')) == 'FINISHED'
        ends = {(round(v.co.x, 4), round(v.co.y, 4)) for v in chain_ends(obj)}
        assert ends == {(1.5, 0.0), (0.0, 1.5)}


def test_the_overlay_snapshot_carries_the_arc_preview():
    with addon():
        obj = _edit_sketch()
        c, *_ = _make_controller(obj, 'ARC_3PT')
        c.start(_event('LEFTMOUSE', at=(1, 0)))
        _click_at(c, 0.7071, 0.7071)
        c.handle(_event('MOUSEMOVE', at=(0, 1)))
        preview, mouse, to_screen = c.snapshot()
        assert len(preview.strips) == 1 and len(preview.strips[0]) > 3
        assert preview.strips[0][-1] == (0.0, 1.0)
        c.abort()
        assert c.snapshot() is None
