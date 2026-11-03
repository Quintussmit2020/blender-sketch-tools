"""sketch.subdivide (plan 8.8, DECISIONS C17): the pure split logic, then the operator on real edit
meshes, undo, the resolver's view of the result, the Extrude and Sweep outputs, the menus and the
Phase 4 guard."""

import types

import bmesh
import bpy

import sketch_tools
from sketch_tools.core import attributes, resolve, subdivide_logic
from sketch_tools.core.resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from sketch_tools.ops import draw_tool, modal_common, subdivide
from sketch_tools.tests.test_addon import (
    INNER, OUTER, SQUARE, _FakeLayout, _edit_sketch, addon, make_open_sketch, make_sketch, make_sweep,
    output_bm, signed_volume,
)
from sketch_tools.ui import menus, panel

MENUS = ("VIEW3D_MT_edit_mesh_edges", "VIEW3D_MT_edit_mesh_context_menu")

# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def test_split_points_are_evenly_spaced_and_exclude_the_ends():
    points = subdivide_logic.split_points((0.0, 0.0), (4.0, 2.0), 3)
    assert points == [(1.0, 0.5), (2.0, 1.0), (3.0, 1.5)]
    assert subdivide_logic.split_points((1.0, 1.0), (1.0, 3.0), 1) == [(1.0, 2.0)]
    assert subdivide_logic.split_points((0.0, 0.0), (1.0, 0.0), 0) == []
    # Direction matters: from b towards a the same points come out in the other order.
    forward = subdivide_logic.split_points((0.0, 0.0), (3.0, 0.0), 2)
    backward = subdivide_logic.split_points((3.0, 0.0), (0.0, 0.0), 2)
    assert backward == forward[::-1]


def test_new_vertex_count_in_count_mode_ignores_length_except_zero():
    count = subdivide_logic.new_vertex_count
    assert count(1.0, 'COUNT', 3, 0.5) == 3
    assert count(100.0, 'COUNT', 1, 0.5) == 1
    assert count(0.0, 'COUNT', 3, 0.5) == 0
    assert count(1e-7, 'COUNT', 3, 0.5) == 0


def test_new_vertex_count_in_length_mode_is_ceil_minus_one():
    count = subdivide_logic.new_vertex_count
    assert count(1.0, 'LENGTH', 1, 0.5) == 1, "exactly twice the maximum is two parts"
    assert count(1.0000001, 'LENGTH', 1, 0.5) == 1, "float noise must not add a part"
    assert count(1.2, 'LENGTH', 1, 0.5) == 2
    assert count(0.5, 'LENGTH', 1, 0.5) == 0, "already short enough"
    assert count(0.3, 'LENGTH', 1, 0.5) == 0
    assert count(0.0, 'LENGTH', 1, 0.5) == 0
    assert count(2.5, 'LENGTH', 1, 1.0) == 2


def test_length_mode_is_bounded_per_edge():
    parts = subdivide_logic.MAX_PARTS
    assert subdivide_logic.new_vertex_count(1000.0, 'LENGTH', 1, 0.001) == parts - 1
    assert subdivide_logic.was_clamped(1000.0, 0.001)
    assert not subdivide_logic.was_clamped(1.0, 0.5)
    assert not subdivide_logic.was_clamped(float(parts), 1.0)
    assert subdivide_logic.was_clamped(parts + 1.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    # Vertices created by hand have no index yet.
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    return bm


def _select(obj, indices):
    """Select exactly these vertices (vertex select mode flush, so edges between them select)."""
    bm = _bm(obj)
    for v in bm.verts:
        v.select_set(False)
    for i in indices:
        bm.verts[i].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def _select_all(obj):
    _select(obj, range(len(_bm(obj).verts)))


def _set_fillet(obj, index, radius):
    _select(obj, [index])
    attributes.write_fillet_radius(obj, radius)


def _snapshot(obj):
    """Coordinates, edges, radii and ids by vertex index, for before and after comparisons."""
    bm = _bm(obj)
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    return ([tuple(round(c, 6) for c in v.co) for v in bm.verts],
            sorted(tuple(sorted((e.verts[0].index, e.verts[1].index))) for e in bm.edges),
            [v[rad] for v in bm.verts], [v[ids] for v in bm.verts])


def _chains(obj):
    resolved = resolve.resolve_bmesh(_bm(obj), obj.sketch_tools.fillet_segments)
    return [(c.closed, len(c.points)) for c in resolved.chains], resolved.warnings


def _run(**kwargs):
    # The third argument asks for an undo push, as a button or menu click does.
    return bpy.ops.sketch.subdivide('EXEC_DEFAULT', True, **kwargs)


def _line_sketch(points):
    """Edit-mode sketch holding one open polyline."""
    obj = _edit_sketch()
    bm = _bm(obj)
    verts = [bm.verts.new((x, y, 0.0)) for x, y in points]
    for a, b in zip(verts, verts[1:]):
        bm.edges.new((a, b))
    bmesh.update_edit_mesh(obj.data)
    return obj


def _edge_ends(edge):
    return {edge.verts[0].index, edge.verts[1].index}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


def test_n_cuts_give_n_even_sharp_vertices_per_selected_edge():
    with addon():
        obj = _edit_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]])
        _set_fillet(obj, 2, 0.5)
        before = _snapshot(obj)
        fillet_id = before[3][2]
        assert fillet_id != 0
        # Edge 0-1 only.
        _select(obj, [0, 1])
        assert _run(mode='COUNT', cuts=3) == {'FINISHED'}
        bm = _bm(obj)
        rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
        ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
        assert len(bm.verts) == 7 and len(bm.edges) == 7
        new = [v for v in bm.verts if v.index >= 4]
        assert sorted(v.co.x for v in new) == [1.0, 2.0, 3.0]
        assert all(v.co.y == 0.0 and v.co.z == 0.0 for v in new)
        assert all(v[rad] == 0.0 and v[ids] == 0 for v in new)
        # Existing vertices keep position, radius and id; the other edges are the same edges.
        after = _snapshot(obj)
        assert after[0][:4] == before[0] and after[2][:4] == before[2] and after[3][:4] == before[3]
        assert after[2][2] == 0.5 and after[3][2] == fillet_id
        edges = set(after[1])
        assert {(1, 2), (2, 3), (0, 3)} <= edges
        assert (0, 1) not in edges
        # The cut edge is now a path 0 - 3 new vertices - 1 with four equal parts.
        lengths = [e.calc_length() for e in bm.edges if e.verts[0].index >= 4 or e.verts[1].index >= 4]
        assert len(lengths) == 4 and all(abs(length - 1.0) < 1e-5 for length in lengths)


def test_every_selected_edge_is_cut_and_unselected_edges_are_left_alone():
    with addon():
        obj = _edit_sketch([SQUARE])
        # Edge select mode: two opposite sides have all four vertices selected, yet the other two
        # sides are not selected and must stay as they are.
        settings = bpy.context.tool_settings
        settings.mesh_select_mode = (False, True, False)
        try:
            bm = _bm(obj)
            for v in bm.verts:
                v.select_set(False)
            for e in bm.edges:
                if _edge_ends(e) in ({0, 1}, {2, 3}):
                    e.select_set(True)
            bmesh.update_edit_mesh(obj.data)
            assert all(v.select for v in _bm(obj).verts)
            assert _run(cuts=2) == {'FINISHED'}
        finally:
            settings.mesh_select_mode = (True, False, False)
        bm = _bm(obj)
        assert len(bm.verts) == 4 + 4 and len(bm.edges) == 4 + 4
        untouched = [e for e in bm.edges if e.verts[0].index < 4 and e.verts[1].index < 4]
        assert sorted(tuple(sorted(_edge_ends(e))) for e in untouched) == [(0, 3), (1, 2)]


def test_selected_vertices_imply_their_edges():
    with addon():
        obj = _edit_sketch([SQUARE])
        bm = _bm(obj)
        # Raw vertex flags, no edge flush, as a script or another tool may leave them.
        for v in bm.verts:
            v.select = v.index in (1, 2)
        bmesh.update_edit_mesh(obj.data)
        assert _run(cuts=1) == {'FINISHED'}
        bm = _bm(obj)
        assert len(bm.verts) == 5
        assert tuple(bm.verts[4].co) == (1.0, 0.5, 0.0)
        # A lone selected vertex implies no edge.
        _select(obj, [0])
        assert _run(cuts=1) == {'CANCELLED'}
        assert len(_bm(obj).verts) == 5


def test_length_mode_gives_equal_segments_no_longer_than_the_maximum():
    with addon():
        obj = _line_sketch([(0, 0), (2.5, 0), (2.5, 1.0), (4.0, 1.0)])
        _select_all(obj)
        assert _run(mode='LENGTH', max_length=1.0) == {'FINISHED'}
        bm = _bm(obj)
        # 2.5 -> 3 parts, 1.0 -> left alone, 1.5 -> 2 parts.
        assert len(bm.verts) == 4 + 2 + 0 + 1
        assert all(e.calc_length() <= 1.0 + 1e-5 for e in bm.edges)
        assert sorted(round(e.calc_length(), 5) for e in bm.edges) == sorted(
            [round(2.5 / 3, 5)] * 3 + [1.0] + [0.75] * 2)
        # Exactly twice the maximum is two parts.
        other = _line_sketch([(10, 0), (12, 0)])
        _select(other, [0, 1])
        assert _run(mode='LENGTH', max_length=1.0) == {'FINISHED'}
        assert len(_bm(other).verts) == 3


def test_length_mode_with_nothing_long_enough_changes_nothing():
    with addon():
        obj = _line_sketch([(0, 0), (1, 0)])
        _select_all(obj)
        before = _snapshot(obj)
        assert _run(mode='LENGTH', max_length=5.0) == {'CANCELLED'}
        assert _snapshot(obj) == before


def test_long_edge_in_length_mode_is_limited():
    with addon():
        obj = _line_sketch([(0, 0), (500, 0)])
        _select_all(obj)
        assert _run(mode='LENGTH', max_length=0.001) == {'FINISHED'}
        assert len(_bm(obj).verts) == 2 + subdivide_logic.MAX_PARTS - 1


def test_only_the_new_vertices_are_selected():
    with addon():
        obj = _edit_sketch([SQUARE])
        _select_all(obj)
        assert _run(cuts=2) == {'FINISHED'}
        bm = _bm(obj)
        assert {v.index for v in bm.verts if v.select} == set(range(4, 12))
        assert not any(f.select for f in bm.faces)
        # An edge is selected only when both its vertices are new: the pair on each side.
        selected_edges = [e for e in bm.edges if e.select]
        assert len(selected_edges) == 4
        assert all(min(_edge_ends(e)) >= 4 for e in selected_edges)
        assert not list(bm.select_history)


def test_selection_ends_in_vertex_mode_when_started_in_edge_mode():
    with addon():
        obj = _edit_sketch([SQUARE])
        settings = bpy.context.tool_settings
        settings.mesh_select_mode = (False, True, False)
        try:
            bm = _bm(obj)
            for e in bm.edges:
                e.select_set(_edge_ends(e) == {0, 1})
            bmesh.update_edit_mesh(obj.data)
            assert _run(cuts=1) == {'FINISHED'}
            assert tuple(settings.mesh_select_mode) == (True, False, False)
            bm = _bm(obj)
            assert len(bm.verts) == 5 and [v.index for v in bm.verts if v.select] == [4]
        finally:
            settings.mesh_select_mode = (True, False, False)


def test_zero_length_and_hidden_edges_are_skipped():
    with addon():
        obj = _edit_sketch()
        bm = _bm(obj)
        a, b, c, d, e_, f, g = (bm.verts.new(p) for p in
                                [(0, 0, 0), (0, 0, 0), (2, 0, 0), (5, 0, 0), (5, 3, 0), (8, 0, 0), (8, 4, 0)])
        bm.edges.new((a, b))
        bm.edges.new((b, c))
        bm.edges.new((d, e_))
        hidden = bm.edges.new((f, g))
        for v in bm.verts:
            v.select_set(True)
        # Hiding an edge hides its vertices too.
        hidden.hide_set(True)
        bmesh.update_edit_mesh(obj.data)
        assert _run(cuts=1) == {'FINISHED'}
        bm = _bm(obj)
        # Two edges were cut (b-c and d-e); the zero-length one and the hidden one were not.
        assert len(bm.verts) == 7 + 2 and len(bm.edges) == 4 + 2
        hidden_edges = [e for e in bm.edges if e.hide]
        assert len(hidden_edges) == 1 and _edge_ends(hidden_edges[0]) == {5, 6}
        zero = [e for e in bm.edges if _edge_ends(e) == {0, 1}]
        assert len(zero) == 1 and zero[0].calc_length() == 0.0

        # Only a zero-length edge selected: nothing happens, nothing is changed.
        other = _edit_sketch()
        bm = _bm(other)
        p, q = bm.verts.new((1, 1, 0)), bm.verts.new((1, 1, 0))
        bm.edges.new((p, q))
        bmesh.update_edit_mesh(other.data)
        _select_all(other)
        before = _snapshot(other)
        assert _run(cuts=2) == {'CANCELLED'}
        assert _snapshot(other) == before


def test_off_plane_end_points_still_give_new_points_on_the_plane():
    with addon():
        obj = _edit_sketch()
        bm = _bm(obj)
        bm.edges.new((bm.verts.new((0, 0, 0.5)), bm.verts.new((4, 0, -0.5))))
        bmesh.update_edit_mesh(obj.data)
        _select_all(obj)
        assert _run(cuts=1) == {'FINISHED'}
        assert tuple(_bm(obj).verts[2].co) == (2.0, 0.0, 0.0)


def test_neighbouring_fillets_do_not_leak_onto_new_vertices():
    with addon():
        obj = _edit_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]])
        _set_fillet(obj, 1, 0.5)
        _set_fillet(obj, 0, 0.25)
        _select(obj, [0, 1])
        assert _run(cuts=3) == {'FINISHED'}
        bm = _bm(obj)
        rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
        ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
        # Blender's own subdivide would interpolate 0.25 and 0.5 onto these.
        assert [v[rad] for v in bm.verts if v.index >= 4] == [0.0, 0.0, 0.0]
        assert [v[ids] for v in bm.verts if v.index >= 4] == [0, 0, 0]
        assert bm.verts[0][rad] == 0.25 and bm.verts[1][rad] == 0.5
        assert bm.verts[0][ids] != 0 and bm.verts[1][ids] != 0


def test_operator_is_one_undo_step():
    with addon():
        obj = _edit_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]])
        _set_fillet(obj, 0, 0.5)
        _select_all(obj)
        bpy.ops.ed.undo_push(message="before subdivide")
        before = _snapshot(obj)
        selection = {v.index for v in _bm(obj).verts if v.select}
        assert _run(cuts=2) == {'FINISHED'}
        assert len(_bm(obj).verts) == 4 + 8
        assert bpy.ops.ed.undo() == {'FINISHED'}
        obj = bpy.context.active_object
        assert obj.mode == 'EDIT'
        assert _snapshot(obj) == before, "one undo did not restore the sketch exactly"
        assert {v.index for v in _bm(obj).verts if v.select} == selection
        assert bpy.ops.ed.redo() == {'FINISHED'}
        assert len(_bm(bpy.context.active_object).verts) == 4 + 8


def test_redo_panel_draws_only_the_relevant_option():
    drawn = []

    class Layout(_FakeLayout):
        def prop(self, data, name, **_kwargs):
            drawn.append(name)

    shell = types.SimpleNamespace(layout=Layout(), mode='COUNT')
    subdivide.SKETCH_OT_subdivide.draw(shell, None)
    assert drawn == ["mode", "cuts"]
    drawn.clear()
    shell.mode = 'LENGTH'
    subdivide.SKETCH_OT_subdivide.draw(shell, None)
    assert drawn == ["mode", "max_length"]


def test_properties_and_options():
    with addon():
        props = bpy.ops.sketch.subdivide.get_rna_type().properties
        assert props["cuts"].default == 1 and props["cuts"].hard_min == 1 and props["cuts"].hard_max == 100
        assert props["mode"].default == 'COUNT'
        assert [item.identifier for item in props["mode"].enum_items] == ['COUNT', 'LENGTH']
        assert props["max_length"].subtype == 'DISTANCE'
        assert 0.0 < props["max_length"].hard_min < props["max_length"].default
        assert subdivide.SKETCH_OT_subdivide.bl_options == {'REGISTER', 'UNDO'}


# ---------------------------------------------------------------------------
# Poll and the Phase 4 guard
# ---------------------------------------------------------------------------


def test_poll_needs_edit_mode_and_an_active_sketch():
    with addon():
        obj = _edit_sketch([SQUARE])
        assert bpy.ops.sketch.subdivide.poll()
        bpy.ops.object.mode_set(mode='OBJECT')
        assert not bpy.ops.sketch.subdivide.poll()
        bpy.ops.object.mode_set(mode='EDIT')
        assert bpy.ops.sketch.subdivide.poll()

        # An ordinary mesh in edit mode.
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        assert not cube.sketch_tools.is_sketch
        bpy.ops.object.mode_set(mode='EDIT')
        assert not bpy.ops.sketch.subdivide.poll()
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.context.view_layer.objects.active = None
        assert not bpy.ops.sketch.subdivide.poll()
        bpy.context.view_layer.objects.active = obj


def test_refused_while_any_sketch_modal_is_live():
    with addon():
        obj = _edit_sketch([SQUARE])
        _select_all(obj)
        before = _snapshot(obj)
        for module in (draw_tool, sketch_tools.ops.fillet, sketch_tools.ops.trim):
            token = object()
            module._live.add(token)
            try:
                assert modal_common.any_sketch_modal_live()
                assert _run(cuts=1) == {'CANCELLED'}
                assert _snapshot(obj) == before
                reports = []
                shell = types.SimpleNamespace(mode='COUNT', cuts=1, max_length=0.5,
                                              report=lambda level, text: reports.append((level, text)))
                ctx = types.SimpleNamespace(active_object=obj, tool_settings=bpy.context.tool_settings)
                assert subdivide.SKETCH_OT_subdivide.execute(shell, ctx) == {'CANCELLED'}
                assert reports == [({'WARNING'}, modal_common.BUSY_MESSAGE)]
            finally:
                module._live.discard(token)
        assert not modal_common.any_sketch_modal_live()
        assert _run(cuts=1) == {'FINISHED'}


# ---------------------------------------------------------------------------
# The resolver and the outputs
# ---------------------------------------------------------------------------


def test_resolver_sees_the_same_chains_before_and_after():
    with addon():
        # A closed rectangle with a line hanging off one corner (a junction), a closed triangle,
        # and a separate open polyline.
        obj = _edit_sketch([[(0, 0), (4, 0), (4, 3), (0, 3)], [(10, 0), (13, 0), (11, 2)]])
        bm = _bm(obj)
        first = bm.verts[0]
        tail = bm.verts.new((-2.0, -2.0, 0.0))
        bm.edges.new((first, tail))
        a, b, c = (bm.verts.new(p) for p in [(20, 0, 0), (22, 1, 0), (24, 0, 0)])
        bm.edges.new((a, b))
        bm.edges.new((b, c))
        bmesh.update_edit_mesh(obj.data)
        before, warnings_before = _chains(obj)
        assert sorted(closed for closed, _n in before) == [False, False, True, True]
        edge_count = len(_bm(obj).edges)
        _select_all(obj)
        assert _run(cuts=2) == {'FINISHED'}
        after, warnings_after = _chains(obj)
        assert [closed for closed, _n in after] == [closed for closed, _n in before]
        # Every chain gained 2 points per edge.
        assert sum(n for _c, n in after) == sum(n for _c, n in before) + 2 * edge_count
        assert warnings_after == warnings_before == []
        # The junction is still a junction.
        assert len(_bm(obj).verts[0].link_edges) == 3


def test_closed_loop_stays_one_closed_loop_when_cut_in_pieces():
    with addon():
        obj = _edit_sketch([SQUARE])
        for indices in ([0, 1], [1, 2], [2, 3]):
            _select(obj, indices)
            assert _run(cuts=1) == {'FINISHED'}
        assert _chains(obj)[0] == [(True, 4 + 3)]


def test_extrude_output_matches_the_unsubdivided_one_with_filleted_corners():
    with addon():
        sketch = make_sketch([[(0, 0), (4, 0), (4, 4), (0, 4)]], mode='EXTRUDE')
        sketch.sketch_tools.fillet_segments = 6
        bpy.ops.object.mode_set(mode='EDIT')
        _set_fillet(sketch, 0, 0.5)
        _set_fillet(sketch, 2, 0.75)
        bpy.ops.object.mode_set(mode='OBJECT')
        bm = output_bm(sketch)
        volume = signed_volume(bm)
        faces_before = len(bm.faces)
        bm.free()
        radii = [d.value for d in sketch.data.attributes[FILLET_RADIUS_ATTR].data]
        ids = [d.value for d in sketch.data.attributes[FILLET_ID_ATTR].data]
        assert radii == [0.5, 0.0, 0.75, 0.0] and ids[0] != 0 and ids[2] != 0

        bpy.ops.object.mode_set(mode='EDIT')
        _select_all(sketch)
        assert _run(cuts=1) == {'FINISHED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        radii_after = [d.value for d in sketch.data.attributes[FILLET_RADIUS_ATTR].data]
        ids_after = [d.value for d in sketch.data.attributes[FILLET_ID_ATTR].data]
        assert radii_after[:4] == radii and ids_after[:4] == ids, "existing fillets changed"
        assert radii_after[4:] == [0.0] * 4 and ids_after[4:] == [0] * 4
        assert sketch.sketch_tools.warnings == ""
        bm = output_bm(sketch)
        try:
            assert abs(signed_volume(bm) - volume) < 1e-4, (signed_volume(bm), volume)
            assert len(bm.faces) > faces_before
        finally:
            bm.free()


def test_extrude_hole_loop_can_be_subdivided():
    with addon():
        sketch = make_sketch([OUTER, INNER], mode='EXTRUDE')
        bm = output_bm(sketch)
        volume = signed_volume(bm)
        bm.free()
        assert abs(volume - 24.0) < 1e-4
        bpy.ops.object.mode_set(mode='EDIT')
        # Only the hole loop (vertices 4 to 7).
        _select(sketch, [4, 5, 6, 7])
        assert _run(cuts=3) == {'FINISHED'}
        assert len(_bm(sketch).verts) == 8 + 12
        chains, warnings = _chains(sketch)
        assert sorted(chains) == [(True, 4), (True, 16)] and warnings == []
        bpy.ops.object.mode_set(mode='OBJECT')
        assert sketch.sketch_tools.warnings == ""
        bm = output_bm(sketch)
        try:
            assert abs(signed_volume(bm) - volume) < 1e-4
            # Still a torus: the hole is still a hole.
            assert len(bm.verts) - len(bm.edges) + len(bm.faces) == 0
        finally:
            bm.free()


def test_sweep_path_closed_and_open_can_be_subdivided():
    with addon():
        closed = make_sweep(make_sketch([SQUARE]), sw_profile='SQUARE')
        bm = output_bm(closed)
        closed_volume = signed_volume(bm)
        bm.free()
        assert abs(closed_volume - 0.16) < 1e-5
        bpy.ops.object.mode_set(mode='EDIT')
        _select_all(closed)
        assert _run(cuts=2) == {'FINISHED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        assert closed.sketch_tools.warnings == ""
        bm = output_bm(closed)
        try:
            assert abs(signed_volume(bm) - closed_volume) < 1e-5, signed_volume(bm)
            assert all(len(e.link_faces) == 2 for e in bm.edges)
        finally:
            bm.free()

        opened = make_sweep(make_open_sketch([(0, 0), (2, 0), (2, 2)]), sw_profile='SQUARE', sw_cap_ends=True)
        bm = output_bm(opened)
        open_volume = signed_volume(bm)
        bm.free()
        bpy.ops.object.mode_set(mode='EDIT')
        _select_all(opened)
        assert _run(mode='LENGTH', max_length=0.5) == {'FINISHED'}
        assert len(_bm(opened).verts) == 3 + 2 * 3
        bpy.ops.object.mode_set(mode='OBJECT')
        assert opened.sketch_tools.warnings == ""
        bm = output_bm(opened)
        try:
            assert abs(signed_volume(bm) - open_volume) < 1e-5, (signed_volume(bm), open_volume)
            assert all(len(e.link_faces) == 2 for e in bm.edges)
        finally:
            bm.free()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class _RecordingLayout(_FakeLayout):
    def __init__(self, operators=None):
        super().__init__()
        self.operators = [] if operators is None else operators

    def _child(self, *_args, **_kwargs):
        return _RecordingLayout(self.operators)

    column = row = box = _child

    def operator(self, idname, **kwargs):
        self.operators.append(idname)
        return super().operator(idname, **kwargs)


def test_menu_entries_appear_only_for_sketches():
    with addon():
        sketch = make_sketch([SQUARE])
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        for obj, expected in ((sketch, ["sketch.subdivide"]), (cube, []), (None, [])):
            layout = _RecordingLayout()
            menus._draw_subdivide(types.SimpleNamespace(layout=layout), types.SimpleNamespace(active_object=obj))
            assert layout.operators == expected, (obj, layout.operators)
        for name in MENUS:
            assert menus._draw_subdivide in getattr(bpy.types, name).draw._draw_funcs, name


def test_menu_entries_are_removed_on_unregister_and_not_doubled():
    def appended():
        return [name for name in MENUS if menus._draw_subdivide in getattr(bpy.types, name).draw._draw_funcs]

    for _ in range(2):
        with addon():
            assert len(appended()) == 2
            for name in MENUS:
                assert getattr(bpy.types, name).draw._draw_funcs.count(menus._draw_subdivide) == 1
        assert appended() == []


def test_panel_offers_subdivide_in_edit_mode_only():
    with addon():
        sketch = make_sketch([SQUARE])
        for mode, expected in (('OBJECT', False), ('EDIT_MESH', True)):
            ctx = types.SimpleNamespace(mode=mode, active_object=sketch, window_manager=bpy.context.window_manager)
            layout = _RecordingLayout()
            panel.SKETCH_PT_sketch.draw(types.SimpleNamespace(layout=layout), ctx)
            assert ("sketch.subdivide" in layout.operators) == expected, (mode, layout.operators)
