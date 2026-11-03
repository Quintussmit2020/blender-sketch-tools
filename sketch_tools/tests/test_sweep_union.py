"""Boolean union sweep option (DECISIONS C22), headless.

The reference for "has intersecting parts" and "is merged" is a BVH self overlap over faces that
share no vertex, so the tests do not trust the builder or the boolean to grade themselves."""

import io
import math
import types
from contextlib import redirect_stdout

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

from sketch_tools import props
from sketch_tools.core import build

from .test_addon import (
    SQUARE,
    _FakeLayout,
    _RecordingLayout,
    addon,
    assert_closed_and_consistent,
    count_rebuilds,
    make_open_sketch,
    make_sketch,
    make_sweep,
    only_event_handlers,
    output_bm,
    signed_volume,
)
from .test_path3d_addon import make_path

BOWTIE_Z = (0.0, 0.0, 1.0)


def figure_eight(n=49, a=2.0, b=1.5):
    """Odd n: the two passes through the middle cross inside an edge. Even n: both pass through
    a vertex at the origin, which the resolver welds into a four-way junction."""
    return [(a * math.sin(2.0 * math.pi * k / n), b * math.sin(4.0 * math.pi * k / n)) for k in range(n)]


def datablock_counts():
    return tuple(len(c) for c in (bpy.data.scenes, bpy.data.objects, bpy.data.meshes, bpy.data.collections))


def temp_leftovers():
    names = []
    for collection in (bpy.data.scenes, bpy.data.objects, bpy.data.meshes, bpy.data.collections):
        names.extend(d.name for d in collection if d.name.startswith(build.UNION_TEMP_NAME))
    return names


def crossing_pairs(bm):
    """Face pairs that share no vertex and still intersect."""
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    tree = BVHTree.FromBMesh(bm)
    return sum(
        1 for a, b in tree.overlap(tree)
        if a < b and set(bm.faces[a].verts).isdisjoint(bm.faces[b].verts)
    )


def shell_count(bm):
    seen = set()
    count = 0
    for start in bm.verts:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            vert = stack.pop()
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return count


def measure(sketch):
    bm = output_bm(sketch)
    try:
        return {
            "faces": len(bm.faces),
            "verts": len(bm.verts),
            "volume": signed_volume(bm),
            "crossings": crossing_pairs(bm),
            "shells": shell_count(bm),
            "closed": all(len(e.link_faces) == 2 for e in bm.edges),
            "coords": sorted(tuple(round(c, 5) for c in v.co) for v in bm.verts),
        }
    finally:
        bm.free()


def assert_no_union_warning(sketch):
    lines = sketch.sketch_tools.warnings.split("\n")
    assert build.UNION_NOT_SOLID not in lines and build.UNION_FAILED not in lines, lines


def set_union(sketch, value):
    sketch.sketch_tools.sw_union = value


def assert_unioned(off, on):
    assert off["crossings"] > 0, "the un-merged sweep should have intersecting parts"
    assert on["crossings"] == 0, f"{on['crossings']} face pairs still intersect"
    assert on["closed"] and on["shells"] == 1, on
    assert 0.0 < on["volume"] < off["volume"] - 1e-4, (on["volume"], off["volume"])


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _figure_eight_case(profile, smooth=False):
    with addon():
        sketch = make_sketch([figure_eight()])
        make_sweep(sketch, sw_profile=profile, sw_smooth=smooth, sw_radius=0.15, sw_width=0.3, sw_height=0.3)
        off = measure(sketch)
        assert sketch.sketch_tools.warnings == ""
        before = datablock_counts()
        set_union(sketch, True)
        assert sketch.sketch_tools.warnings == "", sketch.sketch_tools.warnings
        on = measure(sketch)
        assert datablock_counts() == before and not temp_leftovers()
        assert_unioned(off, on)
        out = sketch.sketch_tools.output_object
        assert len(out.modifiers) == 0 and out.type == 'MESH'
        bm = output_bm(sketch)
        try:
            assert_closed_and_consistent(bm)
        finally:
            bm.free()
        return sketch, off, on


def test_figure_eight_square_is_merged_into_one_manifold_shell():
    _figure_eight_case('SQUARE')


def test_figure_eight_cylinder_is_merged_into_one_manifold_shell():
    _figure_eight_case('CYLINDER')


def test_figure_eight_through_a_welded_junction_is_merged_too():
    with addon():
        sketch = make_sketch([figure_eight(48)])
        make_sweep(sketch, sw_profile='SQUARE', sw_width=0.3, sw_height=0.3)
        off = measure(sketch)
        set_union(sketch, True)
        assert sketch.sketch_tools.warnings == ""
        assert_unioned(off, measure(sketch))


def test_figure_eight_union_off_again_restores_the_unmerged_sweep():
    with addon():
        sketch = make_sketch([figure_eight()])
        make_sweep(sketch, sw_profile='SQUARE')
        off = measure(sketch)
        set_union(sketch, True)
        assert measure(sketch)["coords"] != off["coords"]
        set_union(sketch, False)
        assert measure(sketch) == off


def test_union_property_exists_with_a_tooltip_and_defaults_off():
    with addon():
        prop = props.SketchToolsProps.bl_rna.properties["sw_union"]
        assert prop.type == 'BOOLEAN' and prop.default is False and prop.description
        sketch = make_sketch([SQUARE])
        assert sketch.sketch_tools.sw_union is False


def test_two_crossing_chains_become_one_shell_with_the_exact_volume():
    with addon():
        sketch = make_sketch([])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        for a, b in (((-1.0, 0.0), (1.0, 0.0)), ((0.0, -1.0), (0.0, 1.0))):
            bm.edges.new((bm.verts.new((*a, 0.0)), bm.verts.new((*b, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='SQUARE', sw_width=0.2, sw_height=0.2)
        off = measure(sketch)
        assert off["shells"] == 2 and off["crossings"] > 0
        assert abs(off["volume"] - 2 * 0.08) < 1e-5
        set_union(sketch, True)
        on = measure(sketch)
        assert_unioned(off, on)
        # Two bars of 2 x 0.2 x 0.2 sharing a 0.2 x 0.2 x 0.2 cube.
        assert abs(on["volume"] - (2 * 0.08 - 0.008)) < 1e-5, on["volume"]


def test_two_disjoint_chains_stay_two_shells_untouched():
    with addon():
        sketch = make_sketch([])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        for y in (0.0, 3.0):
            bm.edges.new((bm.verts.new((0.0, y, 0.0)), bm.verts.new((2.0, y, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='SQUARE')
        off = measure(sketch)
        set_union(sketch, True)
        assert measure(sketch) == off
        assert sketch.sketch_tools.warnings == ""


def test_a_sweep_without_overlap_does_not_run_the_boolean():
    with addon():
        sketch = make_open_sketch([(0, 0), (2, 0), (2, 2)])
        make_sweep(sketch, sw_profile='SQUARE')
        off = measure(sketch)
        original = build._evaluate_union

        def forbidden(_bm):
            raise AssertionError("the boolean ran for a sweep that does not overlap")

        build._evaluate_union = forbidden
        try:
            set_union(sketch, True)
        finally:
            build._evaluate_union = original
        assert measure(sketch) == off and sketch.sketch_tools.warnings == ""


def test_overlapping_ends_of_a_closed_loop_with_a_tail_merge():
    with addon():
        # A tail that doubles back over the loop's own side overlaps it.
        sketch = make_sketch([SQUARE])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        bm.edges.new((bm.verts.new((0.5, -0.5, 0.0)), bm.verts.new((0.5, 0.5, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='SQUARE')
        off = measure(sketch)
        set_union(sketch, True)
        on = measure(sketch)
        assert on["crossings"] == 0 and on["closed"] and on["shells"] == 1
        assert on["volume"] < off["volume"]
        assert not temp_leftovers()


def test_3d_path_crossing_itself_on_a_cube_is_merged():
    with addon():
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        for profile in ('SQUARE', 'CYLINDER'):
            # Open bow-tie on the top face (normal +Z) whose last leg cuts across the first.
            points = [
                ((-0.5, -0.5, 1.0), BOWTIE_Z), ((0.5, 0.5, 1.0), BOWTIE_Z),
                ((0.5, -0.5, 1.0), BOWTIE_Z), ((-0.5, 0.5, 1.0), BOWTIE_Z),
                ((0.5, 0.2, 1.0), BOWTIE_Z),
            ]
            path = make_path(points)
            assert path.sketch_tools.is_path_3d
            with build.suspended():
                path.sketch_tools.sw_profile = profile
                path.sketch_tools.sw_width = path.sketch_tools.sw_height = 0.12
                path.sketch_tools.sw_radius = 0.06
            build.rebuild(path)
            off = measure(path)
            before = datablock_counts()
            set_union(path, True)
            assert_no_union_warning(path)
            on = measure(path)
            assert datablock_counts() == before and not temp_leftovers()
            assert_unioned(off, on)
            bpy.data.objects.remove(path.sketch_tools.output_object)
            bpy.data.objects.remove(path)


def test_closed_3d_bow_tie_is_merged():
    with addon():
        points = [
            ((-0.5, -0.5, 1.0), BOWTIE_Z), ((0.5, 0.5, 1.0), BOWTIE_Z),
            ((0.5, -0.5, 1.0), BOWTIE_Z), ((-0.5, 0.5, 1.0), BOWTIE_Z),
        ]
        path = make_path(points, closed=True)
        with build.suspended():
            path.sketch_tools.sw_width = path.sketch_tools.sw_height = 0.1
        build.rebuild(path)
        off = measure(path)
        set_union(path, True)
        assert_no_union_warning(path)
        assert_unioned(off, measure(path))


# ---------------------------------------------------------------------------
# Smooth and sharp flags
# ---------------------------------------------------------------------------

def _flags(sketch):
    bm = output_bm(sketch)
    try:
        bm.normal_update()
        smooth_faces = sum(1 for f in bm.faces if f.smooth)
        sharp_edges = sum(1 for e in bm.edges if not e.smooth)
        folds = [e for e in bm.edges if e.smooth and e.calc_face_angle(0.0) > math.radians(30.0) + 1e-6]
        return len(bm.faces), smooth_faces, sharp_edges, len(folds)
    finally:
        bm.free()


def test_union_keeps_smooth_faces_and_sharpens_the_seam():
    with addon():
        sketch = make_sketch([figure_eight()])
        make_sweep(sketch, sw_profile='CYLINDER', sw_smooth=True, sw_radius=0.15)
        _faces, smooth, sharp_before, folds = _flags(sketch)
        assert smooth > 0 and folds == 0
        set_union(sketch, True)
        faces, smooth, sharp_after, folds = _flags(sketch)
        assert smooth == faces, "every face of a smooth sweep stays smooth"
        assert folds == 0, "an edge folding by more than 30 degrees must be sharp"
        assert sharp_after >= sharp_before


def test_union_of_a_flat_shaded_sweep_stays_flat_shaded():
    with addon():
        sketch = make_sketch([figure_eight()])
        make_sweep(sketch, sw_profile='CYLINDER', sw_smooth=False, sw_radius=0.15)
        set_union(sketch, True)
        faces, smooth, sharp, _folds = _flags(sketch)
        assert smooth == 0 and sharp == 0


def test_smooth_union_keeps_end_caps_flat_and_everything_else_smooth():
    with addon():
        sketch = make_sketch([])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        for a, b in (((-1.0, 0.0), (1.0, 0.0)), ((0.0, -1.0), (0.0, 1.0))):
            bm.edges.new((bm.verts.new((*a, 0.0)), bm.verts.new((*b, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='CYLINDER', sw_radius=0.1, sw_segments=16, sw_smooth=True)
        set_union(sketch, True)
        assert_no_union_warning(sketch)
        bm = output_bm(sketch)
        try:
            bm.normal_update()
            caps = [f for f in bm.faces if len(f.verts) == 16]
            assert len(caps) == 4, len(caps)
            for cap in caps:
                assert all(not e.smooth for e in cap.edges), "a cap boundary lost its sharp edge"
            assert all(f.smooth for f in bm.faces)
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(30.0) + 1e-6:
                    assert not e.smooth
        finally:
            bm.free()


# ---------------------------------------------------------------------------
# Not a closed solid, failures
# ---------------------------------------------------------------------------

def _same_geometry(a, b):
    return a["coords"] == b["coords"] and a["faces"] == b["faces"]


def test_open_profile_ring_keeps_the_strip_and_warns():
    with addon():
        profile = make_open_sketch([(0.0, 0.0), (0.4, 0.0), (0.4, 0.3)])
        sketch = make_sketch([figure_eight(25)])
        make_sweep(sketch, sw_profile='CUSTOM', sw_custom_object=profile)
        off = measure(sketch)
        assert not off["closed"]
        before = datablock_counts()
        set_union(sketch, True)
        assert build.UNION_NOT_SOLID in sketch.sketch_tools.warnings.split("\n")
        assert _same_geometry(measure(sketch), off)
        assert datablock_counts() == before and not temp_leftovers()
        set_union(sketch, False)
        assert build.UNION_NOT_SOLID not in sketch.sketch_tools.warnings


def test_open_chain_without_end_caps_warns_not_closed():
    with addon():
        sketch = make_open_sketch([(0, 0), (2, 0), (2, 2)])
        make_sweep(sketch, sw_profile='SQUARE', sw_cap_ends=False)
        off = measure(sketch)
        set_union(sketch, True)
        assert sketch.sketch_tools.warnings.split("\n").count(build.UNION_NOT_SOLID) == 1
        assert _same_geometry(measure(sketch), off)


def test_union_leaves_extrude_alone():
    with addon():
        sketch = make_sketch([SQUARE], mode='EXTRUDE')
        before = measure(sketch)
        set_union(sketch, True)
        assert measure(sketch) == before and sketch.sketch_tools.warnings == ""


def _crossing_sketch():
    sketch = make_sketch([figure_eight(25)])
    make_sweep(sketch, sw_profile='SQUARE')
    return sketch


def test_a_failing_boolean_keeps_the_unmerged_sweep_and_warns():
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        original = build._evaluate_union
        buffer = io.StringIO()

        def boom(_bm):
            raise RuntimeError("boolean exploded")

        build._evaluate_union = boom
        try:
            with redirect_stdout(buffer):
                set_union(sketch, True)
                set_union(sketch, False)
                set_union(sketch, True)
        finally:
            build._evaluate_union = original
        assert sketch.sketch_tools.warnings.split("\n") == [build.UNION_FAILED]
        assert _same_geometry(measure(sketch), off)
        # Identical errors are printed once.
        assert buffer.getvalue().count("the sweep union failed") == 1, buffer.getvalue()
        # A later good rebuild replaces the warning.
        sketch.sketch_tools.sw_width = 0.21
        assert sketch.sketch_tools.warnings == ""
        assert measure(sketch)["crossings"] == 0


def test_an_empty_open_or_inside_out_boolean_result_is_rejected():
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        original = build._boolean_once
        calls = []

        def empty(_bm):
            calls.append(1)
            return bmesh.new()

        def open_grid(_bm):
            calls.append(1)
            result = bmesh.new()
            bmesh.ops.create_grid(result, x_segments=1, y_segments=1, size=1.0)
            return result

        def inside_out(bm):
            calls.append(1)
            result = original(bm)
            bmesh.ops.reverse_faces(result, faces=list(result.faces))
            return result

        for fake in (empty, open_grid, inside_out):
            del calls[:]
            build._boolean_once = fake
            try:
                set_union(sketch, False)
                set_union(sketch, True)
            finally:
                build._boolean_once = original
            assert sketch.sketch_tools.warnings.split("\n") == [build.UNION_FAILED], fake.__name__
            assert _same_geometry(measure(sketch), off), fake.__name__
            assert len(calls) == len(build._UNION_RETRY_SHIFTS), (fake.__name__, calls)
            assert not temp_leftovers()


def test_a_bad_first_attempt_is_retried_on_a_shifted_copy():
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        original = build._boolean_once
        seen = []

        def flaky(bm):
            seen.append(min(v.co.x for v in bm.verts))
            if len(seen) == 1:
                result = bmesh.new()
                bmesh.ops.create_grid(result, x_segments=1, y_segments=1, size=1.0)
                return result
            return original(bm)

        build._boolean_once = flaky
        try:
            set_union(sketch, True)
        finally:
            build._boolean_once = original
        assert len(seen) == 2 and seen[1] != seen[0], seen
        assert_no_union_warning(sketch)
        on = measure(sketch)
        assert on["crossings"] == 0 and on["closed"] and on["volume"] < off["volume"]
        # The shift is undone: the output is where the sweep is, not offset by it.
        lo, hi = out_bounds(sketch)
        assert abs(min(c[0] for c in (lo, hi)) - -2.0) < 0.3


def out_bounds(sketch):
    verts = sketch.sketch_tools.output_object.data.vertices
    return (
        tuple(min(v.co[i] for v in verts) for i in range(3)),
        tuple(max(v.co[i] for v in verts) for i in range(3)),
    )


def _face_box_pairs(bm):
    """Face pairs that share no vertex and whose axis-aligned boxes overlap (what a gate that
    only compared boxes would flag)."""
    faces = list(bm.faces)
    boxes = [[(min(v.co[k] for v in f.verts), max(v.co[k] for v in f.verts)) for k in range(3)] for f in faces]
    count = 0
    for i, a in enumerate(faces):
        for j in range(i + 1, len(faces)):
            if set(a.verts).isdisjoint(faces[j].verts) and all(
                boxes[i][k][0] <= boxes[j][k][1] and boxes[j][k][0] <= boxes[i][k][1] for k in range(3)
            ):
                count += 1
    return count


def _forbid_boolean():
    original = build._boolean_once

    def forbidden(_bm):
        raise AssertionError("the boolean ran for faces that never touch")

    build._boolean_once = forbidden
    return original


def test_faces_with_overlapping_boxes_but_no_contact_skip_the_boolean_silently():
    """A U of two diagonal arms close together: many face boxes overlap, no two faces touch.
    BVHTree.overlap tests triangles exactly, so the gate reports no crossing and the boolean
    never runs (no warning, no failure, ngons kept)."""
    with addon():
        c = s = math.sqrt(0.5)
        points = [(c * x - s * y, s * x + c * y) for x, y in ((0, 0), (0, 3), (0.3, 3), (0.3, 0))]
        sketch = make_open_sketch(points)
        make_sweep(sketch, sw_profile='SQUARE', sw_width=0.2, sw_height=0.2)
        bm = output_bm(sketch)
        try:
            assert _face_box_pairs(bm) > 0, "the case needs overlapping boxes"
            assert crossing_pairs(bm) == 0 and not build._has_crossing_faces(bm)
        finally:
            bm.free()
        off = measure(sketch)
        original = _forbid_boolean()
        try:
            set_union(sketch, True)
        finally:
            build._boolean_once = original
        assert measure(sketch) == off and sketch.sketch_tools.warnings == ""
        assert not temp_leftovers()


def test_a_tight_ring_does_not_run_the_boolean_either():
    with addon():
        n = 48
        ring = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        sketch = make_sketch([ring])
        make_sweep(sketch, sw_profile='CYLINDER', sw_radius=0.4, sw_segments=12)
        off = measure(sketch)
        original = _forbid_boolean()
        try:
            set_union(sketch, True)
        finally:
            build._boolean_once = original
        assert measure(sketch) == off and sketch.sketch_tools.warnings == ""


def test_a_boolean_that_changes_nothing_is_reported_as_failed():
    """The silent no-op (an older Blender ignoring the throwaway scene, say) returns the input,
    which is closed and has the summed volume: it must not pass for a union."""
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        assert off["crossings"] > 0
        original = build._boolean_once
        calls = []

        def returns_the_input(bm):
            calls.append(1)
            return bm.copy()

        build._boolean_once = returns_the_input
        try:
            set_union(sketch, True)
        finally:
            build._boolean_once = original
        assert sketch.sketch_tools.warnings.split("\n") == [build.UNION_FAILED]
        assert _same_geometry(measure(sketch), off)
        assert len(calls) == len(build._UNION_RETRY_SHIFTS), calls
        assert not temp_leftovers()
        # The real boolean still merges the same sweep afterwards.
        set_union(sketch, False)
        set_union(sketch, True)
        assert_unioned(off, measure(sketch))
        assert_no_union_warning(sketch)


def test_flush_face_contact_is_reported_by_the_gate_and_shown_as_union_failed():
    """A stem whose end cap lies exactly on a bar's top face has no volume overlap, but
    BVHTree.overlap reports a face lying on another face's interior, so the gate lets it
    through. The real solver then returns the mesh unchanged, which is rejected like any other
    no-op: three attempts, the warning, and the un-merged sweep kept."""
    with addon():
        sketch = make_sketch([])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        for a, b in (((-1.0, 0.0), (1.0, 0.0)), ((0.3, 1.0), (0.3, 0.1))):
            bm.edges.new((bm.verts.new((*a, 0.0)), bm.verts.new((*b, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='SQUARE', sw_width=0.2, sw_height=0.2)
        off = measure(sketch)
        assert off["shells"] == 2 and off["verts"] == 16 and off["faces"] == 12, off
        assert off["crossings"] > 0, "the gate should see the flush contact"
        original = build._boolean_once
        calls = []

        def counting(work):
            calls.append(1)
            return original(work)

        build._boolean_once = counting
        try:
            set_union(sketch, True)
        finally:
            build._boolean_once = original
        assert len(calls) == len(build._UNION_RETRY_SHIFTS) == 3, calls
        assert sketch.sketch_tools.warnings.split("\n") == [build.UNION_FAILED]
        assert _same_geometry(measure(sketch), off)
        assert not temp_leftovers()


def test_the_unchanged_check_does_not_flag_a_small_real_overlap():
    with addon():
        sketch = make_sketch([])
        bm = bmesh.new()
        bm.from_mesh(sketch.data)
        # Two bars that only just cross: a small corner overlap.
        for a, b in (((-1.0, 0.0), (1.0, 0.0)), ((0.99, 0.09), (0.99, 1.0))):
            bm.edges.new((bm.verts.new((*a, 0.0)), bm.verts.new((*b, 0.0))))
        bm.to_mesh(sketch.data)
        bm.free()
        sketch.data.update()
        make_sweep(sketch, sw_profile='SQUARE', sw_width=0.2, sw_height=0.2)
        off = measure(sketch)
        assert off["crossings"] > 0
        set_union(sketch, True)
        assert_no_union_warning(sketch)
        assert measure(sketch)["volume"] < off["volume"] - 1e-6


# ---------------------------------------------------------------------------
# Temporary datablocks
# ---------------------------------------------------------------------------

def test_no_temporary_datablocks_survive_rebuilds_or_unregister():
    with addon():
        sketch = _crossing_sketch()
        set_union(sketch, True)
        settled = datablock_counts()
        for width in (0.25, 0.3, 0.2):
            sketch.sketch_tools.sw_width = width
            assert datablock_counts() == settled and not temp_leftovers()
        for _ in range(3):
            build.rebuild(sketch)
        assert datablock_counts() == settled and not temp_leftovers()
        assert len(bpy.data.scenes) == 1
    assert not temp_leftovers()
    assert len(bpy.data.scenes) == 1


def test_temporary_datablocks_are_removed_when_evaluation_fails():
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        settled = datablock_counts()
        real_bpy = build.bpy

        class BrokenContext:
            def temp_override(self, **_kwargs):
                raise RuntimeError("no depsgraph for you")

        class BrokenBpy:
            data = real_bpy.data
            context = BrokenContext()

            def __getattr__(self, name):
                return getattr(real_bpy, name)

        buffer = io.StringIO()
        build.bpy = BrokenBpy()
        try:
            with redirect_stdout(buffer):
                # Everything the union makes exists by the time temp_override is called.
                result = build._union_sweep(_bm_of(sketch), False)
        finally:
            build.bpy = real_bpy
        assert result == (None, build.UNION_FAILED)
        assert datablock_counts() == settled and not temp_leftovers()
        assert "no depsgraph for you" in buffer.getvalue()
        assert _same_geometry(measure(sketch), off)


def _bm_of(sketch):
    return output_bm(sketch)


def test_temporary_datablocks_are_removed_when_reading_the_result_fails():
    with addon():
        sketch = _crossing_sketch()
        settled = datablock_counts()
        real_new = bmesh.new

        def failing_new():
            import sys
            if sys._getframe(1).f_code.co_name == "_boolean_once":
                raise RuntimeError("no bmesh for you")
            return real_new()

        bm = output_bm(sketch)
        bmesh.new = failing_new
        try:
            with redirect_stdout(io.StringIO()):
                result = build._union_sweep(bm, False)
        finally:
            bmesh.new = real_new
            bm.free()
        assert result == (None, build.UNION_FAILED)
        assert datablock_counts() == settled and not temp_leftovers()


def test_rebuilds_with_union_push_no_undo_steps():
    with addon():
        sketch = _crossing_sketch()
        name = sketch.name
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="base")
        set_union(sketch, True)
        sketch.sketch_tools.sw_width = 0.25
        build.rebuild(sketch)
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="after")
        with only_event_handlers():
            bpy.ops.ed.undo()
        sketch = bpy.data.objects[name]
        assert sketch.sketch_tools.sw_union is False
        assert abs(sketch.sketch_tools.sw_width - 0.2) < 1e-6
        assert not temp_leftovers()


# ---------------------------------------------------------------------------
# Handlers, dependents, undo, load, apply
# ---------------------------------------------------------------------------

def test_union_rebuild_does_not_loop_through_the_depsgraph_handler():
    with addon():
        sketch = _crossing_sketch()
        set_union(sketch, True)
        bpy.context.view_layer.update()
        calls, restore = count_rebuilds()
        try:
            for _ in range(4):
                bpy.context.view_layer.update()
            assert calls[0] == 0, "an idle depsgraph pass rebuilt the sketch"
            sketch.data.vertices[3].co.x += 0.01
            sketch.data.update()
            bpy.context.view_layer.update()
            assert calls[0] >= 1
            settled = calls[0]
            for _ in range(4):
                bpy.context.view_layer.update()
            assert calls[0] == settled, "the rebuild kept triggering rebuilds"
        finally:
            restore()
        assert not temp_leftovers() and measure(sketch)["crossings"] == 0


def test_union_evaluation_triggers_no_depsgraph_update_handlers():
    """C22 claims the boolean is evaluated in a throwaway scene that is not the active one,
    so no depsgraph_update_post handler runs during it and the temporary object can never
    reach the add-on handler. The counter covers exactly the evaluation window, by wrapping
    _boolean_once, so the normal output write that follows cannot count."""
    with addon():
        sketch = _crossing_sketch()
        off = measure(sketch)
        updates = []

        def counter(scene, depsgraph):
            updates.append((scene.name, [u.id.name for u in depsgraph.updates]))

        bpy.app.handlers.depsgraph_update_post.append(counter)
        original = build._boolean_once
        windows = []

        def window(bm):
            before = len(updates)
            result = original(bm)
            windows.append(updates[before:])
            return result

        try:
            # The counter must be alive: an ordinary depsgraph pass in the user scene counts.
            sketch.data.update()
            bpy.context.view_layer.update()
            assert updates, "the counting handler never fired, the test would prove nothing"
            build._boolean_once = window
            set_union(sketch, True)
        finally:
            build._boolean_once = original
            bpy.app.handlers.depsgraph_update_post.remove(counter)
        assert windows, "the boolean was never evaluated"
        for seen in windows:
            assert seen == [], f"depsgraph handlers ran during the union evaluation: {seen}"
        # The boolean really merged something, so the window was not an empty no-op.
        assert_no_union_warning(sketch)
        on = measure(sketch)
        assert on["crossings"] == 0 and on["volume"] < off["volume"] - 1e-4
        assert not temp_leftovers()


def test_custom_profile_dependents_rebuild_with_union():
    with addon():
        profile = make_sketch([[(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]])
        path = make_sketch([figure_eight(25)])
        make_sweep(path, sw_profile='CUSTOM', sw_custom_object=profile, sw_pivot='BL')
        set_union(path, True)
        first = measure(path)
        assert first["crossings"] == 0 and first["shells"] == 1

        data = profile.data
        bm = bmesh.new()
        bm.from_mesh(data)
        for v in bm.verts:
            v.co.x *= 2.0
        bm.to_mesh(data)
        bm.free()
        data.update()
        bpy.context.view_layer.update()
        second = measure(path)
        assert second["volume"] > first["volume"] * 1.5, (first["volume"], second["volume"])
        assert second["crossings"] == 0 and second["closed"]
        assert not temp_leftovers()


def test_undo_redo_and_load_rebuild_a_unioned_sweep():
    import os
    import tempfile

    with addon():
        sketch = _crossing_sketch()
        name = sketch.name
        set_union(sketch, True)
        bpy.context.view_layer.update()
        bpy.ops.ed.undo_push(message="base")
        sketch.sketch_tools.sw_width = 0.3
        bpy.ops.ed.undo_push(message="wider")

        with only_event_handlers():
            bpy.ops.ed.undo()
        sketch = bpy.data.objects[name]
        assert abs(sketch.sketch_tools.sw_width - 0.2) < 1e-6 and sketch.sketch_tools.sw_union
        narrow = measure(sketch)
        assert narrow["crossings"] == 0 and narrow["shells"] == 1

        with only_event_handlers():
            bpy.ops.ed.redo()
        sketch = bpy.data.objects[name]
        wide = measure(sketch)
        assert wide["volume"] > narrow["volume"] and wide["crossings"] == 0

        path = os.path.join(tempfile.gettempdir(), "sketch_tools_union_load_test.blend")
        try:
            with only_event_handlers():
                sketch.sketch_tools.output_object.data.clear_geometry()
                bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
                bpy.ops.wm.open_mainfile(filepath=path)
            sketch = bpy.data.objects[name]
            loaded = measure(sketch)
            assert loaded["crossings"] == 0 and loaded["shells"] == 1
            assert abs(loaded["volume"] - wide["volume"]) < 1e-5
            assert not temp_leftovers()
        finally:
            if os.path.exists(path):
                os.remove(path)


def test_apply_works_on_a_unioned_output():
    with addon():
        sketch = _crossing_sketch()
        set_union(sketch, True)
        expected = measure(sketch)
        bpy.context.view_layer.objects.active = sketch
        assert bpy.ops.sketch.apply() == {'FINISHED'}
        applied = bpy.context.view_layer.objects.active
        assert applied.type == 'MESH' and applied.parent is None and len(applied.modifiers) == 0
        bm = bmesh.new()
        bm.from_mesh(applied.data)
        try:
            assert len(bm.faces) == expected["faces"]
            assert abs(signed_volume(bm) - expected["volume"]) < 1e-6
            assert crossing_pairs(bm) == 0
        finally:
            bm.free()
        assert not temp_leftovers()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def test_panel_shows_the_union_toggle_in_the_sweep_section_only():
    from sketch_tools.ui import panel

    with addon():
        sketch = make_sketch([SQUARE], mode='SWEEP')
        ctx = types.SimpleNamespace(
            mode='OBJECT', active_object=sketch, window_manager=bpy.context.window_manager,
        )
        layout = _RecordingLayout()
        panel.SKETCH_PT_output.draw(types.SimpleNamespace(layout=layout), ctx)
        assert "sw_union" in layout.seen

        with build.suspended():
            sketch.sketch_tools.output_mode = 'EXTRUDE'
        layout = _RecordingLayout()
        panel.SKETCH_PT_output.draw(types.SimpleNamespace(layout=layout), ctx)
        assert "sw_union" not in layout.seen

        with build.suspended():
            sketch.sketch_tools.output_mode = 'SWEEP'
            sketch.sketch_tools.warnings = build.UNION_NOT_SOLID
        panel.SKETCH_PT_sketch.draw(types.SimpleNamespace(layout=_FakeLayout()), ctx)
