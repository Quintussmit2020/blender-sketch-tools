"""Rebuild pipeline (plan 7). Uses bpy.data only: no bpy.ops, so handlers can call it without
pushing undo steps. bpy.context is touched in one place, the temporary scene override that
evaluates the sweep union (C22)."""

import traceback
from contextlib import contextmanager

import bmesh
import bpy
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

from ..builders.extrude import build_extrude
from ..builders.sweep import SHARP_ANGLE, build_sweep
from ..builders.sweep3d import build_sweep_3d
from .attributes import cleanup_fillet_ids, write_back
from .profiles import library_profile, profile_from_chains, profile_from_splines, transform_profile
from .resolve import FILLET_RADIUS_ATTR, resolve_bmesh
from .resolve3d import resolve_path_bmesh

REBUILD_FAILED = "Rebuild failed, see the system console"
EXTRUDE_NEEDS_PLANAR = "Extrude needs a planar sketch"
UNION_NOT_SOLID = "Union skipped: the sweep is not a closed solid"
UNION_FAILED = "Union failed; showing the un-merged sweep"

# Name of the throwaway datablocks the union makes; tests look for leftovers by it.
UNION_TEMP_NAME = "sketch_tools_union_tmp"

# Property updates fired by rebuild itself (output_object, warnings) and depsgraph
# updates caused by writing the output must not start another rebuild.
_rebuilding = False

_IDENTITY = Matrix.Identity(4)

# (sketch session_uid, last traceback line) already printed: a broken sketch would
# otherwise log on every depsgraph update.
_logged_errors = set()


def is_rebuilding():
    return _rebuilding


@contextmanager
def suspended():
    """Change sketch properties without their update callbacks rebuilding."""
    global _rebuilding
    previous = _rebuilding
    _rebuilding = True
    try:
        yield
    finally:
        _rebuilding = previous


def is_sketch(obj):
    return obj is not None and obj.type == 'MESH' and obj.sketch_tools.is_sketch


def is_path_3d(obj):
    return is_sketch(obj) and obj.sketch_tools.is_path_3d


def is_planar_sketch(obj):
    return is_sketch(obj) and not obj.sketch_tools.is_path_3d


def rebuild(sketch_obj):
    global _rebuilding
    if _rebuilding or not is_sketch(sketch_obj) or sketch_obj.library is not None:
        return
    _rebuilding = True
    try:
        _rebuild(sketch_obj)
    finally:
        _rebuilding = False


def safe_rebuild(sketch_obj):
    """rebuild() for handlers: one failing sketch must not stop the others."""
    try:
        rebuild(sketch_obj)
    except Exception:
        text = traceback.format_exc()
        key = (sketch_obj.session_uid, text.strip().splitlines()[-1])
        if key not in _logged_errors:
            _logged_errors.add(key)
            print(f"Sketch Tools: rebuild of '{sketch_obj.name}' failed (identical errors are not shown again)")
            print(text)
        # Unchanged text is not rewritten, so the depsgraph update this write causes does not
        # keep re-triggering the failing rebuild; a later successful rebuild replaces it.
        with suspended():
            _store_warnings(sketch_obj.sketch_tools, [REBUILD_FAILED])


def clear_error_log():
    _logged_errors.clear()
    _union_errors_logged.clear()


def rebuild_all():
    for obj in [o for o in bpy.data.objects if is_sketch(o)]:
        safe_rebuild(obj)


def is_profile_source(obj):
    """Objects that can serve as a custom sweep profile (C4). A profile is planar, so 3D paths
    are out."""
    return obj is not None and (obj.type == 'CURVE' or is_planar_sketch(obj))


def dependents(source):
    """Sketches that sweep a custom profile taken from source. Computed on demand: it is a
    scan of bpy.data.objects, cheaper than keeping a lookup in sync with undo and file load."""
    result = []
    for obj in bpy.data.objects:
        if not is_sketch(obj) or obj.library is not None:
            continue
        st = obj.sketch_tools
        if st.output_mode == 'SWEEP' and st.sw_profile == 'CUSTOM' and st.sw_custom_object == source:
            result.append(obj)
    return result


def rebuild_dependents(source, _visited=None):
    """Rebuild every sketch whose custom profile comes from source, then theirs in turn.
    The visited set ends cycles (A uses B uses A)."""
    if _rebuilding:
        return
    visited = _visited if _visited is not None else set()
    visited.add(source.as_pointer())
    for obj in dependents(source):
        pointer = obj.as_pointer()
        if pointer in visited:
            continue
        safe_rebuild(obj)
        rebuild_dependents(obj, visited)


def _rebuild(sketch):
    st = sketch.sketch_tools
    mode = st.output_mode
    path_3d = st.is_path_3d
    if mode == 'NONE':
        # The fillet ids are a planar sketch's business; a 3D path ignores them (C20).
        if not path_3d:
            _clean_fillet_ids(sketch)
        _remove_output(sketch)
        _store_warnings(st, [])
        return

    if path_3d and mode != 'SWEEP':
        _remove_output(sketch)
        _store_warnings(st, [EXTRUDE_NEEDS_PLANAR])
        return

    resolved = _resolve_path(sketch) if path_3d else _resolve(sketch, st.fillet_segments)
    warnings = list(resolved.warnings)

    bm = None
    try:
        if mode == 'SWEEP':
            bm, build_warnings = _build_sweep(sketch, resolved, build_sweep_3d if path_3d else build_sweep)
        else:
            bm, build_warnings = build_extrude(resolved, st.ex_height, st.ex_offset, st.ex_flip, st.ex_caps)
        warnings.extend(build_warnings)
        if bm is None:
            _remove_output(sketch)
        else:
            out = _get_or_create_output(sketch)
            if out is None:
                warnings.append("Sketch is not in any collection; no output created")
            else:
                bm.to_mesh(out.data)
                out.data.update()
    finally:
        if bm is not None:
            bm.free()
    _store_warnings(st, warnings)


def _resolve(sketch, fillet_segments, write_cleanup=True):
    """write_cleanup is off for a profile sketch: the cleanup still applies to what is resolved,
    but only the sketch's own rebuild writes to its mesh."""
    mesh = sketch.data
    if mesh.is_editmode:
        # The edit bmesh is owned by Blender: never free it, never clean it up here.
        return resolve_bmesh(bmesh.from_edit_mesh(mesh), fillet_segments)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if cleanup_fillet_ids(bm) and write_cleanup:
            write_back(bm, mesh)
        return resolve_bmesh(bm, fillet_segments)
    finally:
        bm.free()


def _resolve_path(path):
    mesh = path.data
    if mesh.is_editmode:
        return resolve_path_bmesh(bmesh.from_edit_mesh(mesh))

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        return resolve_path_bmesh(bm)
    finally:
        bm.free()


def _build_sweep(sketch, resolved, builder):
    """Returns (bm or None, warnings). None means there is nothing to show."""
    st = sketch.sketch_tools
    warnings = []
    rings = _profile_rings(sketch, warnings)
    if not rings:
        return None, warnings
    rings = transform_profile(rings, st.sw_pivot, st.sw_offset_u, st.sw_offset_v, st.sw_rotation, st.sw_mirror_u)
    bm, sweep_warnings = builder(resolved, rings, st.sw_mitre_limit, st.sw_smooth, st.sw_cap_ends)
    warnings.extend(sweep_warnings)
    if st.sw_union:
        try:
            merged, union_warning = _union_sweep(bm, st.sw_smooth)
        except BaseException:
            bm.free()
            raise
        if merged is not None:
            bm.free()
            bm = merged
        if union_warning:
            warnings.append(union_warning)
    return bm, warnings


def _profile_rings(sketch, warnings):
    st = sketch.sketch_tools
    if st.sw_profile != 'CUSTOM':
        rings, profile_warnings = library_profile(
            st.sw_profile, st.sw_width, st.sw_height, st.sw_radius, st.sw_thickness, st.sw_segments,
        )
        warnings.extend(profile_warnings)
        return rings

    source = st.sw_custom_object
    if source is None:
        warnings.append("Custom profile: no profile object chosen")
        return None
    if source == sketch:
        warnings.append("Custom profile: a sketch cannot use itself as its profile")
        return None
    if is_path_3d(source):
        warnings.append("Custom profile: a 3D path cannot be a profile")
        return None
    if is_sketch(source):
        resolved = _resolve(source, source.sketch_tools.fillet_segments, write_cleanup=False)
        warnings.extend("Profile: " + w for w in resolved.warnings)
        rings, profile_warnings = profile_from_chains(resolved.chains)
    elif source.type == 'CURVE':
        rings, profile_warnings = profile_from_splines(_read_splines(source.data))
    else:
        warnings.append(f"Custom profile: '{source.name}' is not a sketch or a curve object")
        return None
    warnings.extend(profile_warnings)
    return rings


def _read_splines(curve):
    """Plain-data splines in the curve object's local space (C14)."""
    splines = []
    for spline in curve.splines:
        if spline.type == 'BEZIER':
            points = [
                (tuple(p.co), tuple(p.handle_left), tuple(p.handle_right)) for p in spline.bezier_points
            ]
        else:
            points = [tuple(p.co)[:3] for p in spline.points]
        splines.append({
            "type": spline.type,
            "points": points,
            "cyclic": spline.use_cyclic_u,
            "resolution": spline.resolution_u,
        })
    return splines


# ---------------------------------------------------------------------------
# Boolean union of a sweep (C22)
# ---------------------------------------------------------------------------

_union_errors_logged = set()

# Two crossing round bars meet along ridges at exactly the same height, and the exact solver
# can leave a hole there. The retry moves the input by a few thousandths of its size, which
# rounds every coordinate differently and breaks that tie; the result is moved back. Distinct
# fractions, so no retry repeats another.
_UNION_RETRY_SHIFTS = (None, (1.7e-3, 0.9e-3, 0.5e-3), (-0.6e-3, 1.3e-3, -1.9e-3))


def _union_sweep(bm, smooth):
    """Merge the overlapping and self-intersecting parts of a swept solid into one mesh.
    Returns (new bmesh or None, warning or None). None means: keep bm, which is not touched.
    Never raises: a failing boolean must leave the un-merged sweep on screen."""
    if not bm.faces:
        return None, None
    if not _is_closed_solid(bm):
        return None, UNION_NOT_SOLID
    try:
        # The exact solver is the costly part; a sweep whose parts do not meet is already
        # what the union would give, so it is left alone (and keeps its ngons).
        if not _has_crossing_faces(bm):
            return None, None
        merged = _evaluate_union(bm)
    except Exception:
        text = traceback.format_exc()
        key = text.strip().splitlines()[-1]
        if key not in _union_errors_logged:
            _union_errors_logged.add(key)
            print("Sketch Tools: the sweep union failed (identical errors are not shown again)")
            print(text)
        return None, UNION_FAILED
    if merged is None:
        return None, UNION_FAILED
    if smooth:
        _restore_smooth_shading(merged)
    return merged, None


def _is_closed_solid(bm):
    """Manifold, every edge walked in opposite directions by its two faces, no loose geometry.
    An open profile ring (a strip) or a sweep without end caps fails; so does a union that
    left a seam open."""
    if any(v.is_wire for v in bm.verts):
        return False
    for edge in bm.edges:
        loops = edge.link_loops
        if len(loops) != 2 or loops[0].vert == loops[1].vert:
            return False
    return True


def _signed_volume(bm):
    total = 0.0
    for face in bm.faces:
        co = [v.co for v in face.verts]
        for i in range(1, len(co) - 1):
            total += co[0].dot(co[i].cross(co[i + 1]))
    return total / 6.0


def _has_crossing_faces(bm):
    """True when two faces that share no vertex intersect or touch. BVHTree.overlap tests the
    triangles of both trees exactly (not just their boxes), so a tight ring or a near-touching
    U whose boxes overlap without any contact reports nothing."""
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    tree = BVHTree.FromBMesh(bm)
    faces = bm.faces
    for a, b in tree.overlap(tree):
        if a < b and set(faces[a].verts).isdisjoint(faces[b].verts):
            return True
    return False


def _evaluate_union(bm):
    """The union of bm, or None when no attempt gave a closed, outward, smaller-or-equal
    solid. The caller frees the bmesh it gets."""
    summed = _signed_volume(bm)
    size = max(max(v.co[i] for v in bm.verts) - min(v.co[i] for v in bm.verts) for i in range(3))
    for shift in _UNION_RETRY_SHIFTS:
        work = bm.copy()
        try:
            offset = None if shift is None else tuple(c * size for c in shift)
            originals = None
            if offset is not None:
                originals = _shift(work, offset)
            merged = _boolean_once(work)
        finally:
            work.free()
        try:
            if offset is not None:
                _shift_back(merged, offset, originals)
            # The union can only be smaller than the sum of the overlapping parts.
            volume = _signed_volume(merged)
            if (
                len(merged.faces) and _is_closed_solid(merged) and 0.0 < volume <= summed * (1.0 + 1e-4)
                and not _is_unchanged(merged, bm, volume, summed)
            ):
                return merged
        except BaseException:
            merged.free()
            raise
        merged.free()
    return None


def _is_unchanged(merged, original, volume, summed):
    """The caller only gets here for a sweep in which the gate saw contact, which may be flush
    contact with no volume overlap. A result with the same counts and volume as the input
    means the solver did nothing (an older Blender ignoring the throwaway scene, or faces
    that only touch). A real union adds seam vertices and loses the volume the parts shared."""
    return (
        len(merged.verts) == len(original.verts) and len(merged.faces) == len(original.faces)
        and abs(volume - summed) <= 1e-6 * summed
    )


def _shift(bm, offset):
    """Move every vertex by offset. Returns {moved coordinate: original coordinate}, so the
    vertices the solver keeps can be put back exactly."""
    originals = {}
    for vert in bm.verts:
        original = tuple(vert.co)
        vert.co = (original[0] + offset[0], original[1] + offset[1], original[2] + offset[2])
        originals[tuple(vert.co)] = original
    return originals


def _shift_back(bm, offset, originals):
    for vert in bm.verts:
        original = originals.get(tuple(vert.co))
        vert.co = original if original is not None else (
            vert.co[0] - offset[0], vert.co[1] - offset[1], vert.co[2] - offset[2])


def _boolean_once(bm):
    """A Boolean modifier set to a self union, evaluated on a throwaway object in a throwaway
    scene, so the user's scenes, selection and undo stack are never touched. Returns a new
    bmesh the caller frees. Every datablock made here is removed before returning, on every
    path."""
    mesh = obj = collection = scene = None
    try:
        mesh = bpy.data.meshes.new(UNION_TEMP_NAME)
        bm.to_mesh(mesh)
        obj = bpy.data.objects.new(UNION_TEMP_NAME, mesh)
        # The exact solver resolves self intersections of a mesh when the operand is a
        # collection; an empty one makes the mesh its own and only operand. With an object
        # operand left empty the modifier does nothing.
        collection = bpy.data.collections.new(UNION_TEMP_NAME)
        modifier = obj.modifiers.new("Union", 'BOOLEAN')
        modifier.operation = 'UNION'
        modifier.solver = 'EXACT'
        modifier.operand_type = 'COLLECTION'
        modifier.collection = collection
        modifier.use_self = True
        scene = bpy.data.scenes.new(UNION_TEMP_NAME)
        scene.collection.objects.link(obj)
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        result_mesh = evaluated.to_mesh()
        merged = bmesh.new()
        try:
            merged.from_mesh(result_mesh)
        except BaseException:
            merged.free()
            raise
        finally:
            evaluated.to_mesh_clear()
    finally:
        _remove_temporary(scene, obj, collection, mesh)
    return merged


def _remove_temporary(scene, obj, collection, mesh):
    for datablock, bpy_collection in (
        (scene, bpy.data.scenes), (obj, bpy.data.objects),
        (collection, bpy.data.collections), (mesh, bpy.data.meshes),
    ):
        if datablock is None:
            continue
        try:
            bpy_collection.remove(datablock)
        except Exception:
            traceback.print_exc()


def _restore_smooth_shading(bm):
    """What the boolean does with shading is not dependable: faces it splits come back flat
    shaded, and the seam edges it creates come out smooth, which would shade the crossing as
    a rounded blend. So every face is smooth again, as in a smooth sweep, and every edge
    folding by more than the sweep's own corner angle (C5) is sharp. End caps keep their
    flat look because the builder already made their boundary edges sharp and the boolean
    keeps those; sharp edges are never made smooth."""
    bm.normal_update()
    for face in bm.faces:
        face.smooth = True
    for edge in bm.edges:
        if edge.smooth and len(edge.link_faces) == 2 and edge.calc_face_angle(0.0) > SHARP_ANGLE:
            edge.smooth = False


def _clean_fillet_ids(sketch):
    """C10 cleanup for modes that never resolve. The edit bmesh is left to Blender."""
    mesh = sketch.data
    if mesh.is_editmode or mesh.attributes.get(FILLET_RADIUS_ATTR) is None:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if cleanup_fillet_ids(bm):
            write_back(bm, mesh)
    finally:
        bm.free()


def _store_warnings(st, warnings):
    text = "\n".join(warnings)
    if st.warnings != text:
        st.warnings = text


def _get_or_create_output(sketch):
    st = sketch.sketch_tools
    out = st.output_object
    if out is not None and out.parent == sketch:
        if out.users_scene and out.type == 'MESH':
            _enforce_output_conventions(sketch, out)
            _follow_sketch_name(sketch, out)
            return out
        # Our own output, unlinked from every scene: drop it so the name is free.
        _delete_object(out)
    # Otherwise out is missing, or belongs to the sketch this one was duplicated from.

    collections = sketch.users_collection
    if not collections:
        return None
    name = sketch.name + ".out"
    out = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    collections[0].objects.link(out)
    _enforce_output_conventions(sketch, out)
    st.output_object = out
    return out


def _enforce_output_conventions(sketch, out):
    # Only write on change: every write tags the object for a depsgraph update.
    if out.parent != sketch:
        out.parent = sketch
    if out.matrix_parent_inverse != _IDENTITY:
        out.matrix_parent_inverse = _IDENTITY
    if out.matrix_basis != _IDENTITY:
        out.matrix_basis = _IDENTITY
    if not out.hide_select:
        out.hide_select = True


def _max_name_bytes():
    # 63 before Blender 4.3, 255 since; the property length includes the terminating NUL.
    return bpy.types.Object.bl_rna.properties["name"].length_max - 1


def _stored_name(name):
    """The name Blender keeps: it cuts names to its byte limit (a partial character is dropped)."""
    raw = name.encode("utf-8")
    limit = _max_name_bytes()
    if len(raw) <= limit:
        return name
    return raw[:limit].decode("utf-8", errors="ignore")


def _follow_sketch_name(sketch, out):
    wanted = _stored_name(sketch.name + ".out")
    if out.name == wanted:
        return
    # A taken name gets a numeric suffix; accept it rather than renaming on every rebuild.
    # Blender cuts the base so that base + ".001" still fits, hence the prefix test for a
    # wanted name that was itself cut.
    base, _dot, suffix = out.name.rpartition(".")
    if suffix.isdigit() and wanted.startswith(base):
        if base == wanted or len(wanted.encode("utf-8")) > _max_name_bytes() - len(suffix) - 1:
            return
    out.name = wanted


def _remove_output(sketch):
    st = sketch.sketch_tools
    out = st.output_object
    if out is None:
        return
    if out.parent == sketch:
        _delete_object(out)
    if st.output_object is not None:
        st.output_object = None


def _delete_object(obj):
    data = obj.data
    bpy.data.objects.remove(obj)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)
