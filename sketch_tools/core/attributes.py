"""Per-vertex fillet attributes (plan 3.2, DECISIONS C10). bpy.data and bmesh only:
no context, so core/build.py and props.py can both use it."""

import random

import bmesh

from .resolve import FILLET_ID_ATTR, FILLET_RADIUS_ATTR
from .resolve3d import PATH_NORMAL_ATTR


def new_fillet_id():
    return random.randint(1, 2**31 - 1)


def ensure_attributes(mesh):
    attrs = mesh.attributes
    if attrs.get(FILLET_RADIUS_ATTR) is None:
        attrs.new(FILLET_RADIUS_ATTR, 'FLOAT', 'POINT')
    if attrs.get(FILLET_ID_ATTR) is None:
        attrs.new(FILLET_ID_ATTR, 'INT', 'POINT')


def ensure_path_attributes(mesh):
    """A 3D path carries the fillet attributes too (ignored in Phase 7) plus the surface
    normal captured with each point (C20)."""
    ensure_attributes(mesh)
    if mesh.attributes.get(PATH_NORMAL_ATTR) is None:
        mesh.attributes.new(PATH_NORMAL_ATTR, 'FLOAT_VECTOR', 'POINT')


def ensure_bm_layers(bm):
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    if rad is None:
        rad = bm.verts.layers.float.new(FILLET_RADIUS_ATTR)
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    if ids is None:
        ids = bm.verts.layers.int.new(FILLET_ID_ATTR)
    return rad, ids


def cleanup_fillet_ids(bm):
    """DECISIONS C10: vertices Blender copied from a filleted vertex lose the radius.

    Among vertices with a non-zero radius, one whose id is 0 or repeats the id of a
    lower-index vertex is reset to radius 0, id 0. Returns True if anything changed.
    """
    rad = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    if rad is None:
        return False
    ids = bm.verts.layers.int.get(FILLET_ID_ATTR)
    seen = set()
    changed = False
    for v in bm.verts:
        if v[rad] == 0.0:
            continue
        vid = v[ids] if ids is not None else 0
        if vid == 0 or vid in seen:
            v[rad] = 0.0
            if ids is not None:
                v[ids] = 0
            changed = True
        else:
            seen.add(vid)
    return changed


def write_fillet_radius(obj, radius):
    """Write radius to every selected vertex of a sketch, with a fresh id per filleted vertex."""
    mesh = obj.data
    if mesh.is_editmode:
        bm = bmesh.from_edit_mesh(mesh)
        rad, ids = ensure_bm_layers(bm)
        for v in bm.verts:
            if v.select:
                v[rad] = radius
                v[ids] = new_fillet_id() if radius != 0.0 else 0
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        return

    ensure_attributes(mesh)
    count = len(mesh.vertices)
    selected = [False] * count
    mesh.vertices.foreach_get("select", selected)
    rad_data = mesh.attributes[FILLET_RADIUS_ATTR].data
    id_data = mesh.attributes[FILLET_ID_ATTR].data
    radii = [0.0] * count
    id_values = [0] * count
    rad_data.foreach_get("value", radii)
    id_data.foreach_get("value", id_values)
    for i, sel in enumerate(selected):
        if sel:
            radii[i] = radius
            id_values[i] = new_fillet_id() if radius != 0.0 else 0
    rad_data.foreach_set("value", radii)
    id_data.foreach_set("value", id_values)
    mesh.update()


def write_back(bm, mesh):
    """Copy the per-vertex fillet values of bm (built from mesh, same indices) into mesh."""
    rad_layer = bm.verts.layers.float.get(FILLET_RADIUS_ATTR)
    mesh.attributes[FILLET_RADIUS_ATTR].data.foreach_set("value", [v[rad_layer] for v in bm.verts])
    id_layer = bm.verts.layers.int.get(FILLET_ID_ATTR)
    id_attr = mesh.attributes.get(FILLET_ID_ATTR)
    if id_layer is not None and id_attr is not None:
        id_attr.data.foreach_set("value", [v[id_layer] for v in bm.verts])
    mesh.update()
