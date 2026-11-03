"""Live rebuild triggers (plan 7, DECISIONS C6). Handlers never call operators, so the
rebuilds they cause push no undo steps."""

import traceback

import bpy
from bpy.app.handlers import persistent

from .core import build
from .ops import trim

# Errors already printed; a broken handler would otherwise log on every depsgraph update.
_logged_errors = set()


def _log_once(where):
    text = traceback.format_exc()
    key = (where, text.strip().splitlines()[-1])
    if key in _logged_errors:
        return
    _logged_errors.add(key)
    print(f"Sketch Tools: error in {where} handler (further identical errors are not shown)")
    print(text)


def _changed_objects(depsgraph):
    """Objects whose geometry changed outside edit mode: sketches to rebuild, and sketches or
    curves that other sketches may use as a sweep profile (C4)."""
    # Keyed by pointer: names are not unique across linked libraries. Leaving edit mode
    # needs no tracking of its own, the write-back flags the Mesh or Curve as updated.
    found = {}
    datablocks = []
    for update in depsgraph.updates:
        original = update.id.original
        if isinstance(original, bpy.types.Object):
            if update.is_updated_geometry:
                found[original.as_pointer()] = original
        elif isinstance(original, (bpy.types.Mesh, bpy.types.Curve)):
            datablocks.append(original)

    if datablocks:
        users = bpy.data.user_map(subset=datablocks, key_types={'MESH', 'CURVE'}, value_types={'OBJECT'})
        for objects in users.values():
            for obj in objects:
                found[obj.as_pointer()] = obj

    # Output objects are never sketches or curves, so they drop out here. A 3D path is a sketch
    # but never a profile source, so it is kept for its own rebuild. Edit mode rebuilds wait for
    # the write-back on exit.
    return [obj for obj in found.values()
            if (build.is_sketch(obj) or build.is_profile_source(obj)) and obj.mode != 'EDIT']


@persistent
def _on_depsgraph_update_post(scene, depsgraph):
    if build.is_rebuilding():
        return
    try:
        if not (depsgraph.id_type_updated('OBJECT') or depsgraph.id_type_updated('MESH')
                or depsgraph.id_type_updated('CURVE')):
            return
        for obj in _changed_objects(depsgraph):
            if build.is_sketch(obj):
                build.safe_rebuild(obj)
            if build.is_profile_source(obj):
                build.rebuild_dependents(obj)
    except Exception:
        _log_once("depsgraph_update_post")


def _remove_orphan_snapshots(where):
    # Before the rebuild, in a try of its own: the rebuild must not depend on it. It removes
    # only the Trim/Extend snapshot meshes (ops/trim.py) and touches no object, so it cannot
    # set off a rebuild.
    try:
        trim.remove_orphan_snapshots()
    except Exception:
        _log_once(where)


@persistent
def _on_load_post(*_args):
    _remove_orphan_snapshots("load_post (snapshots)")
    try:
        build.rebuild_all()
    except Exception:
        _log_once("load_post")


@persistent
def _on_undo_redo_post(*_args):
    _remove_orphan_snapshots("undo/redo (snapshots)")
    try:
        build.rebuild_all()
    except Exception:
        _log_once("undo/redo")


# A running Trim or Extend keeps a snapshot mesh alive with a fake user (ops/trim.py). Saved as
# it is, it would come back as a permanent orphan, so the fake user is off while the file is
# written; a mesh with no user is not written.
@persistent
def _on_save_pre(*_args):
    try:
        trim.pin_snapshots(False)
    except Exception:
        _log_once("save_pre")


@persistent
def _on_save_done(*_args):
    try:
        trim.pin_snapshots(True)
    except Exception:
        _log_once("save_post")


def _handler_pairs():
    pairs = [
        (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update_post),
        (bpy.app.handlers.load_post, _on_load_post),
        (bpy.app.handlers.undo_post, _on_undo_redo_post),
        (bpy.app.handlers.redo_post, _on_undo_redo_post),
        (bpy.app.handlers.save_pre, _on_save_pre),
        (bpy.app.handlers.save_post, _on_save_done),
    ]
    # A failed write must not leave the snapshot without its fake user. Not every Blender
    # version has this handler (4.2 may not), and then there is nothing to hook.
    save_post_fail = getattr(bpy.app.handlers, "save_post_fail", None)
    if save_post_fail is not None:
        pairs.append((save_post_fail, _on_save_done))
    return pairs


def _remove_stale(handler_list, func):
    # A reloaded module leaves the old function object behind; match by name too.
    for existing in list(handler_list):
        if (existing is func
                or (getattr(existing, "__name__", None) == func.__name__
                    and getattr(existing, "__module__", None) == func.__module__)):
            handler_list.remove(existing)


def register():
    for handler_list, func in _handler_pairs():
        _remove_stale(handler_list, func)
        handler_list.append(func)


def unregister():
    for handler_list, func in _handler_pairs():
        _remove_stale(handler_list, func)
    _logged_errors.clear()
    build.clear_error_log()
