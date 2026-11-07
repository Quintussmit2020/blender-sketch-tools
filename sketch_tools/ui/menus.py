"""Subdivide entries in Blender's Edit Mode Edge menu and edge context menu (plan 9). The draw
functions check the active object on every draw, so other meshes never see them."""

import bpy

from ..core import build


def _draw_subdivide(self, context):
    # sketch.subdivide refuses 3D paths, so the entry is not offered for them.
    if not build.is_planar_sketch(context.active_object):
        return
    layout = self.layout
    layout.separator()
    layout.operator("sketch.subdivide", icon='MOD_SUBSURF')


def _draw_edges_to_sketch(self, context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH' or build.is_sketch(obj):
        return
    layout = self.layout
    layout.separator()
    layout.operator("sketch.edges_to_sketch", icon='CURVE_PATH')


_DRAWS = (_draw_subdivide, _draw_edges_to_sketch)

# Menu classes whose draw gets the draw functions appended. The context menu picks its content by
# select mode, but appended functions draw in every mode, so the entry is there in vertex mode
# too, where sketches are edited.
_MENUS = ("VIEW3D_MT_edit_mesh_edges", "VIEW3D_MT_edit_mesh_context_menu")


def register():
    for name in _MENUS:
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            for draw in _DRAWS:
                menu.append(draw)


def unregister():
    for name in reversed(_MENUS):
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            for draw in reversed(_DRAWS):
                menu.remove(draw)
