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


# Menu classes whose draw gets _draw_subdivide appended. The context menu picks its content by
# select mode, but appended functions draw in every mode, so the entry is there in vertex mode
# too, where sketches are edited.
_MENUS = ("VIEW3D_MT_edit_mesh_edges", "VIEW3D_MT_edit_mesh_context_menu")


def register():
    for name in _MENUS:
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            menu.append(_draw_subdivide)


def unregister():
    for name in reversed(_MENUS):
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            menu.remove(_draw_subdivide)
