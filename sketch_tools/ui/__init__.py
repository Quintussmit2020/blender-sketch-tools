import bpy

from . import menus, overlay, panel, tool

classes = panel.classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    tool.register()
    menus.register()


def unregister():
    menus.unregister()
    tool.unregister()
    overlay.remove_all()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
