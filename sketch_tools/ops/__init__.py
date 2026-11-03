import bpy

from . import create, draw3d, draw_tool, fillet, output, subdivide, trim

classes = create.classes + output.classes + draw_tool.classes + draw3d.classes + fillet.classes + trim.classes + subdivide.classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    # Running tool modals hold draw handlers and header text that outlive the operator class.
    draw_tool.close_all()
    draw3d.close_all()
    fillet.close_all()
    trim.close_all()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
