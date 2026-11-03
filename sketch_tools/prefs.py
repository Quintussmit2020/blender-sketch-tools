import bpy
from bpy.props import FloatProperty, IntProperty

# __package__ is "sketch_tools" when imported directly (tests) and
# "bl_ext.user_default.sketch_tools" when installed as an extension.
ADDON_ID = __package__


class SketchToolsPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    grid_increment: FloatProperty(
        name="Grid Increment", default=0.1, min=0.0001, subtype='DISTANCE',
        description="Snap increment for the sketch draw tool. Blender stores it in sketch "
        "units and displays it with the scene unit scale",
    )
    circle_segments: IntProperty(
        name="Circle Segments", default=32, min=3, max=256,
        description="Points per circle drawn with the sketch tool",
    )
    default_fillet_segments: IntProperty(
        name="Default Fillet Segments", default=8, min=1, max=64,
        description="Fillet segments given to new sketches",
    )
    snap_pixel_threshold: IntProperty(
        name="Snap Threshold (px)", default=12, min=1, max=100,
        description="Screen distance for vertex snapping and picking",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "grid_increment")
        col.prop(self, "circle_segments")
        col.prop(self, "default_fillet_segments")
        col.prop(self, "snap_pixel_threshold")


def get_prefs(context):
    """None when the package is registered without being enabled as an add-on (tests)."""
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon is not None else None


classes = (SketchToolsPreferences,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
