import traceback

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .core import build
from .core.attributes import write_fillet_radius


# ---------------------------------------------------------------------------
# Object PropertyGroup (plan 3.1)
# ---------------------------------------------------------------------------

def _rebuild(self, context):
    # safe_rebuild, not rebuild: an exception here would print a traceback on every UI change
    # and leave no "Rebuild failed" warning on the sketch.
    build.safe_rebuild(self.id_data)


def _rebuild_with_dependents(self, context):
    # A sketch's fillet segments change its resolved shape, which sketches using it as a
    # custom profile have to follow (C4).
    obj = self.id_data
    build.safe_rebuild(obj)
    build.rebuild_dependents(obj)


def _poll_custom_profile(self, obj):
    return obj != self.id_data and build.is_profile_source(obj)


OUTPUT_MODES = (
    ('NONE', "None", "No output object"),
    ('EXTRUDE', "Extrude", "Extrude closed loops, nested loops become holes"),
    ('SWEEP', "Sweep", "Sweep a profile along the sketch"),
)

SWEEP_PROFILES = (
    ('SQUARE', "Square", "Rectangular bar"),
    ('CYLINDER', "Cylinder", "Round bar"),
    ('HALF_ROUND', "Half Round", "Semicircle with the flat side on the plane"),
    ('TUBE', "Tube", "Hollow round tube"),
    ('CUSTOM', "Custom", "Profile from another sketch or a curve object"),
)

PIVOTS = (
    ('TL', "Top Left", ""), ('TC', "Top Center", ""), ('TR', "Top Right", ""),
    ('ML', "Middle Left", ""), ('MC', "Middle Center", ""), ('MR', "Middle Right", ""),
    ('BL', "Bottom Left", ""), ('BC', "Bottom Center", ""), ('BR', "Bottom Right", ""),
)


class SketchToolsProps(bpy.types.PropertyGroup):
    is_sketch: BoolProperty(name="Is Sketch", default=False, update=_rebuild)
    is_path_3d: BoolProperty(
        name="Is 3D Path", default=False, update=_rebuild,
        description="A non-planar wire path drawn on object surfaces; it can be swept but not extruded",
    )
    output_mode: EnumProperty(name="Output", items=OUTPUT_MODES, default='NONE', update=_rebuild)
    output_object: PointerProperty(name="Output Object", type=bpy.types.Object, update=_rebuild)
    fillet_segments: IntProperty(
        name="Fillet Segments", default=8, min=1, max=64, update=_rebuild_with_dependents,
        description="Segments per fillet arc",
    )

    ex_height: FloatProperty(name="Height", default=2.0, subtype='DISTANCE', update=_rebuild)
    ex_offset: FloatProperty(
        name="Offset", default=0.0, subtype='DISTANCE', update=_rebuild,
        description="Shift along the plane normal before extruding",
    )
    ex_flip: BoolProperty(name="Flip", default=False, update=_rebuild)
    ex_caps: BoolProperty(name="Caps", default=True, update=_rebuild)

    sw_profile: EnumProperty(name="Profile", items=SWEEP_PROFILES, default='SQUARE', update=_rebuild)
    sw_custom_object: PointerProperty(
        name="Custom Profile", type=bpy.types.Object, poll=_poll_custom_profile, update=_rebuild,
    )
    sw_width: FloatProperty(
        name="Width", default=0.2, min=0.0, subtype='DISTANCE', update=_rebuild,
        description="Square width or half round diameter",
    )
    sw_height: FloatProperty(name="Height", default=0.2, min=0.0, subtype='DISTANCE', update=_rebuild)
    sw_radius: FloatProperty(name="Radius", default=0.1, min=0.0, subtype='DISTANCE', update=_rebuild)
    sw_thickness: FloatProperty(name="Thickness", default=0.02, min=0.0, subtype='DISTANCE', update=_rebuild)
    sw_segments: IntProperty(name="Segments", default=16, min=3, max=256, update=_rebuild)
    sw_pivot: EnumProperty(name="Pivot", items=PIVOTS, default='BC', update=_rebuild)
    sw_offset_u: FloatProperty(
        name="Offset U", default=0.0, subtype='DISTANCE', update=_rebuild,
        description="In-plane offset, perpendicular to the path",
    )
    sw_offset_v: FloatProperty(
        name="Offset V", default=0.0, subtype='DISTANCE', update=_rebuild,
        description="Offset along the plane normal",
    )
    sw_rotation: FloatProperty(
        name="Rotation", default=0.0, min=-360.0, max=360.0, update=_rebuild,
        description="Profile rotation about the path tangent, in degrees",
    )
    sw_mirror_u: BoolProperty(name="Mirror U", default=False, update=_rebuild)
    sw_mitre_limit: FloatProperty(
        name="Mitre Limit", default=4.0, min=1.0, update=_rebuild,
        description="Corner offset limit as a multiple of each profile point's own distance from the path",
    )
    sw_smooth: BoolProperty(name="Smooth", default=False, update=_rebuild)
    sw_cap_ends: BoolProperty(
        name="Cap Ends", default=True, update=_rebuild,
        description="Cap the ends of open chains",
    )
    sw_union: BoolProperty(
        name="Union", default=False, update=_rebuild,
        description="Merge overlapping and self-crossing parts into one closed mesh (needs a closed profile and end caps; slower)",
    )

    warnings: StringProperty(name="Warnings", default="")


# ---------------------------------------------------------------------------
# Panel-only state on the WindowManager
# ---------------------------------------------------------------------------

# Shapes of the Sketch tool (C21). The ids are the draw_logic shape constants and the 3D path tool
# reads the same property.
SHAPE_ITEMS = (
    ('POLYLINE', "Polyline", "Click points, each joined to the previous one", 'IPO_LINEAR', 0),
    ('RECTANGLE', "Rectangle", "Two opposite corners, as two clicks or a drag", 'MESH_PLANE', 1),
    ('CIRCLE', "Circle", "Click the centre, then set the radius", 'MESH_CIRCLE', 2),
    ('ARC_3PT', "3-Point Arc", "Click the start, a point on the arc, then the end", 'SPHERECURVE', 3),
    ('ARC_CENTER', "Centre Arc",
     "Click the centre, the start (sets the radius), then the end. Runs counter-clockwise from the start",
     'CON_ROTLIMIT', 4),
)

# Value of sketch.draw's mode property that means "the shape chosen in the header". The tool's LMB
# item sets it; R and O name their shapes, because a mode kept in the tool settings would
# otherwise be merged over them.
FROM_HEADER = 'SETTING'

# Seconds after the last change of the fillet field before its undo step is pushed.
FILLET_UNDO_DELAY = 0.3

_undo_error_shown = False


def _has_window():
    # Background runs have a window manager but no window to push an undo step for.
    return not bpy.app.background and bool(bpy.context.window_manager.windows)


def _undo_window(windows):
    """Timer callbacks run without a window, so pick one; prefer a window showing a 3D view."""
    for win in windows:
        if any(area.type == 'VIEW_3D' for area in win.screen.areas):
            return win
    return windows[0]


def _push_undo():
    global _undo_error_shown
    try:
        win = _undo_window(bpy.context.window_manager.windows)
        with bpy.context.temp_override(window=win, screen=win.screen):
            bpy.ops.ed.undo_push(message="Set Fillet Radius")
    except Exception:
        if not _undo_error_shown:
            _undo_error_shown = True
            print("Sketch Tools: could not push the undo step for the fillet radius (shown once)")
            traceback.print_exc()


def _undo_timer():
    if _has_window():
        _push_undo()
    return None


def _schedule_undo_push():
    """The field fires its update on every tick of a drag; one step after the last tick is
    enough. Re-registering restarts the delay."""
    if not _has_window():
        return
    if bpy.app.timers.is_registered(_undo_timer):
        bpy.app.timers.unregister(_undo_timer)
    bpy.app.timers.register(_undo_timer, first_interval=FILLET_UNDO_DELAY)


def _fillet_radius_update(self, context):
    obj = context.active_object
    if not build.is_planar_sketch(obj):
        return
    write_fillet_radius(obj, self.sketch_tools_fillet_radius)
    build.safe_rebuild(obj)
    # WindowManager edits push no undo step of their own, so the mesh writes above
    # would be lost to the next undo.
    _schedule_undo_push()


def register():
    bpy.utils.register_class(SketchToolsProps)
    bpy.types.Object.sketch_tools = PointerProperty(type=SketchToolsProps)
    bpy.types.WindowManager.sketch_tools_fillet_radius = FloatProperty(
        name="Fillet Radius", default=0.0, min=0.0, subtype='DISTANCE',
        update=_fillet_radius_update,
        description="Fillet radius written to the selected sketch vertices",
    )
    bpy.types.WindowManager.sketch_tools_shape = EnumProperty(
        name="Shape", items=SHAPE_ITEMS, default='POLYLINE',
        description="Shape the Sketch tool draws with the next click",
    )
    bpy.types.WindowManager.sketch_tools_apply_delete_sketch = BoolProperty(
        name="Delete Sketch", default=False,
        description="Delete the sketch on Apply instead of hiding it",
    )


def unregister():
    if bpy.app.timers.is_registered(_undo_timer):
        bpy.app.timers.unregister(_undo_timer)
    del bpy.types.WindowManager.sketch_tools_apply_delete_sketch
    del bpy.types.WindowManager.sketch_tools_shape
    del bpy.types.WindowManager.sketch_tools_fillet_radius
    del bpy.types.Object.sketch_tools
    bpy.utils.unregister_class(SketchToolsProps)
