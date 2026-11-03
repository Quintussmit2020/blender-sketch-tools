"""The Sketch tool in the Mesh Edit Mode toolbar (plan 8.3). It only runs sketch.draw; the
operator's poll keeps it inert on meshes that are not sketches."""

import bpy
from bpy.types import WorkSpaceTool

from ..core import build
from ..props import FROM_HEADER

TOOL_ID = "sketch_tools.sketch"


class SKETCH_TOOL_sketch(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = TOOL_ID
    bl_label = "Sketch"
    bl_description = (
        "Draw polylines, rectangles, circles and arcs on the sketch plane. The header picks the shape.\n"
        "LMB adds a point, RMB or Enter ends, C closes, Shift locks the angle. "
        "Snapping follows the snap magnet (Shift+Tab); hold Ctrl to flip it while held. It snaps to the sketch's own vertices first, then visible vertices and edges of other objects, then the grid. "
        "Type a length in the scene base unit (m for Metric) then Tab for an angle. "
        "R and O start a rectangle or circle on a planar sketch. While drawing, R rectangle, O circle, "
        "P polyline and A arc (again for the other arc) switch the shape.\n"
        "On a 3D path, LMB places points on the surfaces of other objects instead: it snaps to vertices, "
        "edge midpoints and points on edges. Shift or Ctrl turn snapping off "
        "(Shift also locks the angle in a polyline).\n"
        "Click-select is not available while this tool is active: switch to a select tool "
        "to pick vertices, for example for the fillet radius field"
    )
    bl_icon = "ops.gpencil.primitive_polyline"
    bl_cursor = 'CROSSHAIR'
    bl_widget = None
    # R and O start the modal in that mode and wait for the first click. Every item sets its mode
    # explicitly: a mode kept in the tool settings would be merged over the R and O items.
    # LMB accepts Shift and Ctrl (-1 = any) so the first point can angle-lock or snap; Alt stays
    # excluded for view navigation. R and O stay modifier-free so Shift+R, Ctrl+R etc. keep their
    # Blender meaning.
    # The second LMB item serves 3D paths: sketch.draw refuses them and sketch.draw_3d refuses planar
    # sketches, so the item whose poll passes takes the click and one tool covers both.
    bl_keymap = (
        ("sketch.draw", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": -1, "ctrl": -1},
         {"properties": [("mode", FROM_HEADER)]}),
        ("sketch.draw_3d", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": -1, "ctrl": -1}, {"properties": []}),
        ("sketch.draw", {"type": 'R', "value": 'PRESS'}, {"properties": [("mode", 'RECTANGLE')]}),
        ("sketch.draw", {"type": 'O', "value": 'PRESS'}, {"properties": [("mode", 'CIRCLE')]}),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        layout.prop(context.window_manager, "sketch_tools_shape", text="", expand=True, icon_only=True)
        # R and O only start shapes on planar sketches; on 3D paths they switch inside a run and the
        # header enum picks the starting shape.
        if build.is_path_3d(getattr(context, "active_object", None)):
            layout.label(text="Header picks the shape. While drawing: R rectangle, O circle, P polyline, A arc, Shift or Ctrl turn snapping off (Shift also locks the angle in a polyline)")
        else:
            layout.label(text="R rectangle, O circle; while drawing also P polyline, A arc; Shift angle lock, hold Ctrl to flip snapping to points (also other objects), lengths in the scene base unit (m for Metric)")


def register():
    bpy.utils.register_tool(SKETCH_TOOL_sketch, separator=False, group=False)


def unregister():
    bpy.utils.unregister_tool(SKETCH_TOOL_sketch)
