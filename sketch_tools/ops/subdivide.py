"""sketch.subdivide: split the selected sketch edges into equal parts (plan 8.8, DECISIONS C17).

Blender's own Subdivide interpolates the end vertices' fillet radius and id onto the new
points, so a point later dragged into a corner would carry a fillet. This operator writes the
new points sharp. Edges are cut with edge_split, which keeps every other edge, vertex, radius
and id where it was, so chains, loops and junctions stay as they are."""

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty

from ..core import attributes, build, subdivide_logic
from . import modal_common

MODES = [
    (subdivide_logic.COUNT, "Number of Cuts", "Put the same number of new vertices on every selected edge"),
    (subdivide_logic.LENGTH, "Maximum Length", "Cut every selected edge into equal parts no longer than the maximum length"),
]


def selected_edges(bm):
    """Visible edges the user selected. In vertex select mode selected vertices imply their
    edges, so an edge whose two vertices are both selected counts even when its own flag is not
    set yet. In edge and face mode the edge flags decide: two opposite sides of a square have
    all four vertices selected without the other two sides being selected."""
    by_vertex = bm.select_mode == {'VERT'}
    return [e for e in bm.edges
            if not e.hide and not any(v.hide for v in e.verts)
            and (all(v.select for v in e.verts) if by_vertex else e.select)]


def cut_edge(edge, points, rad, ids):
    """Split edge at the given (x, y) points, which run from its first vertex to its second.
    Returns the new vertices, sharp and on the sketch plane."""
    start = edge.verts[0]
    new_verts = []
    remaining = len(points)
    for x, y in points:
        # Each cut takes the first part of what is left, so the fraction grows as it shrinks.
        _piece, vert = bmesh.utils.edge_split(edge, start, 1.0 / (remaining + 1))
        # Position from the pure helper, not the interpolation, so spacing is exact and z is 0.
        vert.co = (x, y, 0.0)
        vert[rad] = 0.0
        vert[ids] = 0
        new_verts.append(vert)
        start = vert
        remaining -= 1
    return new_verts


def select_only(bm, verts):
    for face in bm.faces:
        face.select_set(False)
    for edge in bm.edges:
        edge.select_set(False)
    for vert in bm.verts:
        vert.select_set(False)
    bm.select_history.clear()
    for vert in verts:
        vert.select_set(True)
    # Vertex select mode: an edge between two selected vertices is selected too.
    bm.select_flush_mode()


class SKETCH_OT_subdivide(bpy.types.Operator):
    bl_idname = "sketch.subdivide"
    bl_label = "Subdivide Sketch Edges"
    bl_description = ("Split the selected sketch edges into equal parts. The new vertices are sharp "
                      "(no fillet) and are the only ones selected afterwards")
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", items=MODES, default=subdivide_logic.COUNT)
    cuts: IntProperty(
        name="Number of Cuts", default=1, min=1, max=100,
        description="New vertices on every selected edge",
    )
    max_length: FloatProperty(
        name="Maximum Length", default=0.5, min=0.001, subtype='DISTANCE',
        description="Longest part an edge may be left with. Blender stores it in sketch units "
        "and displays it with the scene unit scale",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and build.is_planar_sketch(context.active_object)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "mode")
        layout.prop(self, "cuts" if self.mode == subdivide_logic.COUNT else "max_length")

    def execute(self, context):
        # Trim, Extend and the draw tool keep a snapshot or a run of their own on this mesh.
        if modal_common.any_sketch_modal_live():
            self.report({'WARNING'}, modal_common.BUSY_MESSAGE)
            return {'CANCELLED'}
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        edges = selected_edges(bm)
        if not edges:
            self.report({'WARNING'}, "Select the sketch edges to subdivide")
            return {'CANCELLED'}

        # Plan against the unchanged mesh first: splitting one edge must not change the plan
        # of another, and a run that would change nothing must not touch the mesh.
        plans = []
        clamped = 0
        for edge in edges:
            a, b = (v.co.xy for v in edge.verts)
            length = subdivide_logic.edge_length(a, b)
            count = subdivide_logic.new_vertex_count(length, self.mode, self.cuts, self.max_length)
            if count:
                plans.append((edge, subdivide_logic.split_points(a, b, count)))
            if self.mode == subdivide_logic.LENGTH and subdivide_logic.was_clamped(length, self.max_length):
                clamped += 1
        if not plans:
            self.report({'WARNING'}, "No selected edge is long enough to subdivide"
                        if self.mode == subdivide_logic.LENGTH else "The selected edges have no length")
            return {'CANCELLED'}

        # The selection is by vertex, which needs vertex select mode. Setting the mode flushes the
        # selection, so it is set after planning (the plan holds on to its edges) and before cutting.
        tool_settings = context.tool_settings
        if tuple(tool_settings.mesh_select_mode) != (True, False, False):
            tool_settings.mesh_select_mode = (True, False, False)

        rad, ids = attributes.ensure_bm_layers(bm)
        new_verts = []
        for edge, points in plans:
            new_verts.extend(cut_edge(edge, points, rad, ids))
        select_only(bm, new_verts)
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)

        if clamped:
            self.report({'WARNING'}, f"{clamped} edge{'s' if clamped != 1 else ''} limited to "
                        f"{subdivide_logic.MAX_PARTS} parts")
        return {'FINISHED'}


classes = (SKETCH_OT_subdivide,)
