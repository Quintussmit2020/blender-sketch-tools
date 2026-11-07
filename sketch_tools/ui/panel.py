"""N-panel (plan 9). draw() only reads stored properties; it never
resolves the sketch, because it runs on every redraw."""

import bpy

from ..core import build


# Only the fields each profile reads.
_SWEEP_PROFILE_FIELDS = {
    'SQUARE': ("sw_width", "sw_height"),
    'CYLINDER': ("sw_radius", "sw_segments"),
    'HALF_ROUND': ("sw_width", "sw_segments"),
    'TUBE': ("sw_radius", "sw_thickness", "sw_segments"),
    'CUSTOM': ("sw_custom_object",),
}


class _SketchToolsPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Sketch Tools"

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'}


class _ActiveSketchPanel(_SketchToolsPanel):
    @classmethod
    def poll(cls, context):
        return super().poll(context) and build.is_sketch(context.active_object)


class SKETCH_PT_create(_SketchToolsPanel, bpy.types.Panel):
    bl_label = "Create"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator_menu_enum("sketch.new", "plane", text="New Sketch", icon='GREASEPENCIL')
        col.operator("sketch.set_plane_from_face", icon='FACESEL')
        col.operator("sketch.new_path_3d", icon='CURVE_PATH')
        obj = context.active_object
        if context.mode == 'EDIT_MESH' and obj is not None and not build.is_sketch(obj):
            col.operator("sketch.edges_to_sketch", icon='EDGESEL')


class SKETCH_PT_sketch(_ActiveSketchPanel, bpy.types.Panel):
    bl_label = "Sketch"

    def draw(self, context):
        layout = self.layout
        st = context.active_object.sketch_tools

        if st.is_path_3d:
            # The 2D tools refuse 3D paths (C20), so none of their controls are shown.
            col = layout.column(align=True)
            if context.mode == 'EDIT_MESH':
                col.label(text="3D path: draw it with the Sketch tool")
                col.label(text="on other objects.")
            else:
                col.label(text="3D path: enter edit mode first, then")
                col.label(text="draw it with the Sketch tool.")
        else:
            col = layout.column()
            col.prop(context.window_manager, "sketch_tools_fillet_radius", text="Fillet Radius")
            col.prop(st, "fillet_segments")

        if context.mode == 'EDIT_MESH':
            # The same enum as the Sketch tool header (C21); both the planar and the 3D tool read it.
            # expand inside a column stacks the items as full-width bars, so the icons go in a row.
            col = layout.column(align=True)
            col.label(text="Shape")
            col.row(align=True).prop(context.window_manager, "sketch_tools_shape", text="", expand=True, icon_only=True)
            col.label(text="Click in the viewport with the Sketch tool")

        if context.mode == 'EDIT_MESH' and not st.is_path_3d:
            col = layout.column(align=True)
            col.operator("sketch.fillet", icon='MOD_BEVEL')
            col.operator("sketch.trim", icon='EDGESEL')
            col.operator("sketch.extend", icon='FORWARD')
            col.operator("sketch.subdivide", text="Subdivide", icon='MOD_SUBSURF')

        if st.warnings:
            box = layout.box()
            for line in st.warnings.split("\n"):
                box.label(text=line, icon='ERROR')


class SKETCH_PT_output(_ActiveSketchPanel, bpy.types.Panel):
    bl_label = "Output"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        st = context.active_object.sketch_tools

        if st.is_path_3d:
            # Extrude needs a plane. A path set to it by script shows the reason in the warnings box.
            row = layout.row()
            row.prop_enum(st, "output_mode", 'NONE')
            row.prop_enum(st, "output_mode", 'SWEEP')
        else:
            layout.row().prop(st, "output_mode", expand=True)

        if st.output_mode == 'EXTRUDE' and not st.is_path_3d:
            col = layout.column()
            col.prop(st, "ex_height")
            col.prop(st, "ex_offset")
            col.prop(st, "ex_flip")
            col.prop(st, "ex_caps")
        elif st.output_mode == 'SWEEP':
            layout.prop(st, "sw_profile")
            col = layout.column()
            for name in _SWEEP_PROFILE_FIELDS[st.sw_profile]:
                col.prop(st, name)
            col.separator()
            col.prop(st, "sw_pivot")
            col.prop(st, "sw_offset_u")
            col.prop(st, "sw_offset_v")
            col.prop(st, "sw_rotation")
            col.prop(st, "sw_mirror_u")
            col.separator()
            col.prop(st, "sw_mitre_limit")
            col.prop(st, "sw_smooth")
            col.prop(st, "sw_cap_ends")
            col.prop(st, "sw_union")

        layout.separator()
        col = layout.column(align=True)
        col.use_property_split = False
        col.operator("sketch.rebuild", icon='FILE_REFRESH')
        wm = context.window_manager
        row = col.row(align=True)
        op = row.operator("sketch.apply", icon='CHECKMARK')
        op.delete_sketch = wm.sketch_tools_apply_delete_sketch
        row.prop(wm, "sketch_tools_apply_delete_sketch", text="Delete Sketch", toggle=False)


classes = (SKETCH_PT_create, SKETCH_PT_sketch, SKETCH_PT_output)
