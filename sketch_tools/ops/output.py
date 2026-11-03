import bpy
from bpy.props import BoolProperty, EnumProperty

from ..core import build
from ..props import OUTPUT_MODES


def _active_sketch(context):
    obj = context.active_object
    return obj if build.is_sketch(obj) else None


class SKETCH_OT_set_output_mode(bpy.types.Operator):
    bl_idname = "sketch.set_output_mode"
    bl_label = "Set Output Mode"
    bl_description = "Choose what the sketch builds"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", items=OUTPUT_MODES, default='EXTRUDE')

    @classmethod
    def poll(cls, context):
        return _active_sketch(context) is not None

    def execute(self, context):
        _active_sketch(context).sketch_tools.output_mode = self.mode
        return {'FINISHED'}


class SKETCH_OT_rebuild(bpy.types.Operator):
    bl_idname = "sketch.rebuild"
    bl_label = "Rebuild"
    bl_description = "Rebuild the sketch's output object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_sketch(context) is not None

    def execute(self, context):
        build.rebuild(_active_sketch(context))
        return {'FINISHED'}


class SKETCH_OT_apply(bpy.types.Operator):
    bl_idname = "sketch.apply"
    bl_label = "Apply"
    bl_description = "Turn the output into a plain mesh and detach it from the sketch"
    bl_options = {'REGISTER', 'UNDO'}

    delete_sketch: BoolProperty(
        name="Delete Sketch", default=False,
        description="Delete the sketch instead of hiding it",
    )

    @classmethod
    def poll(cls, context):
        sketch = _active_sketch(context)
        return sketch is not None and sketch.sketch_tools.output_mode != 'NONE'

    def execute(self, context):
        sketch = _active_sketch(context)
        if sketch.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')

        build.rebuild(sketch)
        st = sketch.sketch_tools
        out = st.output_object
        if out is None or out.parent != sketch:
            self.report({'ERROR'}, "This sketch has no output to apply")
            return {'CANCELLED'}

        world = out.matrix_world.copy()
        out.parent = None
        out.matrix_world = world
        out.hide_select = False

        # Detach first, then switch the mode off, without a rebuild in between that
        # would create a fresh output object.
        with build.suspended():
            st.output_object = None
            st.output_mode = 'NONE'
            st.warnings = ""

        name = sketch.name
        if self.delete_sketch:
            mesh = sketch.data
            bpy.data.objects.remove(sketch)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        else:
            sketch.name = name + ".sketch"
            # Frees the mesh datablock name for the applied mesh.
            sketch.data.name = name + ".sketch"
            sketch.select_set(False)
            sketch.hide_set(True)
        out.name = name
        out.data.name = name

        out.select_set(True)
        context.view_layer.objects.active = out
        return {'FINISHED'}


classes = (SKETCH_OT_set_output_mode, SKETCH_OT_rebuild, SKETCH_OT_apply)
