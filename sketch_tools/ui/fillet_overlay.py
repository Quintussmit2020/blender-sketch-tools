"""Viewport feedback for the modal fillet (plan 8.4): the arc each affected corner will get, in a
second colour where the radius had to be clamped, plus the radius readout next to the cursor.

Built on the draw tool's overlay: same handler ownership (removed by overlay.remove_all when the
add-on unregisters), same region filter, same line and text helpers."""

import bpy
import gpu

from . import overlay

COLOR_ARC = overlay.COLOR_BAND
COLOR_CLAMPED = (1.0, 0.35, 0.2, 1.0)


def screen_geometry(preview, to_screen):
    """{"arcs", "clamped", "corners"}: lists of strips in region pixels. A strip with a point behind
    the camera is dropped. Corners with nothing to round (radius 0) show only their marker."""
    arcs, clamped, corners = [], [], []
    for arc in preview.arcs:
        center = to_screen(arc.corner)
        if center is not None:
            corners.append(overlay._square(center, overlay.MARKER_HALF))
        if len(arc.points) < 2:
            continue
        projected = overlay._project(arc.points, to_screen)
        if projected:
            (clamped if arc.clamped else arcs).append(projected)
    return {"arcs": arcs, "clamped": clamped, "corners": corners}


class FilletOverlay(overlay.DrawOverlay):
    """get_state returns (FilletPreview, mouse_xy or None, to_screen) or None."""

    def _draw(self):
        if self._failed:
            return
        try:
            region = bpy.context.region
            if self._region_pointer is not None and region.as_pointer() != self._region_pointer:
                return
            state = self._get_state()
            if state is None:
                return
            preview, mouse, to_screen = state
            geometry = screen_geometry(preview, to_screen)
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", overlay.LINE_WIDTH)
            previous_blend = gpu.state.blend_get()
            gpu.state.blend_set('ALPHA')
            try:
                overlay._draw_strips(shader, geometry["corners"], overlay.COLOR_GUIDE)
                overlay._draw_strips(shader, geometry["arcs"], COLOR_ARC)
                overlay._draw_strips(shader, geometry["clamped"], COLOR_CLAMPED)
            finally:
                gpu.state.blend_set(previous_blend)
            if preview.lines and mouse is not None:
                overlay._draw_text(preview.lines, mouse)
        except Exception:
            import traceback
            self._failed = True
            print("Sketch Tools: the fillet overlay failed and is disabled for this run")
            traceback.print_exc()
