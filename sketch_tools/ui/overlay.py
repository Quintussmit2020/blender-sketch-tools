"""Viewport feedback for the Sketch draw tool (plan 8.3): rubber band, rectangle and circle
previews, angle lock guide and a text readout next to the cursor.

Everything is drawn in region pixels (POST_PIXEL) from the DrawSession's Preview, projected
through the same projector the tool uses for picking, so one handler is enough. The handler is
registered for every 3D view but draws only in the region the tool runs in."""

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

COLOR_BAND = (1.0, 0.8, 0.2, 1.0)
COLOR_GUIDE = (0.4, 0.8, 1.0, 0.6)
COLOR_SNAP = (0.3, 1.0, 0.4, 1.0)
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)
COLOR_TEXT_SHADOW = (0.0, 0.0, 0.0, 0.8)
LINE_WIDTH = 2.0
MARKER_HALF = 4.0
SNAP_HALF = 8.0
TEXT_SIZE = 15.0
TEXT_OFFSET = (18.0, -22.0)
LINE_HEIGHT = 20.0

# Live overlays, so unregistering the add-on while the tool is modal still removes the handler.
_live = set()


def _project(points, to_screen):
    out = []
    for xy in points:
        screen = to_screen(xy)
        if screen is None:
            return None
        out.append(screen)
    return out


def _square(center, half):
    x, y = center
    return [(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half),
            (x - half, y - half)]


def screen_geometry(preview, to_screen):
    """Preview in region pixels: {"band": [strip], "guide": [strip], "snap": [strip]}, each strip a
    list of points. Strips with a point behind the camera are dropped."""
    band = []
    for strip in preview.strips:
        projected = _project(strip, to_screen)
        if projected:
            band.append(projected)
    for xy in preview.markers:
        center = to_screen(xy)
        if center is not None:
            band.append(_square(center, MARKER_HALF))
    guide = []
    if preview.guide is not None:
        projected = _project(preview.guide, to_screen)
        if projected:
            guide.append(projected)
    snap = []
    if preview.snap_xy is not None:
        center = to_screen(preview.snap_xy)
        if center is not None:
            snap.append(_square(center, SNAP_HALF))
    return {"band": band, "guide": guide, "snap": snap}


def _draw_strips(shader, strips, color):
    shader.uniform_float("color", color)
    for strip in strips:
        # 3D positions: the polyline shader wants a vec3 attribute on Vulkan and Metal.
        pos = [(x, y, 0.0) for x, y in strip]
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pos})
        batch.draw(shader)


def _draw_text(lines, mouse):
    font_id = 0
    blf.size(font_id, TEXT_SIZE)
    for i, line in enumerate(lines):
        x = mouse[0] + TEXT_OFFSET[0]
        y = mouse[1] + TEXT_OFFSET[1] - i * LINE_HEIGHT
        blf.position(font_id, x + 1, y - 1, 0)
        blf.color(font_id, *COLOR_TEXT_SHADOW)
        blf.draw(font_id, line)
        blf.position(font_id, x, y, 0)
        blf.color(font_id, *COLOR_TEXT)
        blf.draw(font_id, line)


class DrawOverlay:
    """Owns one draw handler. get_state returns (preview, mouse_xy, to_screen) or None.
    region_pointer is the as_pointer() of the only region to draw in, None draws in all."""

    def __init__(self, get_state, region_pointer=None):
        self._get_state = get_state
        self._region_pointer = region_pointer
        self._handle = None
        self._failed = False

    def add(self):
        if self._handle is None:
            self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), 'WINDOW', 'POST_PIXEL')
            _live.add(self)

    def remove(self):
        _live.discard(self)
        if self._handle is not None:
            handle, self._handle = self._handle, None
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')

    def _draw(self):
        # A broken draw callback would print on every redraw, so report once and go quiet.
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
            viewport = (region.width, region.height)
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            shader.uniform_float("viewportSize", viewport)
            shader.uniform_float("lineWidth", LINE_WIDTH)
            previous_blend = gpu.state.blend_get()
            gpu.state.blend_set('ALPHA')
            try:
                _draw_strips(shader, geometry["guide"], COLOR_GUIDE)
                _draw_strips(shader, geometry["band"], COLOR_BAND)
                _draw_strips(shader, geometry["snap"], COLOR_SNAP)
            finally:
                gpu.state.blend_set(previous_blend)
            if preview.lines and mouse is not None:
                _draw_text(preview.lines, mouse)
        except Exception:
            import traceback
            self._failed = True
            print("Sketch Tools: the draw tool overlay failed and is disabled for this stroke")
            traceback.print_exc()


def remove_all():
    for overlay in list(_live):
        overlay.remove()
