"""Pieces the sketch modals (draw, fillet, trim, extend) share: the registry that lets any of
them refuse a second start, the event classes they pass through, and the view projector with the
region geometry behind it. Plain module-level helpers so no modal imports another."""

import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

from ..core import draw_logic

# One set of running controllers per modal module, so unregistering can wind each module down
# while any of them can ask whether another tool is live.
_LIVE_SETS = []


def live_set():
    """A new set for one modal module's running controllers, counted by any_sketch_modal_live."""
    controllers = set()
    _LIVE_SETS.append(controllers)
    return controllers


BUSY_MESSAGE = "Finish the running sketch tool first"


def any_sketch_modal_live():
    """True while a draw, fillet, trim or extend run is active. Two runs would edit the same
    bmesh and overwrite each other's snapshots."""
    return any(_LIVE_SETS)


# Regions that lie inside the WINDOW region's rectangle when region overlap is on.
OVERLAY_REGIONS = {'HEADER', 'TOOL_HEADER', 'TOOLS', 'UI', 'HUD', 'ASSET_SHELF', 'ASSET_SHELF_HEADER'}

# Footprint of the viewport navigation gizmo, in units of the UI scale factor, measured on Blender
# 5.1 as the pixels that differ with the gizmo on and off (UI scale 0.8, 1.0 and 1.5 at 150 percent
# OS scaling, sidebar open and closed, gizmo size 80 and 120). It sits in the top-right corner of the
# region's visible area. The axis ball alone is about 0.97 * size + 11 wide and tall; with the column
# of zoom, pan, camera and ortho buttons beneath it the height is about 1.1 * size + 128. Both are
# rounded up and include the gap to the edge of the visible area. Only sizes 80 and 120 were
# measured, so other sizes interpolate or extrapolate. The ball slope is a little above the measured
# 0.97 so that size 200 is covered too (checked at 30, 50, 80, 120 and 200).
GIZMO_BALL_PER_SIZE = 0.99
GIZMO_BALL_FIXED = 11.0
GIZMO_WITH_BUTTONS_PER_SIZE = 1.1
GIZMO_WITH_BUTTONS_FIXED = 128.0
# Pixels added on the open sides, for sub-pixel layout and hit slop.
GIZMO_MARGIN = 4.0

NAVIGATION = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'MOUSESMARTZOOM',
}
POINTER_MOVES = {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}
MODIFIER_KEYS = {'LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_CTRL', 'RIGHT_CTRL'}
PRESSES = {'PRESS', 'DOUBLE_CLICK'}
# Events that are neither keyboard nor mouse (timers, window focus, text input, ...). Swallowing
# them would starve other editors' timers, so they always go on to Blender.
SYSTEM_PREFIXES = ('TIMER', 'WINDOW_', 'ACTIONZONE_', 'EVT_', 'XR_', 'TEXTINPUT', 'NONE')
TOP_ROW_CHARS = {
    'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
    'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
    'PERIOD': '.', 'MINUS': '-',
}
NUMPAD_CHARS = {
    'NUMPAD_0': '0', 'NUMPAD_1': '1', 'NUMPAD_2': '2', 'NUMPAD_3': '3', 'NUMPAD_4': '4',
    'NUMPAD_5': '5', 'NUMPAD_6': '6', 'NUMPAD_7': '7', 'NUMPAD_8': '8', 'NUMPAD_9': '9',
    'NUMPAD_PERIOD': '.', 'NUMPAD_MINUS': '-',
}


class ViewProjector:
    """Mouse ray to sketch plane and sketch point to screen, for one 3D view region."""

    def __init__(self, region, rv3d, matrix_world, overlays=()):
        """overlays: (x, y, width, height) window-space rectangles of the area's header,
        toolbar and sidebar regions, which region overlap lets sit on top of this one."""
        self.region = region
        self.overlays = tuple(overlays)
        self.rv3d = rv3d
        self.world = matrix_world.copy()
        self.inverse = matrix_world.inverted_safe()
        # Why the last ray_to_local returned None (draw_logic.MISS_*).
        self.miss = None

    def contains(self, mouse, window):
        """True when the pointer is over the visible part of this region. mouse is relative to
        the region, window is the same point in window coordinates. Header, toolbar and sidebar
        events arrive here too, and with region overlap their rectangles lie inside this
        region's, so those are excluded explicitly."""
        if not (0.0 <= mouse[0] < self.region.width and 0.0 <= mouse[1] < self.region.height):
            return False
        return not any(x <= window[0] < x + w and y <= window[1] < y + h
                       for x, y, w, h in self.overlays)

    def ray_to_local(self, mouse):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, mouse)
        direction = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, mouse)
        xy, self.miss = draw_logic.ray_plane_hit(
            origin, direction, self.inverse, allow_behind=not self.rv3d.is_perspective)
        return xy

    def to_screen(self, xy):
        point = view3d_utils.location_3d_to_region_2d(
            self.region, self.rv3d, self.world @ Vector((xy[0], xy[1], 0.0)))
        return None if point is None else (point.x, point.y)


def over_region(projector, event):
    """True when the event's pointer is over the projector's visible region. No projector (no
    usable view) counts as over it: there is nothing to test against."""
    return projector is None or projector.contains(
        (event.mouse_region_x, event.mouse_region_y), (event.mouse_x, event.mouse_y))


def find_area(pointer):
    """The Area with this as_pointer() in any open window, or None. A modal keeps the pointer,
    not the Area: once the area is closed its struct dangles without raising ReferenceError,
    and the pointer lookup is the only safe way to ask whether it is still there."""
    for manager in bpy.data.window_managers:
        for window in manager.windows:
            for area in window.screen.areas:
                if area.as_pointer() == pointer:
                    return area
    return None


def area_callbacks(area):
    """(header, redraw) for the area, each looking it up again on every call and doing nothing
    when it is gone. header(text) sets the header text, header(None) clears it."""
    pointer = area.as_pointer()

    def header(text):
        live = find_area(pointer)
        if live is not None:
            live.header_text_set(text)

    def redraw():
        live = find_area(pointer)
        if live is not None:
            live.tag_redraw()

    return header, redraw


def window_region(area):
    """The area's WINDOW region, or None."""
    for region in area.regions:
        if region.type == 'WINDOW':
            return region
    return None


def overlay_rects(area):
    # Hidden regions can still report a size.
    return [(r.x, r.y, r.width, r.height) for r in area.regions
            if r.type in OVERLAY_REGIONS and r.width > 1 and r.height > 1]


def navigation_gizmo_rect(region, area, space, preferences):
    """Window-space rectangle (x, y, width, height) the viewport navigation gizmo covers, or None
    when it is not shown. Python cannot ask which gizmo is under the pointer, so this is the
    measured footprint (see GIZMO_*) anchored to the corner of the region's visible area: below
    the top-aligned header regions and left of the right-aligned sidebar."""
    # An area that is being closed has no active space left.
    if space is None or not (space.show_gizmo and space.show_gizmo_navigate):
        return None
    system, view = preferences.system, preferences.view
    # ui_scale is the factor the interface is really drawn at, OS scaling included. It reads
    # zero where there is no window (background mode); dpi / 72 is the same factor.
    fac = system.ui_scale if system.ui_scale > 0.0 else system.dpi / 72.0
    ball = GIZMO_BALL_PER_SIZE * view.gizmo_size_navigate_v3d + GIZMO_BALL_FIXED
    height = ball
    if view.show_navigate_ui:
        height = GIZMO_WITH_BUTTONS_PER_SIZE * view.gizmo_size_navigate_v3d + GIZMO_WITH_BUTTONS_FIXED
    left, bottom = region.x, region.y
    right, top = region.x + region.width, region.y + region.height
    for other in area.regions:
        if other.type not in OVERLAY_REGIONS or other.width <= 1 or other.height <= 1:
            continue
        if other.alignment == 'RIGHT' and left < other.x < right:
            right = other.x
        elif other.alignment == 'TOP' and bottom < other.y < top:
            top = other.y
    x = max(left, right - ball * fac - GIZMO_MARGIN)
    y = max(bottom, top - height * fac - GIZMO_MARGIN)
    return (x, y, right - x, top - y)


def projector_factory(obj):
    def factory():
        context = bpy.context
        region, rv3d, area = context.region, context.region_data, context.area
        if region is None or rv3d is None or area is None or obj.name not in bpy.data.objects:
            return None
        rects = overlay_rects(area)
        gizmo = navigation_gizmo_rect(region, area, area.spaces.active, context.preferences)
        if gizmo is not None:
            rects.append(gizmo)
        return ViewProjector(region, rv3d, obj.matrix_world, rects)
    return factory
