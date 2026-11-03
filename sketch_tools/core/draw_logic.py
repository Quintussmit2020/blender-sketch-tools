"""Decision logic of the Sketch draw tool (plan 8.3).

Plain tuples and floats only, no bpy.context and no bpy.ops, so every key behaviour runs
headless. The modal operator translates Blender events into the abstract events below and
applies the returned commands to the edit bmesh.

Events handed to DrawSession.handle (xy is sketch local 2D, mods is a Mods or None):
    ("MOVE", xy, mods)
    ("CLICK", xy, mods)                  left mouse press
    ("RELEASE", xy, mods, dragged)       left mouse release; dragged = moved past the click slop
    ("KEY", name)                        ENTER RMB ESC C N R O P A TAB BACKSPACE
    ("TYPE", ch)                         one of 0-9 . -

Commands returned (applied in order):
    ("ADD_VERT", ref, xy)                new sharp vertex
    ("ADD_EDGE", ref_a, ref_b)           edge, skipped by the applier if it already exists
    ("END",)                             chain finished, keep everything added
    ("CANCEL",)                          drop everything added this session

A ref is an int. Refs >= 0 are existing vertices (from a vertex snap); refs < 0 are vertices
this session creates, so a snap back onto the session's first vertex closes the chain.
Closing (C, or a snap onto the first vertex) is an ADD_EDGE from the last vertex to the
first one followed by END.

Mods.ctrl means snapping is on (the controller folds Blender's snap toggle and the Ctrl key into it,
C23), and Mods.snap is the snap target the controller found: a Snap on an existing sketch vertex (ref >= 0,
reused) or a scene snap (ref None, kind set), a point of another object already projected onto the
sketch plane, which the session treats as a free point that becomes a new vertex.
"""

import math
from collections import namedtuple

from mathutils import Vector

from . import shapes
from .shapes import circle_points

POLYLINE = 'POLYLINE'
RECTANGLE = 'RECTANGLE'
CIRCLE = 'CIRCLE'
ARC_3PT = 'ARC_3PT'
ARC_CENTER = 'ARC_CENTER'
ARCS = (ARC_3PT, ARC_CENTER)

ANGLE_STEP = math.radians(45.0)
# Shorter segments are treated as "no segment yet" (mouse has not left the anchor).
MIN_LENGTH = 1e-6
# Fallbacks when the add-on is registered without being enabled through preferences (tests).
DEFAULT_GRID_INCREMENT = 0.1
DEFAULT_CIRCLE_SEGMENTS = 32
DEFAULT_SNAP_PIXELS = 12
# Why a ray found no point on the sketch plane.
MISS_EDGE_ON = "EDGE_ON"
MISS_BEHIND = "BEHIND"
# |ray . plane normal| / |ray| below this is a ray skimming the plane (about 1 degree).
GRAZING_EPS = 0.02
MAX_TYPED_CHARS = 14

Mods = namedtuple("Mods", "shift ctrl snap", defaults=(False, False, None))
# ref is the existing sketch vertex to reuse, or None for a scene snap, which becomes a new vertex at
# xy (the scene point projected onto the sketch plane). kind names what a scene snap sits on, one of
# SCENE_SNAP_LABELS; None for a vertex of the sketch itself.
Snap = namedtuple("Snap", "ref xy kind", defaults=(None,))
# Keys are core/snap3d's VERTEX, MIDPOINT and EDGE kinds.
SCENE_SNAP_LABELS = {
    "VERTEX": "Scene vertex",
    "MIDPOINT": "Scene edge midpoint",
    "EDGE": "Scene point on edge",
}
# A resolved point: xy, ref (None = would become a new vertex).
Target = namedtuple("Target", "xy ref")

_NO_MODS = Mods()


def effective_unit_scale(unit_system, scale_length):
    """Displayed units per local unit. Blender ignores scale_length when the unit system is None."""
    return scale_length if unit_system != 'NONE' and scale_length > 0.0 else 1.0


def grid_snap_point(xy, step):
    return (round(xy[0] / step) * step, round(xy[1] / step) * step)


def grid_round(value, step):
    return round(value / step) * step


def lock_angle(angle, reference, step=ANGLE_STEP):
    """Nearest multiple of 45 degrees from the reference direction. That covers lines at 0, 45
    and 90 degrees to the previous segment, on either side of it and behind it."""
    return reference + round((angle - reference) / step) * step


def turn_angle_deg(vector_angle, reference):
    """Signed turn of a direction against the reference, in (-180, 180] degrees."""
    deg = math.degrees(vector_angle - reference)
    deg = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if deg <= -180.0 + 1e-9 else deg


def nearest_snap(mouse, candidates, threshold):
    """The candidate nearest to the mouse within threshold pixels, else None.

    candidates yields (ref, local_xy, screen_xy); screen_xy None (behind the camera) is skipped."""
    best = None
    best_dist = threshold
    for ref, xy, screen in candidates:
        if screen is None:
            continue
        dist = math.hypot(screen[0] - mouse[0], screen[1] - mouse[1])
        if dist <= best_dist:
            best_dist = dist
            best = Snap(ref, tuple(xy))
    return best


def ray_plane_hit(origin, direction, world_inverse, allow_behind=False, grazing_eps=GRAZING_EPS):
    """(xy, miss): sketch local (x, y) where a world space ray meets the sketch plane (local
    z = 0), or (None, reason) with MISS_EDGE_ON when the ray skims or is parallel to the plane
    and MISS_BEHIND when it meets the plane behind the ray origin (a perspective camera looking
    away). allow_behind is for orthographic views, whose ray origin sits inside the scene."""
    o = world_inverse @ Vector(origin)
    d = world_inverse.to_3x3() @ Vector(direction)
    length = d.length
    if length == 0.0 or abs(d.z) / length < grazing_eps:
        return None, MISS_EDGE_ON
    t = -o.z / d.z
    if t <= 0.0 and not allow_behind:
        return None, MISS_BEHIND
    return (o.x + d.x * t, o.y + d.y * t), None


def ray_to_plane_local(origin, direction, world_inverse, allow_behind=False, grazing_eps=GRAZING_EPS):
    """ray_plane_hit without the reason: the point, or None."""
    return ray_plane_hit(origin, direction, world_inverse, allow_behind, grazing_eps)[0]


class NumericInput:
    """Typed numbers for the rubber-banded value(s): digits, '.', '-', Backspace, Tab.

    One text buffer per field; Tab cycles the active field. '-' toggles the sign of the active
    field wherever it is pressed (as in Blender's transform input). Plain floats, no unit parsing;
    DrawSession reads length fields in the scene base unit (unit scale applied)."""

    def __init__(self, labels):
        self.labels = tuple(labels)
        self.texts = [""] * len(self.labels)
        self.active = 0

    def type(self, ch):
        if not self.labels:
            return
        text = self.texts[self.active]
        if ch == '-':
            self.texts[self.active] = text[1:] if text.startswith('-') else '-' + text
        elif ch == '.':
            if '.' not in text and len(text) < MAX_TYPED_CHARS:
                self.texts[self.active] = text + '.'
        elif ch.isdigit() and len(text) < MAX_TYPED_CHARS:
            self.texts[self.active] = text + ch

    def backspace(self):
        if not self.labels:
            return
        if self.texts[self.active]:
            self.texts[self.active] = self.texts[self.active][:-1]
        elif self.active > 0:
            self.active -= 1

    def next_field(self):
        if not self.labels:
            return
        self.active = (self.active + 1) % len(self.labels)

    def value(self, index):
        """Float typed in a field, None when empty or not a number yet ('-', '.', '-.')."""
        try:
            return float(self.texts[index])
        except ValueError:
            return None

    def has_value(self):
        return any(self.value(i) is not None for i in range(len(self.labels)))

    def is_typing(self):
        return any(self.texts)

    def clear(self):
        self.texts = [""] * len(self.labels)
        self.active = 0


class Pt:
    """A point of the current chain. emitted: the vertex already exists in the sketch."""

    __slots__ = ("ref", "xy", "emitted")

    def __init__(self, ref, xy, emitted):
        self.ref = ref
        self.xy = xy
        self.emitted = emitted


class Preview:
    """What the overlay draws. strips are polylines in sketch local 2D (already closed where
    the shape is closed), markers are small squares (points not in the mesh yet), snap_xy is the
    vertex the cursor would snap to, guide is the angle lock line."""

    def __init__(self):
        self.strips = []
        self.markers = []
        self.snap_xy = None
        self.guide = None
        self.lines = []


_FIELDS = {
    POLYLINE: ("Length", "Angle"),
    RECTANGLE: ("Width", "Height"),
    CIRCLE: ("Radius",),
    # The three-point arc takes no typed values. The centre arc types its radius first, then
    # (once the start is placed) its angle.
    ARC_3PT: (),
    ARC_CENTER: ("Radius",),
}
_ANGLE_FIELD = ("Angle",)

_MODE_KEYS = {"R": RECTANGLE, "O": CIRCLE, "P": POLYLINE}

_MODE_HELP = {
    POLYLINE: "Polyline: LMB point, RMB/Enter end, C close, N end chain",
    RECTANGLE: "Rectangle: click opposite corner or drag, type W Tab H",
    CIRCLE: "Circle: click to set the radius, type R",
    ARC_3PT: "Three-point arc: click start, a point on the arc, end",
    ARC_CENTER: "Centre arc: click centre, start, then end",
}

_TYPE_HELP = {
    ARC_3PT: "Shift locks the end angle, hold Ctrl to flip snapping, no typed values",
    ARC_CENTER: "Shift angle lock, hold Ctrl to flip snapping, type radius then angle (deg), Enter commits",
}


def _fmt(value):
    return f"{value:.3f}"


class DrawSession:
    """One chain, rectangle, circle or arc, from the first point to END or CANCEL.

    The first point is not turned into a vertex until a second point exists, so ending or
    cancelling after one click leaves the mesh untouched, and R / O / P / A can still change the
    mode after the first click. The arcs place nothing until their third click."""

    def __init__(self, mode=POLYLINE, grid=DEFAULT_GRID_INCREMENT, circle_segments=DEFAULT_CIRCLE_SEGMENTS,
                 unit_scale=1.0):
        """grid is the step in sketch local units. unit_scale is the effective scene unit scale
        (effective_unit_scale): typed lengths are in the scene base unit (m for Metric, not length_unit), so a typed L is L / unit_scale
        local units, and the readout shows local lengths times unit_scale."""
        self.grid = grid
        self.unit_scale = unit_scale if unit_scale > 0.0 else 1.0
        self.circle_segments = max(3, int(circle_segments))
        self.points = []
        self.done = False
        self.cancelled = False
        self.cursor = None
        self.mods = _NO_MODS
        self._next_ref = -1
        self._set_mode(mode)

    def _set_mode(self, mode):
        self.mode = mode
        self.numeric = NumericInput(_FIELDS[mode])

    def _alloc(self):
        ref = self._next_ref
        self._next_ref -= 1
        return ref

    # -- event entry point ------------------------------------------------

    def handle(self, event):
        if self.done:
            return []
        kind = event[0]
        if kind == "MOVE":
            self._set_cursor(event[1], event[2] if len(event) > 2 else None)
            return []
        if kind == "CLICK":
            self._set_cursor(event[1], event[2] if len(event) > 2 else None)
            return self._click()
        if kind == "RELEASE":
            self._set_cursor(event[1], event[2] if len(event) > 2 else None)
            dragged = event[3] if len(event) > 3 else False
            # Arcs are click only: a drag has no clear second and third point.
            if dragged and self.mode in (RECTANGLE, CIRCLE) and len(self.points) == 1:
                return self._commit_shape()
            return []
        if kind == "KEY":
            return self._key(event[1])
        if kind == "TYPE":
            if self.points:
                self.numeric.type(event[1])
            return []
        raise ValueError(f"unknown draw event {kind!r}")

    def _typed_length(self, index):
        """Typed length field in sketch local units, None when empty."""
        value = self.numeric.value(index)
        return None if value is None else value / self.unit_scale

    def _set_cursor(self, xy, mods):
        self.cursor = tuple(xy)
        self.mods = mods or _NO_MODS

    def _key(self, name):
        if name == "ESC":
            return self._cancel()
        if name in ("RMB", "N"):
            return self._end()
        if name == "ENTER":
            if not self.numeric.has_value():
                return self._end()
            return self._commit_typed()
        if name == "C":
            return self._close()
        if name == "TAB":
            if self.points:
                self.numeric.next_field()
            return []
        if name == "BACKSPACE":
            self.numeric.backspace()
            return []
        if name in _MODE_KEYS or name == "A":
            # Only before any vertex exists, so nothing has to be undone.
            if len(self.points) <= 1:
                if name == "A":
                    # A alternates the two arcs, and from any other shape starts with the three-point arc.
                    target = ARC_CENTER if self.mode == ARC_3PT else ARC_3PT
                else:
                    target = _MODE_KEYS[name]
                if target != self.mode:
                    self._set_mode(target)
            return []
        raise ValueError(f"unknown draw key {name!r}")

    # -- ending ------------------------------------------------------------

    def _finish(self, commands):
        self.done = True
        return commands

    def _cancel(self):
        self.cancelled = True
        return self._finish([("CANCEL",)])

    def _end(self):
        """RMB, N, or Enter with nothing typed. A lone point, or a rectangle or circle without
        its second point, has nothing to keep."""
        if self.mode == POLYLINE and len(self.points) >= 2:
            return self._finish([("END",)])
        return self._cancel()

    def _close(self):
        if self.mode != POLYLINE or len(self.points) < 3:
            return []
        return self._finish([("ADD_EDGE", self.points[-1].ref, self.points[0].ref), ("END",)])

    # -- clicking ----------------------------------------------------------

    def _click(self):
        if self.mode == POLYLINE:
            if not self.points:
                return self._place_first()
            return self._poly_commit(self._poly_target())
        if not self.points:
            return self._place_first()
        return self._commit_shape()

    def _place_first(self):
        if self.cursor is None:
            return []
        snap = self.mods.snap if self.mods.ctrl else None
        if snap is not None and snap.ref is not None:
            self.points.append(Pt(snap.ref, snap.xy, True))
        elif snap is not None:
            self.points.append(Pt(self._alloc(), snap.xy, False))
        else:
            xy = grid_snap_point(self.cursor, self.grid) if self.mods.ctrl else self.cursor
            self.points.append(Pt(self._alloc(), xy, False))
        self.numeric.clear()
        return []

    @staticmethod
    def _emit(pt, commands):
        if not pt.emitted:
            commands.append(("ADD_VERT", pt.ref, pt.xy))
            pt.emitted = True

    # -- polyline ----------------------------------------------------------

    def _prev_angle(self):
        """Direction of the previous segment; sketch X for the first segment."""
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[-2].xy, self.points[-1].xy
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _poly_vector(self, use_typed=True):
        """(length, angle, target) of the segment from the last point to the cursor.

        length is signed (typed negative lengths run backwards); target.ref is set when the
        cursor snaps to an existing vertex; target is None while the segment has no length."""
        anchor = self.points[-1].xy
        prev = self._prev_angle()
        cur = self.cursor if self.cursor is not None else anchor
        dx, dy = cur[0] - anchor[0], cur[1] - anchor[1]
        raw_len = math.hypot(dx, dy)
        raw_ang = math.atan2(dy, dx) if raw_len > MIN_LENGTH else prev
        typed_len = self._typed_length(0) if use_typed else None
        typed_ang = self.numeric.value(1) if use_typed else None
        mods = self.mods

        if typed_len is None and typed_ang is None and mods.ctrl and mods.snap is not None:
            sx, sy = mods.snap.xy
            length = math.hypot(sx - anchor[0], sy - anchor[1])
            if length < MIN_LENGTH:
                return 0.0, raw_ang, None
            return length, math.atan2(sy - anchor[1], sx - anchor[0]), Target(mods.snap.xy, mods.snap.ref)

        angle = raw_ang
        length = raw_len
        if typed_ang is not None:
            angle = prev + math.radians(typed_ang)
        elif mods.shift:
            angle = lock_angle(raw_ang, prev)

        if typed_len is not None:
            length = typed_len
        elif mods.ctrl:
            if mods.shift or typed_ang is not None:
                length = grid_round(length, self.grid)
            else:
                gx, gy = grid_snap_point(cur, self.grid)
                length = math.hypot(gx - anchor[0], gy - anchor[1])
                if length >= MIN_LENGTH:
                    return length, math.atan2(gy - anchor[1], gx - anchor[0]), Target((gx, gy), None)
                return 0.0, angle, None

        if abs(length) < MIN_LENGTH:
            return length, angle, None
        xy = (anchor[0] + length * math.cos(angle), anchor[1] + length * math.sin(angle))
        return length, angle, Target(xy, None)

    def _poly_target(self):
        return self._poly_vector()[2]

    def _poly_commit(self, target):
        if target is None:
            return []
        anchor = self.points[-1]
        first = self.points[0]
        if target.ref is not None:
            if target.ref == anchor.ref:
                return []
            if target.ref == first.ref and len(self.points) < 3:
                return []
        commands = []
        self._emit(anchor, commands)
        if target.ref is not None:
            pt = Pt(target.ref, target.xy, True)
        else:
            pt = Pt(self._alloc(), target.xy, False)
            self._emit(pt, commands)
        commands.append(("ADD_EDGE", anchor.ref, pt.ref))
        if target.ref is not None and target.ref == first.ref:
            return self._finish(commands + [("END",)])
        self.points.append(pt)
        self.numeric.clear()
        return commands

    def _commit_typed(self):
        if not self.points:
            return []
        if self.mode == POLYLINE:
            return self._poly_commit(self._poly_target())
        return self._commit_shape()

    # -- rectangle and circle ----------------------------------------------

    def _rect_corners(self):
        """[(xy, ref_or_None) x 4] from the first corner to the cursor, or None when degenerate."""
        c0 = self.points[0].xy
        cur = self.cursor if self.cursor is not None else c0
        mods = self.mods
        typed_w, typed_h = self._typed_length(0), self._typed_length(1)
        c1_ref = None
        if typed_w is None and typed_h is None and mods.ctrl and mods.snap is not None:
            cur = mods.snap.xy
            c1_ref = mods.snap.ref
        elif mods.ctrl:
            cur = grid_snap_point(cur, self.grid)
        w, h = cur[0] - c0[0], cur[1] - c0[1]
        # Typed sizes are measured toward the cursor side; a minus flips them.
        if typed_w is not None:
            w = typed_w * (-1.0 if w < 0.0 else 1.0)
            c1_ref = None
        if typed_h is not None:
            h = typed_h * (-1.0 if h < 0.0 else 1.0)
            c1_ref = None
        if abs(w) < MIN_LENGTH or abs(h) < MIN_LENGTH:
            return None
        ring = shapes.rectangle_points(c0, (c0[0] + w, c0[1] + h))
        if w * h < 0.0:
            # Keep the vertex order of Phase 3 (x first, then y) for corners on the other side
            # of the first one; the resolver normalises the winding anyway.
            ring[1], ring[3] = ring[3], ring[1]
        return [(ring[0], self.points[0].ref), (ring[1], None), (ring[2], c1_ref), (ring[3], None)]

    def _circle_points(self):
        c = self.points[0].xy
        cur = self.cursor if self.cursor is not None else c
        mods = self.mods
        typed = self._typed_length(0)
        if typed is not None:
            radius = abs(typed)
        else:
            if mods.ctrl and mods.snap is not None:
                cur = mods.snap.xy
            radius = math.hypot(cur[0] - c[0], cur[1] - c[1])
            if mods.ctrl and mods.snap is None:
                radius = grid_round(radius, self.grid)
        if radius < MIN_LENGTH:
            return None
        return circle_points(c, radius, self.circle_segments)

    # -- arcs --------------------------------------------------------------

    def _snap_or_grid(self, cur):
        """Target for a free point: the snapped vertex, else the grid point under Ctrl."""
        mods = self.mods
        if mods.ctrl and mods.snap is not None:
            return Target(mods.snap.xy, mods.snap.ref)
        return Target(grid_snap_point(cur, self.grid) if mods.ctrl else cur, None)

    def _arc3_end(self):
        """Third point of the three-point arc. Shift locks the chord from the start to a multiple
        of 45 degrees from sketch X (the polyline's first segment rule)."""
        p0 = self.points[0].xy
        cur = self.cursor if self.cursor is not None else p0
        mods = self.mods
        if mods.shift and not (mods.ctrl and mods.snap is not None):
            dx, dy = cur[0] - p0[0], cur[1] - p0[1]
            length = math.hypot(dx, dy)
            if mods.ctrl:
                length = grid_round(length, self.grid)
            angle = lock_angle(math.atan2(dy, dx), 0.0)
            return Target((p0[0] + length * math.cos(angle), p0[1] + length * math.sin(angle)), None)
        return self._snap_or_grid(cur)

    def _centre_start(self):
        """Target for the centre arc's start, the point that sets the radius, or None while the
        radius is zero. Typed radius, Shift and Ctrl follow the circle's rules; Shift also locks
        the start direction to multiples of 45 degrees from sketch X."""
        c = self.points[0].xy
        cur = self.cursor if self.cursor is not None else c
        mods = self.mods
        typed = self._typed_length(0)
        if typed is None and mods.ctrl and mods.snap is not None:
            if math.hypot(mods.snap.xy[0] - c[0], mods.snap.xy[1] - c[1]) < MIN_LENGTH:
                return None
            return Target(mods.snap.xy, mods.snap.ref)
        dx, dy = cur[0] - c[0], cur[1] - c[1]
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if length > MIN_LENGTH else 0.0
        if mods.shift:
            angle = lock_angle(angle, 0.0)
        if typed is not None:
            radius = abs(typed)
        else:
            radius = grid_round(length, self.grid) if mods.ctrl else length
        if radius < MIN_LENGTH:
            return None
        return Target((c[0] + radius * math.cos(angle), c[1] + radius * math.sin(angle)), None)

    def _centre_end(self):
        """Target for the centre arc's end on the start's circle, or None while the direction is
        undefined. Typed angle is degrees counter-clockwise from the start. Shift locks the sweep to
        multiples of 45 degrees. Ctrl turns the cursor into the nearest vertex (the end reuses it
        when it lies on the circle) or grid point."""
        c, start = self.points[0].xy, self.points[1].xy
        radius = math.hypot(start[0] - c[0], start[1] - c[1])
        a_start = math.atan2(start[1] - c[1], start[0] - c[0])
        typed = self.numeric.value(0)
        snap_ref = None
        if typed is not None:
            angle = a_start + math.radians(typed)
        else:
            target = self._snap_or_grid(self.cursor if self.cursor is not None else c)
            snap_ref = target.ref
            dx, dy = target.xy[0] - c[0], target.xy[1] - c[1]
            if math.hypot(dx, dy) < MIN_LENGTH:
                return None
            angle = math.atan2(dy, dx)
            if self.mods.shift and snap_ref is None:
                angle = lock_angle(angle, a_start)
        xy = (c[0] + radius * math.cos(angle), c[1] + radius * math.sin(angle))
        if snap_ref is not None:
            on_circle = math.hypot(xy[0] - target.xy[0], xy[1] - target.xy[1]) <= 1e-6 * max(1.0, radius)
            snap_ref = snap_ref if on_circle else None
        return Target(xy, snap_ref)

    def _arc_points(self):
        """(points, last_ref) of the arc being drawn, from its start to its end: None points
        while it is degenerate. last_ref is an existing vertex the end reuses, else None."""
        if self.mode == ARC_3PT:
            end = self._arc3_end()
            return shapes.arc_three_points(self.points[0].xy, self.points[1].xy, end.xy, self.circle_segments), end.ref
        end = self._centre_end()
        if end is None:
            return None, None
        return shapes.arc_centre(self.points[0].xy, self.points[1].xy, end.xy, self.circle_segments), end.ref

    def _commit_arc(self):
        """Second click places the middle point (three-point) or the start (centre arc); the
        third commits the arc as an open chain from its start to its end."""
        if len(self.points) == 1:
            if self.mode == ARC_3PT:
                mid = self._snap_or_grid(self.cursor).xy
                first = self.points[0].xy
                # A second click on the start would leave no circle through any third point.
                if math.hypot(mid[0] - first[0], mid[1] - first[1]) < MIN_LENGTH:
                    return []
                self.points.append(Pt(None, mid, False))
                return []
            start = self._centre_start()
            if start is None:
                return []
            if start.ref is not None:
                self.points.append(Pt(start.ref, start.xy, True))
            else:
                self.points.append(Pt(self._alloc(), start.xy, False))
            self.numeric = NumericInput(_ANGLE_FIELD)
            return []
        arc, last_ref = self._arc_points()
        if arc is None:
            return []
        first = self.points[0] if self.mode == ARC_3PT else self.points[1]
        commands = []
        self._emit(first, commands)
        refs = [first.ref]
        for i, xy in enumerate(arc[1:], start=1):
            if i == len(arc) - 1 and last_ref is not None:
                refs.append(last_ref)
                continue
            ref = self._alloc()
            commands.append(("ADD_VERT", ref, xy))
            refs.append(ref)
        commands.extend(("ADD_EDGE", refs[i], refs[i + 1]) for i in range(len(refs) - 1))
        commands.append(("END",))
        return self._finish(commands)

    def _commit_shape(self):
        if self.mode in ARCS:
            return self._commit_arc()
        commands = []
        if self.mode == RECTANGLE:
            corners = self._rect_corners()
            if corners is None:
                return []
            refs = []
            for i, (xy, ref) in enumerate(corners):
                if i == 0 and self.points[0].emitted:
                    refs.append(self.points[0].ref)
                elif ref is not None and ref >= 0:
                    refs.append(ref)
                else:
                    pt = self.points[0] if i == 0 else Pt(self._alloc(), xy, False)
                    self._emit(pt, commands)
                    refs.append(pt.ref)
        else:
            pts = self._circle_points()
            if pts is None:
                return []
            refs = []
            for xy in pts:
                ref = self._alloc()
                commands.append(("ADD_VERT", ref, xy))
                refs.append(ref)
        for i in range(len(refs)):
            commands.append(("ADD_EDGE", refs[i], refs[(i + 1) % len(refs)]))
        commands.append(("END",))
        return self._finish(commands)

    # -- feedback ------------------------------------------------------------

    def _snap_label(self):
        """Name of the scene point under the cursor, None for a sketch vertex or no snap."""
        snap = self.mods.snap
        if self.mods.ctrl and snap is not None and snap.kind is not None:
            return SCENE_SNAP_LABELS[snap.kind]
        return None

    def preview(self):
        out = self._preview()
        label = self._snap_label()
        if label is not None and not self.done:
            out.lines = [label] + out.lines
        return out

    def _preview(self):
        out = Preview()
        if self.done or not self.points:
            if self.mods.ctrl and self.mods.snap is not None:
                out.snap_xy = self.mods.snap.xy
            return out
        out.markers.extend(pt.xy for pt in self.points if not pt.emitted)
        if self.mods.ctrl and self.mods.snap is not None:
            out.snap_xy = self.mods.snap.xy
        if self.mode == POLYLINE:
            self._preview_polyline(out)
        elif self.mode == RECTANGLE:
            self._preview_rectangle(out)
        elif self.mode == CIRCLE:
            self._preview_circle(out)
        else:
            self._preview_arc(out)
        return out

    def _field_lines(self, values):
        """values: [(label, number, suffix)], numbers already in the scene base unit. While
        typing, typed text replaces the number and the active field is marked."""
        lines = []
        for i, (label, number, suffix) in enumerate(values):
            if self.numeric.is_typing():
                text = self.numeric.texts[i]
                marker = ">" if i == self.numeric.active else " "
                if i == self.numeric.active:
                    text += "_"
                if not self.numeric.texts[i] and i != self.numeric.active:
                    text = _fmt(number) + suffix
                lines.append(f"{marker} {label}: {text}")
            else:
                lines.append(f"{label}: {_fmt(number)}{suffix}")
        return lines

    def _preview_polyline(self, out):
        anchor = self.points[-1].xy
        length, angle, target = self._poly_vector()
        if self.mods.shift and self.numeric.value(1) is None:
            reach = max(abs(length) * 2.0, 1.0)
            ux, uy = math.cos(angle), math.sin(angle)
            out.guide = ((anchor[0] - ux * reach, anchor[1] - uy * reach),
                         (anchor[0] + ux * reach, anchor[1] + uy * reach))
        if target is not None:
            out.strips.append([anchor, target.xy])
            vec_angle = math.atan2(target.xy[1] - anchor[1], target.xy[0] - anchor[0])
        else:
            vec_angle = angle
        out.lines = self._field_lines([
            ("Length", length * self.unit_scale, ""),
            ("Angle", turn_angle_deg(vec_angle, self._prev_angle()), " deg"),
        ])

    def _preview_rectangle(self, out):
        corners = self._rect_corners()
        c0 = self.points[0].xy
        if corners is None:
            w = h = 0.0
        else:
            ring = [xy for xy, _ in corners]
            out.strips.append(ring + [ring[0]])
            w, h = ring[2][0] - c0[0], ring[2][1] - c0[1]
        out.lines = self._field_lines([("Width", abs(w) * self.unit_scale, ""), ("Height", abs(h) * self.unit_scale, "")])

    def _preview_circle(self, out):
        pts = self._circle_points()
        radius = 0.0
        if pts is not None:
            c = self.points[0].xy
            radius = math.hypot(pts[0][0] - c[0], pts[0][1] - c[1])
            out.strips.append(pts + [pts[0]])
        out.lines = self._field_lines([("Radius", radius * self.unit_scale, "")])

    def _preview_arc(self, out):
        p0 = self.points[0].xy
        if self.mode == ARC_3PT:
            self._preview_arc3(out, p0)
        else:
            self._preview_centre_arc(out, p0)

    def _preview_arc3(self, out, p0):
        if len(self.points) == 1:
            cur = self._snap_or_grid(self.cursor if self.cursor is not None else p0).xy
            out.strips.append([p0, cur])
            out.lines = self._field_lines([("Radius", 0.0, "")])
            return
        mid = self.points[1].xy
        end = self._arc3_end().xy
        arc = shapes.arc_three_points(p0, mid, end, self.circle_segments)
        circle = shapes.circumcircle(p0, mid, end)
        # Collinear or coincident points have no arc yet: show the corner they make.
        out.strips.append(arc if arc is not None else [p0, mid, end])
        out.lines = self._field_lines([("Radius", circle[1] * self.unit_scale if circle else 0.0, "")])

    def _preview_centre_arc(self, out, c):
        if len(self.points) == 1:
            start = self._centre_start()
            radius = 0.0
            if start is not None:
                out.strips.append([c, start.xy])
                radius = math.hypot(start.xy[0] - c[0], start.xy[1] - c[1])
            out.lines = self._field_lines([("Radius", radius * self.unit_scale, "")])
            return
        start = self.points[1].xy
        radius = math.hypot(start[0] - c[0], start[1] - c[1])
        end = self._centre_end()
        arc = shapes.arc_centre(c, start, end.xy, self.circle_segments) if end is not None else None
        sweep = 0.0
        if arc is not None:
            out.strips.append(arc)
            out.strips.append([c, start])
            out.strips.append([c, arc[-1]])
            sweep = math.degrees((math.atan2(arc[-1][1] - c[1], arc[-1][0] - c[0])
                                  - math.atan2(start[1] - c[1], start[0] - c[0])) % (2.0 * math.pi))
        else:
            out.strips.append([c, start])
        out.lines = self._field_lines([("Angle", sweep, " deg")]) + [f"Radius: {_fmt(radius * self.unit_scale)}"]

    def header_text(self):
        parts = [_MODE_HELP[self.mode]]
        if not self.points:
            parts.append("click to place the first point, hold Ctrl to flip snapping")
        elif self.mode in _TYPE_HELP:
            parts.append(_TYPE_HELP[self.mode])
        else:
            parts.append("Shift angle lock, hold Ctrl to flip snapping, type lengths (base unit), Tab next field")
        if len(self.points) <= 1:
            # In an arc shape A goes to the other arc, which is not obvious from "A arc".
            parts.append("R rect, O circle, P poly, " + ("A other arc" if self.mode in ARCS else "A arc"))
        # The tool swallows keys during a run: Esc drops the whole run and Ctrl+Z is off during it.
        parts.append("Esc cancels, Ctrl+Z off")
        return " | ".join(parts)
