"""Decision logic of the modal fillet (plan 8.4). Plain floats, tuples and mathutils vectors: no
bpy.context and no bpy.ops, so every key behaviour runs headless.

find_corners decides which selected vertices can be rounded, FilletSession turns abstract events
into the radius to commit. Events handed to FilletSession.handle:
    ("MOVE", x, units_per_pixel, ctrl)   pointer moved inside the viewport (x in pixels)
    ("LEAVE",)                           pointer left the viewport
    ("TYPE", ch)                         one of 0-9 .  (a minus is ignored: radii are not negative)
    ("KEY", name)                        ENTER LMB (commit), ESC RMB (cancel), BACKSPACE
The session never touches the mesh: the caller writes the radius when session.done is set and
session.cancelled is not, so cancelling has nothing to restore."""

from collections import namedtuple

from mathutils import Vector
from mathutils.kdtree import KDTree

from .draw_logic import NumericInput, grid_round
from .geometry2d import fillet_corner
from .resolve import WELD_DIST

# A selected vertex that can be rounded: its neighbours along the chain, position (all 2D
# Vectors) and the radius it has now.
Corner = namedtuple("Corner", "index prev p next radius")
# Preview of one corner: where it is, the arc that replaces it ([p] when nothing is drawn) and
# whether the radius had to shrink to fit (the resolver's clamp).
Arc = namedtuple("Arc", "corner points clamped")


def _is_roundable(prev, p, nxt):
    """True unless fillet_corner skips the corner for any radius: a zero-length neighbour, a
    near-straight run or a degenerate spike. Asking fillet_corner keeps this in step with it."""
    return len(fillet_corner(prev, p, nxt, 1.0, 1)[0]) > 1


def find_corners(coords, radii, edges, selected):
    """(corners, skipped) for the selected vertex indices of a sketch.

    coords: (x, y) per vertex, radii: current radius per vertex, edges: (a, b) index pairs.
    A vertex is roundable when the resolver would see it as the interior of a chain with a
    real corner: exactly two edges (chain endpoints and junctions are not interior), no other
    vertex within the resolver's weld distance (it would be merged into a junction first, C9)
    and a corner that is neither straight nor degenerate. skipped counts the rest."""
    neighbours = [[] for _ in coords]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    tree = KDTree(len(coords))
    for i, (x, y) in enumerate(coords):
        tree.insert((x, y, 0.0), i)
    tree.balance()

    corners = []
    skipped = 0
    for i in sorted(selected):
        if len(neighbours[i]) != 2:
            skipped += 1
            continue
        if len(tree.find_range((coords[i][0], coords[i][1], 0.0), WELD_DIST)) > 1:
            skipped += 1
            continue
        prev, p, nxt = (Vector(coords[k]) for k in (neighbours[i][0], i, neighbours[i][1]))
        if not _is_roundable(prev, p, nxt):
            skipped += 1
            continue
        corners.append(Corner(i, prev, p, nxt, radii[i]))
    return corners, skipped


def initial_radius(corners):
    """Where the drag starts: the radius all corners share, else 0."""
    if not corners:
        return 0.0
    first = corners[0].radius
    return first if all(abs(c.radius - first) < 1e-9 for c in corners) else 0.0


def arc_preview(corners, radius, segments):
    """[Arc] for every corner at the given radius, built by the function the resolver uses."""
    arcs = []
    for c in corners:
        points, clamped = fillet_corner(c.prev, c.p, c.next, radius, segments)
        arcs.append(Arc(c.p, points, clamped))
    return arcs


class FilletPreview:
    """What the overlay draws: arcs in sketch local 2D and the text lines for the readout."""

    def __init__(self, arcs, lines):
        self.arcs = arcs
        self.lines = lines


def _fmt(value):
    return f"{value:.3f}"


class FilletSession:
    def __init__(self, corners, skipped, segments, grid, unit_scale=1.0):
        """grid is the step in sketch local units. unit_scale is the effective scene unit scale
        (draw_logic.effective_unit_scale): a typed radius is in the scene base unit, so a typed R is
        R / unit_scale local units, and the readout shows local lengths times unit_scale (C15)."""
        self.corners = tuple(corners)
        self.skipped = skipped
        self.segments = max(1, int(segments))
        self.grid = grid
        self.unit_scale = unit_scale if unit_scale > 0.0 else 1.0
        self.numeric = NumericInput(("Radius",))
        self.ctrl = False
        self.done = False
        self.cancelled = False
        self._last_x = None
        # Accumulated drag, kept apart from the shown radius so that Ctrl rounding does not eat
        # small moves and a drag past zero does not leave a dead zone on the way back.
        self._raw = initial_radius(self.corners)

    @property
    def radius(self):
        """Radius in sketch local units: the typed value, else the drag (rounded to the grid
        with Ctrl)."""
        typed = self.numeric.value(0)
        if typed is not None:
            return max(0.0, typed / self.unit_scale)
        raw = grid_round(self._raw, self.grid) if self.ctrl else self._raw
        return max(0.0, raw)

    def handle(self, event):
        if self.done:
            return
        kind = event[0]
        if kind == "MOVE":
            _, x, units_per_pixel, ctrl = event
            self.ctrl = bool(ctrl)
            if self._last_x is not None and units_per_pixel:
                self._raw = max(0.0, self._raw + (x - self._last_x) * units_per_pixel)
            self._last_x = x
        elif kind == "LEAVE":
            # Re-entering must not read the way across the gap as one big drag.
            self._last_x = None
        elif kind == "TYPE":
            if event[1] != '-':
                self.numeric.type(event[1])
        elif kind == "KEY":
            self._key(event[1])
        else:
            raise ValueError(f"unknown fillet event {kind!r}")

    def _key(self, name):
        if name in ("ENTER", "LMB"):
            self.done = True
        elif name in ("ESC", "RMB"):
            self.done = True
            self.cancelled = True
        elif name == "BACKSPACE":
            self.numeric.backspace()
        else:
            raise ValueError(f"unknown fillet key {name!r}")

    # -- feedback ------------------------------------------------------------

    def preview(self):
        arcs = arc_preview(self.corners, self.radius, self.segments)
        lines = [self._radius_line()]
        clamped = sum(1 for a in arcs if a.clamped)
        if clamped:
            lines.append(f"Clamped at {clamped} of {len(arcs)} corners")
        return FilletPreview(arcs, lines)

    def _radius_line(self):
        if self.numeric.is_typing():
            return f"Radius: {self.numeric.texts[0]}_"
        return f"Radius: {_fmt(self.radius * self.unit_scale)}"

    def header_text(self):
        radius = self._radius_line().replace("Radius", "Fillet radius")
        corners = f"{len(self.corners)} corner{'s' if len(self.corners) != 1 else ''}"
        if self.skipped:
            corners += f", {self.skipped} skipped (endpoints, junctions, straight)"
        return " | ".join((
            radius, corners,
            "drag or type the radius in the scene base unit (m for Metric), Ctrl snaps to the grid",
            "Enter/LMB confirm, Esc/RMB cancel",
        ))
