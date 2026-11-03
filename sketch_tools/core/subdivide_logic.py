"""Where Subdivide puts its new vertices (plan 8.8, DECISIONS C17).

Plain floats and tuples only, no bpy and no bmesh, so everything runs headless. Points are
(x, y) in sketch local space: a sketch vertex that is slightly off the plane is treated as if
it lay on it, like the resolver does."""

import math

COUNT = 'COUNT'
LENGTH = 'LENGTH'

# Edges at most this long are zero-length: there is no direction to place a point along.
ZERO_LENGTH = 1e-6
# Slack on length / max_length, so an edge of exactly twice max_length (up to float noise) is two
# parts and not three.
RATIO_SLACK = 1e-6
# One edge never becomes more parts than this, however small max_length is dragged in the redo
# panel: a hair-thin max_length on a long edge would otherwise build millions of vertices.
MAX_PARTS = 1000


def edge_length(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def new_vertex_count(length, mode, cuts, max_length):
    """How many new vertices an edge of this length gets. 0 for a zero-length edge, and in
    LENGTH mode for an edge that is already no longer than max_length."""
    if length <= ZERO_LENGTH:
        return 0
    if mode == COUNT:
        return max(0, cuts)
    parts = math.ceil(length / max_length - RATIO_SLACK)
    return max(0, min(parts, MAX_PARTS) - 1)


def was_clamped(length, max_length):
    """True when LENGTH mode wanted more parts than MAX_PARTS for this edge."""
    return length > ZERO_LENGTH and math.ceil(length / max_length - RATIO_SLACK) > MAX_PARTS


def split_points(a, b, count):
    """count points evenly spaced between a and b, ends excluded, in order from a."""
    return [(a[0] + (b[0] - a[0]) * i / (count + 1), a[1] + (b[1] - a[1]) * i / (count + 1))
            for i in range(1, count + 1)]
