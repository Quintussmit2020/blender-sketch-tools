"""Sketch Tools: sketch polylines, rectangles, circles and arcs on a plane or as 3D paths snapped to
other objects, then extrude or sweep them."""

from . import handlers, ops, prefs, props, ui


def register():
    props.register()
    prefs.register()
    ops.register()
    ui.register()
    handlers.register()


def unregister():
    handlers.unregister()
    ui.unregister()
    ops.unregister()
    prefs.unregister()
    props.unregister()
