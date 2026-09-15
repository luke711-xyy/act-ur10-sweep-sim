"""Scene and component geometry definition (pure XML/NumPy, no MuJoCo import)."""

from .geometries import GEOMETRY_NAMES, ComponentSpec, make_component_spec
from .scene_builder import build_scene_xml, camera_xyaxes

__all__ = ["GEOMETRY_NAMES", "ComponentSpec", "make_component_spec",
           "build_scene_xml", "camera_xyaxes"]
