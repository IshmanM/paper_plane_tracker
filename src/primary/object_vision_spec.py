from enum import Enum, auto
import numpy as np

from src.primary.color import ColorId


class ObjectType(Enum):
    TENNIS_BALL = auto()
    PAPER_PLANE_SHAPES = auto()


class ShapeMarkerSpec:
    def __init__(
        self,
        color_id: ColorId,
        num_sides: int = 3,
        object_vertices_m: np.ndarray | None = None,
        minimum_contour_area_px: float | None = None,
    ):
        """
        num_sides defines the marker shape. A circle uses 0, a triangle uses 3, a square uses 4, etc.

        object_vertices_m contains the known polygon vertices in the object's coordinate frame,
        ordered around the polygon perimeter. For polygons its shape must be (num_sides, 3).

        TODO: num_sides=0 reserves circles as a special non-polygon case; circle detection/measurement is not implemented yet.
        """
        if num_sides != 0 and num_sides < 3:
            raise ValueError("num_sides must be 0 for a circle or at least 3 for a polygon")

        if object_vertices_m is not None:
            object_vertices_m = np.asarray(object_vertices_m, dtype=np.float64)

            if object_vertices_m.ndim != 2 or object_vertices_m.shape[1] != 3:
                raise ValueError("object_vertices_m must have shape (N, 3)")
            if num_sides != 0 and len(object_vertices_m) != num_sides:
                raise ValueError("object_vertices_m vertex count must match num_sides")

        self.color_id = color_id
        self.num_sides = num_sides
        self.object_vertices_m = object_vertices_m
        self.minimum_contour_area_px = minimum_contour_area_px


class ObjectVisionSpec:
    def __init__(
        self,
        color_ids: list[ColorId],
        minimum_contour_area_px: float,
        polygon_epsilon_ratio: float = 0.03,
        shape_group_distance_factor: float = 3.0,
        shape_markers: list[ShapeMarkerSpec] | None = None,
        width=None, height=None, length=None,
    ):
        self.color_ids = color_ids
        self.minimum_contour_area_px = minimum_contour_area_px
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.shape_group_distance_factor = shape_group_distance_factor
        self.shape_markers = shape_markers if shape_markers is not None else []

        self.width = width # m
        self.height = height # m
        self.length = length # m


# TODO:
#   - Replace placeholder/approximate object-frame marker coordinates with final measured values as needed.
#   - A paper plane may mix different polygon markers; each ShapeMarkerSpec carries its own num_sides.
#   - Add num_sides=0 circle-marker support later as a special case rather than approximating circles as many-sided polygons.
#   - Maybe have multiple paper-plane versions/configurations.
#   - Until grouping is made more sophisticated, repeated markers with the same color/shape should use compatible minimum areas.
OBJECT_VISION_SPECS = {
    ObjectType.TENNIS_BALL: ObjectVisionSpec(
        color_ids=[ColorId.TENNIS_GREEN],
        minimum_contour_area_px=100.0,
        width=0.0635, height=0.0635, length=0.0635,
    ),

    ObjectType.PAPER_PLANE_SHAPES: ObjectVisionSpec(
        color_ids=[ColorId.GREEN, ColorId.CYAN, ColorId.MAGENTA],
        polygon_epsilon_ratio=0.04,
        shape_group_distance_factor=3.0,
        minimum_contour_area_px=80.0,
        shape_markers=[
            ShapeMarkerSpec(
                color_id=ColorId.GREEN,
                num_sides=3,
                object_vertices_m=np.array([
                    [0.030, 0.000, -0.020],
                    [-0.090, 0.000, -0.020],
                    [-0.090, 0.000, 0.020],
                ], dtype=np.float64),
                minimum_contour_area_px=60.0,
            ),
            ShapeMarkerSpec(
                color_id=ColorId.CYAN,
                num_sides=3,
                object_vertices_m=np.array([
                    [0.000, -0.024, 0.015],
                    [-0.090, -0.009, 0.030],
                    [-0.095, -0.041, 0.030],
                ], dtype=np.float64),
                minimum_contour_area_px=60.0,
            ),
            # Example future square marker:
            # ShapeMarkerSpec(color_id=ColorId.MAGENTA, num_sides=4, object_vertices_m=..., minimum_contour_area_px=60.0),
            # TODO: Circle marker support is intentionally unimplemented. Use num_sides=0 for that special case.
        ],
        width=None, height=None, length=None,
    ),
}