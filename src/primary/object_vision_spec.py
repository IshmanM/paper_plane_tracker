from enum import Enum, auto
import numpy as np

from src.primary.color import ColorId


class ObjectType(Enum):
    TENNIS_BALL = auto()
    PAPER_PLANE_TRIANGLES = auto()


class TriangleMarkerSpec:
    def __init__(
        self,
        color_id: ColorId,
        object_vertices_m: np.ndarray | None = None,
        minimum_contour_area_px: float | None = None,
    ):
        """
        object_vertices_m contains the known triangle vertices in the
        paper plane's object coordinate frame.

        Expected shape:

            [
                [x1, y1, z1],
                [x2, y2, z2],
                [x3, y3, z3],
            ]

        The vertex order must correspond to the image-vertex order used
        by the measurement creator.
        """
        self.color_id = color_id
        self.object_vertices_m = object_vertices_m
        self.minimum_contour_area_px = minimum_contour_area_px


class ObjectVisionSpec:
    def __init__(
        self,
        color_ids: list[ColorId],
        minimum_contour_area_px: float,

        polygon_epsilon_ratio: float = 0.03,
        triangle_group_distance_factor: float = 3.0,
        triangle_markers=None,

        width=None,
        height=None,
        length=None
    ):
        self.color_ids = color_ids
        self.minimum_contour_area_px = minimum_contour_area_px

        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.triangle_group_distance_factor = triangle_group_distance_factor
        self.triangle_markers = (
            triangle_markers
            if triangle_markers is not None
            else []
        )

        self.width = width # m
        self.height = height # m
        self.length = length # m


# todo 
#   - The three object_vertices_m=None values are placeholders. 
#     After deciding the exact triangle placement, 
#     replace them with measured coordinates in the paper plane’s coordinate frame.
#
#   -adjust the number of triangles doe paper plane
#   -maybe have multiple paper plane versions, 1,2,3
#
#   -until alg is updated, try to give repeated same-color markers the same minimum area.
#
OBJECT_VISION_SPECS = {
    ObjectType.TENNIS_BALL: ObjectVisionSpec(
        color_ids=[
            ColorId.TENNIS_GREEN,
        ],
        minimum_contour_area_px=100.0,
        width=0.0635,
        height=0.0635,
        length=0.0635
    ),

    ObjectType.PAPER_PLANE_TRIANGLES: ObjectVisionSpec(
        color_ids=[
            ColorId.GREEN,
            ColorId.CYAN,
            ColorId.MAGENTA,
        ],
        polygon_epsilon_ratio = 0.04,
        triangle_group_distance_factor = 3.0,
        minimum_contour_area_px=80.0,
        triangle_markers=[
            TriangleMarkerSpec(
                color_id=ColorId.GREEN,
                object_vertices_m=None,
                minimum_contour_area_px=60.0,
            ),
            TriangleMarkerSpec(
                color_id=ColorId.CYAN,
                object_vertices_m=None,
                minimum_contour_area_px=60.0,
            ),
            TriangleMarkerSpec(
                color_id=ColorId.MAGENTA,
                object_vertices_m=None,
                minimum_contour_area_px=60.0,
            ),
        ],
        width=None,
        height=None,
        length=None
    ),
}