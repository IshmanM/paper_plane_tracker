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

        object_vertices_m contains the known polygon vertices as (x, y) coordinates in the owning
        rigid plane's local frame, ordered around the polygon perimeter. Local z is implicitly 0.
        RigidPlaneSpec rotates then translates these points into the paper plane's common object/reference
        frame before PnP. For polygons its shape is (num_sides, 2).

        Polygon detection/refinement currently assumes convex shapes.
        TODO: Add special-case circle support for num_sides=0.
        TODO: Add concave-polygon support if needed.
        """
        if num_sides != 0 and num_sides < 3:
            raise ValueError("num_sides must be 0 for a circle or at least 3 for a polygon")

        if object_vertices_m is not None:
            object_vertices_m = np.asarray(object_vertices_m, dtype=np.float64)
            if object_vertices_m.ndim != 2 or object_vertices_m.shape[1] != 2:
                raise ValueError("object_vertices_m must have shape (N, 2)")
            if num_sides != 0 and len(object_vertices_m) != num_sides:
                raise ValueError("object_vertices_m vertex count must match num_sides")

        self.color_id = color_id
        self.num_sides = num_sides
        self.object_vertices_m = object_vertices_m
        self.minimum_contour_area_px = minimum_contour_area_px


class RigidPlaneSpec:
    def __init__(
        self,
        rotation_object_from_plane: np.ndarray,
        translation_object_from_plane_m: np.ndarray | None = None,
        shape_markers: list[ShapeMarkerSpec] | None = None,
    ):
        """
        A set of markers whose relative geometry is rigid because they lie on the same physical plane.

        rotation_object_from_plane and translation_object_from_plane_m define the nominal plane-local
        frame relative to the paper plane's object/reference frame. Transform plane-local points by
        applying rotation first, then translation, with each marker point embedded as p_plane = [x, y, 0]:

            p_object = rotation_object_from_plane @ p_plane + translation_object_from_plane_m

        A zero translation means the plane-local origin coincides with the object-frame origin, so the
        plane passes through that origin. During measurement, each marker's local (x, y, 0) vertices are
        transformed into the common object/reference frame before the current combined PnP solve.
        """
        rotation_object_from_plane = np.asarray(rotation_object_from_plane, dtype=np.float64)
        if rotation_object_from_plane.shape != (3, 3) or not np.all(np.isfinite(rotation_object_from_plane)):
            raise ValueError("rotation_object_from_plane must be a finite 3x3 matrix")

        if translation_object_from_plane_m is None:
            translation_object_from_plane_m = np.zeros(3, dtype=np.float64)
        else:
            translation_object_from_plane_m = np.asarray(translation_object_from_plane_m, dtype=np.float64)

        if translation_object_from_plane_m.shape != (3,) or not np.all(np.isfinite(translation_object_from_plane_m)):
            raise ValueError("translation_object_from_plane_m must be a finite length-3 vector")

        self.rotation_object_from_plane = rotation_object_from_plane
        self.translation_object_from_plane_m = translation_object_from_plane_m
        self.shape_markers = shape_markers if shape_markers is not None else []


class ObjectVisionSpec:
    def __init__(
        self,
        color_ids: list[ColorId],
        minimum_contour_area_px: float,
        polygon_epsilon_ratio: float = 0.03,
        shape_group_distance_factor: float = 3.0,
        rigid_planes: list[RigidPlaneSpec] | None = None,
        width=None, height=None, length=None,
    ):
        self.color_ids = color_ids
        self.minimum_contour_area_px = minimum_contour_area_px
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.shape_group_distance_factor = shape_group_distance_factor
        self.rigid_planes = rigid_planes if rigid_planes is not None else []

        self.width = width # m
        self.height = height # m
        self.length = length # m

    @property
    def shape_markers(self) -> list[ShapeMarkerSpec]:
        # Current detection treats the whole object as one marker set. Later it can consume rigid_planes directly.
        return [marker for rigid_plane in self.rigid_planes for marker in rigid_plane.shape_markers]


# TODO:
#   - Replace approximate object-frame marker coordinates with carefully measured final values.
#   - Measure/set the nominal rotation_object_from_plane and translation_object_from_plane_m for every actual rigid surface.
#   - Current PnP still combines all selected markers into one rigid solve; later solve each rigid plane separately and fuse.
#   - A paper plane may mix different convex polygon markers; each ShapeMarkerSpec carries its own num_sides.
#   - Add num_sides=0 circle-marker support later as a special case.
#   - Add concave-polygon support later only if needed.
#   - Maybe have multiple paper-plane versions/configurations.
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
        rigid_planes=[
            # TODO: These plane-local coordinates/transforms reproduce the current rough 3D model.
            # Re-measure the final marker geometry and rigid-plane transforms more accurately.
            RigidPlaneSpec(
                rotation_object_from_plane=np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([0.030, 0.000, -0.020], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.GREEN,
                        num_sides=3,
                        object_vertices_m=np.array([
                            [0.019, -0.020],
                            [0.090, -0.020],
                            [0.090, 0.020],
                        ], dtype=np.float64),
                        minimum_contour_area_px=60.0,
                    ),
                ],
            ),
            RigidPlaneSpec(
                rotation_object_from_plane=np.array([
                    [-0.97332853, -0.16413523, 0.16028476],
                    [0.16222142, -0.98643651, -0.02504449],
                    [0.16222142, 0.00162510, 0.98675304],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([0.000, -0.024, 0.015], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.CYAN,
                        num_sides=3,
                        object_vertices_m=np.array([
                            [0.00000000, 0.00000000],
                            [0.09246621, 0.00000000],
                            [0.09214177, 0.03238664],
                        ], dtype=np.float64),
                        minimum_contour_area_px=60.0,
                    ),
                    # Example future square marker on this same rigid plane:
                    # ShapeMarkerSpec(color_id=ColorId.MAGENTA, num_sides=4, object_vertices_m=..., minimum_contour_area_px=60.0),
                ],
            ),
        ],
        width=None, height=None, length=None,
    ),
}