from enum import Enum, auto
import json
from pathlib import Path

import numpy as np

from src.primary.color import ColorId


MODELS_DIR = Path(__file__).resolve().parent/"models"
OBJECT_VISION_MODEL_FORMAT_VERSION = 1


class ObjectType(Enum):
    TENNIS_BALL = auto()
    PAPER_PLANE_SHAPES = auto()


class ObjectVisionSpecId(Enum):
    # ObjectType selects the detection algorithm; ObjectVisionSpecId selects one concrete model/configuration.
    TENNIS_BALL_DEFAULT = auto()
    PAPER_PLANE_SHAPES_1 = auto()


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

        object_vertices_m contains polygon vertices as plane-local (x, y) coordinates in meters,
        ordered around the perimeter. Local z is implicitly 0; RigidPlaneSpec rotates then translates
        these points into the common object/reference frame before PnP.

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

        Each plane-local marker point is embedded as p_plane=[x,y,0], then transformed by rotation
        first and translation second:

            p_object = rotation_object_from_plane @ p_plane + translation_object_from_plane_m

        The object/reference frame convention for the paper plane is +x forward, +y down, +z left.
        """
        rotation_object_from_plane = np.asarray(rotation_object_from_plane, dtype=np.float64)
        if rotation_object_from_plane.shape != (3, 3) or not np.all(np.isfinite(rotation_object_from_plane)):
            raise ValueError("rotation_object_from_plane must be a finite 3x3 matrix")

        translation_object_from_plane_m = (
            np.zeros(3, dtype=np.float64)
            if translation_object_from_plane_m is None
            else np.asarray(translation_object_from_plane_m, dtype=np.float64)
        )
        if translation_object_from_plane_m.shape != (3,) or not np.all(np.isfinite(translation_object_from_plane_m)):
            raise ValueError("translation_object_from_plane_m must be a finite length-3 vector")

        self.rotation_object_from_plane = rotation_object_from_plane
        self.translation_object_from_plane_m = translation_object_from_plane_m
        self.shape_markers = shape_markers if shape_markers is not None else []


class ObjectVisionSpec:
    def __init__(
        self,
        object_type: ObjectType,
        color_ids: list[ColorId],
        minimum_contour_area_px: float,
        polygon_epsilon_ratio: float = 0.03,
        shape_group_distance_factor: float = 3.0,
        rigid_planes: list[RigidPlaneSpec] | None = None,
        width=None, height=None, length=None,
    ):
        self.object_type = object_type
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
        # Current PnP combines all visible markers; later it can solve/fuse rigid planes separately.
        return [marker for rigid_plane in self.rigid_planes for marker in rigid_plane.shape_markers]


def objectVisionSpecToDict(object_vision_spec: ObjectVisionSpec) -> dict:
    return {
        "format_version": OBJECT_VISION_MODEL_FORMAT_VERSION,
        "object_type": object_vision_spec.object_type.name,
        "color_ids": [color_id.name for color_id in object_vision_spec.color_ids],
        "minimum_contour_area_px": object_vision_spec.minimum_contour_area_px,
        "polygon_epsilon_ratio": object_vision_spec.polygon_epsilon_ratio,
        "shape_group_distance_factor": object_vision_spec.shape_group_distance_factor,
        "width": object_vision_spec.width,
        "height": object_vision_spec.height,
        "length": object_vision_spec.length,
        "rigid_planes": [
            {
                "rotation_object_from_plane": rigid_plane.rotation_object_from_plane.tolist(),
                "translation_object_from_plane_m": rigid_plane.translation_object_from_plane_m.tolist(),
                "shape_markers": [
                    {
                        "color_id": marker.color_id.name,
                        "num_sides": marker.num_sides,
                        "object_vertices_m": None if marker.object_vertices_m is None else marker.object_vertices_m.tolist(),
                        "minimum_contour_area_px": marker.minimum_contour_area_px,
                    }
                    for marker in rigid_plane.shape_markers
                ],
            }
            for rigid_plane in object_vision_spec.rigid_planes
        ],
    }


def objectVisionSpecFromDict(data: dict) -> ObjectVisionSpec:
    format_version = data.get("format_version", 1)
    if format_version != OBJECT_VISION_MODEL_FORMAT_VERSION:
        raise ValueError(f"Unsupported ObjectVisionSpec model format version: {format_version}")

    rigid_planes = []
    for plane_data in data.get("rigid_planes", []):
        shape_markers = [
            ShapeMarkerSpec(
                color_id=ColorId[marker_data["color_id"]],
                num_sides=int(marker_data.get("num_sides", 3)),
                object_vertices_m=marker_data.get("object_vertices_m"),
                minimum_contour_area_px=marker_data.get("minimum_contour_area_px"),
            )
            for marker_data in plane_data.get("shape_markers", [])
        ]
        rigid_planes.append(RigidPlaneSpec(
            rotation_object_from_plane=plane_data["rotation_object_from_plane"],
            translation_object_from_plane_m=plane_data.get("translation_object_from_plane_m"),
            shape_markers=shape_markers,
        ))

    return ObjectVisionSpec(
        object_type=ObjectType[data["object_type"]],
        color_ids=[ColorId[color_name] for color_name in data.get("color_ids", [])],
        minimum_contour_area_px=float(data["minimum_contour_area_px"]),
        polygon_epsilon_ratio=float(data.get("polygon_epsilon_ratio", 0.03)),
        shape_group_distance_factor=float(data.get("shape_group_distance_factor", 3.0)),
        rigid_planes=rigid_planes,
        width=data.get("width"), height=data.get("height"), length=data.get("length"),
    )


def saveObjectVisionSpecModel(object_vision_spec: ObjectVisionSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(objectVisionSpecToDict(object_vision_spec), indent=2) + "\n", encoding="utf-8")
    return path


def loadObjectVisionSpecModel(path: str | Path) -> ObjectVisionSpec:
    return objectVisionSpecFromDict(json.loads(Path(path).read_text(encoding="utf-8")))


# Registered application specs. Add a new ObjectVisionSpecId when an experimental model is promoted
# into normal detection use; the visualizer can save/load unlimited unregistered JSON models meanwhile.
OBJECT_VISION_SPECS = {
    ObjectVisionSpecId.TENNIS_BALL_DEFAULT: ObjectVisionSpec(
        object_type=ObjectType.TENNIS_BALL,
        color_ids=[ColorId.TENNIS_GREEN],
        minimum_contour_area_px=100.0,
        width=0.0635, height=0.0635, length=0.0635,
    ),

    ObjectVisionSpecId.PAPER_PLANE_SHAPES_1: ObjectVisionSpec(
        object_type=ObjectType.PAPER_PLANE_SHAPES,
        color_ids=[ColorId.GREEN, ColorId.CYAN, ColorId.MAGENTA],
        polygon_epsilon_ratio=0.04,
        shape_group_distance_factor=3.0,
        minimum_contour_area_px=80.0,
        rigid_planes=[
            # Rough current model; replace with accurately measured final geometry.
            RigidPlaneSpec(
                rotation_object_from_plane=np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([0.000, 0.000, 0.000], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.GREEN,
                        num_sides=3,
                        object_vertices_m=np.array([
                            [0.019, -0.020],
                            [-0.090, -0.020],
                            [-0.090, 0.020],
                        ], dtype=np.float64),
                        minimum_contour_area_px=60.0,
                    ),
                ],
            ),
            RigidPlaneSpec(
                rotation_object_from_plane=np.array([
                    [0.96605224, -0.17034108, 0.19423435],
                    [0.19128349, -0.033728441, -0.98095516],
                    [0.17364818, 0.98480775, 6.0302083e-17],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([-0.004, -0.0125, 0.025], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.CYAN,
                        num_sides=3,
                        object_vertices_m=np.array([
                            [0.000, 0.000],
                            [-0.091, 0.000],
                            [-0.091, 0.035],
                        ], dtype=np.float64),
                        minimum_contour_area_px=60.0,
                    ),
                ],
            ),
        ],
        width=None, height=None, length=None,
    ),
}