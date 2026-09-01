from enum import Enum, auto
import json
from pathlib import Path

import numpy as np
import cv2

from src.primary.color import ColorId


MODELS_DIR = Path(__file__).resolve().parent/"models"
OBJECT_VISION_MODEL_FORMAT_VERSION = 1


class ObjectType(Enum):
    TENNIS_BALL = auto()
    PAPER_PLANE_SHAPES = auto()
    ARUCO_MARKER = auto()


class ObjectVisionSpecId(Enum):
    # ObjectType selects the detection algorithm; ObjectVisionSpecId selects one concrete model/configuration.
    TENNIS_BALL_DEFAULT = auto()
    ARUCO_MARKER_1 = auto()
    PAPER_PLANE_SHAPES_1 = auto()
    PAPER_PLANE_ARUCO_1 = auto()


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
        plane_id: str | None = None,
    ):
        """
        A set of markers whose relative geometry is rigid because they lie on the same physical plane.

        plane_id identifies this plane inside ObjectVisionSpec so flexible connections can reference it.
        If omitted, ObjectVisionSpec assigns a stable default such as "plane_0".

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
        if plane_id is not None and (not isinstance(plane_id, str) or not plane_id.strip()):
            raise ValueError("plane_id must be None or a non-empty string")

        self.plane_id = None if plane_id is None else plane_id.strip()
        self.rotation_object_from_plane = rotation_object_from_plane
        self.translation_object_from_plane_m = translation_object_from_plane_m
        self.shape_markers = shape_markers if shape_markers is not None else []


def getRigidPlaneIntersection(rigid_plane_1: RigidPlaneSpec, rigid_plane_2: RigidPlaneSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return a point and unit direction for the nominal intersection line of two rigid planes."""
    normal_1 = np.asarray(rigid_plane_1.rotation_object_from_plane[:, 2], dtype=np.float64)
    normal_2 = np.asarray(rigid_plane_2.rotation_object_from_plane[:, 2], dtype=np.float64)
    normal_1_norm, normal_2_norm = np.linalg.norm(normal_1), np.linalg.norm(normal_2)

    if normal_1_norm <= 1e-12 or normal_2_norm <= 1e-12:
        raise ValueError("Rigid plane has an invalid zero-length normal")

    normal_1 /= normal_1_norm
    normal_2 /= normal_2_norm
    hinge_direction = np.cross(normal_1, normal_2)
    hinge_direction_norm = np.linalg.norm(hinge_direction)

    if hinge_direction_norm <= 1e-6:
        raise ValueError("Connected rigid planes are parallel or nearly parallel; cannot determine a unique hinge axis")

    hinge_direction /= hinge_direction_norm
    plane_equations = np.vstack((normal_1, normal_2))
    plane_offsets = np.array([
        np.dot(normal_1, rigid_plane_1.translation_object_from_plane_m),
        np.dot(normal_2, rigid_plane_2.translation_object_from_plane_m),
    ], dtype=np.float64)
    hinge_point = np.linalg.lstsq(plane_equations, plane_offsets, rcond=None)[0]
    return hinge_point, hinge_direction


class ArucoMarkerSpec:
    def __init__(self, marker_id: int, marker_length_m: float, dictionary_name: str):
        """
        Defines one physical ArUco marker.

        marker_length_m is the physical side length of the marker's outer square.
        The marker center should be treated as the detected object's position, which
        is convenient when the calibration laser is aimed at the marker center.

        dictionary_name should match an OpenCV predefined dictionary name, e.g.:
            "DICT_4X4_50"
            "DICT_5X5_100"
            "DICT_6X6_250"
        """
        if not isinstance(dictionary_name, str) or not dictionary_name:
            raise ValueError("dictionary_name must be a non-empty string")
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

        if marker_id < 0 or marker_id >= len(dictionary.bytesList):
            raise ValueError(
                f"marker_id must be in [0, {len(dictionary.bytesList) - 1}] "
                f"for {dictionary_name}"
            )
        if not np.isfinite(marker_length_m) or marker_length_m <= 0.0:
            raise ValueError("marker_length_m must be finite and > 0")

        self.marker_id = int(marker_id)
        self.marker_length_m = float(marker_length_m)
        self.dictionary_name = dictionary_name


class ObjectVisionSpec:
    def __init__(
        self,
        object_type: ObjectType,
        color_ids: list[ColorId] | None = None,
        minimum_contour_area_px: float | None = None,
        polygon_epsilon_ratio: float = 0.03,
        shape_group_distance_factor: float = 3.0,
        rigid_planes: list[RigidPlaneSpec] | None = None,
        rigid_plane_connections: list[tuple[str, str, float]] | None = None,
        aruco_marker: ArucoMarkerSpec | None = None,
        width=None, height=None, length=None,
    ):
        self.object_type = object_type
        self.color_ids = color_ids if color_ids is not None else []
        self.minimum_contour_area_px = minimum_contour_area_px
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.shape_group_distance_factor = shape_group_distance_factor
        self.rigid_planes = rigid_planes if rigid_planes is not None else []
        self.aruco_marker = aruco_marker

        # Give every plane a unique stable ID. This keeps older models/specs that predate plane_id usable.
        used_plane_ids = set()
        for plane_index, rigid_plane in enumerate(self.rigid_planes):
            if rigid_plane.plane_id is None:
                base_id = f"plane_{plane_index}"
                plane_id, suffix = base_id, 1
                while plane_id in used_plane_ids:
                    plane_id, suffix = f"{base_id}_{suffix}", suffix + 1
                rigid_plane.plane_id = plane_id

            if rigid_plane.plane_id in used_plane_ids:
                raise ValueError(f"Duplicate rigid plane ID: {rigid_plane.plane_id}")
            used_plane_ids.add(rigid_plane.plane_id)

        # Each connection is (plane_id_1, plane_id_2, max_rotation_deg), where rotation is ±max_rotation_deg.
        self.rigid_plane_connections: list[tuple[str, str, float]] = []
        rigid_planes_by_id = {rigid_plane.plane_id: rigid_plane for rigid_plane in self.rigid_planes}
        seen_connection_pairs = set()

        for connection in rigid_plane_connections if rigid_plane_connections is not None else []:
            if len(connection) != 3:
                raise ValueError("Each rigid_plane_connection must be (plane_id_1, plane_id_2, max_rotation_deg)")

            plane_id_1, plane_id_2, max_rotation_deg = connection
            if not isinstance(plane_id_1, str) or not isinstance(plane_id_2, str):
                raise ValueError("Rigid plane connection IDs must be strings")
            if plane_id_1 == plane_id_2:
                raise ValueError("A rigid plane cannot be connected to itself")
            if plane_id_1 not in rigid_planes_by_id or plane_id_2 not in rigid_planes_by_id:
                raise ValueError(f"Rigid plane connection references unknown plane: {(plane_id_1, plane_id_2)}")

            max_rotation_deg = float(max_rotation_deg)
            if not np.isfinite(max_rotation_deg) or max_rotation_deg < 0.0:
                raise ValueError("Rigid plane connection max_rotation_deg must be finite and >= 0")

            connection_pair = frozenset((plane_id_1, plane_id_2))
            if connection_pair in seen_connection_pairs:
                raise ValueError(f"Duplicate rigid plane connection: {(plane_id_1, plane_id_2)}")

            # A connection uses the nominal plane-plane intersection as its hinge axis.
            getRigidPlaneIntersection(rigid_planes_by_id[plane_id_1], rigid_planes_by_id[plane_id_2])
            seen_connection_pairs.add(connection_pair)
            self.rigid_plane_connections.append((plane_id_1, plane_id_2, max_rotation_deg))

        self.width = width # m
        self.height = height # m
        self.length = length # m

        if object_type == ObjectType.ARUCO_MARKER and aruco_marker is None:
            raise ValueError("ARUCO_MARKER requires aruco_marker")
        if object_type != ObjectType.ARUCO_MARKER and aruco_marker is not None:
            raise ValueError("aruco_marker is only valid for ARUCO_MARKER")

    @property
    def shape_markers(self) -> list[ShapeMarkerSpec]:
        # Current PnP combines all visible markers; later it can search allowed rigid-plane connection rotations.
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
        "aruco_marker": (
            None
            if object_vision_spec.aruco_marker is None
            else {
                "marker_id": object_vision_spec.aruco_marker.marker_id,
                "marker_length_m": object_vision_spec.aruco_marker.marker_length_m,
                "dictionary_name": object_vision_spec.aruco_marker.dictionary_name,
            }
        ),
        "rigid_planes": [
            {
                "plane_id": rigid_plane.plane_id,
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
        "rigid_plane_connections": [list(connection) for connection in object_vision_spec.rigid_plane_connections],
    }


def objectVisionSpecFromDict(data: dict) -> ObjectVisionSpec:
    format_version = data.get("format_version", 1)
    if format_version != OBJECT_VISION_MODEL_FORMAT_VERSION:
        raise ValueError(f"Unsupported ObjectVisionSpec model format version: {format_version}")

    rigid_planes = []
    for plane_index, plane_data in enumerate(data.get("rigid_planes", [])):
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
            plane_id=plane_data.get("plane_id", f"plane_{plane_index}"),
        ))

    aruco_data = data.get("aruco_marker")
    aruco_marker = (
        None
        if aruco_data is None
        else ArucoMarkerSpec(
            marker_id=int(aruco_data["marker_id"]),
            marker_length_m=float(aruco_data["marker_length_m"]),
            dictionary_name=aruco_data["dictionary_name"],
        )
    )

    minimum_contour_area_px = data.get("minimum_contour_area_px")

    return ObjectVisionSpec(
        object_type=ObjectType[data["object_type"]],
        color_ids=[ColorId[color_name] for color_name in data.get("color_ids", [])],
        minimum_contour_area_px=None if minimum_contour_area_px is None else float(minimum_contour_area_px),
        polygon_epsilon_ratio=float(data.get("polygon_epsilon_ratio", 0.03)),
        shape_group_distance_factor=float(data.get("shape_group_distance_factor", 3.0)),
        rigid_planes=rigid_planes,
        rigid_plane_connections=[
            (connection[0], connection[1], float(connection[2]))
            for connection in data.get("rigid_plane_connections", [])
        ],
        aruco_marker=aruco_marker,
        width=data.get("width"),
        height=data.get("height"),
        length=data.get("length"),
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
        minimum_contour_area_px=12.0,
        width=0.06715, height=0.06715, length=0.06715,
    ),

    ObjectVisionSpecId.ARUCO_MARKER_1: ObjectVisionSpec(
        object_type=ObjectType.ARUCO_MARKER,
        aruco_marker=ArucoMarkerSpec(
            marker_id=0,
            marker_length_m=0.100, 
            dictionary_name="DICT_4X4_50",
        ),
    ),

    ObjectVisionSpecId.PAPER_PLANE_ARUCO_1: ObjectVisionSpec(
        object_type=ObjectType.ARUCO_MARKER,
        aruco_marker=ArucoMarkerSpec(
            marker_id=0,
            marker_length_m=0.080, # TODO: replace with exact printed marker side length
            dictionary_name="DICT_4X4_50",
        ),
    ),

    ObjectVisionSpecId.PAPER_PLANE_SHAPES_1: ObjectVisionSpec(
        object_type=ObjectType.PAPER_PLANE_SHAPES,
        color_ids=[ColorId.GREEN, ColorId.CYAN, ColorId.PINK],
        minimum_contour_area_px=50.0,
        polygon_epsilon_ratio=0.030,
        shape_group_distance_factor=3.0,
        rigid_planes=[
            RigidPlaneSpec(
                plane_id='plane_0',
                rotation_object_from_plane=np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.93969262, 0.3420201406416154],
                    [0.0, -0.34202014, 0.939692621762824],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([0.0, 0.0, 0.01], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.GREEN, num_sides=4,
                        object_vertices_m=np.array([[-0.09, 0.02], [0.044000000000000004, 0.02], [0.034, 0.0], [-0.09, -0.02]], dtype=np.float64),
                        minimum_contour_area_px=30.0,
                    ),
                ],
            ),
            RigidPlaneSpec(
                plane_id='plane_1',
                rotation_object_from_plane=np.array([
                    [0.96605224, -0.22650031, 0.12426050957502129],
                    [0.19128349, 0.30381206, -0.9333321268079458],
                    [0.17364818, 0.92541658, 0.3368240888480406],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([-0.004, -0.0075, 0.045], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.CYAN, num_sides=3,
                        object_vertices_m=np.array([[0.0, 0.0], [-0.091, 0.0], [-0.091, 0.035]], dtype=np.float64),
                        minimum_contour_area_px=30.0,
                    ),
                ],
            ),
            RigidPlaneSpec(
                plane_id='plane_2',
                rotation_object_from_plane=np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.8660254, -0.5000000016387101],
                    [0.0, 0.5, 0.8660254028383291],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([0.0, 0.001, -0.01], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.GREEN, num_sides=3,
                        object_vertices_m=np.array([[-0.09, -0.02], [0.162, 0.02], [-0.09, 0.02]], dtype=np.float64),
                    ),
                ],
            ),
            RigidPlaneSpec(
                plane_id='plane_3',
                rotation_object_from_plane=np.array([
                    [0.96605224, 0.22650031, 0.12426050957502129],
                    [0.19128349, -0.30381206, -0.9333321268079458],
                    [-0.17364818, 0.92541658, -0.3368240888480406],
                ], dtype=np.float64),
                translation_object_from_plane_m=np.array([-0.004, -0.0075, -0.045], dtype=np.float64),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=ColorId.PINK, num_sides=3,
                        object_vertices_m=np.array([[0.0, 0.0], [-0.091, 0.0], [-0.091, -0.035]], dtype=np.float64),
                    ),
                ],
            ),
        ],
        rigid_plane_connections=[
            ('plane_0', 'plane_1', 10.0),
            ('plane_0', 'plane_2', 10.0),
            ('plane_2', 'plane_3', 10.0),
        ],
        width=None, height=None, length=None,
    )

}
