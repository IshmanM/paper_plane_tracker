import cv2
import numpy as np
from collections import Counter
from itertools import combinations

import src.primary.config as config
from src.primary.geometry import estimateObjectWorldPosition
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec
from src.primary.color import COLOR_SPECS, ColorId


# Detection data passed between image processing, drawing, and measurement conversion.
class ShapeDetection:
    def __init__(
        self,
        vertices_px: list[list[float]] | np.ndarray,
        color_id: ColorId | None = None,
        num_sides: int = 3,
    ):
        self.vertices_px = np.asarray(vertices_px, dtype=np.float64)
        self.color_id = color_id
        self.num_sides = num_sides


class Detection:
    def __init__(
        self,
        u: float | None, v: float | None, px_w: float | None, px_h: float | None,
        shapes: list[ShapeDetection] | None = None,
    ):
        self.u = u
        self.v = v
        self.px_w = px_w
        self.px_h = px_h
        self.shapes = shapes if shapes is not None else []


class Measurement:
    def __init__(
        self,
        x: float | None, y: float | None, z: float | None = None,
        pitch: float | None = None, roll: float | None = None, yaw: float | None = None,
    ):
        self.x = x # x points right
        self.y = y # y points down
        self.z = z # z points away from the camera

        # probably not used:
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw


class DetectionDebug:
    def __init__(self):
        self.stages: list[tuple[str, np.ndarray]] = []

    def addStage(self, name: str, image: np.ndarray) -> None:
        self.stages.append((name, image.copy()))


# Public detection entry points and shared result helpers.
def detectSingleObject(frame: np.ndarray, object_type: ObjectType) -> tuple[bool, Detection, Measurement]:
    object_vision_spec = OBJECT_VISION_SPECS[object_type]

    if object_type == ObjectType.TENNIS_BALL:
        return detectTennisBall(frame, object_vision_spec,)
    elif object_type == ObjectType.PAPER_PLANE_SHAPES:
        return detectPaperPlaneShapes(frame, object_vision_spec,)

    raise ValueError(f"Unsupported object type: {object_type}")


def detectTennisBall(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_w=object_vision_spec.width)
    measurement = Measurement(x, y, z, None, None, None,)
    return True, detection, measurement


def detectPaperPlaneShapes(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectUsingBestShapeGroup(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    measurement = createMeasurementUsingShapeGroup(detection, object_vision_spec,)

    if measurement.x is None:
        return False, detection, measurement

    return True, detection, measurement


def failedDetectionResult() -> tuple[bool, Detection, Measurement]:
    detection = Detection(None, None, None, None, [],)
    measurement = Measurement(None, None, None, None, None, None,)
    return False, detection, measurement


def drawDetection(frame: np.ndarray, detection: Detection,) -> None:
    if detection.u is None or detection.v is None or detection.px_w is None or detection.px_h is None:
        return

    x_min = int(round(detection.u - detection.px_w/2.0))
    y_min = int(round(detection.v - detection.px_h/2.0))
    x_max = int(round(detection.u + detection.px_w/2.0))
    y_max = int(round(detection.v + detection.px_h/2.0))

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2,)
    cv2.circle(frame, (int(round(detection.u)), int(round(detection.v))), radius=5, color=(0, 255, 0), thickness=-1,)

    for shape in detection.shapes:
        vertices_px = shape.vertices_px.astype(np.int32)
        color_spec = COLOR_SPECS[shape.color_id]
        cv2.polylines(frame, [vertices_px.reshape(-1, 1, 2)], isClosed=True, color=color_spec.draw_bgr, thickness=2,)

        for vertex_u, vertex_v in vertices_px:
            cv2.circle(frame, (int(vertex_u), int(vertex_v)), radius=4, color=color_spec.draw_bgr, thickness=-1,)


# Tennis-ball path: threshold configured colors, clean the mask, and use the largest valid blob.
def findSingleObjectUsingLargestColorBlob(frame: np.ndarray, object_vision_spec: ObjectVisionSpec) -> Detection | None:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8,)

    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            color_mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv,)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask,)

    combined_mask = cv2.medianBlur(combined_mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < object_vision_spec.minimum_contour_area_px:
        return None

    u, v, px_w, px_h = cv2.boundingRect(largest_contour)
    return Detection(u + px_w/2.0, v + px_h/2.0, px_w, px_h,)


# Refine an accepted polygon by fitting its straight edges and intersecting neighboring lines.
def refineShapeVerticesUsingEdges(contour: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rough_vertices = polygon.reshape(-1, 2).astype(np.float64)
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    num_sides = len(rough_vertices)
    edge_distances, fitted_lines = [], []

    # Assign each contour point to the nearest rough polygon edge.
    for edge_index in range(num_sides):
        edge_start = rough_vertices[edge_index]
        edge_end = rough_vertices[(edge_index + 1)%num_sides]
        edge_vector = edge_end - edge_start
        edge_length = np.linalg.norm(edge_vector)

        if edge_length <= 1e-6:
            return rough_vertices

        relative_points = contour_points - edge_start
        distances = np.abs(edge_vector[0]*relative_points[:, 1] - edge_vector[1]*relative_points[:, 0])/edge_length
        edge_distances.append(distances)

    edge_assignments = np.argmin(np.stack(edge_distances, axis=1), axis=1)

    # Fit each line mainly from the middle of its edge so imperfect corners contribute less.
    for edge_index in range(num_sides):
        edge_start = rough_vertices[edge_index]
        edge_end = rough_vertices[(edge_index + 1)%num_sides]
        edge_vector = edge_end - edge_start
        edge_points = contour_points[edge_assignments == edge_index]

        if len(edge_points) < 2:
            return rough_vertices

        projection = ((edge_points - edge_start)@edge_vector)/np.dot(edge_vector, edge_vector)
        middle_points = edge_points[(projection >= 0.15) & (projection <= 0.85)]

        if len(middle_points) >= 2:
            edge_points = middle_points

        vx, vy, x0, y0 = cv2.fitLine(edge_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
        fitted_lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))

    # Reconstruct each corner from the intersection of its two neighboring fitted edges.
    refined_vertices = []

    for vertex_index in range(num_sides):
        point_1, direction_1 = fitted_lines[(vertex_index - 1)%num_sides]
        point_2, direction_2 = fitted_lines[vertex_index]
        cross = direction_1[0]*direction_2[1] - direction_1[1]*direction_2[0]

        if abs(cross) <= 1e-6:
            return rough_vertices

        difference = point_2 - point_1
        t = (difference[0]*direction_2[1] - difference[1]*direction_2[0])/cross
        refined_vertices.append(point_1 + t*direction_1)

    refined_vertices = np.asarray(refined_vertices, dtype=np.float64)
    maximum_edge_length = max(np.linalg.norm(rough_vertices[i] - rough_vertices[(i + 1)%num_sides]) for i in range(num_sides))

    # Fall back to approxPolyDP if a noisy fit produces an implausibly distant intersection.
    if np.any(np.linalg.norm(refined_vertices - rough_vertices, axis=1) > 0.4*maximum_edge_length):
        return rough_vertices

    return refined_vertices


# Shape path: find color-based polygon candidates, group nearby markers, then select the best group.
def findSingleObjectUsingBestShapeGroup(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, debug: DetectionDebug | None = None) -> Detection | None:
    shape_markers = object_vision_spec.shape_markers

    if not shape_markers:
        raise ValueError("object_vision_spec.shape_markers cannot be empty")

    # TODO: Add special-case circle candidate detection for ShapeMarkerSpec(num_sides=0).
    polygon_markers = [marker for marker in shape_markers if marker.num_sides != 0]

    if not polygon_markers:
        return None

    unique_color_ids = list(dict.fromkeys(marker.color_id for marker in polygon_markers))
    expected_num_sides_by_color = {
        color_id: sorted(set(marker.num_sides for marker in polygon_markers if marker.color_id == color_id))
        for color_id in unique_color_ids
    }
    minimum_shape_area_by_color = {
        color_id: min(
            marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
            for marker in polygon_markers if marker.color_id == color_id
        )
        for color_id in unique_color_ids
    }

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shape_candidates: list[ShapeDetection] = []
    combined_raw_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    combined_cleaned_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if debug is not None:
        debug.stages.clear()
        debug.addStage("Original image", frame)
        contour_debug_frame, polygon_debug_frame, candidate_debug_frame = frame.copy(), frame.copy(), frame.copy()
    else:
        contour_debug_frame = polygon_debug_frame = candidate_debug_frame = None

    # Threshold and clean each marker color independently.
    for color_id in unique_color_ids:
        color_spec = COLOR_SPECS[color_id]
        color_name = color_id.name
        draw_bgr = color_spec.draw_bgr
        raw_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            raw_mask = cv2.bitwise_or(raw_mask, cv2.inRange(hsv_frame, lower_hsv, upper_hsv))

        combined_raw_mask = cv2.bitwise_or(combined_raw_mask, raw_mask)

        if debug is not None:
            debug.addStage(f"Raw mask - {color_name}", raw_mask)

        cleaned_mask = cv2.medianBlur(raw_mask, 5)
        # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        combined_cleaned_mask = cv2.bitwise_or(combined_cleaned_mask, cleaned_mask)

        if debug is not None:
            debug.addStage(f"Cleaned mask - {color_name}", cleaned_mask)

        contours, _ = cv2.findContours(cleaned_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate each qualifying contour against every polygon type configured for this color.
        for contour in contours:
            if contour_debug_frame is not None:
                cv2.drawContours(contour_debug_frame, [contour], -1, draw_bgr, 1)

            if cv2.contourArea(contour) < minimum_shape_area_by_color[color_id]:
                continue

            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            base_polygon = cv2.approxPolyDP(hull, object_vision_spec.polygon_epsilon_ratio*perimeter, True)

            if polygon_debug_frame is not None:
                cv2.polylines(polygon_debug_frame, [base_polygon], True, draw_bgr, 2)
                polygon_center = np.mean(base_polygon.reshape(-1, 2), axis=0).astype(np.int32)
                cv2.putText(polygon_debug_frame, f"{color_name}: {len(base_polygon)} vertices", tuple(polygon_center), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_bgr, 1, cv2.LINE_AA)

            expected_num_sides = expected_num_sides_by_color[color_id]

            # Prefer an exact N-sided match. Only use the N+1 retry when no configured shape matches directly.
            if len(base_polygon) in expected_num_sides:
                candidate_polygons = [(len(base_polygon), base_polygon)]
            else:
                candidate_polygons = []
                for num_sides in expected_num_sides:
                    if len(base_polygon) == num_sides + 1:
                        retry_polygon = cv2.approxPolyDP(contour, (object_vision_spec.polygon_epsilon_ratio + 0.02)*perimeter, True)
                        if len(retry_polygon) == num_sides:
                            candidate_polygons.append((num_sides, retry_polygon))

            for num_sides, polygon in candidate_polygons:
                if not cv2.isContourConvex(polygon):
                    continue

                vertices_px = refineShapeVerticesUsingEdges(contour, polygon)
                shape_candidates.append(ShapeDetection(vertices_px=vertices_px, color_id=color_id, num_sides=num_sides))

                if candidate_debug_frame is not None:
                    shape_index = len(shape_candidates) - 1
                    center_px = np.mean(vertices_px, axis=0).astype(np.int32)
                    shape_points = np.round(vertices_px).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(candidate_debug_frame, [shape_points], True, draw_bgr, 3)
                    cv2.circle(candidate_debug_frame, tuple(center_px), 4, draw_bgr, -1)
                    cv2.putText(candidate_debug_frame, f"S{shape_index}: {color_name}, N={num_sides}", (int(center_px[0]) + 5, int(center_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_bgr, 2, cv2.LINE_AA)

    if debug is not None:
        debug.addStage("Combined raw mask", combined_raw_mask)
        debug.addStage("Combined cleaned mask", combined_cleaned_mask)
        debug.addStage("All mask contours", contour_debug_frame)
        debug.addStage("Polygon approximations", polygon_debug_frame)

        if not shape_candidates:
            cv2.putText(candidate_debug_frame, "No accepted shapes", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Accepted shape candidates", candidate_debug_frame)

    if not shape_candidates:
        return None

    shape_groups = groupShapeCandidates(shape_candidates, object_vision_spec)

    if debug is not None:
        group_debug_frame = frame.copy()

        for group_index, shape_group in enumerate(shape_groups):
            all_group_vertices = np.concatenate([shape.vertices_px for shape in shape_group], axis=0)

            for shape in shape_group:
                draw_bgr = COLOR_SPECS[shape.color_id].draw_bgr
                shape_points = np.round(shape.vertices_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(group_debug_frame, [shape_points], True, draw_bgr, 3)

            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_group_vertices.astype(np.float32))
            cv2.rectangle(group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 255, 255), 2)
            cv2.putText(group_debug_frame, f"G{group_index}: {len(shape_group)}/{len(polygon_markers)} markers", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if not shape_groups:
            cv2.putText(group_debug_frame, "No groups matched shape_markers", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Matching shape groups", group_debug_frame)

    if not shape_groups:
        return None

    best_shape_group = selectBestShapeGroup(shape_groups, object_vision_spec)

    if debug is not None:
        best_group_debug_frame = frame.copy()
        all_best_vertices = np.concatenate([shape.vertices_px for shape in best_shape_group], axis=0)

        for shape in best_shape_group:
            draw_bgr = COLOR_SPECS[shape.color_id].draw_bgr
            shape_points = np.round(shape.vertices_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(best_group_debug_frame, [shape_points], True, draw_bgr, 3)

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
        cv2.rectangle(best_group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
        cv2.putText(best_group_debug_frame, f"Selected best group: {len(best_shape_group)}/{len(polygon_markers)} markers", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        debug.addStage("Selected best shape group", best_group_debug_frame)

    all_best_vertices = np.concatenate([shape.vertices_px for shape in best_shape_group], axis=0)
    bbox_x, bbox_y, px_w, px_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
    detection = Detection(
        u=bbox_x + px_w/2.0, v=bbox_y + px_h/2.0,
        px_w=float(px_w), px_h=float(px_h), shapes=best_shape_group,
    )

    if debug is not None:
        final_debug_frame = frame.copy()
        bbox_x, bbox_y = int(round(detection.u - detection.px_w/2.0)), int(round(detection.v - detection.px_h/2.0))
        bbox_x_2, bbox_y_2 = int(round(detection.u + detection.px_w/2.0)), int(round(detection.v + detection.px_h/2.0))
        cv2.rectangle(final_debug_frame, (bbox_x, bbox_y), (bbox_x_2, bbox_y_2), (0, 0, 255), 3)
        cv2.circle(final_debug_frame, (int(round(detection.u)), int(round(detection.v))), 5, (0, 0, 255), -1)
        cv2.putText(final_debug_frame, f"u={detection.u:.1f}, v={detection.v:.1f}, w={detection.px_w:.1f}, h={detection.px_h:.1f}", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        debug.addStage("Final object detection", final_debug_frame)

    return detection


# Shape grouping and selection helpers.
def groupShapeCandidates(shape_candidates: list[ShapeDetection], object_vision_spec: ObjectVisionSpec) -> list[list[ShapeDetection]]:
    polygon_markers = [marker for marker in object_vision_spec.shape_markers if marker.num_sides != 0]
    required_marker_counts = Counter((marker.color_id, marker.num_sides) for marker in polygon_markers)
    maximum_group_size = min(len(shape_candidates), len(polygon_markers))

    marker_minimum_areas: dict[tuple[ColorId, int], list[float]] = {}
    marker_order: dict[tuple[ColorId, int], int] = {}

    for marker_index, marker in enumerate(polygon_markers):
        marker_key = (marker.color_id, marker.num_sides)
        minimum_area = marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
        marker_minimum_areas.setdefault(marker_key, []).append(minimum_area)
        marker_order.setdefault(marker_key, marker_index)

    for minimum_areas in marker_minimum_areas.values():
        minimum_areas.sort()

    # Prefer the largest valid group, then fall back to partial groups.
    for group_size in range(maximum_group_size, 0, -1):
        shape_groups: list[list[ShapeDetection]] = []

        for shape_combination in combinations(shape_candidates, group_size):
            shape_group = list(shape_combination)
            group_marker_counts = Counter((shape.color_id, shape.num_sides) for shape in shape_group)

            if any(count > required_marker_counts[marker_key] for marker_key, count in group_marker_counts.items()):
                continue

            connected_indices, pending_indices = {0}, [0]

            while pending_indices:
                current_index = pending_indices.pop()

                for candidate_index in range(group_size):
                    if candidate_index in connected_indices:
                        continue

                    if shapeCandidatesAreNear(shape_group[current_index], shape_group[candidate_index], object_vision_spec.shape_group_distance_factor):
                        connected_indices.add(candidate_index)
                        pending_indices.append(candidate_index)

            if len(connected_indices) != group_size:
                continue

            valid_group = True

            for marker_key, marker_count in group_marker_counts.items():
                color_id, num_sides = marker_key
                shape_areas = sorted(
                    cv2.contourArea(shape.vertices_px.astype(np.float32))
                    for shape in shape_group if shape.color_id == color_id and shape.num_sides == num_sides
                )
                minimum_areas = marker_minimum_areas[marker_key][:marker_count]

                if any(shape_area < minimum_area for shape_area, minimum_area in zip(shape_areas, minimum_areas)):
                    valid_group = False
                    break

            if not valid_group:
                continue

            shape_group.sort(key=lambda shape: (
                marker_order[(shape.color_id, shape.num_sides)],
                float(np.mean(shape.vertices_px[:, 0])), float(np.mean(shape.vertices_px[:, 1])),
            ))
            shape_groups.append(shape_group)

        if shape_groups:
            return shape_groups

    return []


def shapeCandidatesAreNear(shape_1: ShapeDetection, shape_2: ShapeDetection, distance_factor: float) -> bool:
    center_1, center_2 = np.mean(shape_1.vertices_px, axis=0), np.mean(shape_2.vertices_px, axis=0)
    _, _, width_1, height_1 = cv2.boundingRect(shape_1.vertices_px.astype(np.float32))
    _, _, width_2, height_2 = cv2.boundingRect(shape_2.vertices_px.astype(np.float32))
    reference_size = max(np.hypot(width_1, height_1), np.hypot(width_2, height_2))
    return np.linalg.norm(center_1 - center_2) <= distance_factor*reference_size


def selectBestShapeGroup(
    shape_groups: list[list[ShapeDetection]], object_vision_spec: ObjectVisionSpec,
    marker_count_weight: float = 0.80, compactness_weight: float = 0.15, area_weight: float = 0.05,
) -> list[ShapeDetection] | None:
    # TODO: Perhaps use marker object_vertices_m to reward groups matching the expected physical marker layout.
    if not shape_groups:
        return None

    polygon_marker_count = sum(marker.num_sides != 0 for marker in object_vision_spec.shape_markers)
    maximum_average_area = max(
        sum(cv2.contourArea(shape.vertices_px.astype(np.float32)) for shape in group)/len(group)
        for group in shape_groups
    )
    best_group = None
    best_score = float("-inf")

    for group in shape_groups:
        marker_count_score = len(group)/polygon_marker_count
        maximum_normalized_distance = 0.0

        for i, shape_1 in enumerate(group):
            for shape_2 in group[i + 1:]:
                center_1, center_2 = np.mean(shape_1.vertices_px, axis=0), np.mean(shape_2.vertices_px, axis=0)
                size_1 = np.max(np.linalg.norm(shape_1.vertices_px[:, None] - shape_1.vertices_px[None, :], axis=2))
                size_2 = np.max(np.linalg.norm(shape_2.vertices_px[:, None] - shape_2.vertices_px[None, :], axis=2))
                normalized_distance = np.linalg.norm(center_1 - center_2)/max((size_1 + size_2)/2.0, 1e-6)
                maximum_normalized_distance = max(maximum_normalized_distance, normalized_distance)

        compactness_score = (
            max(0.0, 1.0 - maximum_normalized_distance/object_vision_spec.shape_group_distance_factor)
            if len(group) > 1 else 0.0
        )
        average_area = sum(cv2.contourArea(shape.vertices_px.astype(np.float32)) for shape in group)/len(group)
        area_score = average_area/maximum_average_area if maximum_average_area > 0.0 else 0.0
        score = marker_count_weight*marker_count_score + compactness_weight*compactness_score + area_weight*area_score

        if score > best_score:
            best_group, best_score = group, score

    return best_group


# Convert the selected image-space shape group into a camera-frame measurement.
def createMeasurementUsingShapeGroup(detection: Detection, object_vision_spec: ObjectVisionSpec) -> Measurement:
    failed_measurement = Measurement(None, None, None, None, None, None,)

    if not detection.shapes:
        return failed_measurement

    marker_indices_by_key: dict[tuple[ColorId, int], list[int]] = {}

    for marker_index, marker in enumerate(object_vision_spec.shape_markers):
        if marker.num_sides != 0:
            marker_indices_by_key.setdefault((marker.color_id, marker.num_sides), []).append(marker_index)

    object_point_groups, image_point_groups, used_marker_indices = [], [], set()

    # Match each detected polygon to its configured physical marker and determine vertex correspondence.
    for shape in detection.shapes:
        marker_key = (shape.color_id, shape.num_sides)
        available_marker_indices = [
            marker_index for marker_index in marker_indices_by_key.get(marker_key, [])
            if marker_index not in used_marker_indices
        ]

        if len(available_marker_indices) != 1:
            raise ValueError(f"Detected marker {marker_key} must match exactly one unused shape marker")

        marker_index = available_marker_indices[0]
        marker = object_vision_spec.shape_markers[marker_index]

        if marker.object_vertices_m is None:
            raise ValueError(f"Shape marker {marker_key} has no object_vertices_m")

        marker_vertices_m = np.asarray(marker.object_vertices_m, dtype=np.float64)
        shape_vertices_px = np.asarray(shape.vertices_px, dtype=np.float64)
        num_vertices = marker.num_sides

        if marker_vertices_m.shape != (num_vertices, 3):
            raise ValueError(f"object_vertices_m for {marker_key} must have shape ({num_vertices}, 3)")
        if shape_vertices_px.shape != (num_vertices, 2):
            raise ValueError(f"vertices_px for {marker_key} must have shape ({num_vertices}, 2)")
        if not np.all(np.isfinite(marker_vertices_m)) or not np.all(np.isfinite(shape_vertices_px)):
            return failed_measurement

        edge_pairs = [(i, (i + 1)%num_vertices) for i in range(num_vertices)]
        object_edge_lengths = np.array([np.linalg.norm(marker_vertices_m[i] - marker_vertices_m[j]) for i, j in edge_pairs])
        object_shape_norm = np.linalg.norm(object_edge_lengths)

        if object_shape_norm <= 1e-12:
            raise ValueError(f"object_vertices_m for {marker_key} form a degenerate shape")

        normalized_object_edge_lengths = object_edge_lengths/object_shape_norm
        best_vertices_px, best_shape_error = None, float("inf")

        # Polygon vertices arrive in perimeter order, so only cyclic shifts and reversed cyclic shifts are possible.
        vertex_orders = []
        for start_index in range(num_vertices):
            vertex_orders.append([(start_index + offset)%num_vertices for offset in range(num_vertices)])
            vertex_orders.append([(start_index - offset)%num_vertices for offset in range(num_vertices)])

        for vertex_order in vertex_orders:
            ordered_vertices_px = shape_vertices_px[vertex_order]
            image_edge_lengths = np.array([np.linalg.norm(ordered_vertices_px[i] - ordered_vertices_px[j]) for i, j in edge_pairs])
            image_shape_norm = np.linalg.norm(image_edge_lengths)

            if image_shape_norm <= 1e-12:
                continue

            shape_error = float(np.linalg.norm(image_edge_lengths/image_shape_norm - normalized_object_edge_lengths))

            if shape_error < best_shape_error:
                best_vertices_px, best_shape_error = ordered_vertices_px, shape_error

        if best_vertices_px is None:
            return failed_measurement

        object_point_groups.append(marker_vertices_m)
        image_point_groups.append(best_vertices_px)
        used_marker_indices.add(marker_index)

    object_points = np.concatenate(object_point_groups, axis=0)
    image_points = np.concatenate(image_point_groups, axis=0)

    solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
        object_points, image_points, config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS, flags=cv2.SOLVEPNP_SQPNP,
    )

    if not solution_count:
        return failed_measurement

    best_translation, best_reprojection_error = None, float("inf")

    # Reject poses behind the camera and select the solution that best reproduces the detected vertices.
    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T

        if np.any(camera_points[:, 2] <= 0.0):
            continue

        projected_points, _ = cv2.projectPoints(
            object_points, rotation_vector, translation_vector, config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS,
        )
        reprojection_error = float(np.sqrt(np.mean(np.sum((projected_points.reshape(-1, 2) - image_points)**2, axis=1))))

        if reprojection_error < best_reprojection_error:
            best_translation, best_reprojection_error = translation_vector.reshape(3), reprojection_error

    if best_translation is None or not np.all(np.isfinite(best_translation)):
        return failed_measurement

    return Measurement(float(best_translation[0]), float(best_translation[1]), float(best_translation[2]), None, None, None,)