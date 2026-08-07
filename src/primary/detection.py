import cv2
import numpy as np
from collections import Counter
from itertools import combinations, permutations

import src.primary.config as config
from src.primary.geometry import estimateObjectWorldPosition
from enum import Enum, auto
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec
from src.primary.color import COLOR_SPECS, ColorId

# Detection data passed between image processing, drawing, and measurement conversion.
class TriangleDetection:
    def __init__(
        self, 
        vertices_px: list[list[float]] | np.ndarray, 
        color_id: ColorId | None = None,
    ):
        self.vertices_px = np.asarray(vertices_px, dtype=np.float64)
        self.color_id = color_id


class Detection:
    def __init__(
        self, 
        u: float | None, v: float | None, px_w: float | None, px_h: float | None,
        triangles: list[TriangleDetection] | None = None,
    ):        
        self.u = u 
        self.v = v
        self.px_w = px_w 
        self.px_h = px_h 
        self.triangles = triangles if triangles is not None else []


class Measurement:
    def __init__(
        self, 
        x: float | None, y: float | None, z: float | None = None, 
        pitch: float | None = None, roll: float | None = None, yaw: float | None = None
    ):
                
        self.x = x # x points right
        self.y = y # y points down
        self.z = z # z points away from the camera
        
        #probably not used:
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

    elif object_type == ObjectType.PAPER_PLANE_TRIANGLES:
        return detectPaperPlaneTriangles(frame, object_vision_spec,)

    # elif object_type == ObjectType.PAPER_PLANE_ARUCO:
    # elif...

    raise ValueError(f"Unsupported object type: {object_type}")


def detectTennisBall(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> tuple[bool, Detection, Measurement]:

    detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_w=object_vision_spec.width)
    measurement = Measurement(x, y, z, None, None, None,)

    return True, detection, measurement


def detectPaperPlaneTriangles(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> tuple[bool, Detection, Measurement]:

    detection = findSingleObjectUsingBestTriangleGroup(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    measurement = createMeasurementUsingTriangleGroup(detection, object_vision_spec,)

    if measurement.x is None:
        return False, detection, measurement

    return True, detection, measurement


def failedDetectionResult() -> tuple[bool, Detection, Measurement]:
    detection = Detection(None, None, None, None, [],)
    measurement = Measurement(None, None, None, None, None, None,)
    return False, detection, measurement


def drawDetection(frame: np.ndarray, detection: Detection,) -> None:

    if (detection.u is None or detection.v is None or detection.px_w is None or detection.px_h is None):
        return

    x_min = int(round(detection.u - detection.px_w / 2.0))
    y_min = int(round(detection.v - detection.px_h / 2.0))
    x_max = int(round(detection.u + detection.px_w / 2.0))
    y_max = int(round(detection.v + detection.px_h / 2.0))

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2,)

    cv2.circle(
        frame,
        (int(round(detection.u)), int(round(detection.v)),),
        radius=5, color=(0, 255, 0), thickness=-1,
    )

    # todo: change below triangles implementation as needed...
    for triangle in detection.triangles:
        vertices_px = triangle.vertices_px.astype(
            np.int32,
        )

        color_spec = COLOR_SPECS[
            triangle.color_id
        ]

        cv2.polylines(
            frame,
            [vertices_px.reshape(-1, 1, 2)],
            isClosed=True,
            color=color_spec.draw_bgr,
            thickness=2,
        )

        for vertex_u, vertex_v in vertices_px:
            cv2.circle(
                frame,
                (int(vertex_u), int(vertex_v)),
                radius=4,
                color=color_spec.draw_bgr,
                thickness=-1,
            )


# Tennis-ball path: threshold configured colors, clean the mask, and use the largest valid blob.
def findSingleObjectUsingLargestColorBlob(frame: np.ndarray, object_vision_spec: ObjectVisionSpec) -> Detection | None:
    # Build one mask from every configured HSV range.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8,)

    # Include every HSV range belonging to every configured object color.
    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            color_mask = cv2.inRange(hsv_frame,lower_hsv, upper_hsv,)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask,)

    combined_mask = cv2.medianBlur(combined_mask, 5) # apply blur

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel) # remove small random white specks
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel) # fil small black holes/gaps 

    contours, heirarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key = cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < object_vision_spec.minimum_contour_area_px:
        return None
    
    u, v, px_w, px_h = cv2.boundingRect(largest_contour)
    u = u + px_w/2.0
    v = v + px_h/2.0

    return Detection(u, v, px_w, px_h, )


# Triangle path: find color-based candidates, group nearby markers, then select the best group.
def findSingleObjectUsingBestTriangleGroup(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, debug: DetectionDebug | None = None) -> Detection | None:
    triangle_markers = object_vision_spec.triangle_markers

    if not triangle_markers:
        raise ValueError("object_vision_spec.triangle_markers cannot be empty")

    # Build the per-color search list and lowest allowed triangle area.
    unique_color_ids = list(dict.fromkeys(marker.color_id for marker in triangle_markers))
    minimum_triangle_area_by_color = {
        color_id: min(
            marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
            for marker in triangle_markers if marker.color_id == color_id
        )
        for color_id in unique_color_ids
    }

    # Prepare shared masks, candidate storage, and optional debug frames.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    triangle_candidates: list[TriangleDetection] = []
    combined_raw_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    combined_cleaned_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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

        # Convert qualifying contours into three-vertex triangle candidates.
        contours, _ = cv2.findContours(cleaned_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour_debug_frame is not None:
                cv2.drawContours(contour_debug_frame, [contour], -1, draw_bgr, 1)

            contour_area = cv2.contourArea(contour)

            if contour_area < minimum_triangle_area_by_color[color_id]:
                continue

            # Bridge small outline gaps before approximating the triangle.
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            polygon = cv2.approxPolyDP(hull, object_vision_spec.polygon_epsilon_ratio * perimeter, True)
            if len(polygon) > 3 and len(polygon) < 6: #special case, more leniency
                polygon = cv2.approxPolyDP(contour, (object_vision_spec.polygon_epsilon_ratio + 0.02) * perimeter, True)
            
            if polygon_debug_frame is not None:
                cv2.polylines(polygon_debug_frame, [polygon], True, draw_bgr, 2)
                polygon_center = np.mean(polygon.reshape(-1, 2), axis=0).astype(np.int32)
                cv2.putText(polygon_debug_frame, f"{color_name}: {len(polygon)} vertices", tuple(polygon_center), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_bgr, 1, cv2.LINE_AA)

            if len(polygon) != 3 or not cv2.isContourConvex(polygon):
                continue

            vertices_px = polygon.reshape(3, 2).astype(np.float64)
            triangle_candidates.append(TriangleDetection(vertices_px=vertices_px, color_id=color_id))

            if candidate_debug_frame is not None:
                triangle_index = len(triangle_candidates) - 1
                center_px = np.mean(vertices_px, axis=0).astype(np.int32)
                triangle_points = np.round(vertices_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(candidate_debug_frame, [triangle_points], True, draw_bgr, 3)
                cv2.circle(candidate_debug_frame, tuple(center_px), 4, draw_bgr, -1)
                cv2.putText(candidate_debug_frame, f"T{triangle_index}: {color_name}", (int(center_px[0]) + 5, int(center_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_bgr, 2, cv2.LINE_AA)

    if debug is not None:
        debug.addStage("Combined raw mask", combined_raw_mask)
        debug.addStage("Combined cleaned mask", combined_cleaned_mask)
        debug.addStage("All mask contours", contour_debug_frame)
        debug.addStage("Polygon approximations", polygon_debug_frame)

        if not triangle_candidates:
            cv2.putText(candidate_debug_frame, "No accepted triangles", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Accepted triangle candidates", candidate_debug_frame)

    if not triangle_candidates:
        return None

    # Form the largest valid nearby groups allowed by the marker specification.
    triangle_groups = groupTriangleCandidates(triangle_candidates, object_vision_spec)

    if debug is not None:
        group_debug_frame = frame.copy()

        for group_index, triangle_group in enumerate(triangle_groups):
            all_group_vertices = np.concatenate([triangle.vertices_px for triangle in triangle_group], axis=0)

            for triangle in triangle_group:
                draw_bgr = COLOR_SPECS[triangle.color_id].draw_bgr
                triangle_points = np.round(triangle.vertices_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(group_debug_frame, [triangle_points], True, draw_bgr, 3)

            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_group_vertices.astype(np.float32))
            cv2.rectangle(group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 255, 255), 2)
            cv2.putText(group_debug_frame, f"G{group_index}: {len(triangle_group)}/{len(triangle_markers)} markers", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if not triangle_groups:
            cv2.putText(group_debug_frame, "No groups matched triangle_markers", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Matching triangle groups", group_debug_frame)

    if not triangle_groups:
        return None

    # Prefer more visible markers, then greater combined triangle area.
    best_triangle_group = selectBestTriangleGroup(triangle_groups, object_vision_spec)

    if debug is not None:
        best_group_debug_frame = frame.copy()
        all_best_vertices = np.concatenate([triangle.vertices_px for triangle in best_triangle_group], axis=0)

        for triangle in best_triangle_group:
            draw_bgr = COLOR_SPECS[triangle.color_id].draw_bgr
            triangle_points = np.round(triangle.vertices_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(best_group_debug_frame, [triangle_points], True, draw_bgr, 3)

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
        cv2.rectangle(best_group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
        cv2.putText(best_group_debug_frame, f"Selected best group: {len(best_triangle_group)}/{len(triangle_markers)} markers", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)        
        debug.addStage("Selected best triangle group", best_group_debug_frame)

    # Use the selected markers' combined image bounds as the detection bounds.
    all_best_vertices = np.concatenate([triangle.vertices_px for triangle in best_triangle_group], axis=0)
    bbox_x, bbox_y, px_w, px_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
    detection = Detection(
        u=bbox_x + px_w/2.0, v=bbox_y + px_h/2.0,
        px_w=float(px_w), px_h=float(px_h),
        triangles=best_triangle_group
    )

    if debug is not None:
        final_debug_frame = frame.copy()
        bbox_x, bbox_y = int(round(detection.u - detection.px_w / 2.0)), int(round(detection.v - detection.px_h / 2.0))
        bbox_x_2, bbox_y_2 = int(round(detection.u + detection.px_w / 2.0)), int(round(detection.v + detection.px_h / 2.0))
        cv2.rectangle(final_debug_frame, (bbox_x, bbox_y), (bbox_x_2, bbox_y_2), (0, 0, 255), 3)
        cv2.circle(final_debug_frame, (int(round(detection.u)), int(round(detection.v))), 5, (0, 0, 255), -1)
        cv2.putText(final_debug_frame, f"u={detection.u:.1f}, v={detection.v:.1f}, w={detection.px_w:.1f}, h={detection.px_h:.1f}", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        debug.addStage("Final object detection", final_debug_frame)

    return detection


# Triangle grouping and selection helpers.
def groupTriangleCandidates(triangle_candidates: list[TriangleDetection], object_vision_spec: ObjectVisionSpec) -> list[list[TriangleDetection]]:
    triangle_markers = object_vision_spec.triangle_markers
    required_color_counts = Counter(marker.color_id for marker in triangle_markers)
    maximum_group_size = min(len(triangle_candidates), len(triangle_markers))

    # Collect the configured count and minimum area for each marker color.
    marker_minimum_areas_by_color: dict[ColorId, list[float]] = {}

    for marker in triangle_markers:
        minimum_area = marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
        marker_minimum_areas_by_color.setdefault(marker.color_id, []).append(minimum_area)

    for minimum_areas in marker_minimum_areas_by_color.values():
        minimum_areas.sort()

    marker_color_order: dict[ColorId, int] = {}

    for marker_index, marker in enumerate(triangle_markers):
        marker_color_order.setdefault(marker.color_id, marker_index)

    # Prefer the largest valid group, then fall back to partial groups.
    for group_size in range(maximum_group_size, 0, -1):
        triangle_groups: list[list[TriangleDetection]] = []

        for triangle_combination in combinations(triangle_candidates, group_size):
            triangle_group = list(triangle_combination)
            group_color_counts = Counter(triangle.color_id for triangle in triangle_group)

            # Partial groups may omit markers but cannot exceed a configured color count.
            if any(count > required_color_counts[color_id] for color_id, count in group_color_counts.items()):
                continue

            # Require every marker to connect through the group's nearby-marker graph.
            connected_indices, pending_indices = {0}, [0]

            while pending_indices:
                current_index = pending_indices.pop()

                for candidate_index in range(group_size):
                    if candidate_index in connected_indices:
                        continue

                    if triangleCandidatesAreNear(triangle_group[current_index], triangle_group[candidate_index], object_vision_spec.triangle_group_distance_factor):
                        connected_indices.add(candidate_index)
                        pending_indices.append(candidate_index)

            if len(connected_indices) != group_size:
                continue

            # Enforce the applicable per-marker minimum area for each color.
            valid_group = True

            for color_id, color_count in group_color_counts.items():
                triangle_areas = sorted(cv2.contourArea(triangle.vertices_px.astype(np.float32)) for triangle in triangle_group if triangle.color_id == color_id)
                minimum_areas = marker_minimum_areas_by_color[color_id][:color_count]

                if any(triangle_area < minimum_area for triangle_area, minimum_area in zip(triangle_areas, minimum_areas)):
                    valid_group = False
                    break

            if not valid_group:
                continue

            triangle_group.sort(key=lambda triangle: (marker_color_order[triangle.color_id], float(np.mean(triangle.vertices_px[:, 0])), float(np.mean(triangle.vertices_px[:, 1])),))
            triangle_groups.append(triangle_group)

        if triangle_groups:
            return triangle_groups

    return []


def triangleCandidatesAreNear(triangle_1: TriangleDetection, triangle_2: TriangleDetection, distance_factor: float) -> bool:
    center_1, center_2 = np.mean(triangle_1.vertices_px, axis=0), np.mean(triangle_2.vertices_px, axis=0)
    _, _, width_1, height_1 = cv2.boundingRect(triangle_1.vertices_px.astype(np.float32))
    _, _, width_2, height_2 = cv2.boundingRect(triangle_2.vertices_px.astype(np.float32))
    reference_size = max(np.hypot(width_1, height_1), np.hypot(width_2, height_2))
    return np.linalg.norm(center_1 - center_2) <= distance_factor * reference_size


def selectBestTriangleGroup(
    triangle_groups: list[list[TriangleDetection]], object_vision_spec: ObjectVisionSpec,
    marker_count_weight: float = 0.80, compactness_weight: float = 0.15, area_weight: float = 0.05,
) -> list[TriangleDetection] | None:

    # Todo: Perhaps use object_vision_spec.object_vertices_m to reward groups matching the expected physical marker layout.

    if not triangle_groups:
        return None

    # Normalize triangle area relative to the largest average group area.
    maximum_average_area = max(
        sum(cv2.contourArea(triangle.vertices_px.astype(np.float32)) for triangle in group) / len(group)
        for group in triangle_groups
    )

    best_group = None
    best_score = float("-inf")

    # Score each candidate group using marker count, compactness, and marker area.
    for group in triangle_groups:
        marker_count_score = len(group) / len(object_vision_spec.color_ids)
        maximum_normalized_distance = 0.0

        # Measure the greatest separation between any two triangles relative to their sizes.
        for i, triangle_1 in enumerate(group):
            for triangle_2 in group[i + 1:]:
                center_1, center_2 = np.mean(triangle_1.vertices_px, axis=0), np.mean(triangle_2.vertices_px, axis=0)
                size_1 = np.max(np.linalg.norm(triangle_1.vertices_px[:, None] - triangle_1.vertices_px[None, :], axis=2))
                size_2 = np.max(np.linalg.norm(triangle_2.vertices_px[:, None] - triangle_2.vertices_px[None, :], axis=2))
                normalized_distance = np.linalg.norm(center_1 - center_2) / max((size_1 + size_2) / 2.0, 1e-6)
                maximum_normalized_distance = max(maximum_normalized_distance, normalized_distance)

        # Groups near the permitted distance limit receive a lower compactness score.
        compactness_score = (
            max(0.0, 1.0 - maximum_normalized_distance / object_vision_spec.triangle_group_distance_factor)
            if len(group) > 1 else 0.0
        )

        average_area = sum(cv2.contourArea(triangle.vertices_px.astype(np.float32)) for triangle in group) / len(group)
        area_score = average_area / maximum_average_area if maximum_average_area > 0.0 else 0.0

        score = marker_count_weight * marker_count_score + compactness_weight * compactness_score + area_weight * area_score

        if score > best_score:
            best_group, best_score = group, score

    return best_group


# Convert the selected image-space detection into a world-space measurement.
def createMeasurementUsingTriangleGroup(detection: Detection, object_vision_spec: ObjectVisionSpec) -> Measurement:
    failed_measurement = Measurement(None, None, None, None, None, None,)

    if not detection.triangles:
        return failed_measurement

    marker_indices_by_color: dict[ColorId, list[int]] = {}
    for marker_index, marker in enumerate(object_vision_spec.triangle_markers):
        marker_indices_by_color.setdefault(marker.color_id, []).append(marker_index)

    object_point_groups, image_point_groups, used_marker_indices = [], [], set()
    edge_pairs = ((0, 1), (1, 2), (2, 0))

    # Match every detected triangle to its physical marker and determine its vertex correspondence.
    for triangle in detection.triangles:
        available_marker_indices = [
            marker_index for marker_index in marker_indices_by_color.get(triangle.color_id, [])
            if marker_index not in used_marker_indices
        ]

        if len(available_marker_indices) != 1:
            raise ValueError(f"Detected color {triangle.color_id} must match exactly one unused triangle marker")

        marker_index = available_marker_indices[0]
        marker = object_vision_spec.triangle_markers[marker_index]

        if marker.object_vertices_m is None:
            raise ValueError(f"Triangle marker {triangle.color_id} has no object_vertices_m")

        marker_vertices_m = np.asarray(marker.object_vertices_m, dtype=np.float64)
        triangle_vertices_px = np.asarray(triangle.vertices_px, dtype=np.float64)

        if marker_vertices_m.shape != (3, 3):
            raise ValueError(f"object_vertices_m for {triangle.color_id} must have shape (3, 3)")
        if triangle_vertices_px.shape != (3, 2):
            raise ValueError(f"vertices_px for {triangle.color_id} must have shape (3, 2)")
        if not np.all(np.isfinite(marker_vertices_m)) or not np.all(np.isfinite(triangle_vertices_px)):
            return failed_measurement

        # approxPolyDP does not guarantee a useful vertex order. Select the ordering whose edge-length ratios
        # most closely resemble the known physical triangle.
        object_edge_lengths = np.array([np.linalg.norm(marker_vertices_m[i] - marker_vertices_m[j]) for i, j in edge_pairs])
        object_shape_norm = np.linalg.norm(object_edge_lengths)

        if object_shape_norm <= 1e-12:
            raise ValueError(f"object_vertices_m for {triangle.color_id} form a degenerate triangle")

        normalized_object_edge_lengths = object_edge_lengths/object_shape_norm
        best_vertices_px, best_shape_error = None, float("inf")

        for vertex_order in permutations(range(3)):
            ordered_vertices_px = triangle_vertices_px[list(vertex_order)]
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

    # TODO: Replace these approximate intrinsics and zero distortion with calibrated camera parameters.
    focal_length_px = float(config.PX_FOCAL_LENGTH)
    camera_matrix = np.array([
        [focal_length_px, 0.0, config.FRAME_W/2.0],
        [0.0, focal_length_px, config.FRAME_H/2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    distortion_coefficients = np.zeros((5, 1), dtype=np.float64)

    # SQPnP supports a partial group containing only one three-vertex marker.
    solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, distortion_coefficients, flags=cv2.SOLVEPNP_SQPNP,
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
            object_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients,
        )
        reprojection_error = float(np.sqrt(np.mean(np.sum((projected_points.reshape(-1, 2) - image_points)**2, axis=1))))

        if reprojection_error < best_reprojection_error:
            best_translation, best_reprojection_error = translation_vector.reshape(3), reprojection_error

    if best_translation is None or not np.all(np.isfinite(best_translation)):
        return failed_measurement

    return Measurement(float(best_translation[0]), float(best_translation[1]), float(best_translation[2]), None, None, None,)