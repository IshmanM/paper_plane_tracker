from __future__ import annotations

import cv2
import numpy as np
import time
from collections import Counter
from itertools import combinations, permutations

from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.geometry import estimateObjectWorldPosition
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec, ObjectVisionSpecId, ShapeMarkerSpec, getRigidPlaneIntersection
from src.primary.color import COLOR_SPECS, ColorId


# Small temporal PnP caches. They only warm-start/tie-break; poor geometry still forces a fresh solve.
_PNP_WARM_START_ANGLES_DEG: dict[tuple[int, frozenset[str]], float] = {}
_PNP_WARM_START_ORDERED_VERTICES: dict[tuple[int, frozenset[str]], dict[tuple[str, int], np.ndarray]] = {}
_PNP_PREVIOUS_PLANE_GROUP: dict[int, tuple[str, ...]] = {}
_PNP_PREVIOUS_TRANSLATION_M: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}


# Detection data passed between image processing, drawing, and measurement conversion.
class ShapeDetection:
    def __init__(
        self,
        vertices_px: list[list[float]] | np.ndarray | None = None,
        color_id: ColorId | None = None,
        num_sides: int = 3,
        ellipse_px: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    ):
        if vertices_px is None and ellipse_px is None:
            raise ValueError("ShapeDetection requires vertices_px or ellipse_px")

        if vertices_px is not None and ellipse_px is not None:
            raise ValueError("ShapeDetection cannot have both vertices_px and ellipse_px")

        self.vertices_px = None if vertices_px is None else np.asarray(vertices_px, dtype=np.float64)
        self.color_id = color_id
        self.num_sides = 0 if ellipse_px is not None else num_sides
        self.ellipse_px = ellipse_px


class Detection:
    def __init__(self, u: float | None, v: float | None, px_w: float | None, px_h: float | None, shapes: list[ShapeDetection] | None = None,
                 bbox_center_offset_px: tuple[float, float] | np.ndarray | None = None, plane_ids: tuple[str, ...] | None = None,
                 shape_marker_keys: list[tuple[str, int]] | None = None):
        """
        u, v are the detected object's center in image pixels.

        bbox_center_offset_px is optional and gives the vector from the object center
        to the bounding-box center:
            bbox_center = [u, v] + bbox_center_offset_px

        When omitted, the bounding box is centered on the object center as before.
        """
        self.u = u
        self.v = v
        self.px_w = px_w
        self.px_h = px_h
        self.shapes = shapes if shapes is not None else []
        self.plane_ids = None if plane_ids is None else tuple(plane_ids)
        self.shape_marker_keys = None if shape_marker_keys is None else list(shape_marker_keys)

        if bbox_center_offset_px is None:
            self.bbox_center_offset_px = None
        else:
            bbox_center_offset_px = np.asarray(bbox_center_offset_px, dtype=np.float64)
            if bbox_center_offset_px.shape != (2,) or not np.all(np.isfinite(bbox_center_offset_px)):
                raise ValueError("bbox_center_offset_px must be a finite length-2 vector")
            self.bbox_center_offset_px = bbox_center_offset_px


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
        self.timings_ms: dict[str, float] = {}
        self._reference_image: np.ndarray | None = None
        self._timing_stage_index: int | None = None

    def reset(self, reference_image: np.ndarray | None = None) -> None:
        self.stages.clear()
        self.timings_ms.clear()
        self._reference_image = None if reference_image is None else reference_image.copy()
        self._timing_stage_index = None

    def addStage(self, name: str, image: np.ndarray) -> None:
        self.stages.append((name, image.copy()))

    def setTiming(self, name: str, elapsed_s: float) -> None:
        self.timings_ms[name] = 1000.0*float(elapsed_s)

    def updateTimingStage(self) -> None:
        if self._reference_image is None or not self.timings_ms:
            return

        timing_height = max(self._reference_image.shape[0], 80 + 27*len(self.timings_ms))
        timing_image = np.full((timing_height, self._reference_image.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(timing_image, "TIMING (warmed; debug drawing excluded)", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        y = 62
        for name, elapsed_ms in self.timings_ms.items():
            cv2.putText(timing_image, f"{name}: {elapsed_ms:.2f} ms", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            y += 27

        stage = ("Timing summary", timing_image)
        if self._timing_stage_index is None:
            self._timing_stage_index = len(self.stages)
            self.stages.append(stage)
        else:
            self.stages[self._timing_stage_index] = stage


# Public detection entry points and shared result helpers.
def detectSingleObject(
    frame: np.ndarray, object_vision_spec_id: ObjectVisionSpecId,
    camera_calibration: CameraCalibration,
) -> tuple[bool, Detection, Measurement]:
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]

    if object_vision_spec.object_type == ObjectType.TENNIS_BALL:
        return detectTennisBall(frame, object_vision_spec, camera_calibration)
    elif object_vision_spec.object_type == ObjectType.ARUCO_MARKER:
        return detectArucoMarkerV2(
            frame, object_vision_spec, camera_calibration,
        )
    elif object_vision_spec.object_type == ObjectType.PAPER_PLANE_SHAPES:
        return detectPaperPlaneShapes(frame, object_vision_spec, camera_calibration)
    elif object_vision_spec.object_type == ObjectType.PAPER_PLANE_PURE_COLOR:
        return detectPaperPlanePureColor(frame, object_vision_spec, camera_calibration)

    raise ValueError(f"Unsupported object type for {object_vision_spec_id}: {object_vision_spec.object_type}")


def detectTennisBall(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectSphere(frame, object_vision_spec, camera_calibration)
    # detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_vision_spec.width, camera_calibration,)
    measurement = Measurement(x, y, z, None, None, None)
    return True, detection, measurement



def detectPaperPlanePureColor(
    frame: np.ndarray, object_vision_spec: ObjectVisionSpec,
    camera_calibration: CameraCalibration,
) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectTriangleColorBlob(frame, object_vision_spec)
    if detection is None:
        return failedDetectionResult()
    if object_vision_spec.width is None or object_vision_spec.width <= 0.0:
        raise ValueError("PAPER_PLANE_PURE_COLOR requires width = physical longest triangle span in meters")

    vertices = detection.shapes[0].vertices_px
    normalized_vertices = cv2.undistortPoints(
        vertices.reshape(-1, 1, 2), camera_calibration.camera_matrix,
        camera_calibration.distortion_coefficients,
    ).reshape(-1, 2)
    image_span = max(
        np.linalg.norm(normalized_vertices[i] - normalized_vertices[j])
        for i, j in combinations(range(3), 2)
    )
    if image_span <= 1e-9:
        return failedDetectionResult(detection)

    z = float(object_vision_spec.width/image_span)
    center_normalized = cv2.undistortPoints(
        np.array([[[detection.u, detection.v]]], dtype=np.float64),
        camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
    )[0, 0]
    return True, detection, Measurement(
        x=float(center_normalized[0]*z), y=float(center_normalized[1]*z), z=z,
    )


def findSingleObjectTriangleColorBlob(
    frame: np.ndarray, object_vision_spec: ObjectVisionSpec,
    debug: DetectionDebug | None = None,
) -> Detection | None:
    """
    Pure-color triangle detector using the tennis-ball detector's color pipeline:
        LAB hotspot -> loose HSV seed -> nearby-component retention
        -> LAB-gradient radial boundary refinement -> triangle fit.

    HSV only seeds the object. The final triangle comes from LAB-refined boundary
    evidence, so incomplete HSV coverage is acceptable.
    """
    if not object_vision_spec.color_ids:
        raise ValueError("Pure-color triangle detection requires at least one color_id")
    if object_vision_spec.minimum_contour_area_px is None:
        raise ValueError("Pure-color triangle detection requires minimum_contour_area_px")

    MAX_CANDIDATES = 2
    NUM_RAYS = 120
    MIN_BOUNDARY_POINTS = 20
    LAB_CHROMA_GRADIENT_GAIN = 2.0
    MIN_LAB_EDGE_STRENGTH = 35.0

    GLOBAL_BLUR_KERNEL = (5, 5)
    HOTSPOT_PERCENTILE = 98.5
    MIN_HOTSPOT_RESPONSE_FACTOR = 0.30
    MIN_HOTSPOT_AREA_PX = 6
    # Pure-color paper planes are much larger/asymmetric than a tennis ball, and a
    # LAB hotspot may land on only one bright wing patch. Give HSV enough room to see
    # the rest of the plane before the tennis-ball-style component joining runs.
    HOTSPOT_PADDING_FACTOR = 1.25
    MIN_HOTSPOT_PADDING_PX = 24

    LOOSE_HSV_LOWER_SUBTRACTION = np.array([0, 40, 15], dtype=np.int16)
    MIN_SECONDARY_SEED_AREA_FACTOR = 0.15
    MAX_SECONDARY_SEED_DISTANCE_FACTOR = 1.25

    # Same LAB target-color construction as findSingleObjectSphere().
    lab_direction = np.zeros(2, dtype=np.float32)
    reference_chroma_strengths = []

    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        if color_spec.lab_value is None:
            raise ValueError(f"Pure-color triangle detection requires ColorSpec.lab_value for {color_id}")

        direction = color_spec.lab_value[1:3].astype(np.float32) - 128.0
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0.0:
            lab_direction += direction/direction_norm
            reference_chroma_strengths.append(direction_norm)

    lab_direction_norm = np.linalg.norm(lab_direction)
    if lab_direction_norm == 0.0 or not reference_chroma_strengths:
        raise ValueError("Configured ColorSpec LAB values do not define a valid chroma direction")

    lab_direction /= lab_direction_norm
    lab_a_direction, lab_b_direction = float(lab_direction[0]), float(lab_direction[1])
    reference_chroma_strength = float(np.mean(reference_chroma_strengths))

    blurred_frame = cv2.GaussianBlur(frame, GLOBAL_BLUR_KERNEL, 0)
    lab_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
    _, a_u8, b_u8 = cv2.split(lab_frame)
    a, b = a_u8.astype(np.float32) - 128.0, b_u8.astype(np.float32) - 128.0

    lab_color_response = a*lab_a_direction + b*lab_b_direction
    positive_response = np.maximum(lab_color_response, 0.0)

    minimum_hotspot_response = MIN_HOTSPOT_RESPONSE_FACTOR*reference_chroma_strength
    percentile_hotspot_response = float(np.percentile(positive_response, HOTSPOT_PERCENTILE))
    hotspot_threshold = max(minimum_hotspot_response, percentile_hotspot_response)
    hotspot_mask = (positive_response >= hotspot_threshold).astype(np.uint8)*255

    hotspot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, hotspot_kernel)
    hotspot_mask = cv2.erode(hotspot_mask, hotspot_kernel, iterations=1)
    hotspot_mask = cv2.dilate(hotspot_mask, hotspot_kernel, iterations=1)

    if debug is not None:
        debug.reset(frame)
        debug.addStage("Original", frame)
        response_debug = np.clip(128.0 + 4.0*lab_color_response, 0, 255).astype(np.uint8)
        debug.addStage("LAB color response from ColorSpec LAB", response_debug)
        debug.addStage("LAB hotspot mask", hotspot_mask)

    # Same LAB hotspot ranking / spatial de-duplication as the tennis-ball path.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)
    response_sums = np.bincount(labels.ravel(), weights=positive_response.ravel(), minlength=num_labels)
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_HOTSPOT_AREA_PX:
            continue

        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        mean_response = float(response_sums[label]/max(area, 1))
        candidates.append((mean_response*(area**0.10), area, x, y, w, h, mean_response))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    selected_candidates = []

    for candidate in candidates:
        _, _, x, y, w, h, _ = candidate
        center_x, center_y = x + 0.5*w, y + 0.5*h
        duplicate = False

        for selected_candidate in selected_candidates:
            _, _, sx, sy, sw, sh, _ = selected_candidate
            selected_center_x, selected_center_y = sx + 0.5*sw, sy + 0.5*sh
            dx, dy = center_x - selected_center_x, center_y - selected_center_y
            duplicate_distance = 0.75*max(w, h, sw, sh)

            if dx*dx + dy*dy < duplicate_distance*duplicate_distance:
                duplicate = True
                break

        if not duplicate:
            selected_candidates.append(candidate)
        if len(selected_candidates) >= MAX_CANDIDATES:
            break

    candidates = selected_candidates

    if debug is not None:
        candidate_frame = frame.copy()
        for candidate_index, (score, area, x, y, w, h, mean_response) in enumerate(candidates, start=1):
            cv2.rectangle(candidate_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(
                candidate_frame,
                f"{candidate_index}: area={area} resp={mean_response:.1f} score={score:.1f}",
                (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 255, 255), 1, cv2.LINE_AA,
            )
        debug.addStage("LAB hotspot candidates", candidate_frame)

    angles = np.linspace(0.0, 2.0*np.pi, NUM_RAYS, endpoint=False)
    directions_u, directions_v = np.cos(angles)[:, None], np.sin(angles)[:, None]
    best_result, best_score = None, -np.inf

    for candidate_index, (_, _, hot_x, hot_y, hot_w, hot_h, _) in enumerate(candidates, start=1):
        hotspot_size = max(hot_w, hot_h)
        padding = max(MIN_HOTSPOT_PADDING_PX, int(HOTSPOT_PADDING_FACTOR*hotspot_size))
        x1, y1 = max(0, hot_x - padding), max(0, hot_y - padding)
        x2 = min(frame.shape[1], hot_x + hot_w + padding)
        y2 = min(frame.shape[0], hot_y + hot_h + padding)

        roi_frame = np.ascontiguousarray(frame[y1:y2, x1:x2])
        if roi_frame.size == 0 or roi_frame.shape[0] < 2 or roi_frame.shape[1] < 2:
            continue

        # Same loose-HSV acquisition as the tennis-ball detector.
        roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        seed_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)

        for color_id in object_vision_spec.color_ids:
            color_spec = COLOR_SPECS[color_id]
            for lower_hsv, upper_hsv in color_spec.hsv_ranges:
                loose_lower_hsv = np.clip(
                    lower_hsv.astype(np.int16) - LOOSE_HSV_LOWER_SUBTRACTION, 0, 255,
                ).astype(np.uint8)
                seed_mask = cv2.bitwise_or(
                    seed_mask, cv2.inRange(roi_hsv, loose_lower_hsv, upper_hsv)
                )

        num_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
            seed_mask, connectivity=8
        )
        best_seed = None

        for seed_label in range(1, num_seed_labels):
            seed_area = int(seed_stats[seed_label, cv2.CC_STAT_AREA])
            if seed_area < object_vision_spec.minimum_contour_area_px:
                continue

            sx = int(seed_stats[seed_label, cv2.CC_STAT_LEFT])
            sy = int(seed_stats[seed_label, cv2.CC_STAT_TOP])
            sw = int(seed_stats[seed_label, cv2.CC_STAT_WIDTH])
            sh = int(seed_stats[seed_label, cv2.CC_STAT_HEIGHT])

            if best_seed is None or seed_area > best_seed[0]:
                best_seed = (seed_area, seed_label, sx, sy, sw, sh)

        if debug is not None:
            raw_seed_debug = np.zeros(frame.shape[:2], dtype=np.uint8)
            raw_seed_debug[y1:y2, x1:x2] = seed_mask
            debug.addStage(f"Candidate {candidate_index} loose HSV seed mask - raw", raw_seed_debug)

        if best_seed is None:
            continue

        seed_area, seed_label, sx, sy, sw, sh = best_seed

        if debug is not None:
            component_mask = (seed_labels == seed_label).astype(np.uint8)*255
            local_contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if local_contours:
                selected_debug = frame.copy()
                selected_contour = max(local_contours, key=cv2.contourArea)
                selected_contour += np.array([[[x1, y1]]], dtype=np.int32)
                cv2.rectangle(selected_debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
                cv2.drawContours(selected_debug, [selected_contour], -1, (0, 255, 255), 1)
                debug.addStage(f"Candidate {candidate_index} selected HSV seed", selected_debug)

        # Same nearby-component retention as the tennis-ball path.
        seed_center_x, seed_center_y = sx + 0.5*sw, sy + 0.5*sh
        min_secondary_area = MIN_SECONDARY_SEED_AREA_FACTOR*seed_area
        max_secondary_distance_sq = (MAX_SECONDARY_SEED_DISTANCE_FACTOR*max(sw, sh))**2

        keep_seed_label = np.zeros(num_seed_labels, dtype=np.uint8)
        keep_seed_label[seed_label] = 255

        for other_label in range(1, num_seed_labels):
            if other_label == seed_label:
                continue

            other_area = int(seed_stats[other_label, cv2.CC_STAT_AREA])
            if other_area < min_secondary_area:
                continue

            ox = int(seed_stats[other_label, cv2.CC_STAT_LEFT])
            oy = int(seed_stats[other_label, cv2.CC_STAT_TOP])
            ow = int(seed_stats[other_label, cv2.CC_STAT_WIDTH])
            oh = int(seed_stats[other_label, cv2.CC_STAT_HEIGHT])
            dx = ox + 0.5*ow - seed_center_x
            dy = oy + 0.5*oh - seed_center_y

            if dx*dx + dy*dy <= max_secondary_distance_sq:
                keep_seed_label[other_label] = 255

        seed_mask = keep_seed_label[seed_labels]

        if debug is not None:
            filtered_seed_debug = np.zeros(frame.shape[:2], dtype=np.uint8)
            filtered_seed_debug[y1:y2, x1:x2] = seed_mask
            debug.addStage(
                f"Candidate {candidate_index} joined HSV blobs - filtered",
                filtered_seed_debug,
            )

        # Important for creased planes: after joining the separate HSV blobs, use the
        # FULL joined mask for both center and radial-search extent. Previously the ray
        # length still mostly came from the primary blob, which could stop before
        # reaching another wing fragment even though that fragment had been retained.
        joined_points = cv2.findNonZero(seed_mask)
        if joined_points is None:
            continue

        _, _, joined_w, joined_h = cv2.boundingRect(joined_points)
        seed_moments = cv2.moments(seed_mask, binaryImage=True)
        if seed_moments["m00"] == 0:
            continue

        center_u = x1 + seed_moments["m10"]/seed_moments["m00"]
        center_v = y1 + seed_moments["m01"]/seed_moments["m00"]
        seed_size = max(joined_w, joined_h, hot_w, hot_h)

        # Same combined LAB-gradient edge strength as the tennis-ball detector.
        lab_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
        lab_roi = cv2.GaussianBlur(lab_roi, (3, 3), 0).astype(np.float32)
        grad_u = cv2.Sobel(lab_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_v = cv2.Sobel(lab_roi, cv2.CV_32F, 0, 1, ksize=3)

        lab_edge_strength = np.sqrt(
            grad_u[:, :, 0]**2 + grad_v[:, :, 0]**2 +
            LAB_CHROMA_GRADIENT_GAIN*(
                grad_u[:, :, 1]**2 + grad_v[:, :, 1]**2 +
                grad_u[:, :, 2]**2 + grad_v[:, :, 2]**2
            )
        )

        if debug is not None:
            roi_debug = frame.copy()
            cv2.rectangle(roi_debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
            debug.addStage(f"Candidate {candidate_index} ROI", roi_debug)
            debug.addStage(
                f"Candidate {candidate_index} LAB edge strength",
                cv2.normalize(lab_edge_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            )

        # Same radial HSV expectation + nearby LAB-edge refinement.
        center_roi_u, center_roi_v = center_u - x1, center_v - y1
        radii = np.arange(1, max(1, int(seed_size)) + 1)
        radius_grid = np.broadcast_to(radii, (NUM_RAYS, len(radii)))

        sample_u = np.rint(center_roi_u + directions_u*radii).astype(np.int32)
        sample_v = np.rint(center_roi_v + directions_v*radii).astype(np.int32)
        valid = (
            (sample_u >= 0) & (sample_u < seed_mask.shape[1]) &
            (sample_v >= 0) & (sample_v < seed_mask.shape[0])
        )

        safe_u = np.clip(sample_u, 0, seed_mask.shape[1] - 1)
        safe_v = np.clip(sample_v, 0, seed_mask.shape[0] - 1)
        seed_hits = (seed_mask[safe_v, safe_u] != 0) & valid
        expected_radii = np.where(seed_hits, radius_grid, 0).max(axis=1)

        search_before = 3
        search_after = np.maximum(5, (0.20*expected_radii).astype(np.int32))
        search_band = (
            (radius_grid >= (expected_radii - search_before)[:, None]) &
            (radius_grid <= (expected_radii + search_after)[:, None]) &
            (expected_radii[:, None] > 0) & valid
        )

        sampled_edge_strength = lab_edge_strength[safe_v, safe_u]
        candidate_strength = np.where(search_band, sampled_edge_strength, 0.0)
        best_edge_indices = np.argmax(candidate_strength, axis=1)
        best_edge_strengths = candidate_strength[np.arange(NUM_RAYS), best_edge_indices]
        rays_with_edge = best_edge_strengths >= MIN_LAB_EDGE_STRENGTH

        if np.count_nonzero(rays_with_edge) < MIN_BOUNDARY_POINTS:
            continue

        ray_indices = np.flatnonzero(rays_with_edge)
        edge_indices = best_edge_indices[rays_with_edge]
        boundary_points = np.column_stack((
            sample_u[ray_indices, edge_indices] + x1,
            sample_v[ray_indices, edge_indices] + y1,
        )).astype(np.float64)

        if debug is not None:
            boundary_debug = frame.copy()
            for point_u, point_v in boundary_points:
                cv2.circle(
                    boundary_debug,
                    (int(round(point_u)), int(round(point_v))),
                    2, (255, 0, 255), -1,
                )
            cv2.putText(
                boundary_debug,
                f"LAB-refined boundary points: {len(boundary_points)}/{NUM_RAYS}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 0, 255), 2, cv2.LINE_AA,
            )
            debug.addStage(f"Candidate {candidate_index} LAB-refined boundary", boundary_debug)

        # Triangle-specific finish: hull the LAB-refined boundary, then require the
        # configured polygon epsilon to reduce it to exactly three convex vertices.
        hull = cv2.convexHull(boundary_points.astype(np.float32))
        perimeter = float(cv2.arcLength(hull, True))
        if perimeter <= 0.0:
            continue

        triangle = cv2.approxPolyDP(
            hull, object_vision_spec.polygon_epsilon_ratio*perimeter, True
        )
        if len(triangle) != 3 or not cv2.isContourConvex(triangle):
            if debug is not None:
                rejected = frame.copy()
                cv2.polylines(rejected, [np.round(hull).astype(np.int32)], True, (0, 0, 255), 1)
                cv2.putText(
                    rejected, f"REJECT: hull approximated to {len(triangle)} sides",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )
                debug.addStage(f"Candidate {candidate_index} rejected - triangle topology", rejected)
            continue

        triangle = triangle.reshape(3, 2).astype(np.float64)
        triangle_area = abs(float(cv2.contourArea(triangle.astype(np.float32))))
        hull_area = abs(float(cv2.contourArea(hull)))
        if triangle_area < object_vision_spec.minimum_contour_area_px or hull_area <= 0.0:
            continue

        # A genuine triangle should explain most of its LAB-refined convex hull.
        fill_ratio = min(hull_area, triangle_area)/max(hull_area, triangle_area)
        if fill_ratio < 0.70:
            continue

        moments = cv2.moments(triangle.astype(np.float32))
        if abs(moments["m00"]) <= 1e-9:
            center_u, center_v = np.mean(triangle, axis=0)
        else:
            center_u = moments["m10"]/moments["m00"]
            center_v = moments["m01"]/moments["m00"]

        bx, by, bw, bh = cv2.boundingRect(triangle.astype(np.float32))
        color_id = object_vision_spec.color_ids[0]
        shape = ShapeDetection(vertices_px=triangle, color_id=color_id, num_sides=3)
        detection = Detection(
            float(center_u), float(center_v), float(bw), float(bh), shapes=[shape]
        )

        score = triangle_area*fill_ratio
        if score > best_score:
            best_score = score
            best_result = detection, boundary_points, fill_ratio

        if debug is not None:
            triangle_debug = frame.copy()
            cv2.polylines(
                triangle_debug,
                [np.round(triangle).astype(np.int32).reshape(-1, 1, 2)],
                True, COLOR_SPECS[color_id].draw_bgr, 2, cv2.LINE_AA,
            )
            cv2.putText(
                triangle_debug,
                f"PASS triangle | area={triangle_area:.0f}px | fill={fill_ratio:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                COLOR_SPECS[color_id].draw_bgr, 2, cv2.LINE_AA,
            )
            debug.addStage(f"Candidate {candidate_index} triangle fit", triangle_debug)

    if best_result is None:
        return None

    detection, boundary_points, fill_ratio = best_result

    if debug is not None:
        final = frame.copy()
        for point_u, point_v in boundary_points:
            cv2.circle(final, (int(round(point_u)), int(round(point_v))), 1, (255, 0, 255), -1)
        drawDetection(final, detection)
        cv2.putText(
            final, f"BEST PURE-COLOR TRIANGLE | fill={fill_ratio:.2f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        debug.addStage("Final pure-color triangle", final)

    return detection



def detectArucoMarker(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    aruco_spec = object_vision_spec.aruco_marker

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_spec.dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejected = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if ids is None:
        return failedDetectionResult()

    ids = ids.flatten()
    matching_indices = np.where(ids == aruco_spec.marker_id)[0]

    if len(matching_indices) == 0:
        return failedDetectionResult()

    marker_corners = np.asarray(corners[matching_indices[0]], dtype=np.float64).reshape(4, 2)
    # top_left, top_right, bottom_right, bottom_left = marker_corners

    min_u, max_u = np.min(marker_corners[:, 0]), np.max(marker_corners[:, 0])
    min_v, max_v = np.min(marker_corners[:, 1]), np.max(marker_corners[:, 1])
    detection = Detection(u=(max_u + min_u)/2.0, v=(max_v + min_v)/2.0, px_w=max_u - min_u, px_h=max_v - min_v,)

    half_side = aruco_spec.marker_length_m/2.0

    object_points = np.array([
        [-half_side,  half_side, 0.0],  # top-left
        [ half_side,  half_side, 0.0],  # top-right
        [ half_side, -half_side, 0.0],  # bottom-right
        [-half_side, -half_side, 0.0],  # bottom-left
    ], dtype=np.float64)

    success, rot_vector, trans_vector = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=marker_corners,
        cameraMatrix=camera_calibration.camera_matrix,
        distCoeffs=camera_calibration.distortion_coefficients,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        return failedDetectionResult(detection)

    trans_vector = trans_vector.reshape(3)
    measurement = Measurement(x=float(trans_vector[0]), y=float(trans_vector[1]), z=float(trans_vector[2]),)

    return True, detection, measurement



class ArucoOpticalFlowTracker:
    """Short-term image-space propagation of the four ArUco corners."""

    def __init__(self, max_flow_only_frames: int = 8):
        self.max_flow_only_frames = max_flow_only_frames
        self.previous_gray = self.previous_corners = None
        self.flow_only_frames = 0
        self.last_forward_backward_errors_px = self.last_rejection_reason = None

    def _clear(self) -> None:
        self.previous_gray = self.previous_corners = None
        self.flow_only_frames = 0

    def _reject(self, reason: str) -> None:
        self.last_rejection_reason = reason
        self._clear()

    def reset(self) -> None:
        self._clear()
        self.last_forward_backward_errors_px = self.last_rejection_reason = None

    def seed(self, gray: np.ndarray, corners: np.ndarray) -> None:
        self.previous_gray = gray.copy()
        self.previous_corners = np.asarray(corners, np.float32).reshape(4, 2).copy()
        self.flow_only_frames = 0
        self.last_forward_backward_errors_px = self.last_rejection_reason = None

    def track(self, gray: np.ndarray) -> np.ndarray | None:
        self.last_forward_backward_errors_px = self.last_rejection_reason = None
        if self.previous_gray is None or self.previous_corners is None:
            self.last_rejection_reason = "no optical-flow seed"
            return None
        if self.previous_gray.shape != gray.shape:
            self._reject("frame dimensions changed")
            return None
        if self.flow_only_frames >= self.max_flow_only_frames:
            self._reject(f"flow-only limit reached ({self.max_flow_only_frames})")
            return None

        prev = self.previous_corners.astype(np.float32)
        diag = float(np.linalg.norm(np.ptp(prev, axis=0)))
        win = int(np.clip(round(0.75*diag), 9, 31))
        win += win % 2 == 0
        lk = dict(winSize=(win, win), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                  minEigThreshold=1e-4)

        curr, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, prev.reshape(-1, 1, 2), None, **lk)
        if curr is None or fwd_status is None:
            self._reject("forward LK failed")
            return None
        curr, fwd_status = curr.reshape(4, 2), fwd_status.reshape(4).astype(bool)

        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, curr.reshape(-1, 1, 2), None, **lk)
        if back is None or back_status is None:
            self._reject("backward LK failed")
            return None
        back, back_status = back.reshape(4, 2), back_status.reshape(4).astype(bool)

        fb = np.linalg.norm(back - prev, axis=1)
        self.last_forward_backward_errors_px = fb.copy()
        max_fb = float(np.clip(0.06*diag, 0.75, 2.5))
        if not np.all(fwd_status & back_status):
            self._reject("one or more LK corners lost")
            return None
        if np.max(fb) > max_fb:
            self._reject(f"forward/back error {np.max(fb):.2f}px > {max_fb:.2f}px")
            return None
        if not np.all(np.isfinite(curr)):
            self._reject("non-finite tracked corners")
            return None

        prev_poly = prev.reshape(-1, 1, 2)
        curr_poly = curr.astype(np.float32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(curr_poly):
            self._reject("tracked quad is not convex")
            return None

        prev_area, curr_area = abs(float(cv2.contourArea(prev_poly))), abs(float(cv2.contourArea(curr_poly)))
        if prev_area <= 1e-6 or curr_area <= 1e-6:
            self._reject("degenerate tracked quad")
            return None
        area_ratio = curr_area/prev_area
        if not 0.45 <= area_ratio <= 2.20:
            self._reject(f"per-frame area ratio {area_ratio:.2f} outside [0.45, 2.20]")
            return None

        prev_sides = np.linalg.norm(np.roll(prev, -1, axis=0) - prev, axis=1)
        curr_sides = np.linalg.norm(np.roll(curr, -1, axis=0) - curr, axis=1)
        min_side = max(2.0, 0.35*float(np.min(prev_sides)))
        if np.min(curr_sides) < min_side:
            self._reject(f"tracked side collapsed to {np.min(curr_sides):.2f}px")
            return None

        max_step = max(20.0, 3.0*diag)
        if np.max(np.linalg.norm(curr - prev, axis=1)) > max_step:
            self._reject(f"corner step exceeded {max_step:.1f}px")
            return None

        self.previous_gray, self.previous_corners = gray.copy(), curr.copy()
        self.flow_only_frames += 1
        return curr.astype(np.float64)


_aruco_optical_flow_tracker = ArucoOpticalFlowTracker(max_flow_only_frames=8)
_aruco_optical_flow_spec_key = None


def resetArucoOpticalFlowTracker() -> None:
    global _aruco_optical_flow_spec_key
    _aruco_optical_flow_tracker.reset()
    _aruco_optical_flow_spec_key = None


def solveArucoMarkerPose(corners: np.ndarray, aruco_spec,
                         camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    min_u, min_v = np.min(corners, axis=0)
    max_u, max_v = np.max(corners, axis=0)
    detection = Detection((min_u + max_u)/2, (min_v + max_v)/2, max_u - min_u, max_v - min_v)

    h = aruco_spec.marker_length_m/2
    object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    success, _, t = cv2.solvePnP(
        object_points, corners, camera_calibration.camera_matrix,
        camera_calibration.distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        return failedDetectionResult(detection)

    t = t.reshape(3)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
        return failedDetectionResult(detection)
    return True, detection, Measurement(x=float(t[0]), y=float(t[1]), z=float(t[2]))


def _sharpenAruco(gray: np.ndarray, k: float = 0.5) -> np.ndarray:
    # Unsharp masking: gray + k * (gray - blurred_gray)
    return cv2.addWeighted(gray, 1.0 + k, cv2.GaussianBlur(gray, (0, 0), 1.0), -k, 0)


def _claheAruco(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _claheSharpenAruco(gray: np.ndarray, k: float = 0.5) -> np.ndarray:
    return _sharpenAruco(_claheAruco(gray), k)


def detectArucoMarkerV2(
    frame: np.ndarray, object_vision_spec: ObjectVisionSpec,
    camera_calibration: CameraCalibration, debug: DetectionDebug | None = None,
) -> tuple[bool, Detection, Measurement]:
    """
    ArUco acquisition/tracking priority:
        raw ArUco
        -> LK if seeded
        -> CLAHE ArUco
        -> CLAHE + strong sharpen (k=1.5) ArUco
        -> fail

    Any successful ArUco detection reseeds LK using the original grayscale frame.
    """
    global _aruco_optical_flow_spec_key

    spec = object_vision_spec.aruco_marker
    tracker = _aruco_optical_flow_tracker
    spec_key = (spec.dictionary_name, spec.marker_id, spec.marker_length_m)
    if spec_key != _aruco_optical_flow_spec_key:
        tracker.reset()
        _aruco_optical_flow_spec_key = spec_key

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    if debug is not None:
        debug.reset(frame)
        debug.addStage("Original image", frame)
        debug.addStage("Grayscale", gray)

    def findMarker(image: np.ndarray) -> np.ndarray | None:
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            return None
        matches = np.where(ids.flatten() == spec.marker_id)[0]
        return None if len(matches) == 0 else np.asarray(corners[matches[0]], np.float64).reshape(4, 2)

    def addCornersStage(name: str, label: str, corners: np.ndarray, color, base: np.ndarray | None = None) -> None:
        if debug is None:
            return
        stage = frame.copy() if base is None else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        points = np.round(corners).astype(np.int32)
        cv2.polylines(stage, [points.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        cv2.putText(stage, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        debug.addStage(name, stage)

    # 1) Raw ArUco.
    corners = findMarker(gray)
    if corners is not None:
        tracker.seed(gray, corners)
        addCornersStage("ArUco result - raw", f"RAW ARUCO ID {spec.marker_id}", corners, (0, 255, 0))
        return solveArucoMarkerPose(corners, spec, camera_calibration)

    # 2) If seeded, try the cheap temporal bridge first.
    if tracker.previous_gray is not None and tracker.previous_corners is not None:
        previous, flow_index = tracker.previous_corners.copy(), tracker.flow_only_frames + 1
        corners = tracker.track(gray)

        if corners is not None:
            if debug is not None:
                stage = frame.copy()
                prev_px, curr_px = np.round(previous).astype(int), np.round(corners).astype(int)
                cv2.polylines(stage, [curr_px.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
                for p0, p1 in zip(prev_px, curr_px):
                    cv2.circle(stage, tuple(p0), 3, (0, 165, 255), -1)
                    cv2.line(stage, tuple(p0), tuple(p1), (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(stage, tuple(p1), 3, (255, 0, 255), -1)

                fb = tracker.last_forward_backward_errors_px
                fb_text = "n/a" if fb is None else "/".join(f"{e:.2f}" for e in fb)
                cv2.putText(stage, f"LK BRIDGE {flow_index}/{tracker.max_flow_only_frames}", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(stage, f"forward-back px: {fb_text}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
                debug.addStage("Optical flow - accepted", stage)

            return solveArucoMarkerPose(corners, spec, camera_calibration)

        if debug is not None:
            stage = frame.copy()
            cv2.putText(stage, "LK REJECTED - TRYING ENHANCED ARUCO", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(stage, tracker.last_rejection_reason or "LK unavailable", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            debug.addStage("Optical flow - rejected", stage)

    # 3) CLAHE reacquisition.
    clahe = _claheAruco(gray)
    if debug is not None:
        debug.addStage("ArUco reacquisition - CLAHE", clahe)

    corners = findMarker(clahe)
    if corners is not None:
        tracker.seed(gray, corners)
        addCornersStage("ArUco result - CLAHE reacquisition",
                        "REACQUIRED WITH CLAHE", corners, (0, 255, 255), clahe)
        return solveArucoMarkerPose(corners, spec, camera_calibration)

    # 4) Last chance: CLAHE + strong unsharp mask, k=1.5.
    clahe_sharp = _sharpenAruco(clahe, 1.5)
    if debug is not None:
        debug.addStage("ArUco reacquisition - CLAHE + sharpen k=1.5", clahe_sharp)

    corners = findMarker(clahe_sharp)
    if corners is not None:
        tracker.seed(gray, corners)
        addCornersStage("ArUco result - strong sharpen reacquisition",
                        "REACQUIRED WITH CLAHE + SHARPEN k=1.5",
                        corners, (0, 255, 255), clahe_sharp)
        return solveArucoMarkerPose(corners, spec, camera_calibration)

    if debug is not None:
        stage = frame.copy()
        cv2.putText(stage, "RAW + LK + CLAHE + SHARPEN FAILED", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2, cv2.LINE_AA)
        debug.addStage("ArUco result - acquisition failed", stage)

    return failedDetectionResult()



def detectPaperPlaneShapes(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectUsingBestShapeGroup(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    measurement = createMeasurementUsingShapeGroup(detection, object_vision_spec, camera_calibration)

    if (
        measurement.x is None or measurement.y is None or measurement.z is None
        or not np.all(np.isfinite([measurement.x, measurement.y, measurement.z]))
        or measurement.z <= 0.0
    ):
        return failedDetectionResult(detection)

    # findSingleObjectUsingBestShapeGroup initially centers Detection on the
    # visible-marker bounding box. For paper planes, PnP gives the actual model/object
    # origin, so make Detection.u/v represent that center while preserving the marker
    # bbox through an offset.
    bbox_center_u, bbox_center_v = detection.u, detection.v
    origin_px, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64), np.zeros((3, 1), dtype=np.float64),
        np.array([[measurement.x], [measurement.y], [measurement.z]], dtype=np.float64),
        camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
    )
    center_u, center_v = origin_px.reshape(2)

    if not np.all(np.isfinite([center_u, center_v])):
        return failedDetectionResult(detection)

    # Degenerate/false PnP solutions can put the model origin absurdly far from the
    # marker group even though all solved marker points happen to remain in front of
    # the camera. Reject those before they pollute the tracker or overflow OpenCV's
    # integer drawing coordinates. The limit is intentionally very generous.
    frame_diagonal_px = float(np.hypot(frame.shape[1], frame.shape[0]))
    bbox_diagonal_px = float(np.hypot(detection.px_w, detection.px_h))
    model_origin_offset_px = float(np.hypot(center_u - bbox_center_u, center_v - bbox_center_v))
    maximum_origin_offset_px = max(frame_diagonal_px, 3.0*bbox_diagonal_px)

    if model_origin_offset_px > maximum_origin_offset_px:
        return failedDetectionResult(detection)

    detection.u = float(center_u)
    detection.v = float(center_v)
    detection.bbox_center_offset_px = np.array([
        float(bbox_center_u - center_u),
        float(bbox_center_v - center_v),
    ], dtype=np.float64)

    return True, detection, measurement


def failedDetectionResult(detection: Detection | None = None) -> tuple[bool, Detection, Measurement]:
    if detection is None:
        detection = Detection(None, None, None, None, [],)
    measurement = Measurement(None, None, None, None, None, None,)
    return False, detection, measurement


def drawDetection(frame: np.ndarray, detection: Detection,) -> None:
    if detection.u is None or detection.v is None or detection.px_w is None or detection.px_h is None:
        return
    if not np.all(np.isfinite([detection.u, detection.v, detection.px_w, detection.px_h])):
        return

    bbox_offset = detection.bbox_center_offset_px
    bbox_center_u = detection.u + (0.0 if bbox_offset is None else bbox_offset[0])
    bbox_center_v = detection.v + (0.0 if bbox_offset is None else bbox_offset[1])

    if not np.all(np.isfinite([bbox_center_u, bbox_center_v])):
        return

    # Keep coordinates inside OpenCV's signed 32-bit point range. Normal off-screen
    # coordinates are fine; only absurd values from a bad pose are suppressed.
    int32_limit = np.iinfo(np.int32).max
    bbox_coordinates = (
        bbox_center_u - detection.px_w/2.0, bbox_center_v - detection.px_h/2.0,
        bbox_center_u + detection.px_w/2.0, bbox_center_v + detection.px_h/2.0,
    )

    if all(abs(value) <= int32_limit for value in bbox_coordinates):
        x_min, y_min, x_max, y_max = (int(round(value)) for value in bbox_coordinates)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2,)

    if abs(detection.u) <= int32_limit and abs(detection.v) <= int32_limit:
        cv2.circle(frame, (int(round(detection.u)), int(round(detection.v))), radius=5, color=(0, 255, 0), thickness=-1,)

    for shape in detection.shapes:
        color_spec = COLOR_SPECS[shape.color_id]

        if shape.ellipse_px is not None:
            cv2.ellipse(frame, shape.ellipse_px, color_spec.draw_bgr, 2, cv2.LINE_AA)
            continue

        vertices_px = shape.vertices_px.astype(np.int32)
        cv2.polylines(frame, [vertices_px.reshape(-1, 1, 2)], isClosed=True, color=color_spec.draw_bgr, thickness=1)

        for vertex_u, vertex_v in vertices_px:
            cv2.circle(frame, (int(vertex_u), int(vertex_v)), radius=4, color=color_spec.draw_bgr, thickness=-1)

def drawModelOrigin(frame: np.ndarray, measurement: Measurement, camera_calibration: CameraCalibration) -> None:
    if measurement.x is None or measurement.y is None or measurement.z is None or measurement.z <= 0.0:
        return

    # The PnP translation is the model/object-frame origin in camera coordinates.
    origin_px, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64), np.zeros((3, 1), dtype=np.float64),
        np.array([[measurement.x], [measurement.y], [measurement.z]], dtype=np.float64),
        camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
    )
    origin_u, origin_v = np.rint(origin_px.reshape(2)).astype(np.int32)
    cv2.drawMarker(frame, (int(origin_u), int(origin_v)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    cv2.putText(frame, "model origin", (int(origin_u) + 8, int(origin_v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


#lab version
def findSingleObjectSphere(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration | None = None, debug: DetectionDebug | None = None) -> Detection | None:
    if not object_vision_spec.color_ids:
        raise ValueError("Sphere detection requires at least one color_id")
    if object_vision_spec.minimum_contour_area_px is None:
        raise ValueError("Sphere detection requires minimum_contour_area_px")

    MAX_SPHERE_CANDIDATES = 2
    NUM_RAYS = 120  # originally 120. TODO: tune for speed vs. accuracy; try 90 later.
    NUM_ANGLE_BINS = 12
    MIN_COVERED_ANGLE_BINS = 8
    MIN_BOUNDARY_POINTS = 20
    LAB_CHROMA_GRADIENT_GAIN = 2.0
    MIN_LAB_EDGE_STRENGTH = 35.0
    MAX_CENTER_SHIFT_FACTOR = 0.40
    MIN_MAX_CENTER_SHIFT_PX = 2.5

    GLOBAL_BLUR_KERNEL = (5, 5)
    HOTSPOT_PERCENTILE = 98.5
    MIN_HOTSPOT_RESPONSE_FACTOR = 0.30
    MIN_HOTSPOT_AREA_PX = 6
    HOTSPOT_PADDING_FACTOR = 0.35

    LOOSE_HSV_LOWER_SUBTRACTION = np.array([0, 40, 15], dtype=np.int16)

    # TODO: Tune these together. Lower area / higher distance preserves more seam-
    # separated ball regions, but increases the chance of retaining nearby HSV noise.
    MIN_SECONDARY_SEED_AREA_FACTOR = 0.15
    MAX_SECONDARY_SEED_DISTANCE_FACTOR = 1.25

    # Step 1: Get the target LAB chroma direction and reference chroma strength
    # directly from ColorSpec.
    lab_direction = np.zeros(2, dtype=np.float32)
    reference_chroma_strengths = []

    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        if color_spec.lab_value is None:
            raise ValueError(f"Sphere detection requires ColorSpec.lab_value for {color_id}")

        direction = color_spec.lab_value[1:3].astype(np.float32) - 128.0
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0.0:
            lab_direction += direction/direction_norm
            reference_chroma_strengths.append(direction_norm)

    lab_direction_norm = np.linalg.norm(lab_direction)
    if lab_direction_norm == 0.0 or not reference_chroma_strengths:
        raise ValueError("Configured ColorSpec LAB values do not define a valid chroma direction")

    lab_direction /= lab_direction_norm
    lab_a_direction, lab_b_direction = float(lab_direction[0]), float(lab_direction[1])
    reference_chroma_strength = float(np.mean(reference_chroma_strengths))

    blurred_frame = frame

    # ----- OPTIONAL GLOBAL 5x5 BLUR START: comment/uncomment freely -----
    blurred_frame = cv2.GaussianBlur(blurred_frame, GLOBAL_BLUR_KERNEL, 0)
    # ----- OPTIONAL GLOBAL 5x5 BLUR END ---------------------------------

    lab_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
    _, a_u8, b_u8 = cv2.split(lab_frame)
    a, b = a_u8.astype(np.float32) - 128.0, b_u8.astype(np.float32) - 128.0

    lab_color_response = a*lab_a_direction + b*lab_b_direction
    positive_response = np.maximum(lab_color_response, 0.0)

    # Require LAB response to be both exceptional within the frame and sufficiently
    # strong in absolute terms relative to the configured target chroma.
    minimum_hotspot_response = MIN_HOTSPOT_RESPONSE_FACTOR*reference_chroma_strength
    percentile_hotspot_response = float(np.percentile(positive_response, HOTSPOT_PERCENTILE))
    hotspot_threshold = max(minimum_hotspot_response, percentile_hotspot_response)
    hotspot_mask = (positive_response >= hotspot_threshold).astype(np.uint8)*255

    hotspot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ----- OPTIONAL LAB HOTSPOT CLEANUP START: comment/uncomment freely -----
    hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, hotspot_kernel)
    hotspot_mask = cv2.erode(hotspot_mask, hotspot_kernel, iterations=1)
    hotspot_mask = cv2.dilate(hotspot_mask, hotspot_kernel, iterations=1)
    # ----- OPTIONAL LAB HOTSPOT CLEANUP END ---------------------------------

    if debug is not None:
        debug.stages.clear()
        debug.addStage("Original", frame)
        response_debug = np.clip(128.0 + 4.0*lab_color_response, 0, 255).astype(np.uint8)
        debug.addStage("LAB color response from ColorSpec LAB", response_debug)
        debug.addStage("LAB hotspot mask", hotspot_mask)

    # Step 2: Convert the strongest LAB-response regions into candidate ROIs.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)
    response_sums = np.bincount(labels.ravel(), weights=positive_response.ravel(), minlength=num_labels)
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_HOTSPOT_AREA_PX:
            continue

        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        mean_response = float(response_sums[label]/max(area, 1))
        candidates.append((mean_response*(area**0.10), area, x, y, w, h, mean_response))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    # Keep spatially distinct hotspots.
    selected_candidates = []

    for candidate in candidates:
        _, _, x, y, w, h, _ = candidate
        center_x, center_y = x + 0.5*w, y + 0.5*h
        duplicate = False

        for selected_candidate in selected_candidates:
            _, _, sx, sy, sw, sh, _ = selected_candidate
            selected_center_x, selected_center_y = sx + 0.5*sw, sy + 0.5*sh
            dx, dy = center_x - selected_center_x, center_y - selected_center_y
            duplicate_distance = 0.75*max(w, h, sw, sh)

            if dx*dx + dy*dy < duplicate_distance*duplicate_distance:
                duplicate = True
                break

        if not duplicate:
            selected_candidates.append(candidate)
        if len(selected_candidates) >= MAX_SPHERE_CANDIDATES:
            break

    candidates = selected_candidates

    if debug is not None:
        candidate_frame = frame.copy()

        for candidate_index, (score, area, x, y, w, h, mean_response) in enumerate(candidates, start=1):
            cv2.rectangle(candidate_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(candidate_frame, f"{candidate_index}: area={area} resp={mean_response:.1f} score={score:.1f}",
                        (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        debug.addStage("LAB hotspot candidates", candidate_frame)

    angles = np.linspace(0.0, 2.0*np.pi, NUM_RAYS, endpoint=False)
    directions_u, directions_v = np.cos(angles)[:, None], np.sin(angles)[:, None]
    best_result, best_final_score = None, -np.inf

    # Step 3: Inside each LAB hotspot ROI, use loose HSV for rough acquisition.
    for candidate_index, (_, _, hot_x, hot_y, hot_w, hot_h, _) in enumerate(candidates, start=1):
        hotspot_size = max(hot_w, hot_h)
        padding = max(8, int(HOTSPOT_PADDING_FACTOR*hotspot_size))
        hot_x1, hot_y1 = max(0, hot_x - padding), max(0, hot_y - padding)
        hot_x2, hot_y2 = min(frame.shape[1], hot_x + hot_w + padding), min(frame.shape[0], hot_y + hot_h + padding)

        roi_frame = np.ascontiguousarray(frame[hot_y1:hot_y2, hot_x1:hot_x2])
        if roi_frame.size == 0 or roi_frame.shape[0] < 2 or roi_frame.shape[1] < 2:
            continue

        roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        seed_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)

        for color_id in object_vision_spec.color_ids:
            color_spec = COLOR_SPECS[color_id]

            for lower_hsv, upper_hsv in color_spec.hsv_ranges:
                loose_lower_hsv = np.clip(lower_hsv.astype(np.int16) - LOOSE_HSV_LOWER_SUBTRACTION, 0, 255).astype(np.uint8)
                seed_mask = cv2.bitwise_or(seed_mask, cv2.inRange(roi_hsv, loose_lower_hsv, upper_hsv))

        num_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
        best_seed = None

        for seed_label in range(1, num_seed_labels):
            seed_area = int(seed_stats[seed_label, cv2.CC_STAT_AREA])
            if seed_area < object_vision_spec.minimum_contour_area_px:
                continue

            sx, sy = int(seed_stats[seed_label, cv2.CC_STAT_LEFT]), int(seed_stats[seed_label, cv2.CC_STAT_TOP])
            sw, sh = int(seed_stats[seed_label, cv2.CC_STAT_WIDTH]), int(seed_stats[seed_label, cv2.CC_STAT_HEIGHT])

            if best_seed is None or seed_area > best_seed[0]:
                best_seed = (seed_area, seed_label, sx, sy, sw, sh)

        if best_seed is None:
            if debug is not None:
                debug.addStage(f"Candidate {candidate_index} loose HSV seed mask", seed_mask)
            continue

        seed_area, seed_label, sx, sy, sw, sh = best_seed
        x, y, w, h = hot_x1 + sx, hot_y1 + sy, sw, sh

        if debug is not None:
            component_mask = (seed_labels[sy:sy + sh, sx:sx + sw] == seed_label).astype(np.uint8)*255
            local_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if local_contours:
                seed_contour = max(local_contours, key=cv2.contourArea) + np.array([[[hot_x1 + sx, hot_y1 + sy]]], dtype=np.int32)
                seed_debug = frame.copy()
                cv2.rectangle(seed_debug, (hot_x1, hot_y1), (hot_x2 - 1, hot_y2 - 1), (255, 255, 255), 1)
                cv2.drawContours(seed_debug, [seed_contour], -1, (0, 255, 255), 1)
                debug.addStage(f"Candidate {candidate_index} selected HSV seed", seed_debug)

            debug.addStage(f"Candidate {candidate_index} loose HSV seed mask - raw", seed_mask)

        # Step 4: Keep the primary HSV component plus only substantial nearby
        # components so small HSV speckles do not distort radial geometry.
        seed_center_x, seed_center_y = sx + 0.5*sw, sy + 0.5*sh
        min_secondary_area = MIN_SECONDARY_SEED_AREA_FACTOR*seed_area
        max_secondary_distance_sq = (MAX_SECONDARY_SEED_DISTANCE_FACTOR*max(sw, sh))**2

        keep_seed_label = np.zeros(num_seed_labels, dtype=np.uint8)
        keep_seed_label[seed_label] = 255

        for other_label in range(1, num_seed_labels):
            if other_label == seed_label:
                continue

            other_area = int(seed_stats[other_label, cv2.CC_STAT_AREA])
            if other_area < min_secondary_area:
                continue

            ox, oy = int(seed_stats[other_label, cv2.CC_STAT_LEFT]), int(seed_stats[other_label, cv2.CC_STAT_TOP])
            ow, oh = int(seed_stats[other_label, cv2.CC_STAT_WIDTH]), int(seed_stats[other_label, cv2.CC_STAT_HEIGHT])
            dx, dy = ox + 0.5*ow - seed_center_x, oy + 0.5*oh - seed_center_y

            if dx*dx + dy*dy <= max_secondary_distance_sq:
                keep_seed_label[other_label] = 255

        seed_mask = keep_seed_label[seed_labels]

        if debug is not None:
            debug.addStage(f"Candidate {candidate_index} loose HSV seed mask - filtered", seed_mask)

        seed_moments = cv2.moments(seed_mask, binaryImage=True)
        if seed_moments["m00"] == 0:
            continue

        center_u = hot_x1 + seed_moments["m10"]/seed_moments["m00"]
        center_v = hot_y1 + seed_moments["m01"]/seed_moments["m00"]

        x1, y1, x2, y2 = hot_x1, hot_y1, hot_x2, hot_y2
        color_roi = roi_frame
        seed_size = max(w, h, hot_w, hot_h)

        # Step 5: Compute combined LAB-gradient strength.
        lab_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2LAB)

        # ----- OPTIONAL ROI LAB 3x3 BLUR START: comment/uncomment freely -----
        lab_roi = cv2.GaussianBlur(lab_roi, (3, 3), 0)
        # ----- OPTIONAL ROI LAB 3x3 BLUR END ---------------------------------

        lab_roi = lab_roi.astype(np.float32)
        grad_u = cv2.Sobel(lab_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_v = cv2.Sobel(lab_roi, cv2.CV_32F, 0, 1, ksize=3)

        lab_edge_strength = np.sqrt(
            grad_u[:, :, 0]**2 + grad_v[:, :, 0]**2 +
            LAB_CHROMA_GRADIENT_GAIN*(
                grad_u[:, :, 1]**2 + grad_v[:, :, 1]**2 +
                grad_u[:, :, 2]**2 + grad_v[:, :, 2]**2
            )
        )

        seed_roi = seed_mask

        if debug is not None:
            roi_debug = frame.copy()
            cv2.rectangle(roi_debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
            debug.addStage(f"Candidate {candidate_index} ROI", roi_debug)
            debug.addStage(f"Candidate {candidate_index} LAB edge strength",
                           cv2.normalize(lab_edge_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        # Step 6: Generate radial samples around the candidate center.
        center_roi_u, center_roi_v = center_u - x1, center_v - y1
        radii = np.arange(1, max(1, int(seed_size)) + 1)
        radius_grid = np.broadcast_to(radii, (NUM_RAYS, len(radii)))

        sample_u = np.rint(center_roi_u + directions_u*radii).astype(np.int32)
        sample_v = np.rint(center_roi_v + directions_v*radii).astype(np.int32)
        valid = (sample_u >= 0) & (sample_u < seed_roi.shape[1]) & (sample_v >= 0) & (sample_v < seed_roi.shape[0])

        safe_u = np.clip(sample_u, 0, seed_roi.shape[1] - 1)
        safe_v = np.clip(sample_v, 0, seed_roi.shape[0] - 1)

        # Step 7: Estimate the outer HSV boundary independently along each ray.
        seed_hits = (seed_roi[safe_v, safe_u] != 0) & valid
        expected_radii = np.where(seed_hits, radius_grid, 0).max(axis=1)

        hsv_expected_points = np.column_stack((
            center_u + directions_u[:, 0]*expected_radii,
            center_v + directions_v[:, 0]*expected_radii,
        ))

        # Step 8: Refine each HSV estimate using the strongest nearby LAB gradient.
        search_before = 3
        search_after = np.maximum(5, (0.20*expected_radii).astype(np.int32))
        search_band = (
            (radius_grid >= (expected_radii - search_before)[:, None]) &
            (radius_grid <= (expected_radii + search_after)[:, None]) &
            (expected_radii[:, None] > 0) & valid
        )

        sampled_edge_strength = lab_edge_strength[safe_v, safe_u]
        candidate_strength = np.where(search_band, sampled_edge_strength, 0.0)
        best_edge_indices = np.argmax(candidate_strength, axis=1)
        best_edge_strengths = candidate_strength[np.arange(NUM_RAYS), best_edge_indices]
        rays_with_edge = best_edge_strengths >= MIN_LAB_EDGE_STRENGTH
        num_rays_with_edge = np.count_nonzero(rays_with_edge)

        if num_rays_with_edge < MIN_BOUNDARY_POINTS:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, f"REJECTED: boundary rays {num_rays_with_edge}/{NUM_RAYS}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - boundary points", failure_frame)
            continue

        ray_indices = np.flatnonzero(rays_with_edge)
        edge_indices = best_edge_indices[rays_with_edge]
        boundary_points = np.column_stack((
            sample_u[ray_indices, edge_indices] + x1,
            sample_v[ray_indices, edge_indices] + y1,
        )).astype(np.float64)

        if debug is not None:
            boundary_frame = frame.copy()

            for point_u, point_v in boundary_points:
                cv2.circle(boundary_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

            cv2.putText(boundary_frame, f"Boundary points: {len(boundary_points)}/{NUM_RAYS}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
            debug.addStage(f"Candidate {candidate_index} boundary points", boundary_frame)

        # Step 9: Fit the initial circle to all selected LAB boundary points.
        A = np.column_stack((2*boundary_points[:, 0], 2*boundary_points[:, 1], np.ones(len(boundary_points))))
        b_fit = boundary_points[:, 0]**2 + boundary_points[:, 1]**2
        initial_circle_u, initial_circle_v, c = np.linalg.lstsq(A, b_fit, rcond=None)[0]
        initial_radius = np.sqrt(max(0.0, c + initial_circle_u**2 + initial_circle_v**2))

        if initial_radius <= 0.0:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, "REJECTED: invalid initial circle radius", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - initial circle", failure_frame)
            continue

        # Step 10: Remove LAB points inconsistent with the initial circle.
        point_radii = np.hypot(boundary_points[:, 0] - initial_circle_u, boundary_points[:, 1] - initial_circle_v)
        residuals = np.abs(point_radii - initial_radius)
        residual_limit = max(2.0, 2.5*np.median(residuals))
        inlier_mask = residuals <= residual_limit
        inlier_points = boundary_points[inlier_mask]
        rejected_points = boundary_points[~inlier_mask]

        if len(inlier_points) < MIN_BOUNDARY_POINTS:
            if debug is not None:
                failure_frame = frame.copy()

                for point_u, point_v in rejected_points:
                    cv2.circle(failure_frame, (int(round(point_u)), int(round(point_v))), 2, (0, 0, 255), -1)
                for point_u, point_v in inlier_points:
                    cv2.circle(failure_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

                cv2.circle(failure_frame, (int(round(initial_circle_u)), int(round(initial_circle_v))),
                           int(round(initial_radius)), (255, 0, 0), 1)
                cv2.putText(failure_frame, f"REJECTED: circle inliers {len(inlier_points)}/{len(boundary_points)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"Residual limit: {residual_limit:.2f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - circle inliers", failure_frame)

            continue

        # Refit using only retained LAB boundary points.
        A = np.column_stack((2*inlier_points[:, 0], 2*inlier_points[:, 1], np.ones(len(inlier_points))))
        b_fit = inlier_points[:, 0]**2 + inlier_points[:, 1]**2
        circle_u, circle_v, c = np.linalg.lstsq(A, b_fit, rcond=None)[0]
        radius = np.sqrt(max(0.0, c + circle_u**2 + circle_v**2))

        if radius <= 0.0:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, "REJECTED: invalid refined circle radius", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - refined circle", failure_frame)
            continue

        if debug is not None:
            # Diagnostic 1: HSV expected boundary vs LAB-selected peaks.
            comparison_frame = frame.copy()

            for ray_index in np.flatnonzero(expected_radii > 0):
                point_u, point_v = hsv_expected_points[ray_index]
                cv2.circle(comparison_frame, (int(round(point_u)), int(round(point_v))), 1, (255, 255, 0), -1)

            for point_u, point_v in boundary_points:
                cv2.circle(comparison_frame, (int(round(point_u)), int(round(point_v))), 1, (255, 0, 255), -1)

            cv2.putText(comparison_frame, "Cyan=HSV | Magenta=LAB peak", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(comparison_frame, "Cyan=HSV | Magenta=LAB peak", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            debug.addStage(f"Candidate {candidate_index} HSV vs LAB", comparison_frame)

            # Diagnostic 2: Initial least-squares fit using every LAB boundary point.
            initial_fit_frame = frame.copy()

            for point_u, point_v in boundary_points:
                cv2.circle(initial_fit_frame, (int(round(point_u)), int(round(point_v))), 1, (180, 180, 180), -1)

            cv2.circle(initial_fit_frame, (int(round(initial_circle_u)), int(round(initial_circle_v))),
                       int(round(initial_radius)), (255, 0, 0), 1)

            text = f"Initial fit: r={initial_radius:.2f}px | points={len(boundary_points)}"
            cv2.putText(initial_fit_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(initial_fit_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            debug.addStage(f"Candidate {candidate_index} initial circle fit", initial_fit_frame)

            # Diagnostic 3: Show only which LAB points survive residual filtering.
            rejection_frame = frame.copy()

            for point_u, point_v in inlier_points:
                cv2.circle(rejection_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

            for point_u, point_v in rejected_points:
                cv2.circle(rejection_frame, (int(round(point_u)), int(round(point_v))), 2, (0, 0, 255), -1)

            text = f"Magenta=inlier | Red=rejected ({len(rejected_points)}/{len(boundary_points)})"
            cv2.putText(rejection_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rejection_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            debug.addStage(f"Candidate {candidate_index} circle point rejection", rejection_frame)

            # Diagnostic 4: Final circle using only retained LAB points.
            final_fit_frame = frame.copy()

            for point_u, point_v in inlier_points:
                cv2.circle(final_fit_frame, (int(round(point_u)), int(round(point_v))), 1, (255, 0, 255), -1)

            cv2.circle(final_fit_frame, (int(round(circle_u)), int(round(circle_v))),
                       int(round(radius)), (0, 255, 0), 1)

            radius_change = radius - initial_radius
            text = f"Final fit: r={radius:.2f}px | delta={radius_change:+.2f}px | inliers={len(inlier_points)}"
            cv2.putText(final_fit_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(final_fit_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            debug.addStage(f"Candidate {candidate_index} final circle fit", final_fit_frame)

        # Step 11: Require boundary evidence around most of the fitted circle.
        point_angles = np.arctan2(inlier_points[:, 1] - circle_v, inlier_points[:, 0] - circle_u)
        angle_bins = (((point_angles + np.pi)/(2.0*np.pi))*NUM_ANGLE_BINS).astype(np.int32) % NUM_ANGLE_BINS
        covered_angle_bins = len(np.unique(angle_bins))

        if covered_angle_bins < MIN_COVERED_ANGLE_BINS:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.circle(failure_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 0, 255), 1)
                cv2.putText(failure_frame, f"REJECTED: angular coverage {covered_angle_bins}/{NUM_ANGLE_BINS}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - angular coverage", failure_frame)
            continue

        # Step 12: Keep the refined circle consistent with the rough HSV center.
        center_displacement = np.hypot(circle_u - center_u, circle_v - center_v)
        max_center_displacement = max(MIN_MAX_CENTER_SHIFT_PX, MAX_CENTER_SHIFT_FACTOR*radius)

        if center_displacement > max_center_displacement:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.circle(failure_frame, (int(round(center_u)), int(round(center_v))), 4, (0, 255, 255), -1)
                cv2.circle(failure_frame, (int(round(circle_u)), int(round(circle_v))), 4, (0, 0, 255), -1)
                cv2.line(failure_frame, (int(round(center_u)), int(round(center_v))),
                         (int(round(circle_u)), int(round(circle_v))), (0, 0, 255), 1)
                cv2.putText(failure_frame, f"REJECTED: center shift {center_displacement:.1f}px", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"Maximum: {max_center_displacement:.1f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - center shift", failure_frame)
            continue

        # Step 13: Score candidates that passed every geometric validation check.
        final_point_radii = np.hypot(inlier_points[:, 0] - circle_u, inlier_points[:, 1] - circle_v)
        mean_residual = np.mean(np.abs(final_point_radii - radius))
        coverage_score = covered_angle_bins/NUM_ANGLE_BINS
        support_score = len(inlier_points)/NUM_RAYS
        residual_score = 1.0/(1.0 + mean_residual/max(radius, 1.0))
        final_score = 0.40*coverage_score + 0.30*support_score + 0.20*residual_score

        if debug is not None:
            passed_frame = frame.copy()

            for point_u, point_v in inlier_points:
                cv2.circle(passed_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

            cv2.circle(passed_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 255, 0), 1)
            cv2.putText(passed_frame, f"PASSED candidate {candidate_index}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(passed_frame, f"score: {final_score:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(passed_frame, f"boundary={len(boundary_points)} | inliers={len(inlier_points)} | coverage={covered_angle_bins}/{NUM_ANGLE_BINS}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
            debug.addStage(f"Candidate {candidate_index} passed", passed_frame)

        if final_score > best_final_score:
            best_final_score = final_score
            best_result = candidate_index, float(circle_u), float(circle_v), float(radius), inlier_points, covered_angle_bins, float(final_score)

    if best_result is None:
        return None

    # Step 14: Build the Detection from the best validated sphere candidate.
    best_candidate_index, circle_u, circle_v, radius, inlier_points, covered_angle_bins, final_score = best_result
    diameter = 2.0*radius
    color_id = object_vision_spec.color_ids[0]
    shape = ShapeDetection(vertices_px=None, color_id=color_id, num_sides=0,
                           ellipse_px=((circle_u, circle_v), (diameter, diameter), 0.0))
    detection = Detection(u=circle_u, v=circle_v, px_w=diameter, px_h=diameter, shapes=[shape])

    if debug is not None:
        success_frame = frame.copy()

        for point_u, point_v in inlier_points:
            cv2.circle(success_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

        cv2.circle(success_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 255, 0), 1)
        cv2.putText(success_frame, f"BEST candidate {best_candidate_index}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(success_frame, f"inliers={len(inlier_points)} | coverage={covered_angle_bins}/{NUM_ANGLE_BINS}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        debug.addStage(f"Best candidate {best_candidate_index}", success_frame)

    return detection


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
def selectPolygonTopology(
    hull: np.ndarray, perimeter: float, epsilon_ratio: float, expected_num_sides: set[int],
) -> tuple[np.ndarray | None, int, list[float]]:
    """Infer side count from contour corners, constrained only by VisionSpec-allowed counts."""
    MAX_TOPOLOGY_EPSILON_RATIO = 0.015
    WEAK_CORNER_STRENGTH = 0.12

    allowed_counts = sorted(set(expected_num_sides))
    if not allowed_counts:
        return None, 0, []

    def cornerStrengths(vertices: np.ndarray) -> np.ndarray:
        strengths = np.zeros(len(vertices), dtype=np.float64)

        for i in range(len(vertices)):
            previous, vertex, following = vertices[i - 1], vertices[i], vertices[(i + 1)%len(vertices)]
            chord = following - previous
            chord_length = float(np.linalg.norm(chord))
            local_scale = max(
                float(np.linalg.norm(vertex - previous)),
                float(np.linalg.norm(following - vertex)),
            )

            if chord_length <= 1e-6 or local_scale <= 1e-6:
                continue

            deviation = abs(
                chord[0]*(vertex - previous)[1] - chord[1]*(vertex - previous)[0]
            )/chord_length
            strengths[i] = deviation/local_scale

        return strengths

    def pruneToSupportedTopology(polygon: np.ndarray) -> tuple[np.ndarray | None, list[float]]:
        vertices = polygon.reshape(-1, 2).astype(np.float64)
        removed_strengths: list[float] = []
        minimum_count = allowed_counts[0]

        while len(vertices) > minimum_count:
            strengths = cornerStrengths(vertices)
            weakest_index = int(np.argmin(strengths))
            weakest_strength = float(strengths[weakest_index])

            # Keep a supported count only when every remaining corner has real support.
            # If one corner is weak and the next-lower count is still compatible with
            # the VisionSpec, remove it and reconsider the topology.
            lower_supported_count_exists = any(
                count <= len(vertices) - 1 for count in allowed_counts
            )

            if weakest_strength >= WEAK_CORNER_STRENGTH or not lower_supported_count_exists:
                break

            vertices = np.delete(vertices, weakest_index, axis=0)
            removed_strengths.append(weakest_strength)

        if len(vertices) not in allowed_counts:
            return None, removed_strengths

        return vertices.astype(np.float32).reshape(-1, 1, 2), removed_strengths

    # Topology needs a fine corner proposal even if polygon_epsilon_ratio is temporarily
    # larger for experimentation. Geometry itself will still come from the original contour.
    proposal_ratio = min(epsilon_ratio, MAX_TOPOLOGY_EPSILON_RATIO)
    base_polygon = cv2.approxPolyDP(hull, proposal_ratio*perimeter, True)
    observed_num_sides = len(base_polygon)
    polygon, removed_strengths = pruneToSupportedTopology(base_polygon)

    if polygon is not None:
        return polygon, observed_num_sides, removed_strengths

    # Only needed when the proposal was too coarse to reach the minimum allowed count.
    if observed_num_sides < allowed_counts[0]:
        retry_ratio = max(0.005, 0.5*proposal_ratio)
        retry_polygon = cv2.approxPolyDP(hull, retry_ratio*perimeter, True)
        polygon, retry_removed = pruneToSupportedTopology(retry_polygon)

        if polygon is not None:
            return polygon, observed_num_sides, removed_strengths + retry_removed

    return None, observed_num_sides, removed_strengths




def snapTopologyCornersToHull(hull: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Place the chosen N topology corners using image-space hull evidence."""
    FAR_BBOX_DIAG_PX, CLOSE_BBOX_DIAG_PX = 70.0, 120.0

    proposal = polygon.reshape(-1, 2).astype(np.float64)
    hull_points = hull.reshape(-1, 2).astype(np.float64)
    num_sides = len(proposal)

    if num_sides < 3 or len(hull_points) < num_sides:
        return proposal

    bbox_diagonal_px = float(np.linalg.norm(np.ptp(hull_points, axis=0)))
    if bbox_diagonal_px < FAR_BBOX_DIAG_PX:
        return proposal

    segment_lengths = np.linalg.norm(np.roll(hull_points, -1, axis=0) - hull_points, axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= 1e-6:
        return proposal

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def arcDistance(index_1: int, index_2: int) -> float:
        distance = abs(float(cumulative[index_2] - cumulative[index_1]))
        return min(distance, perimeter - distance)

    corner_window_px = float(np.clip(0.06*bbox_diagonal_px, 4.0, 12.0))
    strengths = np.zeros(len(hull_points), dtype=np.float64)

    for index in range(len(hull_points)):
        previous_index, distance = index, 0.0
        while distance < corner_window_px:
            next_index = (previous_index - 1) % len(hull_points)
            distance += float(np.linalg.norm(hull_points[next_index] - hull_points[previous_index]))
            previous_index = next_index
            if previous_index == index:
                break

        following_index, distance = index, 0.0
        while distance < corner_window_px:
            next_index = (following_index + 1) % len(hull_points)
            distance += float(np.linalg.norm(hull_points[next_index] - hull_points[following_index]))
            following_index = next_index
            if following_index == index:
                break

        previous, vertex, following = (
            hull_points[previous_index], hull_points[index], hull_points[following_index],
        )
        chord = following - previous
        chord_length = float(np.linalg.norm(chord))
        local_scale = 0.5*(
            float(np.linalg.norm(vertex - previous))
            + float(np.linalg.norm(following - vertex))
        )

        if chord_length <= 1e-6 or local_scale <= 1e-6:
            continue

        deviation = abs(
            chord[0]*(vertex - previous)[1] - chord[1]*(vertex - previous)[0]
        )/chord_length
        strengths[index] = deviation/local_scale

    if bbox_diagonal_px < CLOSE_BBOX_DIAG_PX:
        # Mid range: proposal remains the prior, but may snap to a stronger nearby
        # actual hull turn.
        snap_radius_px = float(np.clip(0.10*bbox_diagonal_px, 5.0, 12.0))
        snapped = []
        used_indices = set()

        for vertex in proposal:
            distances = np.linalg.norm(hull_points - vertex, axis=1)
            candidate_indices = np.flatnonzero(distances <= snap_radius_px)
            if len(candidate_indices) == 0:
                candidate_indices = np.array([int(np.argmin(distances))])

            candidates = sorted(
                candidate_indices,
                key=lambda index: (-strengths[index], distances[index]),
            )
            selected_index = next(
                (int(index) for index in candidates if int(index) not in used_indices),
                None,
            )
            if selected_index is None:
                return proposal

            used_indices.add(selected_index)
            snapped.append(hull_points[selected_index])

        return np.asarray(snapped, dtype=np.float64)

    # Close range: use the N strongest well-separated actual hull turns directly.
    minimum_corner_separation_px = float(np.clip(0.05*perimeter, 6.0, 20.0))
    selected_indices = []

    for index in np.argsort(-strengths):
        index = int(index)
        if all(
            arcDistance(index, selected_index) >= minimum_corner_separation_px
            for selected_index in selected_indices
        ):
            selected_indices.append(index)

        if len(selected_indices) == num_sides:
            break

    if len(selected_indices) != num_sides:
        return proposal

    selected_indices.sort()
    return hull_points[selected_indices]



def refineShapeVerticesUsingEdges(contour: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rough_vertices = polygon.reshape(-1, 2).astype(np.float64)
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    num_sides, num_contour_points = len(rough_vertices), len(contour_points)

    if num_sides < 3 or num_contour_points < num_sides:
        return rough_vertices

    # Topology vertices only identify where the contour should be split. Never assign
    # contour pixels by distance to the proposed straight edges: a rough proposal can
    # be visibly offset even when the underlying HSV contour is excellent.
    corner_indices = np.array([
        int(np.argmin(np.sum((contour_points - vertex)**2, axis=1)))
        for vertex in rough_vertices
    ])

    if len(set(corner_indices.tolist())) != num_sides:
        return rough_vertices

    forward_steps = np.array([
        (corner_indices[(i + 1)%num_sides] - corner_indices[i]) % num_contour_points
        for i in range(num_sides)
    ])
    reverse_steps = np.array([
        (corner_indices[i] - corner_indices[(i + 1)%num_sides]) % num_contour_points
        for i in range(num_sides)
    ])
    contour_direction = 1 if np.sum(forward_steps) <= np.sum(reverse_steps) else -1

    bbox_diagonal_px = float(np.linalg.norm(np.ptp(rough_vertices, axis=0)))
    short_edge_px = max(10.0, 0.18*bbox_diagonal_px)
    fitted_lines = []

    for edge_index in range(num_sides):
        start = rough_vertices[edge_index]
        end = rough_vertices[(edge_index + 1)%num_sides]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))

        if edge_length <= 1e-6:
            return rough_vertices

        start_index = int(corner_indices[edge_index])
        end_index = int(corner_indices[(edge_index + 1)%num_sides])
        indices = []
        index = start_index

        while True:
            indices.append(index)
            if index == end_index:
                break
            index = (index + contour_direction) % num_contour_points
            if len(indices) > num_contour_points:
                return rough_vertices

        edge_points = contour_points[np.asarray(indices)]
        trim = int(round(0.10*len(edge_points)))

        if trim > 0 and len(edge_points) - 2*trim >= 3:
            edge_points = edge_points[trim:-trim]

        if len(edge_points) < 3:
            return rough_vertices

        edge_dir = edge/edge_length

        if edge_length <= short_edge_px:
            # Short edges do not have enough lever arm for a stable free angle fit.
            # Keep the topology direction, but use this edge's OWN contiguous contour arc
            # to determine its normal position.
            normal = np.array([-edge_dir[1], edge_dir[0]])
            normal_offsets = (edge_points - start)@normal
            median_offset = float(np.median(normal_offsets))
            line_point = 0.5*(start + end) + median_offset*normal
            line_direction = edge_dir
            residuals = np.abs(normal_offsets - median_offset)
        else:
            vx, vy, x0, y0 = cv2.fitLine(
                edge_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01,
            ).reshape(4)
            line_point = np.array([x0, y0], dtype=np.float64)
            line_direction = np.array([vx, vy], dtype=np.float64)
            direction_norm = float(np.linalg.norm(line_direction))

            if direction_norm <= 1e-6:
                return rough_vertices

            line_direction /= direction_norm
            relative = edge_points - line_point
            residuals = np.abs(
                line_direction[0]*relative[:, 1] - line_direction[1]*relative[:, 0]
            )

        edge_fit_error = float(np.sqrt(np.mean(residuals**2)))
        maximum_edge_fit_error_px = min(3.0, max(1.5, 0.01*edge_length))

        if edge_fit_error > maximum_edge_fit_error_px:
            return rough_vertices

        fitted_lines.append((line_point, line_direction))

    refined_vertices = []

    for vertex_index in range(num_sides):
        point_1, direction_1 = fitted_lines[(vertex_index - 1)%num_sides]
        point_2, direction_2 = fitted_lines[vertex_index]
        cross = direction_1[0]*direction_2[1] - direction_1[1]*direction_2[0]

        if abs(cross) <= 1e-4:
            return rough_vertices

        difference = point_2 - point_1
        t = (difference[0]*direction_2[1] - difference[1]*direction_2[0])/cross
        refined_vertices.append(point_1 + t*direction_1)

    refined_vertices = np.asarray(refined_vertices, dtype=np.float64)
    maximum_edge_length = max(
        np.linalg.norm(rough_vertices[i] - rough_vertices[(i + 1)%num_sides])
        for i in range(num_sides)
    )

    if np.any(np.linalg.norm(refined_vertices - rough_vertices, axis=1) > 0.50*maximum_edge_length):
        return rough_vertices

    refined_polygon = refined_vertices.astype(np.float32).reshape(-1, 1, 2)
    rough_area = cv2.contourArea(rough_vertices.astype(np.float32))
    refined_area = cv2.contourArea(refined_polygon)

    if (
        rough_area <= 0.0
        or not cv2.isContourConvex(refined_polygon)
        or not 0.65 <= refined_area/rough_area <= 1.35
    ):
        return rough_vertices

    return refined_vertices



# Refine straight marker edges using the same idea as the tennis-ball LAB rays:
# HSV gives an approximate boundary; short edge-normal rays find the strongest
# target-color LAB transition, then straight lines are refit and intersected.
def refineShapeVerticesUsingLabRays(
    frame: np.ndarray, vertices_px: np.ndarray, color_spec, debug_frame: np.ndarray | None = None,
    draw_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    CLOSE_BBOX_DIAG_PX, MID_BBOX_DIAG_PX = 120.0, 70.0
    MAX_RAYS_PER_EDGE, EDGE_SAMPLE_MARGIN = 10, 0.10
    MAX_EDGE_ANGLE_CHANGE_DEG = 20.0
    ACUTE_VERTEX_MAX_ANGLE_DEG = 45.0
    SHORT_EDGE_FACTOR, MIN_SHORT_EDGE_PX = 0.18, 10.0

    rough = np.asarray(vertices_px, dtype=np.float64).reshape(-1, 2)
    if len(rough) < 3 or color_spec.lab_value is None:
        return rough

    bbox_diagonal_px = float(np.linalg.norm(np.ptp(rough, axis=0)))

    # Close markers already have enough contour geometry; LAB rays mostly add wobble there.
    if bbox_diagonal_px >= CLOSE_BBOX_DIAG_PX:
        return rough

    blend_alpha = (
        1.0 if bbox_diagonal_px < MID_BBOX_DIAG_PX
        else float((CLOSE_BBOX_DIAG_PX - bbox_diagonal_px)/(CLOSE_BBOX_DIAG_PX - MID_BBOX_DIAG_PX))
    )

    search_radius_px = int(np.clip(round(0.10*bbox_diagonal_px), 3, 6))
    pad = search_radius_px + 3
    x1, y1 = max(0, int(np.floor(rough[:, 0].min())) - pad), max(0, int(np.floor(rough[:, 1].min())) - pad)
    x2 = min(frame.shape[1], int(np.ceil(rough[:, 0].max())) + pad + 1)
    y2 = min(frame.shape[0], int(np.ceil(rough[:, 1].max())) + pad + 1)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return rough

    frame_roi = frame[y1:y2, x1:x2]
    if (
        frame_roi.size == 0 or frame_roi.ndim != 3 or frame_roi.shape[0] < 3
        or frame_roi.shape[1] < 3 or frame_roi.shape[2] != 3
    ):
        return rough

    frame_roi = np.ascontiguousarray(frame_roi)
    try:
        lab_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2LAB)
        lab_roi = cv2.GaussianBlur(lab_roi, (3, 3), 0)
    except cv2.error:
        return rough

    direction = color_spec.lab_value[1:3].astype(np.float32) - 128.0
    reference_chroma = float(np.linalg.norm(direction))
    if reference_chroma <= 1e-6:
        return rough
    direction /= reference_chroma

    a = lab_roi[:, :, 1].astype(np.float32) - 128.0
    b = lab_roi[:, :, 2].astype(np.float32) - 128.0
    response = a*float(direction[0]) + b*float(direction[1])
    minimum_edge_drop = max(1.5, 0.04*reference_chroma)
    minimum_inner_response = max(2.0, 0.12*reference_chroma)

    centroid = np.mean(rough, axis=0)
    offsets = np.arange(-search_radius_px, search_radius_px + 0.5, 0.5, dtype=np.float32)
    short_edge_px = max(MIN_SHORT_EDGE_PX, SHORT_EDGE_FACTOR*bbox_diagonal_px)
    fitted_lines = []

    for edge_index in range(len(rough)):
        start, end = rough[edge_index], rough[(edge_index + 1)%len(rough)]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 2.0:
            return rough

        rays_per_edge = int(np.clip(round(edge_length/4.0), 4, MAX_RAYS_PER_EDGE))
        minimum_good_rays = max(3, min(rays_per_edge, 5))

        edge_dir = edge/edge_length
        normal = np.array([-edge_dir[1], edge_dir[0]])
        midpoint = 0.5*(start + end)
        if np.dot(normal, midpoint - centroid) < 0.0:
            normal = -normal

        fractions = np.linspace(EDGE_SAMPLE_MARGIN, 1.0 - EDGE_SAMPLE_MARGIN, rays_per_edge)
        bases = start + fractions[:, None]*edge
        samples = bases[:, None, :] + offsets[None, :, None]*normal
        local = samples - np.array([x1, y1])

        ray_response = cv2.remap(
            response, local[:, :, 0].astype(np.float32), local[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        smoothed_response = ray_response.copy()
        if ray_response.shape[1] >= 3:
            smoothed_response[:, 1:-1] = (
                0.25*ray_response[:, :-2] + 0.50*ray_response[:, 1:-1] + 0.25*ray_response[:, 2:]
            )

        drops = smoothed_response[:, :-1] - smoothed_response[:, 1:]
        best_indices = np.argmax(drops, axis=1)
        rows = np.arange(rays_per_edge)
        best_drops = drops[rows, best_indices]
        inner_response = smoothed_response[rows, best_indices]
        good = (best_drops >= minimum_edge_drop) & (inner_response >= minimum_inner_response)
        if np.count_nonzero(good) < minimum_good_rays:
            return rough

        drop_offsets = 0.5*(offsets[:-1] + offsets[1:])
        boundary_offsets = np.empty(rays_per_edge, dtype=np.float64)
        for ray_index, best_index in enumerate(best_indices):
            lo, hi = max(0, best_index - 1), min(drops.shape[1], best_index + 2)
            weights = np.maximum(drops[ray_index, lo:hi], 0.0)
            boundary_offsets[ray_index] = (
                float(np.sum(weights*drop_offsets[lo:hi])/np.sum(weights))
                if np.sum(weights) > 1e-6 else float(drop_offsets[best_index])
            )

        points = bases + boundary_offsets[:, None]*normal

        if edge_length <= short_edge_px:
            median_offset = float(np.median(boundary_offsets[good]))
            line_point = midpoint + median_offset*normal
            fitted_lines.append((line_point, edge_dir))
        else:
            good_points = points[good]
            vx, vy, x0, y0 = cv2.fitLine(
                good_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01,
            ).reshape(4)
            line_dir = np.array([vx, vy], dtype=np.float64)
            line_dir /= max(float(np.linalg.norm(line_dir)), 1e-12)
            if abs(float(np.dot(line_dir, edge_dir))) < np.cos(np.deg2rad(MAX_EDGE_ANGLE_CHANGE_DEG)):
                return rough
            fitted_lines.append((np.array([x0, y0], dtype=np.float64), line_dir))

        if debug_frame is not None:
            for base, point, is_good in zip(bases, points, good):
                if not is_good:
                    continue
                cv2.line(debug_frame, tuple(np.round(base).astype(int)), tuple(np.round(point).astype(int)), (220, 220, 220), 1)
                cv2.circle(debug_frame, tuple(np.round(point).astype(int)), 2, (255, 0, 255), -1)

    refined = []
    for vertex_index in range(len(rough)):
        p1, d1 = fitted_lines[(vertex_index - 1)%len(rough)]
        p2, d2 = fitted_lines[vertex_index]
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) <= 1e-4:
            return rough
        delta = p2 - p1
        t = (delta[0]*d2[1] - delta[1]*d2[0])/cross
        refined.append(p1 + t*d1)

    refined = np.asarray(refined, dtype=np.float64)

    # Direct acute-tip search: use LAB only for how far the tip extends along its axis.
    TIP_SEARCH_INWARD_PX = 2.0
    tip_search_outward_px = float(np.clip(round(0.20*bbox_diagonal_px), 4, 10))
    tip_offsets = np.arange(-TIP_SEARCH_INWARD_PX, tip_search_outward_px + 0.5, 0.5, dtype=np.float32)

    for vertex_index in range(len(rough)):
        previous_vertex, rough_vertex = rough[(vertex_index - 1)%len(rough)], rough[vertex_index]
        next_vertex = rough[(vertex_index + 1)%len(rough)]
        side_1, side_2 = previous_vertex - rough_vertex, next_vertex - rough_vertex
        side_1_norm, side_2_norm = np.linalg.norm(side_1), np.linalg.norm(side_2)
        if side_1_norm <= 1e-6 or side_2_norm <= 1e-6:
            continue

        cosine_angle = np.clip(float(np.dot(side_1, side_2)/(side_1_norm*side_2_norm)), -1.0, 1.0)
        vertex_angle_deg = float(np.degrees(np.arccos(cosine_angle)))
        if vertex_angle_deg >= ACUTE_VERTEX_MAX_ANGLE_DEG:
            continue

        tip_direction = rough_vertex - centroid
        tip_direction_norm = float(np.linalg.norm(tip_direction))
        if tip_direction_norm <= 1e-6:
            continue
        tip_direction /= tip_direction_norm

        tip_samples = rough_vertex[None, :] + tip_offsets[:, None]*tip_direction[None, :]
        tip_local = tip_samples - np.array([x1, y1])
        tip_response = cv2.remap(
            response,
            tip_local[:, 0].astype(np.float32).reshape(1, -1),
            tip_local[:, 1].astype(np.float32).reshape(1, -1),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)

        if len(tip_response) < 4:
            continue

        smoothed_tip_response = tip_response.copy()
        smoothed_tip_response[1:-1] = (
            0.25*tip_response[:-2] + 0.50*tip_response[1:-1] + 0.25*tip_response[2:]
        )
        tip_drops = smoothed_tip_response[:-1] - smoothed_tip_response[1:]
        tip_drop_offsets = 0.5*(tip_offsets[:-1] + tip_offsets[1:])

        eligible = tip_drop_offsets >= -1.0
        if not np.any(eligible):
            continue

        eligible_indices = np.flatnonzero(eligible)
        best_tip_index = int(eligible_indices[np.argmax(tip_drops[eligible])])
        best_tip_drop = float(tip_drops[best_tip_index])
        best_tip_inner_response = float(smoothed_tip_response[best_tip_index])
        if best_tip_drop < minimum_edge_drop or best_tip_inner_response < minimum_inner_response:
            continue

        lo, hi = max(0, best_tip_index - 1), min(len(tip_drops), best_tip_index + 2)
        weights = np.maximum(tip_drops[lo:hi], 0.0)
        direct_tip_offset = (
            float(np.sum(weights*tip_drop_offsets[lo:hi])/np.sum(weights))
            if np.sum(weights) > 1e-6 else float(tip_drop_offsets[best_tip_index])
        )
        direct_tip = rough_vertex + direct_tip_offset*tip_direction

        perpendicular = np.array([-tip_direction[1], tip_direction[0]])
        line_intersection = refined[vertex_index]
        refined[vertex_index] = (
            centroid
            + np.dot(direct_tip - centroid, tip_direction)*tip_direction
            + np.dot(line_intersection - centroid, perpendicular)*perpendicular
        )

        if debug_frame is not None:
            cv2.line(
                debug_frame,
                tuple(np.round(rough_vertex - TIP_SEARCH_INWARD_PX*tip_direction).astype(int)),
                tuple(np.round(rough_vertex + tip_search_outward_px*tip_direction).astype(int)),
                (255, 255, 0), 1,
            )
            cv2.circle(debug_frame, tuple(np.round(direct_tip).astype(int)), 3, (0, 165, 255), -1)

    # Mid-range blending: let LAB help, but do not let it fully override already decent contour geometry.
    if blend_alpha < 1.0:
        refined = (1.0 - blend_alpha)*rough + blend_alpha*refined

    refined_polygon = refined.astype(np.float32).reshape(-1, 1, 2)
    rough_area = cv2.contourArea(rough.astype(np.float32))
    refined_area = cv2.contourArea(refined_polygon)
    max_edge = max(np.linalg.norm(rough[i] - rough[(i + 1)%len(rough)]) for i in range(len(rough)))
    if (
        rough_area <= 0.0 or not cv2.isContourConvex(refined_polygon)
        or not 0.60 <= refined_area/rough_area <= 1.50
        or np.any(np.linalg.norm(refined - rough, axis=1) > 0.50*max_edge)
    ):
        return rough

    if debug_frame is not None:
        cv2.polylines(debug_frame, [np.round(rough).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 1)
        cv2.polylines(debug_frame, [np.round(refined).astype(np.int32).reshape(-1, 1, 2)], True, draw_bgr, 2)
    return refined

# Shape path: detect HSV contours, infer polygon topology, then fit geometry from the original contours.
def findSingleObjectUsingBestShapeGroup(
    frame: np.ndarray, object_vision_spec: ObjectVisionSpec, debug: DetectionDebug | None = None,
    _timing_profile: dict[str, float] | None = None,
) -> Detection | None:
    shape_markers = object_vision_spec.shape_markers

    if not shape_markers:
        raise ValueError("object_vision_spec.shape_markers cannot be empty")

    # TODO: Add special-case circle candidate detection for ShapeMarkerSpec(num_sides=0).
    polygon_markers = [marker for marker in shape_markers if marker.num_sides != 0]

    if not polygon_markers:
        return None

    # Paper-plane acquisition mirrors the useful first stages of the tennis-ball detector:
    # LAB chroma response -> LAB hotspots -> candidate ROIs -> loose HSV seed. The existing
    # polygon/straight-edge refinement remains responsible for the actual marker geometry.
    LAB_ACQUISITION_SCALE = 0.4
    GLOBAL_BLUR_KERNEL = (5, 5)
    HOTSPOT_PERCENTILE = 98.5
    MIN_HOTSPOT_RESPONSE_FACTOR = 0.30
    MIN_LAB_DIRECTION_COSINE = 0.75
    MIN_HOTSPOT_AREA_PX_FULL_RES = 6
    HOTSPOT_PADDING_FACTOR = 0.75
    MIN_HOTSPOT_PADDING_PX = 10
    EXTRA_HOTSPOT_CANDIDATES = 1

    # Full-resolution HSV is evaluated only around LAB hotspots. Start generously, then
    # automatically expand and retry if a selected HSV component reaches an ROI edge.
    # This keeps the ROI optimization from clipping long/acute marker shapes.
    HSV_ROI_PADDING_FACTOR = 0.80
    MIN_HSV_ROI_PADDING_PX = 16
    HSV_ROI_EXPANSION_FACTOR = 0.50
    MAX_HSV_ROI_EXPANSIONS = 4
    LOOSE_HSV_LOWER_SUBTRACTION = np.array([0, 40, 15], dtype=np.int16)

    # Optional seam/shadow recovery analogous to the tennis-ball path. Off initially because
    # the paper-plane markers should normally be continuous colored regions on white paper.
    KEEP_FRAGMENTED_HSV_COMPONENTS = False
    MIN_SECONDARY_SEED_AREA_FACTOR = 0.15
    MAX_SECONDARY_SEED_DISTANCE_FACTOR = 1.25

    # Optional console profiling. DetectionDebug uses the same timers automatically by
    # running one timing-only pass with debug=None, so debug image construction does not
    # contaminate the production-path timings shown by test_detection_image.py.
    PRINT_SHAPE_DETECTION_TIMING = False

    if debug is not None and _timing_profile is None:
        debug.reset(frame)

        # Remove first-call OpenCV/NumPy initialization from the static-image timing result.
        # This warm-up only happens in the debug/test path, never in the live detector.
        findSingleObjectUsingBestShapeGroup(frame, object_vision_spec, debug=None)

        timing_profile: dict[str, float] = {}
        findSingleObjectUsingBestShapeGroup(frame, object_vision_spec, debug=None, _timing_profile=timing_profile)
        for timing_name, elapsed_s in timing_profile.items():
            debug.setTiming(timing_name, elapsed_s)

    profile_shape_detection = PRINT_SHAPE_DETECTION_TIMING or _timing_profile is not None
    timing_start = time.perf_counter() if profile_shape_detection else None
    timing_lab_seconds = timing_hsv_polygon_seconds = None
    timing_model_setup_s = timing_resize_blur_s = timing_lab_prep_s = None
    timing_hsv_conversion_s = timing_frame_setup_s = None
    timing_hsv_threshold_s = timing_hsv_cleanup_s = timing_hsv_components_s = 0.0
    timing_hsv_association_s = timing_contour_s = timing_polygon_refine_s = 0.0

    model_setup_start = time.perf_counter() if profile_shape_detection else None

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

    # Cap LAB candidates from the model rather than using one fixed global number. Because the
    # normal view contains one plane or one connected pair, allow the maximum expected number of
    # same-color markers in either case plus a small amount of clutter headroom.
    planes_by_id = {plane.plane_id: plane for plane in object_vision_spec.rigid_planes}
    color_counts_by_plane = {
        plane.plane_id: Counter(
            marker.color_id for marker in plane.shape_markers if marker.num_sides != 0
        )
        for plane in object_vision_spec.rigid_planes
    }
    max_hotspot_candidates_by_color = {}

    for color_id in unique_color_ids:
        max_expected = max((counts[color_id] for counts in color_counts_by_plane.values()), default=0)

        for plane_id_1, plane_id_2, _ in object_vision_spec.rigid_plane_connections:
            if plane_id_1 in planes_by_id and plane_id_2 in planes_by_id:
                pair_count = color_counts_by_plane[plane_id_1][color_id] + color_counts_by_plane[plane_id_2][color_id]
                max_expected = max(max_expected, pair_count)

        max_hotspot_candidates_by_color[color_id] = max(1, max_expected) + EXTRA_HOTSPOT_CANDIDATES

    if profile_shape_detection:
        timing_model_setup_s = time.perf_counter() - model_setup_start
        resize_blur_start = time.perf_counter()

    # LAB is only used for rough candidate acquisition, so do it at half resolution.
    # HSV/polygon geometry stays full-resolution.
    acquisition_frame = cv2.resize(
        frame, None, fx=LAB_ACQUISITION_SCALE, fy=LAB_ACQUISITION_SCALE, interpolation=cv2.INTER_AREA,
    )
    blurred_acquisition_frame = cv2.GaussianBlur(acquisition_frame, GLOBAL_BLUR_KERNEL, 0)

    if profile_shape_detection:
        timing_resize_blur_s = time.perf_counter() - resize_blur_start
        lab_prep_start = time.perf_counter()

    lab_frame = cv2.cvtColor(blurred_acquisition_frame, cv2.COLOR_BGR2LAB)
    _, a_u8, b_u8 = cv2.split(lab_frame)
    a = a_u8.astype(np.float32) - 128.0
    b = b_u8.astype(np.float32) - 128.0
    pixel_chroma_strength = np.sqrt(a*a + b*b)

    if profile_shape_detection:
        timing_lab_prep_s = time.perf_counter() - lab_prep_start
        hsv_conversion_start = time.perf_counter()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if profile_shape_detection:
        timing_hsv_conversion_s = time.perf_counter() - hsv_conversion_start
        frame_setup_start = time.perf_counter()

    acquisition_to_full_x = frame.shape[1]/acquisition_frame.shape[1]
    acquisition_to_full_y = frame.shape[0]/acquisition_frame.shape[0]
    min_hotspot_area_px = max(1, int(np.ceil(
        MIN_HOTSPOT_AREA_PX_FULL_RES*LAB_ACQUISITION_SCALE*LAB_ACQUISITION_SCALE
    )))

    shape_candidates: list[ShapeDetection] = []
    combined_raw_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8) if debug is not None else None
    combined_cleaned_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8) if debug is not None else None
    hsv_cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 3x3 at half resolution is approximately the old 5x5 full-resolution cleanup footprint.
    hotspot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if profile_shape_detection:
        timing_frame_setup_s = time.perf_counter() - frame_setup_start
        timing_lab_seconds = 0.0
        timing_hsv_polygon_seconds = 0.0

    if debug is not None:
        debug.addStage("Original image", frame)
        contour_debug_frame, polygon_debug_frame = frame.copy(), frame.copy()
        snapped_corner_debug_frame, lab_ray_debug_frame = frame.copy(), frame.copy()
        candidate_debug_frame = frame.copy()
    else:
        contour_debug_frame = polygon_debug_frame = snapped_corner_debug_frame = None
        lab_ray_debug_frame = candidate_debug_frame = None

    for color_id in unique_color_ids:
        color_spec = COLOR_SPECS[color_id]
        color_name = color_id.name
        draw_bgr = color_spec.draw_bgr

        if color_spec.lab_value is None:
            raise ValueError(f"Paper-plane shape detection requires ColorSpec.lab_value for {color_id}")

        lab_stage_start = time.perf_counter() if profile_shape_detection else None

        # Step 1: Continuous LAB chroma response for this marker color.
        lab_direction = color_spec.lab_value[1:3].astype(np.float32) - 128.0
        reference_chroma_strength = float(np.linalg.norm(lab_direction))

        if reference_chroma_strength <= 0.0:
            raise ValueError(f"ColorSpec.lab_value for {color_id} does not define a valid chroma direction")

        lab_direction /= reference_chroma_strength
        lab_color_response = a*float(lab_direction[0]) + b*float(lab_direction[1])

        # Unlike the single-color tennis-ball case, multiple marker colors can have positive
        # projections onto one another's LAB directions. For positive response,
        # response/chroma >= cosine_threshold is exactly equivalent to
        # response >= cosine_threshold*chroma, so avoid a full-image floating-point division.
        direction_matches = (
            (lab_color_response > 0.0)
            & (lab_color_response >= MIN_LAB_DIRECTION_COSINE*pixel_chroma_strength)
        )
        aligned_response = np.where(direction_matches, lab_color_response, 0.0)

        # Step 2: Find strong target-color LAB hotspots using both an absolute chroma floor and
        # a high within-frame percentile, then lightly clean the hotspot mask.
        minimum_hotspot_response = MIN_HOTSPOT_RESPONSE_FACTOR*reference_chroma_strength
        percentile_hotspot_response = float(np.percentile(aligned_response, HOTSPOT_PERCENTILE))
        hotspot_threshold = max(minimum_hotspot_response, percentile_hotspot_response)
        hotspot_mask = (aligned_response >= hotspot_threshold).astype(np.uint8)*255
        hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, hotspot_kernel)
        hotspot_mask = cv2.erode(hotspot_mask, hotspot_kernel, iterations=1)
        hotspot_mask = cv2.dilate(hotspot_mask, hotspot_kernel, iterations=1)

        if debug is not None:
            response_debug_small = np.clip(4.0*aligned_response, 0, 255).astype(np.uint8)
            response_debug = cv2.resize(
                response_debug_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR,
            )
            hotspot_debug_mask = cv2.resize(
                hotspot_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST,
            )
            debug.addStage(f"LAB color response - {color_name}", response_debug)
            debug.addStage(f"LAB hotspot mask - {color_name}", hotspot_debug_mask)

        # Step 3: Rank spatially distinct hotspot components and keep only as many as the model
        # says could plausibly exist for this color, plus two clutter candidates.
        num_hotspot_labels, hotspot_labels, hotspot_stats, _ = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)
        response_sums = np.bincount(
            hotspot_labels.ravel(), weights=aligned_response.ravel(), minlength=num_hotspot_labels,
        )
        hotspot_candidates = []

        for hotspot_label in range(1, num_hotspot_labels):
            area = int(hotspot_stats[hotspot_label, cv2.CC_STAT_AREA])
            if area < min_hotspot_area_px:
                continue

            x = int(hotspot_stats[hotspot_label, cv2.CC_STAT_LEFT])
            y = int(hotspot_stats[hotspot_label, cv2.CC_STAT_TOP])
            w = int(hotspot_stats[hotspot_label, cv2.CC_STAT_WIDTH])
            h = int(hotspot_stats[hotspot_label, cv2.CC_STAT_HEIGHT])
            mean_response = float(response_sums[hotspot_label]/max(area, 1))
            score = mean_response*(area**0.10)
            hotspot_candidates.append((score, area, x, y, w, h, mean_response, hotspot_label))

        hotspot_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        selected_hotspots = []

        for candidate in hotspot_candidates:
            _, _, x, y, w, h, _, _ = candidate
            center_x, center_y = x + 0.5*w, y + 0.5*h
            duplicate = False

            for selected in selected_hotspots:
                _, _, sx, sy, sw, sh, _, _ = selected
                selected_center_x, selected_center_y = sx + 0.5*sw, sy + 0.5*sh
                duplicate_distance = 0.75*max(w, h, sw, sh)

                if (center_x - selected_center_x)**2 + (center_y - selected_center_y)**2 < duplicate_distance**2:
                    duplicate = True
                    break

            if not duplicate:
                selected_hotspots.append(candidate)
            if len(selected_hotspots) >= max_hotspot_candidates_by_color[color_id]:
                break

        # Convert the selected half-resolution hotspot boxes to full-resolution coordinates once.
        selected_hotspots_full = []
        for score, area, x, y, w, h, mean_response, hotspot_label in selected_hotspots:
            full_x1 = int(np.floor(x*acquisition_to_full_x))
            full_y1 = int(np.floor(y*acquisition_to_full_y))
            full_x2 = int(np.ceil((x + w)*acquisition_to_full_x))
            full_y2 = int(np.ceil((y + h)*acquisition_to_full_y))
            full_x1, full_y1 = max(0, full_x1), max(0, full_y1)
            full_x2, full_y2 = min(frame.shape[1], full_x2), min(frame.shape[0], full_y2)
            selected_hotspots_full.append((
                score, area, full_x1, full_y1, max(1, full_x2 - full_x1), max(1, full_y2 - full_y1),
                mean_response, hotspot_label, x, y, w, h,
            ))

        if profile_shape_detection:
            timing_lab_seconds += time.perf_counter() - lab_stage_start
            hsv_polygon_stage_start = time.perf_counter()

        if debug is not None:
            hotspot_debug = frame.copy()
            for candidate_index, (score, area, x, y, w, h, mean_response, *_rest) in enumerate(selected_hotspots_full, start=1):
                cv2.rectangle(hotspot_debug, (x, y), (x + w, y + h), draw_bgr, 1)
                cv2.putText(
                    hotspot_debug,
                    f"{candidate_index}: area={area} resp={mean_response:.1f} score={score:.1f}",
                    (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, draw_bgr, 1, cv2.LINE_AA,
                )
            debug.addStage(
                f"LAB hotspot candidates - {color_name} ({len(selected_hotspots_full)}/{max_hotspot_candidates_by_color[color_id]} cap)",
                hotspot_debug,
            )

        if not selected_hotspots_full:
            continue

        # Step 4: Evaluate loose HSV only inside generous full-resolution ROIs around the
        # selected LAB hotspots. Overlapping ROIs are merged so nearby/same-color markers are
        # segmented together. If a selected HSV component touches an expandable ROI edge, enlarge
        # that ROI and retry before extracting any contour, preventing ROI-induced shape clipping.
        initial_hsv_rois = []

        for hotspot_index, hotspot in enumerate(selected_hotspots_full):
            _, _, hot_x, hot_y, hot_w, hot_h, *_ = hotspot
            hotspot_size = max(hot_w, hot_h)
            padding = max(MIN_HSV_ROI_PADDING_PX, int(HSV_ROI_PADDING_FACTOR*hotspot_size))
            x1 = max(0, hot_x - padding)
            y1 = max(0, hot_y - padding)
            x2 = min(frame.shape[1], hot_x + hot_w + padding)
            y2 = min(frame.shape[0], hot_y + hot_h + padding)

            if x2 > x1 and y2 > y1:
                initial_hsv_rois.append([x1, y1, x2, y2, [hotspot_index]])

        # Merge overlapping padded ROIs transitively. Multiple HSV connected components inside a
        # merged ROI remain separate, so nearby same-color markers are still independently usable.
        merged_hsv_rois = []

        for x1, y1, x2, y2, hotspot_indices in initial_hsv_rois:
            merged = True

            while merged:
                merged = False

                for merged_index, (mx1, my1, mx2, my2, merged_hotspot_indices) in enumerate(merged_hsv_rois):
                    overlaps = x1 < mx2 and x2 > mx1 and y1 < my2 and y2 > my1

                    if not overlaps:
                        continue

                    x1, y1 = min(x1, mx1), min(y1, my1)
                    x2, y2 = max(x2, mx2), max(y2, my2)
                    hotspot_indices = list(dict.fromkeys(hotspot_indices + merged_hotspot_indices))
                    merged_hsv_rois.pop(merged_index)
                    merged = True
                    break

            merged_hsv_rois.append([x1, y1, x2, y2, hotspot_indices])

        roi_results = []
        color_raw_debug_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8) if debug is not None else None
        color_cleaned_debug_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8) if debug is not None else None

        for initial_x1, initial_y1, initial_x2, initial_y2, hotspot_indices in merged_hsv_rois:
            x1, y1, x2, y2 = initial_x1, initial_y1, initial_x2, initial_y2
            final_result = None

            for expansion_index in range(MAX_HSV_ROI_EXPANSIONS + 1):
                hsv_roi = hsv_frame[y1:y2, x1:x2]

                if hsv_roi.size == 0:
                    break

                threshold_start = time.perf_counter() if profile_shape_detection else None
                raw_loose_hsv_roi = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)

                for lower_hsv, upper_hsv in color_spec.hsv_ranges:
                    loose_lower_hsv = np.clip(
                        lower_hsv.astype(np.int16) - LOOSE_HSV_LOWER_SUBTRACTION, 0, 255,
                    ).astype(np.uint8)
                    raw_loose_hsv_roi = cv2.bitwise_or(
                        raw_loose_hsv_roi, cv2.inRange(hsv_roi, loose_lower_hsv, upper_hsv),
                    )

                if profile_shape_detection:
                    timing_hsv_threshold_s += time.perf_counter() - threshold_start
                    cleanup_start = time.perf_counter()

                # Keep all local HSV cleanup kernels at 3x3 so tiny distant polygon
                # features are not blurred/eroded away.
                cleaned_loose_hsv_roi = cv2.medianBlur(raw_loose_hsv_roi, 3)
                cleaned_loose_hsv_roi = cv2.morphologyEx(
                    cleaned_loose_hsv_roi, cv2.MORPH_OPEN, hsv_cleanup_kernel,
                )
                cleaned_loose_hsv_roi = cv2.morphologyEx(
                    cleaned_loose_hsv_roi, cv2.MORPH_CLOSE, hsv_cleanup_kernel,
                )

                if profile_shape_detection:
                    timing_hsv_cleanup_s += time.perf_counter() - cleanup_start
                    components_start = time.perf_counter()

                num_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
                    cleaned_loose_hsv_roi, connectivity=8,
                )

                if profile_shape_detection:
                    timing_hsv_components_s += time.perf_counter() - components_start
                    association_start = time.perf_counter()

                selected_seeds = []
                used_primary_seed_labels = set()
                selected_component_touches_expandable_edge = False

                for hotspot_index in hotspot_indices:
                    hotspot = selected_hotspots_full[hotspot_index]
                    _, _, hot_x, hot_y, hot_w, hot_h, _, hotspot_label, *_ = hotspot

                    # Upscale only this hotspot-label crop into the current HSV ROI coordinates.
                    low_x1 = max(0, int(np.floor(x1/acquisition_to_full_x)))
                    low_y1 = max(0, int(np.floor(y1/acquisition_to_full_y)))
                    low_x2 = min(hotspot_labels.shape[1], int(np.ceil(x2/acquisition_to_full_x)))
                    low_y2 = min(hotspot_labels.shape[0], int(np.ceil(y2/acquisition_to_full_y)))
                    low_component_roi = (
                        hotspot_labels[low_y1:low_y2, low_x1:low_x2] == hotspot_label
                    ).astype(np.uint8)

                    if low_component_roi.size == 0:
                        continue

                    hotspot_component_roi = cv2.resize(
                        low_component_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    best_seed = None

                    for seed_label in np.unique(seed_labels[hotspot_component_roi]):
                        seed_label = int(seed_label)

                        if seed_label == 0 or seed_label in used_primary_seed_labels:
                            continue

                        seed_area = int(seed_stats[seed_label, cv2.CC_STAT_AREA])

                        if seed_area < minimum_shape_area_by_color[color_id]:
                            continue

                        overlap = int(np.count_nonzero((seed_labels == seed_label) & hotspot_component_roi))

                        if overlap <= 0:
                            continue

                        if best_seed is None or (overlap, seed_area) > (best_seed[0], best_seed[1]):
                            best_seed = (overlap, seed_area, seed_label)

                    if best_seed is None:
                        continue

                    _, seed_area, primary_seed_label = best_seed
                    used_primary_seed_labels.add(primary_seed_label)
                    selected_seeds.append((hotspot_index, seed_area, primary_seed_label))

                    sx = int(seed_stats[primary_seed_label, cv2.CC_STAT_LEFT])
                    sy = int(seed_stats[primary_seed_label, cv2.CC_STAT_TOP])
                    sw = int(seed_stats[primary_seed_label, cv2.CC_STAT_WIDTH])
                    sh = int(seed_stats[primary_seed_label, cv2.CC_STAT_HEIGHT])
                    touches_left = sx <= 0 and x1 > 0
                    touches_top = sy <= 0 and y1 > 0
                    touches_right = sx + sw >= x2 - x1 and x2 < frame.shape[1]
                    touches_bottom = sy + sh >= y2 - y1 and y2 < frame.shape[0]

                    if touches_left or touches_top or touches_right or touches_bottom:
                        selected_component_touches_expandable_edge = True

                if profile_shape_detection:
                    timing_hsv_association_s += time.perf_counter() - association_start

                final_result = (
                    x1, y1, x2, y2, raw_loose_hsv_roi, cleaned_loose_hsv_roi,
                    seed_labels, seed_stats, num_seed_labels, selected_seeds,
                )

                if not selected_component_touches_expandable_edge or expansion_index >= MAX_HSV_ROI_EXPANSIONS:
                    break

                roi_width, roi_height = x2 - x1, y2 - y1
                expand_x = max(MIN_HSV_ROI_PADDING_PX, int(HSV_ROI_EXPANSION_FACTOR*roi_width))
                expand_y = max(MIN_HSV_ROI_PADDING_PX, int(HSV_ROI_EXPANSION_FACTOR*roi_height))
                new_x1, new_y1 = max(0, x1 - expand_x), max(0, y1 - expand_y)
                new_x2 = min(frame.shape[1], x2 + expand_x)
                new_y2 = min(frame.shape[0], y2 + expand_y)

                if (new_x1, new_y1, new_x2, new_y2) == (x1, y1, x2, y2):
                    break

                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

            if final_result is None:
                continue

            roi_results.append(final_result)

            if debug is not None:
                x1, y1, x2, y2, raw_roi, cleaned_roi, *_ = final_result
                color_raw_debug_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    color_raw_debug_mask[y1:y2, x1:x2], raw_roi,
                )
                color_cleaned_debug_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    color_cleaned_debug_mask[y1:y2, x1:x2], cleaned_roi,
                )

        if debug is not None:
            combined_raw_mask = cv2.bitwise_or(combined_raw_mask, color_raw_debug_mask)
            combined_cleaned_mask = cv2.bitwise_or(combined_cleaned_mask, color_cleaned_debug_mask)
            debug.addStage(f"Loose HSV mask - {color_name}", color_raw_debug_mask)
            debug.addStage(f"Cleaned loose HSV mask - {color_name}", color_cleaned_debug_mask)

        # Step 5/6: cleaned HSV chooses the real component; complete overlapping RAW
        # HSV components supply geometry. Noise rejection and edge geometry stay separate.
        for (
            x1, y1, x2, y2, raw_loose_hsv_roi, cleaned_loose_hsv_roi,
            seed_labels, seed_stats, num_seed_labels, selected_seeds,
        ) in roi_results:
            for hotspot_index, seed_area, primary_seed_label in selected_seeds:
                candidate_index = hotspot_index + 1
                hotspot = selected_hotspots_full[hotspot_index]
                _, _, hot_x, hot_y, hot_w, hot_h, *_ = hotspot
                keep_labels = [primary_seed_label]

                # Optional/off: recover substantial nearby HSV components if lighting later splits
                # one physical marker. All coordinates here are local to the already-safe HSV ROI.
                if KEEP_FRAGMENTED_HSV_COMPONENTS:
                    sx = int(seed_stats[primary_seed_label, cv2.CC_STAT_LEFT])
                    sy = int(seed_stats[primary_seed_label, cv2.CC_STAT_TOP])
                    sw = int(seed_stats[primary_seed_label, cv2.CC_STAT_WIDTH])
                    sh = int(seed_stats[primary_seed_label, cv2.CC_STAT_HEIGHT])
                    seed_center_x, seed_center_y = sx + 0.5*sw, sy + 0.5*sh
                    min_secondary_area = MIN_SECONDARY_SEED_AREA_FACTOR*seed_area
                    max_secondary_distance_sq = (MAX_SECONDARY_SEED_DISTANCE_FACTOR*max(sw, sh))**2

                    for other_label in range(1, num_seed_labels):
                        if other_label == primary_seed_label:
                            continue

                        other_area = int(seed_stats[other_label, cv2.CC_STAT_AREA])

                        if other_area < min_secondary_area:
                            continue

                        ox = int(seed_stats[other_label, cv2.CC_STAT_LEFT])
                        oy = int(seed_stats[other_label, cv2.CC_STAT_TOP])
                        ow = int(seed_stats[other_label, cv2.CC_STAT_WIDTH])
                        oh = int(seed_stats[other_label, cv2.CC_STAT_HEIGHT])
                        dx = ox + 0.5*ow - seed_center_x
                        dy = oy + 0.5*oh - seed_center_y

                        if dx*dx + dy*dy <= max_secondary_distance_sq:
                            keep_labels.append(other_label)

                contour_start = time.perf_counter() if profile_shape_detection else None

                # CLEANED mask decides which component is real; RAW mask supplies geometry.
                # Recover the COMPLETE raw HSV component(s) that overlap the selected cleaned
                # component. This restores sharp tips removed by cleanup without using dilation,
                # so unrelated nearby raw pixels cannot be pulled into the polygon merely because
                # they happen to lie within a support radius.
                selected_clean_component_roi = np.isin(seed_labels, keep_labels).astype(np.uint8)*255
                num_raw_labels, raw_labels = cv2.connectedComponents(raw_loose_hsv_roi, connectivity=8)
                overlapping_raw_labels = np.unique(raw_labels[selected_clean_component_roi != 0])
                overlapping_raw_labels = overlapping_raw_labels[overlapping_raw_labels != 0]

                if len(overlapping_raw_labels) == 0:
                    continue

                overlapping_raw_component_roi = np.isin(raw_labels, overlapping_raw_labels).astype(np.uint8)*255

                # A raw component can contain a thin noisy tendril connected by only a
                # pixel or two. Keep the sharp raw boundary, but only within a small,
                # scale-aware recovery distance from the trusted CLEANED component.
                clean_points = cv2.findNonZero(selected_clean_component_roi)
                if clean_points is None:
                    continue

                _, _, clean_w, clean_h = cv2.boundingRect(clean_points)
                max_raw_recovery_px = int(np.clip(round(0.06*np.hypot(clean_w, clean_h)), 2, 4))
                distance_from_clean = cv2.distanceTransform(
                    cv2.bitwise_not(selected_clean_component_roi), cv2.DIST_L2, 3,
                )
                raw_geometry_roi = cv2.bitwise_and(
                    overlapping_raw_component_roi,
                    (distance_from_clean <= max_raw_recovery_px).astype(np.uint8)*255,
                )
                geometry_points = cv2.findNonZero(raw_geometry_roi)
                if geometry_points is None:
                    continue

                geometry_x, geometry_y, geometry_w, geometry_h = cv2.boundingRect(geometry_points)
                geometry_crop = raw_geometry_roi[
                    geometry_y:geometry_y + geometry_h,
                    geometry_x:geometry_x + geometry_w,
                ]

                contours, _ = cv2.findContours(
                    geometry_crop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    offset=(x1 + geometry_x, y1 + geometry_y),
                )
                if profile_shape_detection:
                    timing_contour_s += time.perf_counter() - contour_start

                if debug is not None:
                    roi_debug = frame.copy()
                    cv2.rectangle(roi_debug, (x1, y1), (x2 - 1, y2 - 1), draw_bgr, 1)
                    cv2.rectangle(
                        roi_debug, (hot_x, hot_y), (hot_x + hot_w, hot_y + hot_h), (255, 255, 255), 1,
                    )
                    debug.addStage(f"{color_name} candidate {candidate_index} ROI", roi_debug)
                    debug.addStage(
                        f"{color_name} candidate {candidate_index} loose HSV seed mask",
                        cleaned_loose_hsv_roi,
                    )

                    selected_seed_debug = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                    selected_seed_debug[y1:y2, x1:x2] = selected_clean_component_roi
                    debug.addStage(
                        f"{color_name} candidate {candidate_index} selected CLEANED component",
                        selected_seed_debug,
                    )

                    raw_geometry_debug = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                    raw_geometry_debug[y1:y2, x1:x2] = raw_geometry_roi
                    debug.addStage(
                        f"{color_name} candidate {candidate_index} RAW geometry mask",
                        raw_geometry_debug,
                    )

                if not contours:
                    continue

                # Cleanup may have joined several legitimate raw fragments of the same marker.
                # All retained raw fragments overlap the selected cleaned component, so combine
                # them before the existing convex-polygon path.
                contour = contours[0] if len(contours) == 1 else cv2.convexHull(np.concatenate(contours, axis=0))
                contour_area = cv2.contourArea(contour)

                if contour_debug_frame is not None:
                    cv2.drawContours(contour_debug_frame, [contour], -1, draw_bgr, 1)

                if contour_area < minimum_shape_area_by_color[color_id]:
                    continue

                polygon_start = time.perf_counter() if profile_shape_detection else None
                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)

                if perimeter <= 0:
                    if profile_shape_detection:
                        timing_polygon_refine_s += time.perf_counter() - polygon_start
                    continue

                epsilon_ratio = object_vision_spec.polygon_epsilon_ratio
                expected_num_sides = expected_num_sides_by_color[color_id]
                polygon, observed_num_sides, removed_corner_strengths = selectPolygonTopology(
                    hull, perimeter, epsilon_ratio, expected_num_sides,
                )

                if polygon_debug_frame is not None:
                    polygon_center = np.mean(hull.reshape(-1, 2), axis=0).astype(np.int32)
                    if polygon is None:
                        debug_text = f"{color_name}: observed={observed_num_sides}, expected={expected_num_sides}, REJECT"
                    else:
                        for vertex_u, vertex_v in np.round(polygon.reshape(-1, 2)).astype(np.int32):
                            cv2.circle(
                                polygon_debug_frame, (int(vertex_u), int(vertex_v)),
                                4, draw_bgr, -1,
                            )
                        pruned_text = (
                            f", pruned={len(removed_corner_strengths)}"
                            if removed_corner_strengths else ""
                        )
                        debug_text = (
                            f"{color_name}: observed={observed_num_sides} -> N={len(polygon)}"
                            f"{pruned_text}, allowed={expected_num_sides}"
                        )
                    cv2.putText(
                        polygon_debug_frame, debug_text, tuple(polygon_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, draw_bgr, 1, cv2.LINE_AA,
                    )

                if polygon is not None and cv2.isContourConvex(polygon):
                    num_sides = len(polygon)
                    geometry_corners = snapTopologyCornersToHull(hull, polygon)

                    if snapped_corner_debug_frame is not None:
                        for proposal_vertex, snapped_vertex in zip(polygon.reshape(-1, 2), geometry_corners):
                            proposal_point = tuple(np.round(proposal_vertex).astype(int))
                            snapped_point = tuple(np.round(snapped_vertex).astype(int))
                            cv2.line(snapped_corner_debug_frame, proposal_point, snapped_point, (220, 220, 220), 1)
                            cv2.circle(snapped_corner_debug_frame, proposal_point, 3, draw_bgr, -1)
                            cv2.circle(snapped_corner_debug_frame, snapped_point, 6, draw_bgr, 2)

                    # VisionSpec/topology chooses N. Pixel-scale hull evidence chooses where
                    # those N corners sit. Final edge geometry comes from the ORIGINAL HSV
                    # contour arcs between those corners.
                    geometry_polygon = geometry_corners.astype(np.float32).reshape(-1, 1, 2)
                    vertices_px = refineShapeVerticesUsingEdges(contour, geometry_polygon)
                    vertices_px = refineShapeVerticesUsingLabRays(
                        frame, vertices_px, color_spec, lab_ray_debug_frame, draw_bgr,
                    )
                    shape_candidates.append(
                        ShapeDetection(vertices_px=vertices_px, color_id=color_id, num_sides=num_sides),
                    )

                    if candidate_debug_frame is not None:
                        shape_index = len(shape_candidates) - 1
                        center_px = np.mean(vertices_px, axis=0).astype(np.int32)
                        shape_points = np.round(vertices_px).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(candidate_debug_frame, [shape_points], True, draw_bgr, 1)
                        cv2.circle(candidate_debug_frame, tuple(center_px), 4, draw_bgr, -1)
                        cv2.putText(
                            candidate_debug_frame, f"S{shape_index}: {color_name}, N={num_sides}",
                            (int(center_px[0]) + 5, int(center_px[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_bgr, 1, cv2.LINE_AA,
                        )

                    if debug is not None:
                        topology_debug = frame.copy()
                        for vertex_u, vertex_v in np.round(polygon.reshape(-1, 2)).astype(np.int32):
                            cv2.circle(topology_debug, (int(vertex_u), int(vertex_v)), 5, draw_bgr, -1)
                        debug.addStage(
                            f"{color_name} candidate {candidate_index} topology corners (not geometry)",
                            topology_debug,
                        )

                        refined_debug = frame.copy()
                        refined_points = np.round(vertices_px).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(refined_debug, [refined_points], True, draw_bgr, 2)
                        for vertex_u, vertex_v in np.round(vertices_px).astype(np.int32):
                            cv2.circle(refined_debug, (int(vertex_u), int(vertex_v)), 4, draw_bgr, -1)
                        debug.addStage(f"{color_name} candidate {candidate_index} refined polygon", refined_debug)

                if profile_shape_detection:
                    timing_polygon_refine_s += time.perf_counter() - polygon_start

        if profile_shape_detection:
            timing_hsv_polygon_seconds += time.perf_counter() - hsv_polygon_stage_start

    if debug is not None:
        debug.addStage("Combined raw mask", combined_raw_mask)
        debug.addStage("Combined cleaned mask", combined_cleaned_mask)
        debug.addStage("All selected HSV contours", contour_debug_frame)
        debug.addStage("Topology corner proposals (not geometry)", polygon_debug_frame)
        debug.addStage(
            "Scale-aware geometry corners (dot=proposal, ring=used corner)",
            snapped_corner_debug_frame,
        )
        debug.addStage(
            "LAB edge rays (magenta=edge, cyan=acute-tip ray, orange=direct tip)",
            lab_ray_debug_frame,
        )

        if not shape_candidates:
            cv2.putText(candidate_debug_frame, "No accepted shapes", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("All raw shape candidates (pre-plane selection)", candidate_debug_frame)

    if not shape_candidates:
        if profile_shape_detection:
            timing_total_s = time.perf_counter() - timing_start
            if _timing_profile is not None:
                _timing_profile.update({
                    "2D model bookkeeping": timing_model_setup_s,
                    "2D resize + blur": timing_resize_blur_s,
                    "2D LAB prep + chroma norm": timing_lab_prep_s,
                    "2D HSV conversion": timing_hsv_conversion_s,
                    "2D frame setup": timing_frame_setup_s,
                    "2D LAB acquisition": timing_lab_seconds,
                    "2D HSV threshold (ROI)": timing_hsv_threshold_s,
                    "2D HSV cleanup (ROI)": timing_hsv_cleanup_s,
                    "2D HSV components (ROI)": timing_hsv_components_s,
                    "2D HSV hotspot association": timing_hsv_association_s,
                    "2D component contours": timing_contour_s,
                    "2D polygon + refinement": timing_polygon_refine_s,
                    "2D HSV + polygons": timing_hsv_polygon_seconds,
                    "2D detection total": timing_total_s,
                })
            if PRINT_SHAPE_DETECTION_TIMING:
                print(
                    f"shape timing: model={timing_model_setup_s*1000.0:.1f} ms | resize+blur={timing_resize_blur_s*1000.0:.1f} ms | "
                    f"LABprep={timing_lab_prep_s*1000.0:.1f} ms | HSVconv={timing_hsv_conversion_s*1000.0:.1f} ms | "
                    f"LAB={timing_lab_seconds*1000.0:.1f} ms | HSV+polygon={timing_hsv_polygon_seconds*1000.0:.1f} ms | "
                    f"total={timing_total_s*1000.0:.1f} ms (no shapes)"
                )
        if debug is not None:
            debug.updateTimingStage()
        return None

    if debug is not None:
        debug.addStage("Plane / hinge selection diagnostics", makeTextDebugStage(frame, getPlaneSelectionDebugLines(shape_candidates, object_vision_spec)))
    grouping_start = time.perf_counter() if profile_shape_detection else None
    selected_group = selectBestPlaneShapeGroup(shape_candidates, object_vision_spec)

    if selected_group is None:
        if profile_shape_detection:
            timing_total_s = time.perf_counter() - timing_start
            timing_grouping_s = time.perf_counter() - grouping_start
            if _timing_profile is not None:
                _timing_profile.update({
                    "2D model bookkeeping": timing_model_setup_s,
                    "2D resize + blur": timing_resize_blur_s,
                    "2D LAB prep + chroma norm": timing_lab_prep_s,
                    "2D HSV conversion": timing_hsv_conversion_s,
                    "2D frame setup": timing_frame_setup_s,
                    "2D LAB acquisition": timing_lab_seconds,
                    "2D HSV threshold (ROI)": timing_hsv_threshold_s,
                    "2D HSV cleanup (ROI)": timing_hsv_cleanup_s,
                    "2D HSV components (ROI)": timing_hsv_components_s,
                    "2D HSV hotspot association": timing_hsv_association_s,
                    "2D component contours": timing_contour_s,
                    "2D polygon + refinement": timing_polygon_refine_s,
                    "2D HSV + polygons": timing_hsv_polygon_seconds,
                    "2D plane selection": timing_grouping_s,
                    "2D detection total": timing_total_s,
                })
        if debug is not None:
            debug.updateTimingStage()
        return None

    best_shape_group, selected_plane_ids, selected_marker_keys = selected_group
    all_best_vertices = np.concatenate([shape.vertices_px for shape in best_shape_group], axis=0)
    bbox_x, bbox_y, px_w, px_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
    detection = Detection(
        u=bbox_x + px_w/2.0, v=bbox_y + px_h/2.0, px_w=float(px_w), px_h=float(px_h),
        shapes=best_shape_group, plane_ids=selected_plane_ids, shape_marker_keys=selected_marker_keys,
    )

    if debug is not None:
        selected_debug_frame = frame.copy()
        for shape in best_shape_group:
            draw_bgr = COLOR_SPECS[shape.color_id].draw_bgr
            shape_points = np.round(shape.vertices_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(selected_debug_frame, [shape_points], True, draw_bgr, 2)
        cv2.rectangle(
            selected_debug_frame, (bbox_x, bbox_y), (bbox_x + px_w, bbox_y + px_h), (0, 255, 0), 2,
        )
        selection_mode = "HINGE" if len(selected_plane_ids) == 2 else "SINGLE-PLANE FALLBACK"
        cv2.putText(
            selected_debug_frame,
            f"{selection_mode}: planes={'+'.join(selected_plane_ids)} | markers={len(best_shape_group)}",
            (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
        debug.addStage("Selected best two planes / single-plane fallback", selected_debug_frame)

        final_debug_frame = frame.copy()
        drawDetection(final_debug_frame, detection)
        debug.addStage("Final object detection", final_debug_frame)

    if profile_shape_detection:
        timing_total_s = time.perf_counter() - timing_start
        timing_grouping_s = time.perf_counter() - grouping_start
        if _timing_profile is not None:
            _timing_profile.update({
                "2D model bookkeeping": timing_model_setup_s,
                "2D resize + blur": timing_resize_blur_s,
                "2D LAB prep + chroma norm": timing_lab_prep_s,
                "2D HSV conversion": timing_hsv_conversion_s,
                "2D frame setup": timing_frame_setup_s,
                "2D LAB acquisition": timing_lab_seconds,
                "2D HSV threshold (ROI)": timing_hsv_threshold_s,
                "2D HSV cleanup (ROI)": timing_hsv_cleanup_s,
                "2D HSV components (ROI)": timing_hsv_components_s,
                "2D HSV hotspot association": timing_hsv_association_s,
                "2D component contours": timing_contour_s,
                "2D polygon + refinement": timing_polygon_refine_s,
                "2D HSV + polygons": timing_hsv_polygon_seconds,
                "2D plane selection": timing_grouping_s,
                "2D detection total": timing_total_s,
            })
        if PRINT_SHAPE_DETECTION_TIMING:
            print(
                f"shape timing: LAB={timing_lab_seconds*1000.0:.1f} ms | "
                f"HSV+polygon={timing_hsv_polygon_seconds*1000.0:.1f} ms | "
                f"plane-select={timing_grouping_s*1000.0:.1f} ms | total={timing_total_s*1000.0:.1f} ms"
            )

    if debug is not None:
        detection._debug = debug
        debug.updateTimingStage()

    return detection


# Explain marker compatibility only for debug; the normal selector still uses getShapeMarkerError().
def describeShapeMarkerMatch(shape: ShapeDetection, marker: ShapeMarkerSpec, minimum_area_px: float | None) -> str:
    if shape.color_id != marker.color_id:
        return "color"
    if shape.num_sides != marker.num_sides:
        return f"N {shape.num_sides}!={marker.num_sides}"
    if shape.vertices_px is None or marker.object_vertices_m is None:
        return "missing vertices"
    vertices = np.asarray(shape.vertices_px, dtype=np.float64)
    if vertices.shape != (shape.num_sides, 2) or not np.all(np.isfinite(vertices)):
        return "bad image vertices"
    area = cv2.contourArea(vertices.astype(np.float32))
    if minimum_area_px is not None and area < minimum_area_px:
        return f"area {area:.0f}<{minimum_area_px:.0f}"
    error = getShapeMarkerError(shape, marker, minimum_area_px)
    return "geometry" if error is None else f"OK err={error:.3f}"


# Compare one detected polygon with one physical marker. None means incompatible.
def getShapeMarkerError(shape: ShapeDetection, marker: ShapeMarkerSpec, minimum_area_px: float | None) -> float | None:
    if shape.color_id != marker.color_id or shape.num_sides != marker.num_sides or shape.vertices_px is None or marker.object_vertices_m is None:
        return None
    image_vertices = np.asarray(shape.vertices_px, dtype=np.float64)
    if image_vertices.shape != (shape.num_sides, 2) or not np.all(np.isfinite(image_vertices)):
        return None
    if minimum_area_px is not None and cv2.contourArea(image_vertices.astype(np.float32)) < minimum_area_px:
        return None
    marker_vertices = np.asarray(marker.object_vertices_m, dtype=np.float64)
    if marker_vertices.shape != (marker.num_sides, 2) or not np.all(np.isfinite(marker_vertices)):
        return None
    image_edges = np.linalg.norm(image_vertices - np.roll(image_vertices, -1, axis=0), axis=1)
    marker_edges = np.linalg.norm(marker_vertices - np.roll(marker_vertices, -1, axis=0), axis=1)
    image_norm, marker_norm = np.linalg.norm(image_edges), np.linalg.norm(marker_edges)
    if image_norm <= 1e-12 or marker_norm <= 1e-12:
        return None
    image_edges, marker_edges = image_edges/image_norm, marker_edges/marker_norm
    reversed_edges = image_edges[::-1]
    return min(
        min(float(np.linalg.norm(np.roll(image_edges, shift) - marker_edges)) for shift in range(marker.num_sides)),
        min(float(np.linalg.norm(np.roll(reversed_edges, shift) - marker_edges)) for shift in range(marker.num_sides)),
    )


def findBestShapeMarkerAssignment(shape_candidates: list[ShapeDetection], marker_entries: list[tuple[str, int, ShapeMarkerSpec, float | None]],
                                  required_plane_ids: set[str] | None = None) -> tuple[list[tuple[int, int]], float] | None:
    pair_errors = {}
    for shape_index, shape in enumerate(shape_candidates):
        for entry_index, (_, _, marker, minimum_area_px) in enumerate(marker_entries):
            error = getShapeMarkerError(shape, marker, minimum_area_px)
            if error is not None:
                pair_errors[(shape_index, entry_index)] = error

    for match_count in range(min(len(shape_candidates), len(marker_entries)), 0, -1):
        best_assignment, best_error = None, float('inf')
        for shape_indices in combinations(range(len(shape_candidates)), match_count):
            for entry_indices in combinations(range(len(marker_entries)), match_count):
                for ordered_entry_indices in permutations(entry_indices):
                    assignment = list(zip(shape_indices, ordered_entry_indices))
                    if any(pair not in pair_errors for pair in assignment):
                        continue
                    represented_planes = {marker_entries[entry_index][0] for _, entry_index in assignment}
                    if required_plane_ids is not None and not required_plane_ids.issubset(represented_planes):
                        continue
                    error = sum(pair_errors[pair] for pair in assignment)
                    if error < best_error:
                        best_assignment, best_error = assignment, error
        if best_assignment is not None:
            return best_assignment, best_error
    return None


def evaluatePlaneGroups(shape_candidates: list[ShapeDetection], object_vision_spec: ObjectVisionSpec, plane_groups: list[tuple[str, ...]],
                        require_all_planes: bool) -> tuple[list[ShapeDetection], tuple[str, ...], list[tuple[str, int]]] | None:
    planes_by_id = {plane.plane_id: plane for plane in object_vision_spec.rigid_planes}
    previous_plane_group = _PNP_PREVIOUS_PLANE_GROUP.get(id(object_vision_spec))
    best_result, best_rank = None, None

    for plane_group in plane_groups:
        marker_entries = []
        for plane_id in plane_group:
            plane = planes_by_id.get(plane_id)
            if plane is None:
                continue
            for marker_index, marker in enumerate(plane.shape_markers):
                if marker.num_sides == 0:
                    continue
                minimum_area = marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
                marker_entries.append((plane_id, marker_index, marker, minimum_area))
        if not marker_entries:
            continue

        assignment_result = findBestShapeMarkerAssignment(shape_candidates, marker_entries, set(plane_group) if require_all_planes else None)
        if assignment_result is None:
            continue
        assignment, assignment_error = assignment_result
        matched_counts = Counter(marker_entries[entry_index][0] for _, entry_index in assignment)
        marker_counts = Counter(plane_id for plane_id, _, _, _ in marker_entries)
        matched_count = len(assignment)
        full_plane_count = sum(matched_counts[plane_id] == marker_counts[plane_id] for plane_id in matched_counts)
        coverage = matched_count/max(len(marker_entries), 1)
        previous_bonus = int(previous_plane_group is not None and frozenset(plane_group) == frozenset(previous_plane_group))
        rank = (matched_count, full_plane_count, coverage, previous_bonus, -assignment_error/matched_count)
        if best_rank is not None and rank <= best_rank:
            continue

        selected_shapes = [shape_candidates[shape_index] for shape_index, _ in assignment]
        selected_marker_keys = [(marker_entries[entry_index][0], marker_entries[entry_index][1]) for _, entry_index in assignment]
        best_result, best_rank = (selected_shapes, tuple(plane_group), selected_marker_keys), rank

    return best_result


def selectBestPlaneShapeGroup(shape_candidates: list[ShapeDetection], object_vision_spec: ObjectVisionSpec) -> tuple[list[ShapeDetection], tuple[str, ...], list[tuple[str, int]]] | None:
    plane_ids = [plane.plane_id for plane in object_vision_spec.rigid_planes]
    plane_id_set = set(plane_ids)
    hinge_groups = [(p1, p2) for p1, p2, _ in object_vision_spec.rigid_plane_connections if p1 in plane_id_set and p2 in plane_id_set]
    result = evaluatePlaneGroups(shape_candidates, object_vision_spec, hinge_groups, require_all_planes=True)
    if result is not None:
        return result
    return evaluatePlaneGroups(shape_candidates, object_vision_spec, [(plane_id,) for plane_id in plane_ids], require_all_planes=False)


def getPlaneSelectionDebugLines(shape_candidates: list[ShapeDetection], object_vision_spec: ObjectVisionSpec) -> list[str]:
    planes = {plane.plane_id: plane for plane in object_vision_spec.rigid_planes}
    lines = ["RAW SHAPES"]

    for shape_index, shape in enumerate(shape_candidates):
        area = cv2.contourArea(np.asarray(shape.vertices_px, dtype=np.float32)) if shape.vertices_px is not None else 0.0
        color_name = getattr(shape.color_id, "name", str(shape.color_id))
        lines.append(f"S{shape_index}: {color_name} N={shape.num_sides} area={area:.0f}")
        matches = []
        for plane_id, plane in planes.items():
            for marker_index, marker in enumerate(plane.shape_markers):
                if marker.num_sides == 0:
                    continue
                minimum_area = marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
                status = describeShapeMarkerMatch(shape, marker, minimum_area)
                if status != "color":
                    marker_color = getattr(marker.color_id, "name", str(marker.color_id))
                    matches.append(f"{plane_id}:M{marker_index} {marker_color}/N{marker.num_sides} {status}")
        lines.extend([f"  {match}" for match in matches] or ["  no same-color physical marker"] )

    lines.append("HINGES")
    for plane_id_1, plane_id_2, _ in object_vision_spec.rigid_plane_connections:
        if plane_id_1 not in planes or plane_id_2 not in planes:
            lines.append(f"{plane_id_1}<->{plane_id_2}: FAIL missing plane")
            continue
        entries = []
        for plane_id in (plane_id_1, plane_id_2):
            for marker_index, marker in enumerate(planes[plane_id].shape_markers):
                if marker.num_sides == 0:
                    continue
                minimum_area = marker.minimum_contour_area_px if marker.minimum_contour_area_px is not None else object_vision_spec.minimum_contour_area_px
                entries.append((plane_id, marker_index, marker, minimum_area))
        result = findBestShapeMarkerAssignment(shape_candidates, entries, {plane_id_1, plane_id_2})
        if result is None:
            compatible_planes = set()
            for shape in shape_candidates:
                for plane_id, _, marker, minimum_area in entries:
                    if getShapeMarkerError(shape, marker, minimum_area) is not None:
                        compatible_planes.add(plane_id)
            missing = [plane_id for plane_id in (plane_id_1, plane_id_2) if plane_id not in compatible_planes]
            reason = "missing evidence: " + ",".join(missing) if missing else "no valid one-to-one assignment"
            lines.append(f"{plane_id_1}<->{plane_id_2}: FAIL {reason}")
            continue
        assignment, error = result
        mapping = ", ".join(f"S{s}->{entries[m][0]}:M{entries[m][1]}" for s, m in assignment)
        lines.append(f"{plane_id_1}<->{plane_id_2}: PASS {mapping} | err={error:.3f}")

    selected = selectBestPlaneShapeGroup(shape_candidates, object_vision_spec)
    lines.append("SELECTED: none" if selected is None else f"SELECTED: {'+'.join(selected[1])}")
    return lines


def makeTextDebugStage(reference_frame: np.ndarray, lines: list[str]) -> np.ndarray:
    width = max(reference_frame.shape[1], 900)
    image = np.full((max(160, 34 + 22*len(lines)), width, 3), 25, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (12, 25 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
    return image


# Build the exact physical marker data selected in 2D. PnP never reselects planes, hinges, or marker identities.
def buildSelectedPnPMarkers(detection: Detection, object_vision_spec: ObjectVisionSpec) -> list[tuple] | None:
    if not detection.shapes or not detection.plane_ids or len(detection.plane_ids) > 2 or detection.shape_marker_keys is None:
        return None
    if len(detection.shapes) != len(detection.shape_marker_keys):
        return None
    planes_by_id = {plane.plane_id: plane for plane in object_vision_spec.rigid_planes}
    selected_plane_ids = set(detection.plane_ids)
    markers = []

    for shape, (plane_id, marker_index) in zip(detection.shapes, detection.shape_marker_keys):
        plane = planes_by_id.get(plane_id)
        if plane is None or plane_id not in selected_plane_ids or marker_index < 0 or marker_index >= len(plane.shape_markers):
            return None
        marker = plane.shape_markers[marker_index]
        if marker.num_sides == 0 or marker.object_vertices_m is None or shape.color_id != marker.color_id or shape.num_sides != marker.num_sides:
            return None
        vertices_xy = np.asarray(marker.object_vertices_m, dtype=np.float64)
        if vertices_xy.shape != (marker.num_sides, 2) or not np.all(np.isfinite(vertices_xy)):
            return None
        vertices_plane = np.column_stack((vertices_xy, np.zeros(marker.num_sides, dtype=np.float64)))
        object_points = (plane.rotation_object_from_plane@vertices_plane.T).T + plane.translation_object_from_plane_m
        edge_lengths = np.linalg.norm(vertices_xy - np.roll(vertices_xy, -1, axis=0), axis=1)
        edge_norm = np.linalg.norm(edge_lengths)
        if edge_norm <= 1e-12 or not np.all(np.isfinite(object_points)):
            return None
        markers.append((shape, plane_id, marker_index, marker, object_points, edge_lengths/edge_norm))
    return markers


def getVertexOrderings(shape: ShapeDetection, normalized_object_edges: np.ndarray) -> list[tuple[np.ndarray, float]]:
    vertices = np.asarray(shape.vertices_px, dtype=np.float64)
    if vertices.shape != (shape.num_sides, 2) or not np.all(np.isfinite(vertices)):
        return []
    orderings = []
    for start in range(shape.num_sides):
        for direction in (1, -1):
            order = [(start + direction*offset)%shape.num_sides for offset in range(shape.num_sides)]
            ordered = vertices[order]
            image_edges = np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1)
            image_norm = np.linalg.norm(image_edges)
            if image_norm > 1e-12:
                orderings.append((ordered, float(np.linalg.norm(image_edges/image_norm - normalized_object_edges))))
    orderings.sort(key=lambda item: item[1])
    return orderings


def normalizeOrderedVertices(vertices_px: np.ndarray) -> np.ndarray | None:
    centered = vertices_px - np.mean(vertices_px, axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered*centered, axis=1))))
    return None if scale <= 1e-12 else centered/scale


def chooseInitialCorrespondences(pnp_markers: list[tuple], marker_matches: list[list[tuple[np.ndarray, float]]], cached_vertices: dict[tuple[str, int], np.ndarray]) -> list[int]:
    indices = []
    for marker_data, orderings in zip(pnp_markers, marker_matches):
        marker_key = (marker_data[1], marker_data[2])
        previous_vertices = cached_vertices.get(marker_key)
        if previous_vertices is None:
            indices.append(0)
            continue
        best_index, best_error = 0, float('inf')
        for index, (ordered_vertices, _) in enumerate(orderings):
            normalized = normalizeOrderedVertices(ordered_vertices)
            if normalized is None:
                continue
            error = float(np.sqrt(np.mean(np.sum((normalized - previous_vertices)**2, axis=1))))
            if error < best_error:
                best_index, best_error = index, error
        indices.append(best_index)
    return indices


def getFlexedObjectPoints(pnp_markers: list[tuple], flex_angle_deg: float, connection: tuple[str, str, float] | None,
                          hinge_point: np.ndarray | None, hinge_direction: np.ndarray | None) -> list[np.ndarray]:
    if connection is None:
        return [marker_data[4] for marker_data in pnp_markers]
    plane_id_1, plane_id_2, _ = connection
    half_angle_rad = np.deg2rad(flex_angle_deg/2.0)
    rotation_1, _ = cv2.Rodrigues(-half_angle_rad*hinge_direction)
    rotation_2, _ = cv2.Rodrigues(+half_angle_rad*hinge_direction)
    result = []
    for marker_data in pnp_markers:
        plane_id, points = marker_data[1], marker_data[4]
        rotation = rotation_1 if plane_id == plane_id_1 else rotation_2 if plane_id == plane_id_2 else None
        result.append(points if rotation is None else (rotation@(points - hinge_point).T).T + hinge_point)
    return result


def getSQPnPSolutions(
    object_points: np.ndarray, image_points: np.ndarray, camera_calibration: CameraCalibration,
) -> list[tuple[np.ndarray, float]]:
    result = cv2.solvePnPGeneric(
        object_points, image_points, camera_calibration.camera_matrix,
        camera_calibration.distortion_coefficients, flags=cv2.SOLVEPNP_SQPNP,
    )
    solution_count, rotation_vectors, translation_vectors = result[:3]
    if not solution_count:
        return []

    solutions = []
    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T
        if np.any(camera_points[:, 2] <= 0.0):
            continue

        projected, _ = cv2.projectPoints(
            object_points, rotation_vector, translation_vector,
            camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
        )
        error = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - image_points)**2, axis=1,
        ))))
        solutions.append((translation_vector.reshape(3), error))

    return solutions


def solvePnPAtFlexAngle(
    flex_angle_deg: float, pnp_markers: list[tuple],
    marker_matches: list[list[tuple[np.ndarray, float]]], correspondence_indices: list[int],
    connection: tuple[str, str, float] | None, hinge_point: np.ndarray | None,
    hinge_direction: np.ndarray | None, camera_calibration: CameraCalibration,
    previous_translation: np.ndarray | None,
) -> tuple[np.ndarray | None, float]:
    object_groups = getFlexedObjectPoints(
        pnp_markers, flex_angle_deg, connection, hinge_point, hinge_direction,
    )
    object_points = np.concatenate(object_groups, axis=0)
    image_points = np.concatenate([
        marker_matches[i][correspondence_indices[i]][0]
        for i in range(len(pnp_markers))
    ], axis=0)

    valid_solutions = getSQPnPSolutions(object_points, image_points, camera_calibration)
    if not valid_solutions:
        return None, float("inf")

    minimum_error = min(error for _, error in valid_solutions)
    near_best = [
        (translation, error) for translation, error in valid_solutions
        if error <= minimum_error + 0.50
    ]

    if previous_translation is None or len(near_best) == 1:
        return min(near_best, key=lambda item: item[1])

    return min(
        near_best,
        key=lambda item: (
            float(np.linalg.norm(item[0] - previous_translation)),
            item[1],
        ),
    )



def searchFlexAngles(search_angles_deg: np.ndarray, solve_angle) -> tuple[np.ndarray | None, float, float]:
    best_translation, best_error, best_angle = None, float('inf'), 0.0
    for angle in search_angles_deg:
        translation, error = solve_angle(float(angle))
        if error < best_error:
            best_translation, best_error, best_angle = translation, error, float(angle)
    return best_translation, best_error, best_angle


def searchFullFlex(max_rotation_deg: float, coarse_angles_deg: np.ndarray, solve_angle) -> tuple[np.ndarray | None, float, float]:
    best_translation, best_error, best_angle = searchFlexAngles(coarse_angles_deg, solve_angle)
    if max_rotation_deg <= 0.0 or best_translation is None:
        return best_translation, best_error, best_angle
    coarse_step = 2.0*max_rotation_deg/6.0
    fine_min, fine_max = max(-max_rotation_deg, best_angle - coarse_step), min(max_rotation_deg, best_angle + coarse_step)
    fine_angles = np.unique(np.concatenate((np.arange(fine_min, fine_max + 0.5, 1.0), np.array([fine_min, best_angle, fine_max]))))
    fine_translation, fine_error, fine_angle = searchFlexAngles(fine_angles, solve_angle)
    return (fine_translation, fine_error, fine_angle) if fine_translation is not None and fine_error < best_error else (best_translation, best_error, best_angle)


def findBestCorrespondenceAtAngle(flex_angle_deg: float, pnp_markers: list[tuple], marker_matches: list[list[tuple[np.ndarray, float]]],
                                  connection: tuple[str, str, float] | None, hinge_point: np.ndarray | None,
                                  hinge_direction: np.ndarray | None, camera_calibration: CameraCalibration) -> tuple[list[int] | None, float]:
    object_groups = getFlexedObjectPoints(pnp_markers, flex_angle_deg, connection, hinge_point, hinge_direction)
    anchor_index = max(range(len(pnp_markers)), key=lambda i: pnp_markers[i][3].num_sides)
    anchor_object_points = object_groups[anchor_index]
    best_indices, best_error = None, float('inf')

    for anchor_ordering_index, (anchor_image_points, _) in enumerate(marker_matches[anchor_index]):
        if len(anchor_object_points) == 3:
            solution_count, rotation_vectors, translation_vectors = cv2.solveP3P(anchor_object_points, anchor_image_points,
                camera_calibration.camera_matrix, camera_calibration.distortion_coefficients, flags=cv2.SOLVEPNP_AP3P)
        else:
            result = cv2.solvePnPGeneric(anchor_object_points, anchor_image_points, camera_calibration.camera_matrix,
                                         camera_calibration.distortion_coefficients, flags=cv2.SOLVEPNP_SQPNP)
            solution_count, rotation_vectors, translation_vectors = result[:3]
        if not solution_count:
            continue

        for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            trial_indices, total_squared_error, total_points, valid = [0]*len(pnp_markers), 0.0, 0, True
            trial_indices[anchor_index] = anchor_ordering_index
            for i, object_points in enumerate(object_groups):
                camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T
                if np.any(camera_points[:, 2] <= 0.0):
                    valid = False
                    break
                projected, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_calibration.camera_matrix,
                                                 camera_calibration.distortion_coefficients)
                projected = projected.reshape(-1, 2)
                if i == anchor_index:
                    ordering_index = anchor_ordering_index
                    squared_error = float(np.sum((anchor_image_points - projected)**2))
                else:
                    ordering_index, squared_error = min(
                        ((j, float(np.sum((vertices - projected)**2))) for j, (vertices, _) in enumerate(marker_matches[i])),
                        key=lambda item: item[1],
                    )
                trial_indices[i] = ordering_index
                total_squared_error += squared_error
                total_points += len(object_points)
            if valid and total_points:
                error = float(np.sqrt(total_squared_error/total_points))
                if error < best_error:
                    best_indices, best_error = trial_indices, error
    return best_indices, best_error


def rescueCorrespondences(best_flex_angle_deg: float, coarse_angles_deg: np.ndarray, current_indices: list[int], pnp_markers: list[tuple],
                          marker_matches: list[list[tuple[np.ndarray, float]]], connection: tuple[str, str, float] | None,
                          hinge_point: np.ndarray | None, hinge_direction: np.ndarray | None, camera_calibration: CameraCalibration,
                          rescue_error_px: float) -> tuple[list[int], float, float, int]:
    best_indices, best_angle, best_error, tested = current_indices.copy(), best_flex_angle_deg, float('inf'), set()

    def tryAngles(angles) -> None:
        nonlocal best_indices, best_angle, best_error
        for raw_angle in angles:
            angle = float(raw_angle)
            if connection is not None:
                angle = float(np.clip(angle, -connection[2], connection[2]))
            key = round(angle, 6)
            if key in tested:
                continue
            tested.add(key)
            indices, error = findBestCorrespondenceAtAngle(angle, pnp_markers, marker_matches, connection, hinge_point, hinge_direction, camera_calibration)
            if indices is not None and error < best_error:
                best_indices, best_angle, best_error = indices, angle, error

    tryAngles([best_flex_angle_deg])
    if best_error > rescue_error_px and connection is not None:
        tryAngles([best_flex_angle_deg - 2.0, best_flex_angle_deg + 2.0])
    if best_error > rescue_error_px and connection is not None:
        tryAngles(coarse_angles_deg)
    return best_indices, best_angle, best_error, len(tested)



def addPnPDiagnosticsStage(
    debug: DetectionDebug, detection: Detection, pnp_markers: list[tuple],
    marker_matches: list[list[tuple[np.ndarray, float]]], correspondence_indices: list[int],
    connection: tuple[str, str, float] | None, hinge_point: np.ndarray | None,
    hinge_direction: np.ndarray | None, camera_calibration: CameraCalibration,
    best_translation: np.ndarray, best_error: float, best_angle: float,
    previous_translation: np.ndarray | None, search_trace: list[tuple[float, float | None, float]],
    accepted_warm: bool, rescue_angles_tested: int,
) -> None:
    plane_group = tuple(detection.plane_ids)
    object_groups = getFlexedObjectPoints(
        pnp_markers, best_angle, connection, hinge_point, hinge_direction,
    )
    image_groups = [
        marker_matches[i][correspondence_indices[i]][0]
        for i in range(len(pnp_markers))
    ]

    joint_solutions = getSQPnPSolutions(
        np.concatenate(object_groups, axis=0),
        np.concatenate(image_groups, axis=0),
        camera_calibration,
    )

    lines = [
        "PNP DIAGNOSTICS (debug only; no estimator behavior changes)",
        f"planes={'+'.join(plane_group)} | bbox_diag={np.hypot(detection.px_w or 0.0, detection.px_h or 0.0):.1f}px",
        f"final: z={best_translation[2]:.3f}m | xyz=({best_translation[0]:+.3f}, {best_translation[1]:+.3f}, {best_translation[2]:+.3f})m",
        f"reproj={best_error:.3f}px | flex={best_angle:+.1f}deg | warm_accepted={accepted_warm} | rescue_angles={rescue_angles_tested}",
        f"previous z={'none' if previous_translation is None else f'{previous_translation[2]:.3f}m'}",
        f"correspondence indices={correspondence_indices}",
        "",
        f"FINAL JOINT SQPnP BRANCHES ({len(joint_solutions)} valid):",
    ]

    for index, (translation, error) in enumerate(sorted(joint_solutions, key=lambda item: item[1])):
        chosen = (
            np.linalg.norm(translation - best_translation) < 1e-6
            and abs(error - best_error) < 1e-6
        )
        lines.append(
            f"  {'*' if chosen else ' '} branch {index}: "
            f"z={translation[2]:.3f}m | xyz=({translation[0]:+.3f},{translation[1]:+.3f},{translation[2]:+.3f}) "
            f"| err={error:.3f}px"
        )

    lines.append("")
    lines.append("SINGLE-PLANE SQPnP AT FINAL FLEX (diagnostic only):")

    for plane_id in plane_group:
        indices = [i for i, marker_data in enumerate(pnp_markers) if marker_data[1] == plane_id]
        if not indices:
            lines.append(f"  {plane_id}: no selected marker")
            continue

        object_points = np.concatenate([object_groups[i] for i in indices], axis=0)
        image_points = np.concatenate([image_groups[i] for i in indices], axis=0)
        plane_solutions = getSQPnPSolutions(object_points, image_points, camera_calibration)

        if not plane_solutions:
            lines.append(f"  {plane_id}: no valid SQPnP solution")
            continue

        branch_text = ", ".join(
            f"z={translation[2]:.3f}m/e={error:.2f}px"
            for translation, error in sorted(plane_solutions, key=lambda item: item[1])
        )
        lines.append(f"  {plane_id}: {branch_text}")

    if search_trace:
        lines.append("")
        lines.append("FLEX SEARCH BEST SOLUTION PER SOLVE CALL:")
        for angle, z, error in search_trace[-20:]:
            z_text = "none" if z is None else f"{z:.3f}m"
            error_text = "inf" if not np.isfinite(error) else f"{error:.2f}px"
            lines.append(f"  angle={angle:+5.1f}deg -> z={z_text:>8} | err={error_text}")

        if len(search_trace) > 20:
            lines.append(f"  ... {len(search_trace) - 20} earlier solve calls omitted")

    width = max(900, debug._reference_image.shape[1] if debug._reference_image is not None else 900)
    height = max(480, 48 + 24*len(lines))
    image = np.full((height, width, 3), 25, dtype=np.uint8)

    y = 30
    for index, line in enumerate(lines):
        font_scale = 0.58 if index == 0 else 0.48
        thickness = 2 if index == 0 else 1
        cv2.putText(
            image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (235, 235, 235), thickness, cv2.LINE_AA,
        )
        y += 24

    debug.addStage("PnP diagnostics", image)



# Convert the already-selected one-plane or one-hinge shape group into a camera-frame measurement.
def createMeasurementUsingShapeGroup(detection: Detection, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration,
                                     _timing_warmup: bool = False) -> Measurement:
    failed = Measurement(None, None, None, None, None, None)
    measurement_debug = None if _timing_warmup else getattr(detection, '_debug', None)
    if measurement_debug is not None:
        createMeasurementUsingShapeGroup(detection, object_vision_spec, camera_calibration, _timing_warmup=True)
    timing_start = time.perf_counter()

    pnp_markers = buildSelectedPnPMarkers(detection, object_vision_spec)
    if pnp_markers is None:
        return failed
    plane_group = tuple(detection.plane_ids)
    planes_by_id = {plane.plane_id: plane for plane in object_vision_spec.rigid_planes}
    connections = {frozenset((p1, p2)): (p1, p2, max_angle) for p1, p2, max_angle in object_vision_spec.rigid_plane_connections}
    connection = connections.get(frozenset(plane_group)) if len(plane_group) == 2 else None
    if len(plane_group) == 2 and connection is None:
        return failed

    marker_matches = [getVertexOrderings(marker_data[0], marker_data[5]) for marker_data in pnp_markers]
    if any(not orderings for orderings in marker_matches):
        return failed
    bbox_diagonal_px = float(np.hypot(detection.px_w or 0.0, detection.px_h or 0.0))
    severe_error_px = max(12.0, 0.040*bbox_diagonal_px)
    rescue_error_px = max(8.0, 0.025*bbox_diagonal_px)

    hinge_point = hinge_direction = None
    max_rotation_deg = 0.0
    if connection is not None:
        plane_id_1, plane_id_2, max_rotation_deg = connection
        hinge_point, hinge_direction = getRigidPlaneIntersection(planes_by_id[plane_id_1], planes_by_id[plane_id_2])
    coarse_angles_deg = np.linspace(-max_rotation_deg, max_rotation_deg, 7) if max_rotation_deg > 0.0 else np.array([0.0])

    spec_state_key = id(object_vision_spec)
    warm_key = (spec_state_key, frozenset(plane_group))
    warm_angle = _PNP_WARM_START_ANGLES_DEG.get(warm_key) if connection is not None else None
    cached_vertices = _PNP_WARM_START_ORDERED_VERTICES.get(warm_key, {})
    correspondence_indices = chooseInitialCorrespondences(pnp_markers, marker_matches, cached_vertices)
    previous_translation = _PNP_PREVIOUS_TRANSLATION_M.get((spec_state_key, plane_group))
    pnp_setup_seconds = time.perf_counter() - timing_start
    pnp_warm_seconds = pnp_full_seconds = pnp_rescue_seconds = 0.0
    rescue_angles_tested = 0
    pnp_search_trace = [] if measurement_debug is not None else None

    def solveAngle(angle: float, indices: list[int] | None = None):
        translation, error = solvePnPAtFlexAngle(
            angle, pnp_markers, marker_matches,
            correspondence_indices if indices is None else indices,
            connection, hinge_point, hinge_direction, camera_calibration, previous_translation,
        )
        if pnp_search_trace is not None:
            pnp_search_trace.append((
                float(angle),
                None if translation is None else float(translation[2]),
                float(error),
            ))
        return translation, error

    best_translation, best_error, best_angle = None, float('inf'), 0.0
    accepted_warm = False
    needs_full_search = True
    if connection is not None and warm_angle is not None and max_rotation_deg > 0.0:
        warm_min, warm_max = max(-max_rotation_deg, warm_angle - 2.0), min(max_rotation_deg, warm_angle + 2.0)
        warm_angles = np.unique(np.append(np.arange(warm_min, warm_max + 0.5, 1.0), np.clip(warm_angle, -max_rotation_deg, max_rotation_deg)))
        t = time.perf_counter()
        best_translation, best_error, best_angle = searchFlexAngles(warm_angles, solveAngle)
        pnp_warm_seconds = time.perf_counter() - t
        at_lower_edge = abs(best_angle - warm_min) <= 0.51 and warm_min > -max_rotation_deg + 1e-9
        at_upper_edge = abs(best_angle - warm_max) <= 0.51 and warm_max < max_rotation_deg - 1e-9
        needs_full_search = best_translation is None or at_lower_edge or at_upper_edge or best_error > severe_error_px
        accepted_warm = not needs_full_search

    if needs_full_search:
        t = time.perf_counter()
        best_translation, best_error, best_angle = searchFullFlex(max_rotation_deg, coarse_angles_deg, solveAngle)
        pnp_full_seconds = time.perf_counter() - t
        accepted_warm = False

    if not accepted_warm and best_translation is not None and best_error > rescue_error_px and len(pnp_markers) >= 2:
        t = time.perf_counter()
        rescued_indices, rescue_angle, _, rescue_angles_tested = rescueCorrespondences(
            best_angle, coarse_angles_deg, correspondence_indices, pnp_markers, marker_matches, connection,
            hinge_point, hinge_direction, camera_calibration, rescue_error_px,
        )
        if rescued_indices != correspondence_indices:
            local_angles = np.array([rescue_angle])
            if connection is not None and max_rotation_deg > 0.0:
                coarse_step = 2.0*max_rotation_deg/6.0
                fine_min, fine_max = max(-max_rotation_deg, rescue_angle - coarse_step), min(max_rotation_deg, rescue_angle + coarse_step)
                local_angles = np.unique(np.concatenate((np.arange(fine_min, fine_max + 0.5, 1.0), np.array([fine_min, rescue_angle, fine_max]))))
            rescued_translation, rescued_error, rescued_angle = searchFlexAngles(local_angles, lambda angle: solveAngle(angle, rescued_indices))
            if rescued_translation is not None and rescued_error < best_error:
                best_translation, best_error, best_angle = rescued_translation, rescued_error, rescued_angle
                correspondence_indices = rescued_indices
        pnp_rescue_seconds = time.perf_counter() - t

    if best_translation is None or not np.all(np.isfinite(best_translation)):
        return failed

    _PNP_PREVIOUS_PLANE_GROUP[spec_state_key] = plane_group
    if best_error <= severe_error_px:
        _PNP_PREVIOUS_TRANSLATION_M[(spec_state_key, plane_group)] = best_translation.copy()
        ordered_vertices_by_marker = {}
        for marker_data, orderings, correspondence_index in zip(pnp_markers, marker_matches, correspondence_indices):
            normalized = normalizeOrderedVertices(orderings[correspondence_index][0])
            if normalized is not None:
                ordered_vertices_by_marker[(marker_data[1], marker_data[2])] = normalized.copy()
        if ordered_vertices_by_marker:
            _PNP_WARM_START_ORDERED_VERTICES[warm_key] = ordered_vertices_by_marker
        if connection is not None:
            _PNP_WARM_START_ANGLES_DEG[warm_key] = best_angle

    total_group_marker_count = sum(marker.num_sides != 0 for plane_id in plane_group for marker in planes_by_id[plane_id].shape_markers)
    if connection is not None and not _timing_warmup:
        print(f'flex {connection[0]}<->{connection[1]}: {best_angle:+.1f} deg | markers={len(pnp_markers)}/{total_group_marker_count} | '
              f'reprojection={best_error:.2f} px | hinges_considered=1')

    pnp_total_seconds = time.perf_counter() - timing_start
    if measurement_debug is not None:
        addPnPDiagnosticsStage(
            measurement_debug, detection, pnp_markers, marker_matches,
            correspondence_indices, connection, hinge_point, hinge_direction,
            camera_calibration, best_translation, best_error, best_angle,
            previous_translation, pnp_search_trace or [], accepted_warm,
            rescue_angles_tested,
        )
        measurement_debug.setTiming('PnP setup + correspondences', pnp_setup_seconds)
        measurement_debug.setTiming('PnP warm-start search', pnp_warm_seconds)
        measurement_debug.setTiming('PnP full flex search', pnp_full_seconds)
        measurement_debug.setTiming(f'PnP correspondence rescue ({rescue_angles_tested} angles)', pnp_rescue_seconds)
        measurement_debug.setTiming('PnP total', pnp_total_seconds)
        detection_total_ms = measurement_debug.timings_ms.get('2D detection total')
        if detection_total_ms is not None:
            measurement_debug.timings_ms['TOTAL vision'] = detection_total_ms + 1000.0*pnp_total_seconds
        measurement_debug.updateTimingStage()

    return Measurement(float(best_translation[0]), float(best_translation[1]), float(best_translation[2]), None, None, None)

