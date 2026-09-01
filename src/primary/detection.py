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
def detectSingleObject(frame: np.ndarray, object_vision_spec_id: ObjectVisionSpecId, camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]

    if object_vision_spec.object_type == ObjectType.TENNIS_BALL:
        return detectTennisBall(frame, object_vision_spec, camera_calibration)
    elif object_vision_spec.object_type == ObjectType.ARUCO_MARKER:
        return detectArucoMarker(frame, object_vision_spec, camera_calibration)
    elif object_vision_spec.object_type == ObjectType.PAPER_PLANE_SHAPES:
        return detectPaperPlaneShapes(frame, object_vision_spec, camera_calibration)

    raise ValueError(f"Unsupported object type for {object_vision_spec_id}: {object_vision_spec.object_type}")


def detectTennisBall(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration) -> tuple[bool, Detection, Measurement]:
    detection = findSingleObjectSphere(frame, object_vision_spec, camera_calibration)
    # detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_vision_spec.width, camera_calibration,)
    measurement = Measurement(x, y, z, None, None, None)
    return True, detection, measurement


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
def refineShapeVerticesUsingEdges(contour: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rough_vertices = polygon.reshape(-1, 2).astype(np.float64)
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    num_sides = len(rough_vertices)
    edge_distances, fitted_lines = [], []

    # Assign contour points to the nearest rough polygon edge.
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

    # Fit each edge from its straight middle section and measure how well the contour supports that line.
    for edge_index in range(num_sides):
        edge_start = rough_vertices[edge_index]
        edge_end = rough_vertices[(edge_index + 1)%num_sides]
        edge_vector = edge_end - edge_start
        edge_length_squared = np.dot(edge_vector, edge_vector)
        edge_points = contour_points[edge_assignments == edge_index]

        if len(edge_points) < 3:
            return rough_vertices

        projection = ((edge_points - edge_start)@edge_vector)/edge_length_squared
        edge_points = edge_points[(projection >= 0.10) & (projection <= 0.90)]

        if len(edge_points) < 3:
            return rough_vertices

        vx, vy, x0, y0 = cv2.fitLine(edge_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
        line_point = np.array([x0, y0], dtype=np.float64)
        line_direction = np.array([vx, vy], dtype=np.float64)
        direction_norm = np.linalg.norm(line_direction)

        if direction_norm <= 1e-6:
            return rough_vertices

        line_direction /= direction_norm
        relative_points = edge_points - line_point
        residuals = np.abs(line_direction[0]*relative_points[:, 1] - line_direction[1]*relative_points[:, 0])
        edge_fit_error = float(np.sqrt(np.mean(residuals**2)))

        # Allow modest raster/mask roughness, especially on long marker edges. A fixed 1 px RMS
        # cutoff was too strict: one slightly jagged edge could cancel refinement for the entire
        # polygon and prevent acute corners from being recovered by line intersection.
        maximum_edge_fit_error_px = min(3.0, max(1.5, 0.01*edge_length))
        if edge_fit_error > maximum_edge_fit_error_px:
            return rough_vertices

        fitted_lines.append((line_point, line_direction))

    # Intersect neighboring fitted edges to recover refined corners. Acute corners may move substantially
    # beyond approxPolyDP when both neighboring edge fits are clean.
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

    # Keep only an emergency displacement limit; fit residuals above are the main refinement confidence test.
    maximum_edge_length = max(
        np.linalg.norm(rough_vertices[i] - rough_vertices[(i + 1)%num_sides])
        for i in range(num_sides)
    )

    if np.any(np.linalg.norm(refined_vertices - rough_vertices, axis=1) > 0.50*maximum_edge_length):
        return rough_vertices

    # Reject geometry that changes the polygon topology or area implausibly.
    refined_polygon = refined_vertices.astype(np.float32).reshape(-1, 1, 2)

    # Refinement currently supports convex polygons only. A future concave-shape path
    # would need different edge assignment/topology validation.
    if not cv2.isContourConvex(refined_polygon):
        return rough_vertices

    rough_area = cv2.contourArea(rough_vertices.astype(np.float32))
    refined_area = cv2.contourArea(refined_polygon)

    if rough_area <= 0.0 or not 0.65 <= refined_area/rough_area <= 1.35:
        return rough_vertices

    return refined_vertices

# Shape path: find color-based convex polygon candidates, group nearby markers, then select the best group.
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
        contour_debug_frame, polygon_debug_frame, candidate_debug_frame = frame.copy(), frame.copy(), frame.copy()
    else:
        contour_debug_frame = polygon_debug_frame = candidate_debug_frame = None

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

                # Same speck-removal pattern that made the tennis-ball mask quiet:
                # median -> OPEN removes isolated white noise -> CLOSE fills tiny holes.
                # Keep the existing 3x3 kernel here so narrow/acute paper-marker geometry
                # is not eroded as aggressively as it would be by the tennis ball's 5x5.
                # Generic speck suppression for every marker color. Match the tennis-ball
                # detector's 5x5 median filtering, while keeping morphology at 3x3 so acute
                # polygon tips are not unnecessarily eroded.
                cleaned_loose_hsv_roi = cv2.medianBlur(raw_loose_hsv_roi, 5)
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

        # Step 5/6: Extract only the selected component's own bounding rectangle, not a full-frame
        # binary mask. The HSV ROI has already been expanded if the component reached its edge, so
        # this local contour extraction does not crop the marker geometry.
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
                component_x1 = min(int(seed_stats[label, cv2.CC_STAT_LEFT]) for label in keep_labels)
                component_y1 = min(int(seed_stats[label, cv2.CC_STAT_TOP]) for label in keep_labels)
                component_x2 = max(
                    int(seed_stats[label, cv2.CC_STAT_LEFT] + seed_stats[label, cv2.CC_STAT_WIDTH])
                    for label in keep_labels
                )
                component_y2 = max(
                    int(seed_stats[label, cv2.CC_STAT_TOP] + seed_stats[label, cv2.CC_STAT_HEIGHT])
                    for label in keep_labels
                )
                component_labels = seed_labels[component_y1:component_y2, component_x1:component_x2]

                if len(keep_labels) == 1:
                    selected_component_mask = (component_labels == keep_labels[0]).astype(np.uint8)*255
                else:
                    selected_component_mask = np.isin(component_labels, keep_labels).astype(np.uint8)*255

                contours, _ = cv2.findContours(
                    selected_component_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    offset=(x1 + component_x1, y1 + component_y1),
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
                    global_component_x1 = x1 + component_x1
                    global_component_y1 = y1 + component_y1
                    global_component_x2 = x1 + component_x2
                    global_component_y2 = y1 + component_y2
                    selected_seed_debug[
                        global_component_y1:global_component_y2,
                        global_component_x1:global_component_x2,
                    ] = selected_component_mask
                    debug.addStage(
                        f"{color_name} candidate {candidate_index} selected HSV component", selected_seed_debug,
                    )

                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(contour)

                if contour_debug_frame is not None:
                    cv2.drawContours(contour_debug_frame, [contour], -1, draw_bgr, 1)

                if contour_area < minimum_shape_area_by_color[color_id]:
                    continue

                polygon_start = time.perf_counter() if profile_shape_detection else None

                # Existing polygon geometry path: convex hull -> approxPolyDP -> straight-edge fitting
                # and line intersection. This is intentionally retained before trying LAB edge refinement.
                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)

                if perimeter <= 0:
                    if profile_shape_detection:
                        timing_polygon_refine_s += time.perf_counter() - polygon_start
                    continue

                base_epsilon_ratio = object_vision_spec.polygon_epsilon_ratio
                base_polygon = cv2.approxPolyDP(hull, base_epsilon_ratio*perimeter, True)
                expected_num_sides = expected_num_sides_by_color[color_id]
                initial_num_sides = len(base_polygon)

                # Keep the same computational ceiling as before: one normal approxPolyDP call,
                # plus at most ONE retry when the initial count misses a configured polygon by
                # exactly one vertex. The retry moves epsilon in the useful direction:
                #   too few vertices -> smaller epsilon, recover a shallow/missing corner
                #   too many vertices -> larger epsilon, suppress one extra/noisy corner
                if initial_num_sides in expected_num_sides:
                    candidate_polygons = [(initial_num_sides, base_polygon)]
                    retry_polygon = None
                    retry_epsilon_ratio = None
                else:
                    candidate_polygons = []
                    retry_polygon = None
                    retry_epsilon_ratio = None

                    nearest_expected_num_sides = min(
                        expected_num_sides,
                        key=lambda num_sides: abs(num_sides - initial_num_sides),
                    )

                    if abs(nearest_expected_num_sides - initial_num_sides) == 1:
                        epsilon_direction = -1.0 if initial_num_sides < nearest_expected_num_sides else +1.0
                        retry_epsilon_ratio = max(0.005, base_epsilon_ratio + epsilon_direction*0.02)
                        retry_polygon = cv2.approxPolyDP(hull, retry_epsilon_ratio*perimeter, True)

                        if len(retry_polygon) in expected_num_sides:
                            candidate_polygons.append((len(retry_polygon), retry_polygon))

                if polygon_debug_frame is not None:
                    cv2.polylines(polygon_debug_frame, [base_polygon], True, draw_bgr, 1)
                    polygon_center = np.mean(base_polygon.reshape(-1, 2), axis=0).astype(np.int32)

                    if retry_polygon is not None and candidate_polygons:
                        recovered_num_sides = len(candidate_polygons[0][1])
                        debug_text = (
                            f"{color_name}: observed={initial_num_sides}, expected={expected_num_sides}, "
                            f"recovered={recovered_num_sides}"
                        )
                        cv2.polylines(polygon_debug_frame, [candidate_polygons[0][1]], True, draw_bgr, 2)
                    else:
                        debug_text = f"{color_name}: observed={initial_num_sides}, expected={expected_num_sides}"

                    cv2.putText(
                        polygon_debug_frame, debug_text, tuple(polygon_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, draw_bgr, 1, cv2.LINE_AA,
                    )

                for num_sides, polygon in candidate_polygons:
                    if not cv2.isContourConvex(polygon):
                        continue

                    vertices_px = refineShapeVerticesUsingEdges(contour, polygon)
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
                        rough_debug = frame.copy()
                        cv2.polylines(rough_debug, [polygon], True, draw_bgr, 2)
                        debug.addStage(f"{color_name} candidate {candidate_index} rough polygon", rough_debug)

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
        debug.addStage("Polygon approximations", polygon_debug_frame)

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


def solvePnPAtFlexAngle(flex_angle_deg: float, pnp_markers: list[tuple], marker_matches: list[list[tuple[np.ndarray, float]]],
                        correspondence_indices: list[int], connection: tuple[str, str, float] | None, hinge_point: np.ndarray | None,
                        hinge_direction: np.ndarray | None, camera_calibration: CameraCalibration,
                        previous_translation: np.ndarray | None) -> tuple[np.ndarray | None, float]:
    object_groups = getFlexedObjectPoints(pnp_markers, flex_angle_deg, connection, hinge_point, hinge_direction)
    object_points = np.concatenate(object_groups, axis=0)
    image_points = np.concatenate([marker_matches[i][correspondence_indices[i]][0] for i in range(len(pnp_markers))], axis=0)
    result = cv2.solvePnPGeneric(object_points, image_points, camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                                 flags=cv2.SOLVEPNP_SQPNP)
    solution_count, rotation_vectors, translation_vectors = result[:3]
    if not solution_count:
        return None, float('inf')

    valid_solutions = []
    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T
        if np.any(camera_points[:, 2] <= 0.0):
            continue
        projected, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_calibration.camera_matrix,
                                         camera_calibration.distortion_coefficients)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points)**2, axis=1))))
        valid_solutions.append((translation_vector.reshape(3), error))
    if not valid_solutions:
        return None, float('inf')

    minimum_error = min(error for _, error in valid_solutions)
    near_best = [(translation, error) for translation, error in valid_solutions if error <= minimum_error + 0.50]
    if previous_translation is None or len(near_best) == 1:
        return min(near_best, key=lambda item: item[1])
    return min(near_best, key=lambda item: (float(np.linalg.norm(item[0] - previous_translation)), item[1]))


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

    def solveAngle(angle: float, indices: list[int] | None = None):
        return solvePnPAtFlexAngle(angle, pnp_markers, marker_matches, correspondence_indices if indices is None else indices,
                                   connection, hinge_point, hinge_direction, camera_calibration, previous_translation)

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

