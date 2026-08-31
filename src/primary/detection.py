import cv2
import numpy as np
import time
from collections import Counter
from itertools import combinations, permutations

from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.geometry import estimateObjectWorldPosition
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec, ObjectVisionSpecId, getRigidPlaneIntersection
from src.primary.color import COLOR_SPECS, ColorId


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
    def __init__(
        self,
        u: float | None, v: float | None, px_w: float | None, px_h: float | None,
        shapes: list[ShapeDetection] | None = None,
        bbox_center_offset_px: tuple[float, float] | np.ndarray | None = None,
    ):
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

        timing_image = np.full_like(self._reference_image, 25)
        cv2.putText(timing_image, "TIMING (warmed; debug drawing excluded)", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        y = 62
        for name, elapsed_ms in self.timings_ms.items():
            cv2.putText(timing_image, f"{name}: {elapsed_ms:.2f} ms", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            y += 27
            if y > timing_image.shape[0] - 15:
                break

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

    if measurement.x is None:
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

    bbox_offset = detection.bbox_center_offset_px
    bbox_center_u = detection.u + (0.0 if bbox_offset is None else bbox_offset[0])
    bbox_center_v = detection.v + (0.0 if bbox_offset is None else bbox_offset[1])

    x_min = int(round(bbox_center_u - detection.px_w/2.0))
    y_min = int(round(bbox_center_v - detection.px_h/2.0))
    x_max = int(round(bbox_center_u + detection.px_w/2.0))
    y_max = int(round(bbox_center_v + detection.px_h/2.0))

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2,)
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
    LAB_ACQUISITION_SCALE = 0.5
    GLOBAL_BLUR_KERNEL = (5, 5)
    HOTSPOT_PERCENTILE = 98.5
    MIN_HOTSPOT_RESPONSE_FACTOR = 0.30
    MIN_LAB_DIRECTION_COSINE = 0.75
    MIN_HOTSPOT_AREA_PX_FULL_RES = 6
    HOTSPOT_PADDING_FACTOR = 0.75
    MIN_HOTSPOT_PADDING_PX = 10
    EXTRA_HOTSPOT_CANDIDATES = 2
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
    combined_raw_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    combined_cleaned_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
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
        # projections onto one another's LAB directions. Require the pixel chroma direction to
        # actually point near this target color before letting its magnitude compete for hotspots.
        lab_direction_cosine = np.divide(
            lab_color_response, pixel_chroma_strength,
            out=np.full_like(lab_color_response, -1.0), where=pixel_chroma_strength > 1e-6,
        )
        aligned_response = np.where(
            lab_direction_cosine >= MIN_LAB_DIRECTION_COSINE,
            np.maximum(lab_color_response, 0.0), 0.0,
        )

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

        # Step 4: Build one loose HSV mask for this color. Each LAB hotspot ROI selects the HSV
        # connected component that actually overlaps that hotspot; using global component labels
        # avoids clipping a long triangle merely because the strongest LAB hotspot was small.
        raw_loose_hsv_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            loose_lower_hsv = np.clip(
                lower_hsv.astype(np.int16) - LOOSE_HSV_LOWER_SUBTRACTION, 0, 255,
            ).astype(np.uint8)
            raw_loose_hsv_mask = cv2.bitwise_or(raw_loose_hsv_mask, cv2.inRange(hsv_frame, loose_lower_hsv, upper_hsv))

        cleaned_loose_hsv_mask = cv2.medianBlur(raw_loose_hsv_mask, 3)
        cleaned_loose_hsv_mask = cv2.morphologyEx(cleaned_loose_hsv_mask, cv2.MORPH_CLOSE, hsv_cleanup_kernel)
        combined_raw_mask = cv2.bitwise_or(combined_raw_mask, raw_loose_hsv_mask)
        combined_cleaned_mask = cv2.bitwise_or(combined_cleaned_mask, cleaned_loose_hsv_mask)

        if debug is not None:
            debug.addStage(f"Loose HSV mask - {color_name}", raw_loose_hsv_mask)
            debug.addStage(f"Cleaned loose HSV mask - {color_name}", cleaned_loose_hsv_mask)

        num_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(cleaned_loose_hsv_mask, connectivity=8)
        used_primary_seed_labels = set()

        for candidate_index, (
            _, _, hot_x, hot_y, hot_w, hot_h, _, hotspot_label, low_x, low_y, low_w, low_h,
        ) in enumerate(selected_hotspots_full, start=1):
            hotspot_size = max(hot_w, hot_h)
            padding = max(MIN_HOTSPOT_PADDING_PX, int(HOTSPOT_PADDING_FACTOR*hotspot_size))
            x1, y1 = max(0, hot_x - padding), max(0, hot_y - padding)
            x2, y2 = min(frame.shape[1], hot_x + hot_w + padding), min(frame.shape[0], hot_y + hot_h + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            # Upscale only this small hotspot-label ROI to full resolution for overlap testing.
            low_x1 = max(0, int(np.floor(x1/acquisition_to_full_x)))
            low_y1 = max(0, int(np.floor(y1/acquisition_to_full_y)))
            low_x2 = min(hotspot_labels.shape[1], int(np.ceil(x2/acquisition_to_full_x)))
            low_y2 = min(hotspot_labels.shape[0], int(np.ceil(y2/acquisition_to_full_y)))
            low_component_roi = (hotspot_labels[low_y1:low_y2, low_x1:low_x2] == hotspot_label).astype(np.uint8)

            if low_component_roi.size == 0:
                continue

            hotspot_component_roi = cv2.resize(
                low_component_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            seed_labels_roi = seed_labels[y1:y2, x1:x2]
            best_seed = None

            for seed_label in np.unique(seed_labels_roi[hotspot_component_roi]):
                seed_label = int(seed_label)
                if seed_label == 0 or seed_label in used_primary_seed_labels:
                    continue

                seed_area = int(seed_stats[seed_label, cv2.CC_STAT_AREA])
                if seed_area < minimum_shape_area_by_color[color_id]:
                    continue

                overlap = int(np.count_nonzero((seed_labels_roi == seed_label) & hotspot_component_roi))
                if overlap <= 0:
                    continue

                if best_seed is None or (overlap, seed_area) > (best_seed[0], best_seed[1]):
                    best_seed = (overlap, seed_area, seed_label)

            if debug is not None:
                roi_debug = frame.copy()
                cv2.rectangle(roi_debug, (x1, y1), (x2 - 1, y2 - 1), draw_bgr, 1)
                cv2.rectangle(roi_debug, (hot_x, hot_y), (hot_x + hot_w, hot_y + hot_h), (255, 255, 255), 1)
                debug.addStage(f"{color_name} candidate {candidate_index} ROI", roi_debug)
                debug.addStage(
                    f"{color_name} candidate {candidate_index} loose HSV seed mask",
                    cleaned_loose_hsv_mask[y1:y2, x1:x2],
                )

            if best_seed is None:
                continue

            _, seed_area, primary_seed_label = best_seed
            used_primary_seed_labels.add(primary_seed_label)
            keep_seed_label = np.zeros(num_seed_labels, dtype=np.uint8)
            keep_seed_label[primary_seed_label] = 255

            # Step 5 (optional/off): recover substantial nearby HSV components if real lighting
            # later splits a marker into pieces. This is deliberately inactive for the first tests.
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
                        keep_seed_label[other_label] = 255

            selected_seed_mask = keep_seed_label[seed_labels]

            if debug is not None:
                debug.addStage(f"{color_name} candidate {candidate_index} selected HSV component", selected_seed_mask)

            contours, _ = cv2.findContours(selected_seed_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour)

            if contour_debug_frame is not None:
                cv2.drawContours(contour_debug_frame, [contour], -1, draw_bgr, 1)

            if contour_area < minimum_shape_area_by_color[color_id]:
                continue

            # Existing polygon geometry path: convex hull -> approxPolyDP -> straight-edge fitting
            # and line intersection. This is intentionally retained before trying LAB edge refinement.
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            base_polygon = cv2.approxPolyDP(hull, object_vision_spec.polygon_epsilon_ratio*perimeter, True)

            if polygon_debug_frame is not None:
                cv2.polylines(polygon_debug_frame, [base_polygon], True, draw_bgr, 1)
                polygon_center = np.mean(base_polygon.reshape(-1, 2), axis=0).astype(np.int32)
                cv2.putText(
                    polygon_debug_frame, f"{color_name}: {len(base_polygon)} vertices", tuple(polygon_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_bgr, 1, cv2.LINE_AA,
                )

            expected_num_sides = expected_num_sides_by_color[color_id]

            # Prefer an exact N-sided match. Only use the N+1 retry when no configured shape matches directly.
            if len(base_polygon) in expected_num_sides:
                candidate_polygons = [(len(base_polygon), base_polygon)]
            else:
                candidate_polygons = []
                for num_sides in expected_num_sides:
                    if len(base_polygon) == num_sides + 1:
                        retry_polygon = cv2.approxPolyDP(
                            contour, (object_vision_spec.polygon_epsilon_ratio + 0.02)*perimeter, True,
                        )
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
            timing_hsv_polygon_seconds += time.perf_counter() - hsv_polygon_stage_start

    if debug is not None:
        debug.addStage("Combined raw mask", combined_raw_mask)
        debug.addStage("Combined cleaned mask", combined_cleaned_mask)
        debug.addStage("All selected HSV contours", contour_debug_frame)
        debug.addStage("Polygon approximations", polygon_debug_frame)

        if not shape_candidates:
            cv2.putText(candidate_debug_frame, "No accepted shapes", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Accepted shape candidates", candidate_debug_frame)

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
    shape_groups = groupShapeCandidates(shape_candidates, object_vision_spec)

    if debug is not None:
        group_debug_frame = frame.copy()

        for group_index, shape_group in enumerate(shape_groups):
            all_group_vertices = np.concatenate([shape.vertices_px for shape in shape_group], axis=0)

            for shape in shape_group:
                draw_bgr = COLOR_SPECS[shape.color_id].draw_bgr
                shape_points = np.round(shape.vertices_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(group_debug_frame, [shape_points], True, draw_bgr, 1)

            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_group_vertices.astype(np.float32))
            cv2.rectangle(group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 255, 255), 2)
            cv2.putText(group_debug_frame, f"G{group_index}: {len(shape_group)}/{len(polygon_markers)} markers", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if not shape_groups:
            cv2.putText(group_debug_frame, "No groups matched shape_markers", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        debug.addStage("Matching shape groups", group_debug_frame)

    if not shape_groups:
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
                    "2D HSV + polygons": timing_hsv_polygon_seconds,
                    "2D grouping + selection": timing_grouping_s,
                    "2D detection total": timing_total_s,
                })
            if PRINT_SHAPE_DETECTION_TIMING:
                print(
                    f"shape timing: model={timing_model_setup_s*1000.0:.1f} ms | resize+blur={timing_resize_blur_s*1000.0:.1f} ms | "
                    f"LABprep={timing_lab_prep_s*1000.0:.1f} ms | HSVconv={timing_hsv_conversion_s*1000.0:.1f} ms | "
                    f"LAB={timing_lab_seconds*1000.0:.1f} ms | HSV+polygon={timing_hsv_polygon_seconds*1000.0:.1f} ms | "
                    f"grouping={timing_grouping_s*1000.0:.1f} ms | total={timing_total_s*1000.0:.1f} ms (no groups)"
                )
        if debug is not None:
            debug.updateTimingStage()
        return None

    best_shape_group = selectBestShapeGroup(shape_groups, object_vision_spec)

    if debug is not None:
        best_group_debug_frame = frame.copy()
        all_best_vertices = np.concatenate([shape.vertices_px for shape in best_shape_group], axis=0)

        for shape in best_shape_group:
            draw_bgr = COLOR_SPECS[shape.color_id].draw_bgr
            shape_points = np.round(shape.vertices_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(best_group_debug_frame, [shape_points], True, draw_bgr, 1)

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(all_best_vertices.astype(np.float32))
        cv2.rectangle(best_group_debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 2)
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
        cv2.rectangle(final_debug_frame, (bbox_x, bbox_y), (bbox_x_2, bbox_y_2), (0, 0, 255), 2)
        cv2.circle(final_debug_frame, (int(round(detection.u)), int(round(detection.v))), 5, (0, 0, 255), -1)
        cv2.putText(final_debug_frame, f"u={detection.u:.1f}, v={detection.v:.1f}, w={detection.px_w:.1f}, h={detection.px_h:.1f}", (bbox_x, max(20, bbox_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
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
                    "2D HSV + polygons": timing_hsv_polygon_seconds,
                "2D grouping + selection": timing_grouping_s,
                "2D detection total": timing_total_s,
            })
        if PRINT_SHAPE_DETECTION_TIMING:
            print(
                f"shape timing: model={timing_model_setup_s*1000.0:.1f} ms | resize+blur={timing_resize_blur_s*1000.0:.1f} ms | "
                f"LABprep={timing_lab_prep_s*1000.0:.1f} ms | HSVconv={timing_hsv_conversion_s*1000.0:.1f} ms | "
                f"LAB={timing_lab_seconds*1000.0:.1f} ms | HSV+polygon={timing_hsv_polygon_seconds*1000.0:.1f} ms | "
                f"grouping+select={timing_grouping_s*1000.0:.1f} ms | total={timing_total_s*1000.0:.1f} ms"
            )

    if debug is not None:
        detection._debug = debug
        debug.updateTimingStage()

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
def createMeasurementUsingShapeGroup(
    detection: Detection, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration,
    _timing_warmup: bool = False,
) -> Measurement:
    failed_measurement = Measurement(None, None, None, None, None, None,)
    measurement_debug = None if _timing_warmup else getattr(detection, "_debug", None)

    # Warm the PnP/flex path once before timing it in static-image debug mode.
    # Normal live calls have no debug object, so there is no extra solve in main.py.
    if measurement_debug is not None:
        createMeasurementUsingShapeGroup(
            detection, object_vision_spec, camera_calibration, _timing_warmup=True,
        )

    measurement_timing_start = time.perf_counter()

    if not detection.shapes:
        return failed_measurement

    # Precompute every physical polygon marker in the model. Marker identity is intentionally
    # plane-level rather than globally unique: different planes may reuse the same color/shape.
    plane_marker_specs = []
    marker_counts_by_plane = Counter()

    for rigid_plane in object_vision_spec.rigid_planes:
        for marker in rigid_plane.shape_markers:
            if marker.num_sides == 0:
                continue
            if marker.object_vertices_m is None:
                raise ValueError(f"Shape marker {(marker.color_id, marker.num_sides)} has no object_vertices_m")

            marker_vertices_plane_xy_m = np.asarray(marker.object_vertices_m, dtype=np.float64)
            num_vertices = marker.num_sides

            if marker_vertices_plane_xy_m.shape != (num_vertices, 2):
                raise ValueError(f"object_vertices_m for {(marker.color_id, marker.num_sides)} must have shape ({num_vertices}, 2)")
            if not np.all(np.isfinite(marker_vertices_plane_xy_m)):
                return failed_measurement

            marker_vertices_plane_m = np.column_stack((marker_vertices_plane_xy_m, np.zeros(num_vertices, dtype=np.float64)))
            marker_vertices_object_m = (rigid_plane.rotation_object_from_plane@marker_vertices_plane_m.T).T + rigid_plane.translation_object_from_plane_m

            edge_pairs = [(i, (i + 1)%num_vertices) for i in range(num_vertices)]
            object_edge_lengths = np.array([
                np.linalg.norm(marker_vertices_plane_xy_m[i] - marker_vertices_plane_xy_m[j])
                for i, j in edge_pairs
            ])
            object_shape_norm = np.linalg.norm(object_edge_lengths)

            if object_shape_norm <= 1e-12:
                raise ValueError(f"object_vertices_m for {(marker.color_id, marker.num_sides)} form a degenerate shape")
            if not np.all(np.isfinite(marker_vertices_object_m)):
                return failed_measurement

            plane_marker_specs.append((
                rigid_plane, marker, marker_vertices_object_m,
                object_edge_lengths/object_shape_norm,
            ))
            marker_counts_by_plane[rigid_plane.plane_id] += 1

    if not plane_marker_specs:
        return failed_measurement

    # Keep every cyclic/reversed vertex correspondence. 2D edge-length ratios are only
    # used as a cheap initial ordering; perspective can distort those ratios enough to pick
    # the wrong vertex mapping, so high-reprojection PnP fits are rescued below by testing
    # alternate orderings and letting reprojection error decide.
    marker_matches = {}

    for shape_index, shape in enumerate(detection.shapes):
        shape_vertices_px = np.asarray(shape.vertices_px, dtype=np.float64)

        if shape_vertices_px.shape != (shape.num_sides, 2) or not np.all(np.isfinite(shape_vertices_px)):
            return failed_measurement

        for marker_index, (_, marker, _, normalized_object_edge_lengths) in enumerate(plane_marker_specs):
            if shape.color_id != marker.color_id or shape.num_sides != marker.num_sides:
                continue

            num_vertices = marker.num_sides
            edge_pairs = [(i, (i + 1)%num_vertices) for i in range(num_vertices)]
            orderings = []

            # Polygon vertices arrive in perimeter order, so only cyclic shifts and reversed cyclic shifts are possible.
            for start_index in range(num_vertices):
                for direction in (1, -1):
                    vertex_order = [(start_index + direction*offset)%num_vertices for offset in range(num_vertices)]
                    ordered_vertices_px = shape_vertices_px[vertex_order]
                    image_edge_lengths = np.array([
                        np.linalg.norm(ordered_vertices_px[i] - ordered_vertices_px[j])
                        for i, j in edge_pairs
                    ])
                    image_shape_norm = np.linalg.norm(image_edge_lengths)

                    if image_shape_norm <= 1e-12:
                        continue

                    shape_error = float(np.linalg.norm(image_edge_lengths/image_shape_norm - normalized_object_edge_lengths))
                    orderings.append((ordered_vertices_px, shape_error))

            if orderings:
                orderings.sort(key=lambda item: item[1])
                marker_matches[(shape_index, marker_index)] = orderings

    if not marker_matches:
        return failed_measurement

    planes_by_id = {rigid_plane.plane_id: rigid_plane for rigid_plane in object_vision_spec.rigid_planes}
    marker_indices_by_plane = {
        plane_id: [
            marker_index for marker_index, (rigid_plane, _, _, _) in enumerate(plane_marker_specs)
            if rigid_plane.plane_id == plane_id
        ]
        for plane_id in planes_by_id
    }
    connections_by_pair = {
        frozenset((plane_id_1, plane_id_2)): (plane_id_1, plane_id_2, max_rotation_deg)
        for plane_id_1, plane_id_2, max_rotation_deg in object_vision_spec.rigid_plane_connections
    }

    # Try each plane independently and each configured hinge pair. If a model has no flexible
    # connections, also preserve the old behavior of allowing all rigid planes to participate together.
    candidate_plane_groups = [(plane_id,) for plane_id in planes_by_id]
    candidate_plane_groups += [(plane_id_1, plane_id_2) for plane_id_1, plane_id_2, _ in object_vision_spec.rigid_plane_connections]

    if not object_vision_spec.rigid_plane_connections and len(planes_by_id) > 1:
        candidate_plane_groups.append(tuple(planes_by_id))

    pnp_setup_seconds = time.perf_counter() - measurement_timing_start
    pnp_assignment_seconds = 0.0
    pnp_initial_search_seconds = 0.0
    pnp_rescue_seconds = 0.0

    best_result = None
    best_rank = None

    for plane_group in candidate_plane_groups:
        assignment_timing_start = time.perf_counter()
        group_marker_indices = [marker_index for plane_id in plane_group for marker_index in marker_indices_by_plane[plane_id]]
        if not group_marker_indices:
            continue

        # Find the largest one-to-one assignment between detected shapes and the marker multiset
        # on this plane/group. Partial marker visibility is allowed, but only after larger matches fail.
        best_assignment, best_assignment_shape_error = None, float("inf")
        maximum_match_count = min(len(detection.shapes), len(group_marker_indices))

        for match_count in range(maximum_match_count, 0, -1):
            for shape_indices in combinations(range(len(detection.shapes)), match_count):
                for selected_marker_indices in combinations(group_marker_indices, match_count):
                    for ordered_marker_indices in permutations(selected_marker_indices):
                        assignment = list(zip(shape_indices, ordered_marker_indices))

                        if any(pair not in marker_matches for pair in assignment):
                            continue

                        represented_plane_ids = {
                            plane_marker_specs[marker_index][0].plane_id
                            for _, marker_index in assignment
                        }

                        # A hinge candidate only counts as a two-plane observation if both sides are actually represented.
                        if len(plane_group) == 2 and len(represented_plane_ids) != 2:
                            continue

                        total_shape_error = sum(marker_matches[pair][0][1] for pair in assignment)
                        if total_shape_error < best_assignment_shape_error:
                            best_assignment = assignment
                            best_assignment_shape_error = total_shape_error

            if best_assignment is not None:
                break

        pnp_assignment_seconds += time.perf_counter() - assignment_timing_start

        if best_assignment is None:
            continue

        matched_counts_by_plane = Counter(
            plane_marker_specs[marker_index][0].plane_id
            for _, marker_index in best_assignment
        )
        matched_count = len(best_assignment)
        represented_plane_count = len(matched_counts_by_plane)
        full_plane_count = sum(
            matched_counts_by_plane[plane_id] == marker_counts_by_plane[plane_id]
            for plane_id in matched_counts_by_plane
        )
        total_group_marker_count = sum(marker_counts_by_plane[plane_id] for plane_id in plane_group)
        marker_coverage = matched_count/max(total_group_marker_count, 1)
        mean_shape_error = best_assignment_shape_error/matched_count

        connection = connections_by_pair.get(frozenset(plane_group)) if len(plane_group) == 2 else None
        hinge_point = hinge_direction = None
        coarse_angles_deg = np.array([0.0])

        if connection is not None:
            plane_id_1, plane_id_2, max_rotation_deg = connection
            hinge_point, hinge_direction = getRigidPlaneIntersection(planes_by_id[plane_id_1], planes_by_id[plane_id_2])
            coarse_angles_deg = (
                np.linspace(-max_rotation_deg, max_rotation_deg, 7)
                if max_rotation_deg > 0.0 else np.array([0.0])
            )

        def evaluateFlexAngle(
            flex_angle_deg: float, correspondence_indices: dict[tuple[int, int], int] | None = None,
            pnp_flag: int = cv2.SOLVEPNP_SQPNP,
        ) -> tuple[np.ndarray | None, float]:
            object_point_groups, image_point_groups = [], []
            rotation_1 = rotation_2 = None

            if connection is not None:
                half_angle_rad = np.deg2rad(flex_angle_deg/2.0)
                rotation_1, _ = cv2.Rodrigues(-half_angle_rad*hinge_direction)
                rotation_2, _ = cv2.Rodrigues(+half_angle_rad*hinge_direction)

            for shape_index, marker_index in best_assignment:
                rigid_plane, _, nominal_object_points, _ = plane_marker_specs[marker_index]
                object_points = nominal_object_points

                if connection is not None:
                    if rigid_plane.plane_id == plane_id_1:
                        object_points = (rotation_1@(object_points - hinge_point).T).T + hinge_point
                    elif rigid_plane.plane_id == plane_id_2:
                        object_points = (rotation_2@(object_points - hinge_point).T).T + hinge_point

                pair = (shape_index, marker_index)
                correspondence_index = 0 if correspondence_indices is None else correspondence_indices.get(pair, 0)
                object_point_groups.append(object_points)
                image_point_groups.append(marker_matches[pair][correspondence_index][0])

            object_points = np.concatenate(object_point_groups, axis=0)
            image_points = np.concatenate(image_point_groups, axis=0)

            # EPNP is used only as a fast ranking pass during correspondence rescue.
            # The final pose/flex result is always recomputed with SQPNP.
            if pnp_flag == cv2.SOLVEPNP_EPNP:
                if len(object_points) < 4:
                    return None, float("inf")
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points, image_points, camera_calibration.camera_matrix,
                    camera_calibration.distortion_coefficients, flags=pnp_flag,
                )
                if not success:
                    return None, float("inf")
                rotation_vectors, translation_vectors = [rotation_vector], [translation_vector]
            else:
                solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
                    object_points, image_points,
                    camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                    flags=pnp_flag,
                )
                if not solution_count:
                    return None, float("inf")

            best_translation, best_reprojection_error = None, float("inf")

            for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T

                if np.any(camera_points[:, 2] <= 0.0):
                    continue

                projected_points, _ = cv2.projectPoints(
                    object_points, rotation_vector, translation_vector,
                    camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                )
                reprojection_error = float(np.sqrt(np.mean(np.sum(
                    (projected_points.reshape(-1, 2) - image_points)**2, axis=1,
                ))))

                if reprojection_error < best_reprojection_error:
                    best_translation = translation_vector.reshape(3)
                    best_reprojection_error = reprojection_error

            return best_translation, best_reprojection_error

        correspondence_indices = {pair: 0 for pair in best_assignment}

        def searchFlex() -> tuple[np.ndarray | None, float, float]:
            best_translation_local, best_error_local, best_angle_local = None, float("inf"), 0.0

            for flex_angle_deg in coarse_angles_deg:
                translation, reprojection_error = evaluateFlexAngle(float(flex_angle_deg), correspondence_indices)
                if reprojection_error < best_error_local:
                    best_translation_local = translation
                    best_error_local = reprojection_error
                    best_angle_local = float(flex_angle_deg)

            if connection is not None and max_rotation_deg > 0.0 and best_translation_local is not None:
                coarse_step_deg = 2.0*max_rotation_deg/6.0
                fine_min = max(-max_rotation_deg, best_angle_local - coarse_step_deg)
                fine_max = min(+max_rotation_deg, best_angle_local + coarse_step_deg)
                fine_angles_deg = np.unique(np.concatenate((
                    np.arange(fine_min, fine_max + 0.5, 1.0),
                    np.array([fine_min, best_angle_local, fine_max]),
                )))

                for flex_angle_deg in fine_angles_deg:
                    translation, reprojection_error = evaluateFlexAngle(float(flex_angle_deg), correspondence_indices)
                    if reprojection_error < best_error_local:
                        best_translation_local = translation
                        best_error_local = reprojection_error
                        best_angle_local = float(flex_angle_deg)

            return best_translation_local, best_error_local, best_angle_local

        initial_search_timing_start = time.perf_counter()
        best_translation, best_reprojection_error, best_flex_angle_deg = searchFlex()
        pnp_initial_search_seconds += time.perf_counter() - initial_search_timing_start

        # If the cheap edge-ratio ordering produced a poor fit, let actual pose reprojection
        # choose the correspondence. Use one detected marker as a PnP anchor, try each of its
        # cyclic/reversed orderings over the coarse flex angles, then project every other marker
        # and choose the ordering that lands closest to that pose. This avoids an exponential
        # Cartesian search over all marker orderings.
        CORRESPONDENCE_RESCUE_ERROR_PX = 8.0
        if best_translation is not None and best_reprojection_error > CORRESPONDENCE_RESCUE_ERROR_PX and len(best_assignment) >= 2:
            rescue_timing_start = time.perf_counter()
            anchor_pair = max(best_assignment, key=lambda pair: plane_marker_specs[pair[1]][1].num_sides)
            best_rescue_indices = correspondence_indices.copy()
            best_rescue_error = float("inf")
            best_rescue_angle_deg = best_flex_angle_deg

            for rescue_angle_deg in coarse_angles_deg:
                rotation_1 = rotation_2 = None
                if connection is not None:
                    half_angle_rad = np.deg2rad(float(rescue_angle_deg)/2.0)
                    rotation_1, _ = cv2.Rodrigues(-half_angle_rad*hinge_direction)
                    rotation_2, _ = cv2.Rodrigues(+half_angle_rad*hinge_direction)

                object_points_by_pair = {}
                for pair in best_assignment:
                    _, marker_index = pair
                    rigid_plane, _, nominal_object_points, _ = plane_marker_specs[marker_index]
                    object_points = nominal_object_points

                    if connection is not None:
                        if rigid_plane.plane_id == plane_id_1:
                            object_points = (rotation_1@(object_points - hinge_point).T).T + hinge_point
                        elif rigid_plane.plane_id == plane_id_2:
                            object_points = (rotation_2@(object_points - hinge_point).T).T + hinge_point

                    object_points_by_pair[pair] = object_points

                anchor_object_points = object_points_by_pair[anchor_pair]

                for anchor_ordering_index, (anchor_image_points, _) in enumerate(marker_matches[anchor_pair]):
                    if len(anchor_object_points) == 3:
                        solution_count, rotation_vectors, translation_vectors = cv2.solveP3P(
                            anchor_object_points, anchor_image_points,
                            camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                            flags=cv2.SOLVEPNP_AP3P,
                        )
                    else:
                        solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
                            anchor_object_points, anchor_image_points,
                            camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                            flags=cv2.SOLVEPNP_SQPNP,
                        )

                    if not solution_count:
                        continue

                    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                        trial_indices = {anchor_pair: anchor_ordering_index}
                        total_squared_error = 0.0
                        total_point_count = 0
                        valid_pose = True

                        for pair in best_assignment:
                            object_points = object_points_by_pair[pair]
                            camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T
                            if np.any(camera_points[:, 2] <= 0.0):
                                valid_pose = False
                                break

                            projected_points, _ = cv2.projectPoints(
                                object_points, rotation_vector, translation_vector,
                                camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
                            )
                            projected_points = projected_points.reshape(-1, 2)

                            if pair == anchor_pair:
                                ordering_index = anchor_ordering_index
                                squared_error = float(np.sum((anchor_image_points - projected_points)**2))
                            else:
                                ordering_index, squared_error = min(
                                    (
                                        (ordering_index, float(np.sum((ordered_vertices_px - projected_points)**2)))
                                        for ordering_index, (ordered_vertices_px, _) in enumerate(marker_matches[pair])
                                    ),
                                    key=lambda item: item[1],
                                )

                            trial_indices[pair] = ordering_index
                            total_squared_error += squared_error
                            total_point_count += len(object_points)

                        if not valid_pose or total_point_count == 0:
                            continue

                        rescue_error = float(np.sqrt(total_squared_error/total_point_count))
                        if rescue_error < best_rescue_error:
                            best_rescue_error = rescue_error
                            best_rescue_indices = trial_indices
                            best_rescue_angle_deg = float(rescue_angle_deg)

            if best_rescue_indices != correspondence_indices:
                correspondence_indices = best_rescue_indices

                # The anchor search already identified the best coarse flex bin, so only
                # run accurate joint SQPNP locally around it rather than repeating the full grid.
                local_angles_deg = np.array([best_rescue_angle_deg])
                if connection is not None and max_rotation_deg > 0.0:
                    coarse_step_deg = 2.0*max_rotation_deg/6.0
                    fine_min = max(-max_rotation_deg, best_rescue_angle_deg - coarse_step_deg)
                    fine_max = min(+max_rotation_deg, best_rescue_angle_deg + coarse_step_deg)
                    local_angles_deg = np.unique(np.concatenate((
                        np.arange(fine_min, fine_max + 0.5, 1.0),
                        np.array([fine_min, best_rescue_angle_deg, fine_max]),
                    )))

                rescued_translation, rescued_error, rescued_angle_deg = None, float("inf"), best_rescue_angle_deg
                for flex_angle_deg in local_angles_deg:
                    translation, reprojection_error = evaluateFlexAngle(float(flex_angle_deg), correspondence_indices)
                    if reprojection_error < rescued_error:
                        rescued_translation = translation
                        rescued_error = reprojection_error
                        rescued_angle_deg = float(flex_angle_deg)

                if rescued_translation is not None and rescued_error < best_reprojection_error:
                    best_translation = rescued_translation
                    best_reprojection_error = rescued_error
                    best_flex_angle_deg = rescued_angle_deg

            pnp_rescue_seconds += time.perf_counter() - rescue_timing_start

        if best_translation is None or not np.all(np.isfinite(best_translation)):
            continue

        # Ordered criteria avoid arbitrary weighted coefficients:
        #   1) more matched markers, 2) more fully observed planes, 3) greater marker-set
        #   completeness, 4) evidence from more planes, then geometric fit quality.
        candidate_rank = (
            matched_count,
            full_plane_count,
            marker_coverage,
            represented_plane_count,
            -best_reprojection_error,
            -mean_shape_error,
        )

        if best_rank is None or candidate_rank > best_rank:
            best_rank = candidate_rank
            best_result = (
                best_translation, best_reprojection_error, best_flex_angle_deg,
                connection, matched_count, total_group_marker_count,
            )

    if best_result is None:
        return failed_measurement

    best_translation, best_reprojection_error, best_flex_angle_deg, best_connection, matched_count, total_group_marker_count = best_result

    if best_connection is not None and not _timing_warmup:
        plane_id_1, plane_id_2, _ = best_connection
        print(
            f"flex {plane_id_1}<->{plane_id_2}: {best_flex_angle_deg:+.1f} deg | "
            f"markers={matched_count}/{total_group_marker_count} | reprojection={best_reprojection_error:.2f} px"
        )

    pnp_total_seconds = time.perf_counter() - measurement_timing_start
    if measurement_debug is not None:
        measurement_debug.setTiming("PnP setup + correspondences", pnp_setup_seconds)
        measurement_debug.setTiming("PnP marker assignment", pnp_assignment_seconds)
        measurement_debug.setTiming("PnP initial flex search", pnp_initial_search_seconds)
        measurement_debug.setTiming("PnP correspondence rescue", pnp_rescue_seconds)
        measurement_debug.setTiming("PnP total", pnp_total_seconds)
        detection_total_ms = measurement_debug.timings_ms.get("2D detection total")
        if detection_total_ms is not None:
            measurement_debug.timings_ms["TOTAL vision"] = detection_total_ms + 1000.0*pnp_total_seconds
        measurement_debug.updateTimingStage()

    return Measurement(float(best_translation[0]), float(best_translation[1]), float(best_translation[2]), None, None, None,)

