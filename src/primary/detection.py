import cv2
import numpy as np
from collections import Counter
from itertools import combinations

from src.primary.camera_calibration import CameraCalibration
from src.primary.geometry import estimateObjectWorldPosition
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec, ObjectVisionSpecId
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
    detection = findSingleObjectSphere(frame, object_vision_spec,)
    # detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_vision_spec.width, camera_calibration)
    measurement = Measurement(x, y, z, None, None, None,)
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

    return True, detection, measurement


def failedDetectionResult(detection: Detection | None = None) -> tuple[bool, Detection, Measurement]:
    if detection is None:
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


def findSingleObjectSphere(frame: np.ndarray, object_vision_spec: ObjectVisionSpec, debug: DetectionDebug | None = None) -> Detection | None:
    if not object_vision_spec.color_ids:
        raise ValueError("Sphere detection requires at least one color_id")
    if object_vision_spec.minimum_contour_area_px is None:
        raise ValueError("Sphere detection requires minimum_contour_area_px")

    MAX_SPHERE_CANDIDATES = 2
    NUM_RAYS = 120
    NUM_ANGLE_BINS = 12
    MIN_COVERED_ANGLE_BINS = 6
    MIN_BOUNDARY_POINTS = 20
    LAB_CHROMA_GRADIENT_GAIN = 2.0
    MIN_LAB_EDGE_STRENGTH = 35.0

    # Step 1: Build an HSV mask containing rough sphere candidates.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower_hsv, upper_hsv))

    if debug is not None:
        debug.stages.clear()
        debug.addStage("Original", frame)
        debug.addStage("HSV seed mask", mask)

    # Step 2: Find connected HSV regions without extracting full contours.
    # Connected-component statistics give candidate area and bounding boxes in
    # one full-mask pass; exact contours are extracted only inside shortlisted ROIs.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < object_vision_spec.minimum_contour_area_px:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        candidates.append((float(area), int(label), x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    candidates = candidates[:MAX_SPHERE_CANDIDATES]

    if debug is not None:
        candidate_frame = frame.copy()

        for candidate_index, (area, _, x, y, w, h) in enumerate(candidates, start=1):
            cv2.rectangle(candidate_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(candidate_frame, f"{candidate_index}: area={area:.0f}", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        debug.addStage("HSV candidates", candidate_frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angles = np.linspace(0.0, 2.0*np.pi, NUM_RAYS, endpoint=False)
    directions_u, directions_v = np.cos(angles)[:, None], np.sin(angles)[:, None]

    best_result = None
    best_final_score = -np.inf

    # Step 3: Extract and refine the exact contour only inside each shortlisted ROI.
    for candidate_index, (seed_area, label, x, y, w, h) in enumerate(candidates, start=1):
        component_mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8)*255
        local_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not local_contours:
            continue

        seed_contour = max(local_contours, key=cv2.contourArea)
        seed_contour = seed_contour + np.array([[[x, y]]], dtype=seed_contour.dtype)

        moments = cv2.moments(seed_contour)

        if moments["m00"] == 0:
            continue

        center_u = moments["m10"]/moments["m00"]
        center_v = moments["m01"]/moments["m00"]
        seed_size = max(w, h)

        # Step 4: Create a tight ROI and compute a combined LAB edge-strength image.
        # A 3x3 blur preserves small/far-away sphere boundaries better than the old
        # 5x5 grayscale blur. L contributes brightness edges while a/b add color edges.
        padding = max(8, int(0.35*seed_size))
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)

        color_roi = frame[y1:y2, x1:x2]

        # A malformed/noisy candidate should never crash the entire live detector.
        if color_roi.size == 0 or color_roi.shape[0] < 2 or color_roi.shape[1] < 2:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, f"REJECTED: invalid ROI {color_roi.shape}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - invalid ROI", failure_frame)
            continue

        color_roi = np.ascontiguousarray(color_roi)
        lab_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2LAB)
        l_roi, a_roi, b_roi = cv2.split(lab_roi)

        l_roi = cv2.GaussianBlur(l_roi, (3, 3), 0).astype(np.float32)
        a_roi = cv2.GaussianBlur(a_roi, (3, 3), 0).astype(np.float32)
        b_roi = cv2.GaussianBlur(b_roi, (3, 3), 0).astype(np.float32)

        grad_lu = cv2.Sobel(l_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_lv = cv2.Sobel(l_roi, cv2.CV_32F, 0, 1, ksize=3)
        grad_au = cv2.Sobel(a_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_av = cv2.Sobel(a_roi, cv2.CV_32F, 0, 1, ksize=3)
        grad_bu = cv2.Sobel(b_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_bv = cv2.Sobel(b_roi, cv2.CV_32F, 0, 1, ksize=3)

        lab_edge_strength = np.sqrt(
            grad_lu*grad_lu + grad_lv*grad_lv +
            LAB_CHROMA_GRADIENT_GAIN*(grad_au*grad_au + grad_av*grad_av + grad_bu*grad_bu + grad_bv*grad_bv)
        )

        seed_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        contour_roi = seed_contour - np.array([[[x1, y1]]], dtype=seed_contour.dtype)
        cv2.drawContours(seed_roi, [contour_roi], -1, 255, -1)

        if debug is not None:
            roi_frame = frame.copy()
            cv2.drawContours(roi_frame, [seed_contour], -1, (0, 255, 255), 1)
            cv2.rectangle(roi_frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
            debug.addStage(f"Candidate {candidate_index} ROI", roi_frame)

            edge_strength_debug = cv2.normalize(lab_edge_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            debug.addStage(f"Candidate {candidate_index} LAB edge strength", edge_strength_debug)

        # Step 5: Generate radial samples around the candidate center.
        center_roi_u, center_roi_v = center_u - x1, center_v - y1
        radii = np.arange(1, max(1, int(seed_size)) + 1)
        radius_grid = np.broadcast_to(radii, (NUM_RAYS, len(radii)))

        sample_u = np.rint(center_roi_u + directions_u*radii).astype(np.int32)
        sample_v = np.rint(center_roi_v + directions_v*radii).astype(np.int32)

        valid = (
            (sample_u >= 0) & (sample_u < seed_roi.shape[1]) &
            (sample_v >= 0) & (sample_v < seed_roi.shape[0])
        )

        safe_u = np.clip(sample_u, 0, seed_roi.shape[1] - 1)
        safe_v = np.clip(sample_v, 0, seed_roi.shape[0] - 1)

        # Step 6: Estimate the outer HSV boundary independently along each ray.
        seed_hits = (seed_roi[safe_v, safe_u] != 0) & valid
        expected_radii = np.where(seed_hits, radius_grid, 0).max(axis=1)

        # Step 7: Refine the approximate HSV boundary using the strongest nearby
        # combined LAB gradient on each ray. Using the strongest gradient avoids the
        # inward-radius bias that can occur when a thick binary edge band is sampled
        # by simply taking its first pixel.
        search_before = 3
        search_after = np.maximum(5, (0.20*expected_radii).astype(np.int32))

        search_band = (
            (radius_grid >= (expected_radii - search_before)[:, None]) &
            (radius_grid <= (expected_radii + search_after)[:, None]) &
            (expected_radii[:, None] > 0) &
            valid
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

        # Step 8: Fit the initial circle.
        A = np.column_stack((2*boundary_points[:, 0], 2*boundary_points[:, 1], np.ones(len(boundary_points))))
        b = boundary_points[:, 0]**2 + boundary_points[:, 1]**2
        circle_u, circle_v, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(max(0.0, c + circle_u**2 + circle_v**2))

        if radius <= 0.0:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, "REJECTED: invalid initial circle radius", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - initial circle", failure_frame)
            continue

        # Step 9: Remove inconsistent boundary points and refit.
        point_radii = np.hypot(boundary_points[:, 0] - circle_u, boundary_points[:, 1] - circle_v)
        residuals = np.abs(point_radii - radius)
        residual_limit = max(2.0, 2.5*np.median(residuals))
        inlier_points = boundary_points[residuals <= residual_limit]

        if len(inlier_points) < MIN_BOUNDARY_POINTS:
            if debug is not None:
                failure_frame = frame.copy()

                for point_u, point_v in boundary_points:
                    cv2.circle(failure_frame, (int(round(point_u)), int(round(point_v))), 2, (100, 100, 100), -1)

                for point_u, point_v in inlier_points:
                    cv2.circle(failure_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

                cv2.putText(failure_frame, f"REJECTED: circle inliers {len(inlier_points)}/{len(boundary_points)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"Residual limit: {residual_limit:.2f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - circle inliers", failure_frame)

            continue

        A = np.column_stack((2*inlier_points[:, 0], 2*inlier_points[:, 1], np.ones(len(inlier_points))))
        b = inlier_points[:, 0]**2 + inlier_points[:, 1]**2
        circle_u, circle_v, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(max(0.0, c + circle_u**2 + circle_v**2))

        if radius <= 0.0:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.putText(failure_frame, "REJECTED: invalid refined circle radius", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - refined circle", failure_frame)
            continue

        # Step 10: Require boundary evidence around most of the fitted circle.
        point_angles = np.arctan2(inlier_points[:, 1] - circle_v, inlier_points[:, 0] - circle_u)
        angle_bins = (((point_angles + np.pi)/(2.0*np.pi))*NUM_ANGLE_BINS).astype(np.int32) % NUM_ANGLE_BINS
        covered_angle_bins = len(np.unique(angle_bins))

        if covered_angle_bins < MIN_COVERED_ANGLE_BINS:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.circle(failure_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 0, 255), 2)
                cv2.putText(failure_frame, f"REJECTED: angular coverage {covered_angle_bins}/{NUM_ANGLE_BINS}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                debug.addStage(f"Candidate {candidate_index} rejected - angular coverage", failure_frame)
            continue

        
        # Step 11: Require the refined circle to remain reasonably consistent
        # with the rough size and center estimated from the HSV candidate.
        seed_radius = np.sqrt(seed_area/np.pi)
        center_displacement = np.hypot(circle_u - center_u, circle_v - center_v)

        if radius < 0.70*seed_radius or radius > 1.40*seed_radius:
            if debug is not None:
                failure_frame = frame.copy()
                cv2.circle(failure_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 0, 255), 2)

                cv2.putText(failure_frame, f"REJECTED: radius {radius:.1f}px", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"HSV seed radius: {seed_radius:.1f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"Allowed: {0.70*seed_radius:.1f}-{1.40*seed_radius:.1f}px", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                debug.addStage(f"Candidate {candidate_index} rejected - radius", failure_frame)

            continue

        if center_displacement > 0.40*radius:
            if debug is not None:
                failure_frame = frame.copy()

                cv2.circle(failure_frame, (int(round(center_u)), int(round(center_v))), 4, (0, 255, 255), -1)
                cv2.circle(failure_frame, (int(round(circle_u)), int(round(circle_v))), 4, (0, 0, 255), -1)
                cv2.line(
                    failure_frame,
                    (int(round(center_u)), int(round(center_v))),
                    (int(round(circle_u)), int(round(circle_v))),
                    (0, 0, 255), 1,
                )

                cv2.putText(failure_frame, f"REJECTED: center shift {center_displacement:.1f}px", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(failure_frame, f"Maximum: {0.40*radius:.1f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                debug.addStage(f"Candidate {candidate_index} rejected - center shift", failure_frame)

            continue

        # Step 12: Score candidates that passed every geometric validation check.
        final_point_radii = np.hypot(inlier_points[:, 0] - circle_u, inlier_points[:, 1] - circle_v)
        mean_residual = np.mean(np.abs(final_point_radii - radius))

        coverage_score = covered_angle_bins/NUM_ANGLE_BINS
        support_score = len(inlier_points)/NUM_RAYS
        residual_score = 1.0/(1.0 + mean_residual/max(radius, 1.0))
        hsv_fill_ratio = min(1.0, seed_area/(np.pi*radius*radius))

        final_score = (
            0.40*coverage_score +
            0.30*support_score +
            0.20*residual_score +
            0.10*hsv_fill_ratio
        )

        # Reaching this point means the candidate passed every rejection gate.
        if debug is not None:
            passed_frame = frame.copy()

            for point_u, point_v in inlier_points:
                cv2.circle(passed_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

            cv2.circle(
                passed_frame,
                (int(round(circle_u)), int(round(circle_v))),
                int(round(radius)),
                (0, 255, 0), 2,
            )

            cv2.putText(passed_frame, f"PASSED candidate {candidate_index}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(passed_frame, f"score: {final_score:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                passed_frame,
                f"boundary={len(boundary_points)} | inliers={len(inlier_points)} | coverage={covered_angle_bins}/{NUM_ANGLE_BINS}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA,
            )

            debug.addStage(f"Candidate {candidate_index} passed", passed_frame)

        if final_score > best_final_score:
            best_final_score = final_score
            best_result = (
                float(circle_u), float(circle_v), float(radius),
                inlier_points, covered_angle_bins, float(final_score),
            )

    if best_result is None:
        return None

    # Step 13: Build the final Detection from the best validated sphere candidate.
    circle_u, circle_v, radius, inlier_points, covered_angle_bins, final_score = best_result
    diameter = 2.0*radius
    ellipse_px = ((circle_u, circle_v), (diameter, diameter), 0.0)

    shape = ShapeDetection(vertices_px=None, color_id=object_vision_spec.color_ids[0], num_sides=0, ellipse_px=ellipse_px)
    detection = Detection(u=circle_u, v=circle_v, px_w=diameter, px_h=diameter, shapes=[shape])

    if debug is not None:
        success_frame = frame.copy()

        for point_u, point_v in inlier_points:
            cv2.circle(success_frame, (int(round(point_u)), int(round(point_v))), 2, (255, 0, 255), -1)

        cv2.circle(success_frame, (int(round(circle_u)), int(round(circle_v))), int(round(radius)), (0, 255, 0), 2)

        cv2.putText(success_frame, f"PASSED candidate {candidate_index}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(success_frame, f"inliers={len(inlier_points)} | coverage={covered_angle_bins}/{NUM_ANGLE_BINS}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        debug.addStage(f"Candidate {candidate_index} passed", success_frame)

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

        # Poorly supported lines are too risky to extrapolate into a corner.
        if edge_fit_error > 1.0:
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

        cleaned_mask = cv2.medianBlur(raw_mask, 3)
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

            # Current polygon detection assumes convex markers. The convex hull bridges small contour
            # defects but intentionally removes concavities. TODO: handle concave markers separately if needed.
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            base_polygon = cv2.approxPolyDP(hull, object_vision_spec.polygon_epsilon_ratio*perimeter, True)

            if polygon_debug_frame is not None:
                cv2.polylines(polygon_debug_frame, [base_polygon], True, draw_bgr, 1)
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
                # Concave polygon markers are currently unsupported.
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
                    cv2.putText(candidate_debug_frame, f"S{shape_index}: {color_name}, N={num_sides}", (int(center_px[0]) + 5, int(center_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_bgr, 1, cv2.LINE_AA)

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
                cv2.polylines(group_debug_frame, [shape_points], True, draw_bgr, 1)

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
def createMeasurementUsingShapeGroup(detection: Detection, object_vision_spec: ObjectVisionSpec, camera_calibration: CameraCalibration) -> Measurement:
    failed_measurement = Measurement(None, None, None, None, None, None,)

    if not detection.shapes:
        return failed_measurement

    # Keep each marker paired with the rigid plane that defines its local-to-object transform.
    plane_marker_specs = [
        (rigid_plane, marker)
        for rigid_plane in object_vision_spec.rigid_planes
        for marker in rigid_plane.shape_markers
        if marker.num_sides != 0
    ]
    marker_indices_by_key: dict[tuple[ColorId, int], list[int]] = {}

    for marker_index, (_, marker) in enumerate(plane_marker_specs):
        marker_indices_by_key.setdefault((marker.color_id, marker.num_sides), []).append(marker_index)

    object_point_groups, image_point_groups, used_marker_indices = [], [], set()

    # Match each detected polygon to its physical marker and determine vertex correspondence.
    for shape in detection.shapes:
        marker_key = (shape.color_id, shape.num_sides)
        available_marker_indices = [
            marker_index for marker_index in marker_indices_by_key.get(marker_key, [])
            if marker_index not in used_marker_indices
        ]

        if len(available_marker_indices) != 1:
            raise ValueError(f"Detected marker {marker_key} must match exactly one unused shape marker")

        marker_index = available_marker_indices[0]
        rigid_plane, marker = plane_marker_specs[marker_index]

        if marker.object_vertices_m is None:
            raise ValueError(f"Shape marker {marker_key} has no object_vertices_m")

        marker_vertices_plane_xy_m = np.asarray(marker.object_vertices_m, dtype=np.float64)
        shape_vertices_px = np.asarray(shape.vertices_px, dtype=np.float64)
        num_vertices = marker.num_sides

        if marker_vertices_plane_xy_m.shape != (num_vertices, 2):
            raise ValueError(f"object_vertices_m for {marker_key} must have shape ({num_vertices}, 2)")
        if shape_vertices_px.shape != (num_vertices, 2):
            raise ValueError(f"vertices_px for {marker_key} must have shape ({num_vertices}, 2)")
        if not np.all(np.isfinite(marker_vertices_plane_xy_m)) or not np.all(np.isfinite(shape_vertices_px)):
            return failed_measurement

        # Marker vertices are plane-local (x, y), with local z=0. Rotate first, then translate into the common object/reference frame.
        marker_vertices_plane_m = np.column_stack((marker_vertices_plane_xy_m, np.zeros(num_vertices, dtype=np.float64)))
        marker_vertices_object_m = (rigid_plane.rotation_object_from_plane@marker_vertices_plane_m.T).T + rigid_plane.translation_object_from_plane_m

        if not np.all(np.isfinite(marker_vertices_object_m)):
            return failed_measurement

        edge_pairs = [(i, (i + 1)%num_vertices) for i in range(num_vertices)]
        object_edge_lengths = np.array([np.linalg.norm(marker_vertices_plane_xy_m[i] - marker_vertices_plane_xy_m[j]) for i, j in edge_pairs])
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

        object_point_groups.append(marker_vertices_object_m)
        image_point_groups.append(best_vertices_px)
        used_marker_indices.add(marker_index)

    # All visible markers are now expressed in one common object frame, including each rigid plane's R and t.
    object_points = np.concatenate(object_point_groups, axis=0)
    image_points = np.concatenate(image_point_groups, axis=0)

    solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_calibration.camera_matrix, camera_calibration.distortion_coefficients, flags=cv2.SOLVEPNP_SQPNP,
    )

    if not solution_count:
        return failed_measurement

    # FOR DEBUG ONLY
    # print(f"solutions: {solution_count}")

    # for i, (rotation_vector, translation_vector) in enumerate(zip(rotation_vectors, translation_vectors)):
    #     projected_points, _ = cv2.projectPoints(
    #         object_points, rotation_vector, translation_vector,
    #         camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
    #     )
    #     reprojection_error = float(np.sqrt(np.mean(np.sum(
    #         (projected_points.reshape(-1, 2) - image_points)**2, axis=1,
    #     ))))
    #     print(
    #         f"{i}: x={translation_vector[0, 0]:.3f}, "
    #         f"y={translation_vector[1, 0]:.3f}, "
    #         f"z={translation_vector[2, 0]:.3f}, "
    #         f"error={reprojection_error:.17f}"
    #     )

    best_translation, best_reprojection_error = None, float("inf")

    # PnP translation is the common object-frame origin in camera coordinates.
    # Reject poses behind the camera and select the solution that best reproduces the detected vertices.
    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        camera_points = (rotation_matrix@object_points.T + translation_vector.reshape(3, 1)).T

        if np.any(camera_points[:, 2] <= 0.0):
            continue

        projected_points, _ = cv2.projectPoints(
            object_points, rotation_vector, translation_vector, camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
        )
        reprojection_error = float(np.sqrt(np.mean(np.sum((projected_points.reshape(-1, 2) - image_points)**2, axis=1))))

        if reprojection_error < best_reprojection_error:
            best_translation, best_reprojection_error = translation_vector.reshape(3), reprojection_error

    if best_translation is None or not np.all(np.isfinite(best_translation)):
        return failed_measurement

    return Measurement(float(best_translation[0]), float(best_translation[1]), float(best_translation[2]), None, None, None,)