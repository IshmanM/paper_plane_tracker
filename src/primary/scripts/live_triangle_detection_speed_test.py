import argparse
import time
from collections import deque

import cv2
import numpy as np

from src.primary.config import FRAME_W, FRAME_H
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType
from src.primary.detection import findSingleObjectUsingBestTriangleGroup, createMeasurementUsingTriangleGroup


WINDOW_NAME = "Live triangle detection speed test"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_FPS = 60
DEFAULT_TIMING_WINDOW_FRAMES = 120


def drawDetection(frame: np.ndarray, detection) -> None:
    if detection is None:
        return

    # Draw each detected marker triangle, followed by the resulting group bounds and center.
    for triangle in detection.triangles:
        vertices_px = np.rint(triangle.vertices_px).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [vertices_px], True, (255, 255, 255), 2, cv2.LINE_AA)

    top_left = (int(round(detection.u - detection.px_w/2)), int(round(detection.v - detection.px_h/2)))
    bottom_right = (int(round(detection.u + detection.px_w/2)), int(round(detection.v + detection.px_h/2)))
    center = (int(round(detection.u)), int(round(detection.v)))
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
    cv2.circle(frame, center, 4, (0, 0, 255), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live paper-plane triangle detection and measure its speed.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index. Default: {DEFAULT_CAMERA_INDEX}")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
                        help=f"Requested camera FPS. Default: {DEFAULT_CAMERA_FPS}")
    parser.add_argument("--timing-window", type=int, default=DEFAULT_TIMING_WINDOW_FRAMES,
                        help=f"Frames used for rolling timing averages. Default: {DEFAULT_TIMING_WINDOW_FRAMES}")
    args = parser.parse_args()

    if args.camera_fps < 1:
        parser.error("--camera-fps must be at least 1.")
    if args.timing_window < 1:
        parser.error("--timing-window must be at least 1.")

    # Configure the camera to match the resolution expected by the detector.
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    object_vision_spec = OBJECT_VISION_SPECS[ObjectType.PAPER_PLANE_TRIANGLES]
    detection_times_s = deque(maxlen=args.timing_window)
    loop_periods_s = deque(maxlen=args.timing_window)
    previous_loop_start_s = None

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"Camera: {actual_width}x{actual_height} at reported {actual_fps:.1f} FPS")
    print("Press Q or Esc to quit.")

    try:
        while True:
            loop_start_s = time.perf_counter()

            if previous_loop_start_s is not None:
                loop_periods_s.append(loop_start_s - previous_loop_start_s)

            previous_loop_start_s = loop_start_s

            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            frame_height, frame_width = frame.shape[:2]

            if frame_width != FRAME_W or frame_height != FRAME_H:
                raise ValueError(
                    f"Camera produced {frame_width}x{frame_height}, "
                    f"but config expects {FRAME_W}x{FRAME_H}."
                )

            # Time only triangle detection; capture, measurement, drawing, and display are excluded.
            detection_start_s = time.perf_counter()
            detection = findSingleObjectUsingBestTriangleGroup(frame, object_vision_spec)
            detection_times_s.append(time.perf_counter() - detection_start_s)

            measurement = (
                createMeasurementUsingTriangleGroup(detection, object_vision_spec)
                if detection is not None else None
            )

            drawDetection(frame, detection)

            average_detection_s = sum(detection_times_s)/len(detection_times_s)
            detector_rate_hz = 1.0/average_detection_s if average_detection_s > 0.0 else 0.0
            live_fps = 1.0/(sum(loop_periods_s)/len(loop_periods_s)) if loop_periods_s else 0.0
            triangle_count = len(detection.triangles) if detection is not None else 0

            cv2.putText(
                frame,
                f"Detection: {average_detection_s*1000.0:.2f} ms | detector rate: {detector_rate_hz:.1f} Hz",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Live loop: {live_fps:.1f} FPS | triangles: {triangle_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA,
            )

            if measurement is not None and measurement.x is not None and measurement.y is not None and measurement.z is not None:
                measurement_text = f"x: {measurement.x:.3f} m | y: {measurement.y:.3f} m | z: {measurement.z:.3f} m"
            else:
                measurement_text = "Measurement: unavailable"

            cv2.putText(
                frame, measurement_text,
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()