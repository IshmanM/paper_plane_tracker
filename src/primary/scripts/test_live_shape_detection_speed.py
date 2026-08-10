import argparse
import time
from collections import deque

import cv2
import numpy as np

import src.primary.config as config
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpecId
from src.primary.detection import findSingleObjectUsingBestShapeGroup, createMeasurementUsingShapeGroup, drawModelOrigin


WINDOW_NAME = "Live shape detection speed test"
DEFAULT_CAMERA_INDEX = config.CAMERA_INDEX
DEFAULT_CAMERA_FPS = config.FPS
DEFAULT_TIMING_WINDOW_FRAMES = 120

DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.PAPER_PLANE_SHAPES_1
PAPER_PLANE_SHAPE_SPEC_IDS = [
    spec_id for spec_id, spec in OBJECT_VISION_SPECS.items()
    if spec.object_type == ObjectType.PAPER_PLANE_SHAPES
]


def drawDetection(frame: np.ndarray, detection) -> None:
    if detection is None:
        return

    # Draw each detected marker shape, followed by the resulting group bounds and bbox center.
    for shape in detection.shapes:
        vertices_px = np.rint(shape.vertices_px).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [vertices_px], True, (255, 255, 255), 1, cv2.LINE_AA)

    top_left = (int(round(detection.u - detection.px_w/2)), int(round(detection.v - detection.px_h/2)))
    bottom_right = (int(round(detection.u + detection.px_w/2)), int(round(detection.v + detection.px_h/2)))
    center = (int(round(detection.u)), int(round(detection.v)))
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
    cv2.circle(frame, center, 4, (0, 0, 255), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live paper-plane shape detection and measure its speed.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index. Default: {DEFAULT_CAMERA_INDEX}")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
                        help=f"Requested camera FPS. Default: {DEFAULT_CAMERA_FPS}")
    parser.add_argument("--timing-window", type=int, default=DEFAULT_TIMING_WINDOW_FRAMES,
                        help=f"Frames used for rolling timing averages. Default: {DEFAULT_TIMING_WINDOW_FRAMES}")
    parser.add_argument(
        "--spec", default=DEFAULT_OBJECT_VISION_SPEC_ID.name,
        choices=[spec_id.name for spec_id in PAPER_PLANE_SHAPE_SPEC_IDS],
        help=f"Registered paper-plane ObjectVisionSpecId. Default: {DEFAULT_OBJECT_VISION_SPEC_ID.name}",
    )
    args = parser.parse_args()

    if args.camera_fps < 1:
        parser.error("--camera-fps must be at least 1.")
    if args.timing_window < 1:
        parser.error("--timing-window must be at least 1.")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Set auto exposure:", camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25))
    if not config.CAMERA_AUTO_EXPOSURE:
        print("Set exposure:", camera.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE))
        print("Set gain:", camera.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN))

    print("Set auto WB:", camera.set(cv2.CAP_PROP_AUTO_WB, 1 if config.CAMERA_AUTO_WHITE_BALANCE else 0))
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        print("Set WB temperature:", camera.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAMERA_WHITE_BALANCE_TEMPERATURE))

    print("Backend:", camera.getBackendName())
    print("Auto exposure:", camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("Exposure:", camera.get(cv2.CAP_PROP_EXPOSURE))
    print("Gain:", camera.get(cv2.CAP_PROP_GAIN))
    print("Auto WB:", camera.get(cv2.CAP_PROP_AUTO_WB))
    print("WB temperature:", camera.get(cv2.CAP_PROP_WB_TEMPERATURE))

    object_vision_spec_id = ObjectVisionSpecId[args.spec]
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]
    detection_times_s = deque(maxlen=args.timing_window)
    loop_periods_s = deque(maxlen=args.timing_window)
    previous_loop_start_s = None

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"Camera: {actual_width}x{actual_height} at reported {actual_fps:.1f} FPS")
    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print("Red dot = bbox center; yellow X = model origin. Press Q or Esc to quit.")

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

            if frame_width != config.FRAME_W or frame_height != config.FRAME_H:
                raise ValueError(f"Camera produced {frame_width}x{frame_height}, but config expects {config.FRAME_W}x{config.FRAME_H}.")

            # Time only shape detection; capture, measurement, drawing, and display are excluded.
            detection_start_s = time.perf_counter()
            detection = findSingleObjectUsingBestShapeGroup(frame, object_vision_spec)
            detection_times_s.append(time.perf_counter() - detection_start_s)

            measurement = createMeasurementUsingShapeGroup(detection, object_vision_spec) if detection is not None else None
            drawDetection(frame, detection)

            if measurement is not None:
                drawModelOrigin(frame, measurement)

            average_detection_s = sum(detection_times_s)/len(detection_times_s)
            detector_rate_hz = 1.0/average_detection_s if average_detection_s > 0.0 else 0.0
            live_fps = 1.0/(sum(loop_periods_s)/len(loop_periods_s)) if loop_periods_s else 0.0
            shape_count = len(detection.shapes) if detection is not None else 0

            cv2.putText(frame, f"Detection: {average_detection_s*1000.0:.2f} ms | detector rate: {detector_rate_hz:.1f} Hz",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Live loop: {live_fps:.1f} FPS | shapes: {shape_count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

            if measurement is not None and measurement.x is not None and measurement.y is not None and measurement.z is not None:
                measurement_text = f"x: {measurement.x:.3f} m | y: {measurement.y:.3f} m | z: {measurement.z:.3f} m"
            else:
                measurement_text = "Measurement: unavailable"

            cv2.putText(frame, measurement_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()