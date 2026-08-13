import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera_calibration import CameraCalibration
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectVisionSpecId


WINDOW_NAME = "Live detection test"
DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.TENNIS_BALL_DEFAULT
DEFAULT_TIMING_WINDOW_FRAMES = 120
POSITION_AVERAGING_WINDOW_FRAMES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live object detection and display current and averaged measurements.")
    parser.add_argument("--spec", default=DEFAULT_OBJECT_VISION_SPEC_ID.name,
                        choices=[spec_id.name for spec_id in OBJECT_VISION_SPECS],
                        help=f"Registered ObjectVisionSpecId. Default: {DEFAULT_OBJECT_VISION_SPEC_ID.name}")
    parser.add_argument("--timing-window", type=int, default=DEFAULT_TIMING_WINDOW_FRAMES,
                        help=f"Frames used for rolling timing average. Default: {DEFAULT_TIMING_WINDOW_FRAMES}")
    args = parser.parse_args()

    if args.timing_window < 1:
        parser.error("--timing-window must be at least 1.")

    object_vision_spec_id = ObjectVisionSpecId[args.spec]
    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)

    if (camera_calibration.image_width_px, camera_calibration.image_height_px) != (config.FRAME_W, config.FRAME_H):
        raise ValueError(
            f"Camera calibration resolution {camera_calibration.image_width_px}x{camera_calibration.image_height_px} "
            f"does not match configured frame resolution {config.FRAME_W}x{config.FRAME_H}"
        )

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, config.FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25)
    if not config.CAMERA_AUTO_EXPOSURE:
        camera.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE)
        camera.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN)

    camera.set(cv2.CAP_PROP_AUTO_WB, 1 if config.CAMERA_AUTO_WHITE_BALANCE else 0)
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAMERA_WHITE_BALANCE_TEMPERATURE)

    if hasattr(config, "CAMERA_AUTOFOCUS"):
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if config.CAMERA_AUTOFOCUS else 0)
        if not config.CAMERA_AUTOFOCUS and hasattr(config, "CAMERA_FOCUS"):
            camera.set(cv2.CAP_PROP_FOCUS, config.CAMERA_FOCUS)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (config.FRAME_W, config.FRAME_H):
        camera.release()
        raise ValueError(
            f"Camera produced {actual_width}x{actual_height}, "
            f"but config expects {config.FRAME_W}x{config.FRAME_H}"
        )

    detection_times_s = deque(maxlen=args.timing_window)
    loop_periods_s = deque(maxlen=args.timing_window)
    measurement_buffer = deque(maxlen=POSITION_AVERAGING_WINDOW_FRAMES)

    previous_loop_start_s = None
    paused = False
    last_display_frame = None
    last_raw_frame = None

    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print(f"Position average: last {POSITION_AVERAGING_WINDOW_FRAMES} valid measurements")
    print("P: pause/resume | S: save raw frame | Q/Esc: quit")

    try:
        while True:
            if not paused:
                loop_start_s = time.perf_counter()
                if previous_loop_start_s is not None:
                    loop_periods_s.append(loop_start_s - previous_loop_start_s)
                previous_loop_start_s = loop_start_s

                success, raw_frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                last_raw_frame = raw_frame.copy()
                frame = raw_frame.copy()

                detection_start_s = time.perf_counter()
                object_detected, detection, measurement = detectSingleObject(raw_frame, object_vision_spec_id, camera_calibration)
                detection_times_s.append(time.perf_counter() - detection_start_s)

                if object_detected:
                    drawDetection(frame, detection)

                valid_measurement = (
                    object_detected and measurement is not None and
                    measurement.x is not None and measurement.y is not None and measurement.z is not None
                )

                if valid_measurement:
                    position = np.array([measurement.x, measurement.y, measurement.z], dtype=np.float64)
                    if np.all(np.isfinite(position)):
                        measurement_buffer.append(position)

                average_position = np.mean(np.asarray(measurement_buffer), axis=0) if measurement_buffer else None
                average_detection_s = sum(detection_times_s)/len(detection_times_s) if detection_times_s else 0.0
                detector_rate_hz = 1.0/average_detection_s if average_detection_s > 0.0 else 0.0
                live_fps = 1.0/(sum(loop_periods_s)/len(loop_periods_s)) if loop_periods_s else 0.0

                if valid_measurement:
                    current_text = f"Current: x={measurement.x:.3f} m | y={measurement.y:.3f} m | z={measurement.z:.3f} m"
                else:
                    current_text = "Current: unavailable"

                if average_position is not None:
                    average_text = (
                        f"Avg {len(measurement_buffer)}/{POSITION_AVERAGING_WINDOW_FRAMES}: "
                        f"x={average_position[0]:.3f} m | y={average_position[1]:.3f} m | z={average_position[2]:.3f} m"
                    )
                else:
                    average_text = f"Avg 0/{POSITION_AVERAGING_WINDOW_FRAMES}: unavailable"

                cv2.putText(frame, f"Detection: {average_detection_s*1000.0:.2f} ms | rate: {detector_rate_hz:.1f} Hz | loop: {live_fps:.1f} FPS",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, current_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, average_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

                last_display_frame = frame.copy()
            else:
                if last_display_frame is None:
                    continue
                frame = last_display_frame.copy()
                cv2.putText(frame, "PAUSED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
                if not paused:
                    previous_loop_start_s = None
                    loop_periods_s.clear()
                print("Paused" if paused else "Resumed")
            elif key == ord("s") and last_raw_frame is not None:
                save_path = Path("images")/"primary_detection_references"/object_vision_spec_id.name.lower()/"reference_1.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(save_path), last_raw_frame):
                    raise RuntimeError(f"Could not save image: {save_path}")
                print(f"Saved {save_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()