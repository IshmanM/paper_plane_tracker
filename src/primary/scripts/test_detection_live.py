import argparse
import time
from collections import deque
from pathlib import Path

import cv2

import src.primary.config as config
from src.primary.camera_calibration import CameraCalibration
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpecId
from src.primary.detection import detectSingleObject, drawDetection, drawModelOrigin


WINDOW_NAME = "Live detection test"

DEFAULT_CAMERA_INDEX = config.CAMERA_INDEX
DEFAULT_CAMERA_FPS = config.FPS
DEFAULT_TIMING_WINDOW_FRAMES = 120
DEFAULT_REFERENCE_DIRECTORY = Path("images/primary_detection_references")


def chooseObjectVisionSpecId() -> ObjectVisionSpecId:
    spec_ids = list(OBJECT_VISION_SPECS)

    print("Available ObjectVisionSpecIds:")
    for index, spec_id in enumerate(spec_ids, start=1):
        print(f"  {index}. {spec_id.name}")

    while True:
        choice = input("Select ObjectVisionSpecId: ").strip()

        try:
            index = int(choice)
            if 1 <= index <= len(spec_ids):
                return spec_ids[index - 1]
        except ValueError:
            pass

        try:
            spec_id = ObjectVisionSpecId[choice.upper()]
            if spec_id in OBJECT_VISION_SPECS:
                return spec_id
        except KeyError:
            pass

        print("Enter one of the listed numbers or ObjectVisionSpecId names.")


def saveReferenceFrame(frame, reference_directory: Path) -> None:
    reference_directory.mkdir(parents=True, exist_ok=True)
    output_path = reference_directory/"reference_1.png"

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save reference image: {output_path}")

    print(f"Saved unprocessed reference frame: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live object detection and measure its speed.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index. Default: {DEFAULT_CAMERA_INDEX}")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
                        help=f"Requested camera FPS. Default: {DEFAULT_CAMERA_FPS}")
    parser.add_argument("--timing-window", type=int, default=DEFAULT_TIMING_WINDOW_FRAMES,
                        help=f"Frames used for rolling timing averages. Default: {DEFAULT_TIMING_WINDOW_FRAMES}")
    parser.add_argument("--spec", choices=[spec_id.name for spec_id in OBJECT_VISION_SPECS],
                        help="Registered ObjectVisionSpecId. If omitted, you will be required to select one.")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIRECTORY,
                        help=f"Base reference-image directory. Default: {DEFAULT_REFERENCE_DIRECTORY}")
    args = parser.parse_args()

    if args.camera_fps < 1:
        parser.error("--camera-fps must be at least 1.")
    if args.timing_window < 1:
        parser.error("--timing-window must be at least 1.")

    object_vision_spec_id = ObjectVisionSpecId[args.spec] if args.spec is not None else chooseObjectVisionSpecId()
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]
    reference_directory = args.reference_dir/object_vision_spec_id.name.lower()

    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)

    if (camera_calibration.image_width_px, camera_calibration.image_height_px) != (config.FRAME_W, config.FRAME_H):
        raise ValueError(
            f"Camera calibration resolution {camera_calibration.image_width_px}x{camera_calibration.image_height_px} "
            f"does not match configured frame resolution {config.FRAME_W}x{config.FRAME_H}"
        )

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

    print("Set autofocus:", camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if config.CAMERA_AUTOFOCUS else 0))
    if not config.CAMERA_AUTOFOCUS:
        print("Set focus:", camera.set(cv2.CAP_PROP_FOCUS, config.CAMERA_FOCUS))

    print("Backend:", camera.getBackendName())
    print("Auto exposure:", camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("Exposure:", camera.get(cv2.CAP_PROP_EXPOSURE))
    print("Gain:", camera.get(cv2.CAP_PROP_GAIN))
    print("Auto WB:", camera.get(cv2.CAP_PROP_AUTO_WB))
    print("WB temperature:", camera.get(cv2.CAP_PROP_WB_TEMPERATURE))
    print("Autofocus:", camera.get(cv2.CAP_PROP_AUTOFOCUS))
    print("Focus:", camera.get(cv2.CAP_PROP_FOCUS))

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    if (actual_width, actual_height) != (config.FRAME_W, config.FRAME_H):
        camera.release()
        raise ValueError(
            f"Camera produced {actual_width}x{actual_height}, "
            f"but config expects {config.FRAME_W}x{config.FRAME_H}."
        )

    detection_times_s = deque(maxlen=args.timing_window)
    loop_periods_s = deque(maxlen=args.timing_window)
    previous_loop_start_s = None

    paused = False
    last_display_frame = None
    last_raw_frame = None

    print(f"Camera: {actual_width}x{actual_height} at reported {actual_fps:.1f} FPS")
    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print(f"ObjectType: {object_vision_spec.object_type.name}")
    print(f"Reference directory: {reference_directory.resolve()}")
    print("P = pause/resume | S = save unprocessed reference frame | Q/Esc = quit")

    try:
        while True:
            # While paused, keep displaying the last processed frame. The matching
            # unprocessed camera frame is retained separately and can still be saved with S.
            if paused:
                paused_frame = last_display_frame.copy()
                cv2.putText(paused_frame, "PAUSED", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, paused_frame)

                key = cv2.waitKey(30) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord("p"):
                    paused = False
                    previous_loop_start_s = None
                elif key == ord("s") and last_raw_frame is not None:
                    saveReferenceFrame(last_raw_frame, reference_directory)

                continue

            loop_start_s = time.perf_counter()

            if previous_loop_start_s is not None:
                loop_periods_s.append(loop_start_s - previous_loop_start_s)

            previous_loop_start_s = loop_start_s
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            frame_height, frame_width = frame.shape[:2]

            if frame_width != config.FRAME_W or frame_height != config.FRAME_H:
                raise ValueError(
                    f"Camera produced {frame_width}x{frame_height}, "
                    f"but config expects {config.FRAME_W}x{config.FRAME_H}."
                )

            # Preserve the untouched camera image before detection results, boxes,
            # measurements, or other debug information are drawn onto the display frame.
            raw_frame = frame.copy()

            detection_start_s = time.perf_counter()
            detection_success, detection, measurement = detectSingleObject(
                frame, object_vision_spec_id, camera_calibration
            )
            detection_times_s.append(time.perf_counter() - detection_start_s)

            drawDetection(frame, detection)

            if object_vision_spec.object_type == ObjectType.PAPER_PLANE_SHAPES and measurement.x is not None:
                drawModelOrigin(frame, measurement, camera_calibration)

            average_detection_s = sum(detection_times_s)/len(detection_times_s)
            detector_rate_hz = 1.0/average_detection_s if average_detection_s > 0.0 else 0.0
            live_fps = 1.0/(sum(loop_periods_s)/len(loop_periods_s)) if loop_periods_s else 0.0
            shape_count = len(detection.shapes)

            cv2.putText(
                frame,
                f"Detection pipeline: {average_detection_s*1000.0:.2f} ms | rate: {detector_rate_hz:.1f} Hz",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Live loop: {live_fps:.1f} FPS | shapes: {shape_count} | success: {detection_success}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA,
            )

            if measurement.x is not None and measurement.y is not None and measurement.z is not None:
                measurement_text = f"x: {measurement.x:.3f} m | y: {measurement.y:.3f} m | z: {measurement.z:.3f} m"
            else:
                measurement_text = "Measurement: unavailable"

            cv2.putText(frame, measurement_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

            # These two frames correspond to the same camera capture: one untouched
            # for later static analysis and one containing the live detection overlay.
            last_raw_frame = raw_frame
            last_display_frame = frame.copy()

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = True
            elif key == ord("s"):
                saveReferenceFrame(last_raw_frame, reference_directory)

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()