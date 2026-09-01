import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.detection import (
    detectArucoMarkerV2, detectSingleObject, drawDetection,
    resetArucoOpticalFlowTracker,
)
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpecId


WINDOW_NAME = "Live detection test"
# DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.TENNIS_BALL_DEFAULT
DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.PAPER_PLANE_SHAPES_1
DEFAULT_TIMING_WINDOW_FRAMES = 120
POSITION_AVERAGING_WINDOW_FRAMES = 60
DISPLAY_SCALES = (1.0, 1.5, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live object detection and display current and averaged measurements."
    )
    parser.add_argument(
        "--spec", default=DEFAULT_OBJECT_VISION_SPEC_ID.name,
        choices=[spec_id.name for spec_id in OBJECT_VISION_SPECS],
        help=f"Registered ObjectVisionSpecId. Default: {DEFAULT_OBJECT_VISION_SPEC_ID.name}",
    )
    parser.add_argument(
        "--timing-window", type=int, default=DEFAULT_TIMING_WINDOW_FRAMES,
        help=f"Frames used for rolling timing average. Default: {DEFAULT_TIMING_WINDOW_FRAMES}",
    )
    args = parser.parse_args()

    if args.timing_window < 1:
        parser.error("--timing-window must be at least 1.")

    object_vision_spec_id = ObjectVisionSpecId[args.spec]
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]

    switchable_spec_ids = list(OBJECT_VISION_SPECS.keys())[:9]
    spec_hotkeys = {
        ord(str(index + 1)): spec_id
        for index, spec_id in enumerate(switchable_spec_ids)
    }

    camera_calibration = CameraCalibration(
        config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H,
    )

    if (
        camera_calibration.image_width_px, camera_calibration.image_height_px
    ) != (config.FRAME_W, config.FRAME_H):
        raise ValueError(
            f"Camera calibration resolution "
            f"{camera_calibration.image_width_px}x{camera_calibration.image_height_px} "
            f"does not match configured frame resolution {config.FRAME_W}x{config.FRAME_H}"
        )

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, config.FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera.set(
        cv2.CAP_PROP_AUTO_EXPOSURE,
        0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25,
    )
    if not config.CAMERA_AUTO_EXPOSURE:
        camera.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE)
        camera.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN)

    camera.set(
        cv2.CAP_PROP_AUTO_WB,
        1 if config.CAMERA_AUTO_WHITE_BALANCE else 0,
    )
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        camera.set(
            cv2.CAP_PROP_WB_TEMPERATURE,
            config.CAMERA_WHITE_BALANCE_TEMPERATURE,
        )

    if hasattr(config, "CAMERA_AUTOFOCUS"):
        camera.set(
            cv2.CAP_PROP_AUTOFOCUS,
            1 if config.CAMERA_AUTOFOCUS else 0,
        )
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
    display_scale_index = 0

    last_display_frame = None
    last_raw_frame = None

    timing_text = ""
    current_text = ""
    average_text = ""

    def drawDisplayText(
        image: np.ndarray, text: str, position: tuple[int, int],
        color: tuple[int, int, int], display_scale: float,
    ) -> None:
        font_scale = 0.52*display_scale
        thickness = max(1, round(display_scale))
        pos = (
            round(position[0]*display_scale),
            round(position[1]*display_scale),
        )

        cv2.putText(
            image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness, cv2.LINE_AA,
        )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.FRAME_W, config.FRAME_H)

    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    if object_vision_spec.object_type == ObjectType.ARUCO_MARKER:
        print("ArUco path: standard ArUco + four-corner pyramidal LK")
    print(
        f"Position average: last "
        f"{POSITION_AVERAGING_WINDOW_FRAMES} valid measurements"
    )
    print("Object switching:")
    for index, spec_id in enumerate(switchable_spec_ids, start=1):
        print(f"  {index}: {spec_id.name}")

    print(
        "1-9: switch object | Tab: display size | P: pause/resume | "
        "S: save raw frame | Q/Esc: quit"
    )

    try:
        while True:
            if not paused:
                loop_start_s = time.perf_counter()

                if previous_loop_start_s is not None:
                    loop_periods_s.append(
                        loop_start_s - previous_loop_start_s
                    )

                previous_loop_start_s = loop_start_s

                success, raw_frame = camera.read()
                if not success:
                    raise RuntimeError(
                        "Could not read a frame from the camera."
                    )

                last_raw_frame = raw_frame.copy()
                frame = raw_frame.copy()

                detection_start_s = time.perf_counter()

                # Explicitly exercise V2 for ArUco while keeping the existing
                # generic path unchanged for every other ObjectVisionSpec.
                if object_vision_spec.object_type == ObjectType.ARUCO_MARKER:
                    object_detected, detection, measurement = detectArucoMarkerV2(
                        raw_frame, object_vision_spec, camera_calibration,
                    )
                else:
                    object_detected, detection, measurement = detectSingleObject(
                        raw_frame, object_vision_spec_id, camera_calibration,
                    )

                detection_times_s.append(
                    time.perf_counter() - detection_start_s
                )

                if object_detected:
                    drawDetection(frame, detection)

                valid_measurement = (
                    object_detected
                    and measurement is not None
                    and measurement.x is not None
                    and measurement.y is not None
                    and measurement.z is not None
                )

                if valid_measurement:
                    position = np.array(
                        [measurement.x, measurement.y, measurement.z],
                        dtype=np.float64,
                    )
                    if np.all(np.isfinite(position)):
                        measurement_buffer.append(position)

                average_position = (
                    np.mean(np.asarray(measurement_buffer), axis=0)
                    if measurement_buffer else None
                )
                average_detection_s = (
                    sum(detection_times_s)/len(detection_times_s)
                    if detection_times_s else 0.0
                )
                detector_rate_hz = (
                    1.0/average_detection_s
                    if average_detection_s > 0.0 else 0.0
                )
                live_fps = (
                    1.0/(sum(loop_periods_s)/len(loop_periods_s))
                    if loop_periods_s else 0.0
                )

                timing_text = (
                    f"Detection: {average_detection_s*1000.0:.2f} ms | "
                    f"rate: {detector_rate_hz:.1f} Hz | "
                    f"loop: {live_fps:.1f} FPS"
                )

                current_text = (
                    f"Current: x={measurement.x:.3f} m | "
                    f"y={measurement.y:.3f} m | "
                    f"z={measurement.z:.3f} m"
                    if valid_measurement
                    else "Current: unavailable"
                )

                if average_position is not None:
                    average_text = (
                        f"Avg {len(measurement_buffer)}/"
                        f"{POSITION_AVERAGING_WINDOW_FRAMES}: "
                        f"x={average_position[0]:.3f} m | "
                        f"y={average_position[1]:.3f} m | "
                        f"z={average_position[2]:.3f} m"
                    )
                else:
                    average_text = (
                        f"Avg 0/{POSITION_AVERAGING_WINDOW_FRAMES}: "
                        "unavailable"
                    )

                frame = cv2.flip(frame, 1)
                last_display_frame = frame.copy()

            elif last_display_frame is None:
                continue

            display_scale = DISPLAY_SCALES[display_scale_index]
            display_frame = cv2.resize(
                last_display_frame, None,
                fx=display_scale, fy=display_scale,
                interpolation=cv2.INTER_LINEAR,
            )

            drawDisplayText(
                display_frame, timing_text, (10, 25),
                (0, 255, 0), display_scale,
            )
            drawDisplayText(
                display_frame, current_text, (10, 50),
                (0, 255, 0), display_scale,
            )
            drawDisplayText(
                display_frame, average_text, (10, 75),
                (255, 255, 0), display_scale,
            )

            if paused:
                drawDisplayText(
                    display_frame, "PAUSED", (10, 100),
                    (0, 0, 255), display_scale,
                )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key in spec_hotkeys:
                new_object_vision_spec_id = spec_hotkeys[key]

                if new_object_vision_spec_id != object_vision_spec_id:
                    object_vision_spec_id = new_object_vision_spec_id
                    object_vision_spec = OBJECT_VISION_SPECS[
                        object_vision_spec_id
                    ]

                    # Do not allow image-space state from the previous selected
                    # object/spec to leak into the newly selected one.
                    resetArucoOpticalFlowTracker()

                    detection_times_s.clear()
                    loop_periods_s.clear()
                    measurement_buffer.clear()
                    previous_loop_start_s = None

                    timing_text = ""
                    current_text = ""
                    average_text = ""
                    last_display_frame = None

                    print(
                        f"ObjectVisionSpecId: "
                        f"{object_vision_spec_id.name}"
                    )
                    if (
                        object_vision_spec.object_type
                        == ObjectType.ARUCO_MARKER
                    ):
                        print(
                            "ArUco path: standard ArUco + "
                            "four-corner pyramidal LK"
                        )

            elif key == 9:
                display_scale_index = (
                    display_scale_index + 1
                ) % len(DISPLAY_SCALES)
                display_scale = DISPLAY_SCALES[display_scale_index]

                cv2.resizeWindow(
                    WINDOW_NAME,
                    round(config.FRAME_W*display_scale),
                    round(config.FRAME_H*display_scale),
                )
                print(f"Display scale: {display_scale:.1f}x")

            elif key == ord("p"):
                paused = not paused

                if not paused:
                    previous_loop_start_s = None
                    loop_periods_s.clear()

                print("Paused" if paused else "Resumed")

            elif key == ord("s") and last_raw_frame is not None:
                save_path = (
                    Path("images")
                    / "primary_detection_references"
                    / object_vision_spec_id.name.lower()
                    / "reference_1.png"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if not cv2.imwrite(str(save_path), last_raw_frame):
                    raise RuntimeError(
                        f"Could not save image: {save_path}"
                    )

                print(f"Saved {save_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
