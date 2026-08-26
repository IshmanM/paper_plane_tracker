import cv2
import numpy as np
import os
import threading
from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.camera_to_platform_calibration import CameraToPlatformCalibration
from src.primary.tracking import TrackStatus, SingleObjectTracker, drawTrack
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.object_vision_spec import ObjectVisionSpecId
import src.primary.config as config
from datetime import datetime
import time
from src.primary.comm_buffer import CommBuffer, cmd_thread_main
from src.primary.platform import Platform
from src.primary.platform_geometry_spec import PlatformGeometrySpecId

from src.comm.link import UdpLink
from src.comm.network_config import (
    PRIMARY_IP,
    ENDPOINT_IP,
    UDP_PORT,
    PRIMARY_NODE_ID,
    ENDPOINT_NODE_ID,
    DEFAULT_MAX_PACKET_BYTES,
)


WINDOW_NAME = "Webcam Feed"
DISPLAY_SCALES = (1.0, 1.5, 2.0)


def drawDisplayText(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int], display_scale: float) -> None:
    """Draw HUD text after display scaling so the text itself remains sharp."""
    pos = (round(position[0]*display_scale), round(position[1]*display_scale))
    font_scale = 0.4*display_scale
    thickness = max(1, round(display_scale))

    # Black outline makes colored text readable over bright/dark backgrounds.
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    # object_vision_spec_id = ObjectVisionSpecId.TENNIS_BALL_DEFAULT
    object_vision_spec_id = ObjectVisionSpecId.ARUCO_MARKER_1
    # object_vision_spec_id = ObjectVisionSpecId.PAPER_PLANE_SHAPES_1

    platform_geometry_spec_id = PlatformGeometrySpecId.PLATFORM_1

    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)
    if (camera_calibration.image_width_px, camera_calibration.image_height_px) != (config.FRAME_W, config.FRAME_H):
        raise ValueError(
            f"Camera calibration resolution {camera_calibration.image_width_px}x{camera_calibration.image_height_px} "
            f"does not match configured frame resolution {config.FRAME_W}x{config.FRAME_H}"
        )

    camera_to_platform_calibration = CameraToPlatformCalibration(config.CAMERA_TO_PLATFORM_CALIBRATION_PATH)

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25)
    if not config.CAMERA_AUTO_EXPOSURE:
        cap.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE)
        cap.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN)

    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if config.CAMERA_AUTO_WHITE_BALANCE else 0)
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAMERA_WHITE_BALANCE_TEMPERATURE)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    tracker = SingleObjectTracker(
        min_hits=3,
        max_missed_on_confirmed=15,
        max_missed_on_tentative=1,
        # params...
    )

    comm_buffer = CommBuffer()
    platform = Platform(
        comm_buffer=comm_buffer,
        platform_geometry_spec_id=platform_geometry_spec_id,
        camera_to_platform_calibration=camera_to_platform_calibration,
    )

    link = UdpLink(
        local_ip=PRIMARY_IP,
        remote_ip=ENDPOINT_IP,
        port=UDP_PORT,
        local_node_id=PRIMARY_NODE_ID,
        remote_node_id=ENDPOINT_NODE_ID,
        max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
        check_remote_ip=True,
    )

    # link = None # FOR DEBUG ONLY

    stop_event = threading.Event()
    cmd_thread = threading.Thread(target=cmd_thread_main, args=(comm_buffer, stop_event, link), daemon=True)
    cmd_thread.start()

    # os.makedirs("images/primary_main_screenshots", exist_ok=True)

    last_detection_px_w = 0
    last_detection_px_h = 0
    last_view_frame = None

    detection_label = "No detection"
    track_label = "Dead track"

    tracker_paused = False   # OpenCV/tracker runs by default
    platform_paused = True   # Platform OFF by default

    display_scale_index = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.FRAME_W, config.FRAME_H)

    print("Tab: display size | P: pause all | R: resume tracker | L/O: platform off/on | H/F: triggering | S: screenshot | Q/Esc: quit")

    try:
        while cap.isOpened():
            if not tracker_paused:
                frame_time = time.perf_counter()
                ret, frame = cap.read()  # doesn't always give latest frame but that's a future optimization.

                if not ret:
                    print("Possible camera failure")
                    break

                # Detect the object and produce a measurement.
                object_detected, detection, measurement = detectSingleObject(frame, object_vision_spec_id, camera_calibration)

                if object_detected:
                    last_detection_px_w = detection.px_w
                    last_detection_px_h = detection.px_h
                    drawDetection(frame, detection)

                if not object_detected:
                    detection_label = "No detection"
                else:
                    detection_label = f"Measurement: x={measurement.x:.4f}, y={measurement.y:.4f}, z={measurement.z:.4f}"

                # Track the object state and update the platform planner.
                track_status = tracker.update(object_detected, measurement, frame_time)
                platform.update(tracker=tracker)

                track_label = "Dead track"
                if track_status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
                    drawTrack(frame, tracker.track, last_detection_px_w, last_detection_px_h, camera_calibration)
                    status_label = "Confirmed" if track_status == TrackStatus.CONFIRMED else "Tentative"
                    track_label = (
                        f"{status_label} track: (x: {tracker.track.x:.4f}, y: {tracker.track.y:.4f}, z: {tracker.track.z:.4f}, "
                        f"dx: {tracker.track.dx:.4f}, dy: {tracker.track.dy:.4f}, dz: {tracker.track.dz:.4f})"
                    )

                # Viewing-only flip. Detection/tracking/platform all used the original frame.
                frame = cv2.flip(frame, 1)

                # Store native-resolution visualization WITHOUT HUD text. HUD is
                # rendered after display resizing so the text remains sharp.
                last_view_frame = frame.copy()

            elif last_view_frame is None:
                continue

            # DISPLAY ONLY: resize the visualization. This has no effect on camera
            # resolution, detection, measurement, tracking, calibration, or platform.
            display_scale = DISPLAY_SCALES[display_scale_index]
            display_frame = cv2.resize(last_view_frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)

            # Render text after resize rather than scaling already-rendered text.
            drawDisplayText(display_frame, detection_label, (10, 20), (0, 255, 0), display_scale)
            drawDisplayText(display_frame, track_label, (10, 50), (0, 0, 255), display_scale)

            if tracker_paused:
                drawDisplayText(display_frame, "TRACKER PAUSED", (10, 80), (0, 0, 255), display_scale)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:  # FOR DEBUG ONLY
                print(f"key pressed: {key}, char: {chr(key) if key < 128 else '?'}")

            # q / Esc = quit
            if key in (ord("q"), 27):
                print("Quitting...")
                platform.halt_triggering()
                time.sleep(0.1) #todo: make a permanent fix instead of this temporary one
                platform.turn_off()
                stop_event.set()
                break

            # Tab = cycle display scale only.
            elif key == 9:
                display_scale_index = (display_scale_index + 1) % len(DISPLAY_SCALES)
                display_scale = DISPLAY_SCALES[display_scale_index]
                cv2.resizeWindow(WINDOW_NAME, round(config.FRAME_W*display_scale), round(config.FRAME_H*display_scale))
                print(f"Display scale: {display_scale:.1f}x")

            # p = pause BOTH OpenCV/tracker and platform
            elif key == ord("p"):
                tracker_paused = True
                platform_paused = True
                platform.turn_off()
                print("Paused tracker/OpenCV + platform OFF")

            # r = resume OpenCV/tracker only
            elif key == ord("r"):
                tracker_paused = False
                print("Tracker/OpenCV resumed")

            # l = pause platform only
            elif key == ord("l"):
                platform_paused = True
                platform.turn_off()
                print("Platform OFF")

            # o = resume platform only if OpenCV/tracker is running
            elif key == ord("o"):
                if tracker_paused:
                    print("Cannot turn platform ON while tracker/OpenCV is paused")
                else:
                    platform_paused = False
                    platform.turn_on()
                    print("Platform ON")

            # h = halt triggering
            elif key == ord("h"):
                platform.halt_triggering()
                print("Triggering HALTED")

            # f = allow triggering
            elif key == ord("f"):
                platform.allow_triggering()
                print("Triggering allowed")

            # s = screenshot of exactly what is currently being displayed.
            elif key == ord("s"):
                filename = datetime.now().strftime(f"screenshot_{object_vision_spec_id.name.lower()}_%Y%m%d_%H%M%S.png")
                filepath = "images/primary_main_screenshots/" + filename
                cv2.imwrite(filepath, display_frame)
                print(f"Saved {filepath}")

    finally:
        print("Cleaning up...")

        stop_event.set()

        if cmd_thread.is_alive():
            cmd_thread.join(timeout=1.0)

        if cmd_thread.is_alive():
            print("Warning: command thread did not stop before link close")

        try:
            link.close()
        except Exception as e:
            print(f"Failed to close UDP link: {e}")

        cap.release()
        cv2.destroyAllWindows()
        print("Done.")