import argparse
import time
from pathlib import Path

import cv2

import src.primary.config as config


IMAGE_DIRECTORY = Path(f"images/camera_calibration/{config.CAMERA_CALIBRATION_NAME}")
WINDOW_NAME = "Camera calibration capture"
DEFAULT_PHOTO_COUNT = 25
DEFAULT_INTERVAL_S = 2.0
INITIAL_COUNTDOWN_S = 5.0
CAPTURE_MESSAGE_S = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically capture checkerboard images for camera calibration.")
    parser.add_argument("--photos", type=int, default=DEFAULT_PHOTO_COUNT,
                        help=f"Number of photos to capture. Default: {DEFAULT_PHOTO_COUNT}")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"Seconds between photos. Default: {DEFAULT_INTERVAL_S}")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help=f"Camera index. Default: {config.CAMERA_INDEX}")
    parser.add_argument("--camera-fps", type=int, default=config.FPS,
                        help=f"Requested camera FPS. Default: {config.FPS}")
    parser.add_argument("--mode", choices=("overwrite", "append"), default=None,
                        help="Whether to overwrite existing calibration photos or append new ones.")
    args = parser.parse_args()

    if args.photos < 1:
        parser.error("--photos must be at least 1.")
    if args.interval <= 0.0:
        parser.error("--interval must be greater than 0.")

    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    existing_images = sorted(IMAGE_DIRECTORY.glob("calibration_*.png"))

    # Ask how existing calibration images should be handled if --mode was not provided.
    if args.mode is None:
        while True:
            choice = input("Existing photos: [O]verwrite or [A]ppend? ").strip().lower()

            if choice in ("o", "overwrite"):
                args.mode = "overwrite"
                break
            if choice in ("a", "append"):
                args.mode = "append"
                break

            print("Enter O for overwrite or A for append.")

    # Overwrite removes old calibration images; append continues their numbering.
    if args.mode == "overwrite":
        for image_path in existing_images:
            image_path.unlink()

        image_index = 0
        print(f"Removed {len(existing_images)} existing calibration photos.")
    else:
        existing_indices = []

        for image_path in existing_images:
            try:
                existing_indices.append(int(image_path.stem.split("_")[-1]))
            except ValueError:
                pass

        image_index = max(existing_indices, default=-1) + 1

    # Configure the same camera settings used by the detection system.
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25)
    if not config.CAMERA_AUTO_EXPOSURE:
        camera.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE)
        camera.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN)

    camera.set(cv2.CAP_PROP_AUTO_WB, 1 if config.CAMERA_AUTO_WHITE_BALANCE else 0)
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAMERA_WHITE_BALANCE_TEMPERATURE)

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if config.CAMERA_AUTOFOCUS else 0)
    if not config.CAMERA_AUTOFOCUS:
        camera.set(cv2.CAP_PROP_FOCUS, config.CAMERA_FOCUS)

    print(f"Backend: {camera.getBackendName()}")
    print(f"Auto exposure: {camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"Exposure: {camera.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Gain: {camera.get(cv2.CAP_PROP_GAIN)}")
    print(f"Auto WB: {camera.get(cv2.CAP_PROP_AUTO_WB)}")
    print(f"WB temperature: {camera.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
    print(f"Autofocus: {camera.get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"Focus: {camera.get(cv2.CAP_PROP_FOCUS)}")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_width != config.FRAME_W or actual_height != config.FRAME_H:
        camera.release()
        raise ValueError(f"Camera produced {actual_width}x{actual_height}, but config expects {config.FRAME_W}x{config.FRAME_H}.")

    print(f"Capturing {args.photos} photos every {args.interval:.1f} s.")
    print("Move the checkerboard to a different position/orientation after each photo.")
    print("Press Q or Esc to stop early.")

    captured_count = 0
    start_time = time.perf_counter()
    next_capture_time = start_time + INITIAL_COUNTDOWN_S
    last_capture_time = None

    try:
        while captured_count < args.photos:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            current_time = time.perf_counter()
            display_frame = frame.copy()

            # Save the unmodified camera frame when the countdown reaches zero.
            if current_time >= next_capture_time:
                output_path = IMAGE_DIRECTORY/f"calibration_{image_index:03d}.png"

                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not save image: {output_path}")

                captured_count += 1
                image_index += 1
                last_capture_time = current_time
                next_capture_time = current_time + args.interval

                print(f"Captured {captured_count}/{args.photos}: {output_path}")

            # Clearly indicate the instant after a photo was captured.
            if last_capture_time is not None and current_time - last_capture_time < CAPTURE_MESSAGE_S:
                status_text = f"PHOTO TAKEN  {captured_count}/{args.photos}  -  MOVE CHECKERBOARD"
                text_color = (0, 255, 0)
            else:
                seconds_remaining = max(0.0, next_capture_time - current_time)

                if captured_count == 0:
                    status_text = f"Starting in {seconds_remaining:.1f} s"
                else:
                    status_text = f"Next photo in {seconds_remaining:.1f} s  -  MOVE CHECKERBOARD"

                text_color = (0, 255, 255)

            cv2.putText(display_frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display_frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)

            cv2.putText(display_frame, f"Captured: {captured_count}/{args.photos} | Q/Esc: stop", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display_frame, f"Captured: {captured_count}/{args.photos} | Q/Esc: stop", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Capture stopped early.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Captured {captured_count} photos.")
    print(f"Calibration images: {IMAGE_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()