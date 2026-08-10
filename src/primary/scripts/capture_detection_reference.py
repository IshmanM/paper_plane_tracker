from pathlib import Path

import cv2

from src.primary.config import FRAME_W, FRAME_H


CAMERA_INDEX = 0
OUTPUT_PATH = Path("images/primary_detection_references/reference_1.png")
WINDOW_NAME = "Capture detection reference"


def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Press Space or S to save the reference image.")
    print("Press Q or Esc to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            frame_height, frame_width = frame.shape[:2]

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key in (ord("s"), ord(" ")):
                if frame_width != FRAME_W or frame_height != FRAME_H:
                    raise RuntimeError(
                        f"Camera returned {frame_width}x{frame_height}, but config expects "
                        f"{FRAME_W}x{FRAME_H}."
                    )

                if not cv2.imwrite(str(OUTPUT_PATH), frame):
                    raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

                print(f"Saved {frame_width} x {frame_height} image to {OUTPUT_PATH.resolve()}")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()