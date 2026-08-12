import argparse
from pathlib import Path

import cv2

import src.primary.config as config
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectVisionSpecId


WINDOW_NAME = "Detection reference capture"
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


def getNextReferencePath(reference_directory: Path) -> Path:
    maximum_index = 0

    for path in reference_directory.glob("reference_*.png"):
        try:
            maximum_index = max(maximum_index, int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    return reference_directory/f"reference_{maximum_index + 1}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture reference images for a registered ObjectVisionSpecId.")
    parser.add_argument("--spec", choices=[spec_id.name for spec_id in OBJECT_VISION_SPECS],
                        help="Registered ObjectVisionSpecId. If omitted, you will be required to select one.")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIRECTORY,
                        help=f"Base reference-image directory. Default: {DEFAULT_REFERENCE_DIRECTORY}")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help=f"Camera index. Default: {config.CAMERA_INDEX}")
    parser.add_argument("--camera-fps", type=int, default=config.FPS,
                        help=f"Requested camera FPS. Default: {config.FPS}")
    args = parser.parse_args()

    object_vision_spec_id = ObjectVisionSpecId[args.spec] if args.spec is not None else chooseObjectVisionSpecId()
    reference_directory = args.reference_dir/object_vision_spec_id.name.lower()
    reference_directory.mkdir(parents=True, exist_ok=True)

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

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (actual_width, actual_height) != (config.FRAME_W, config.FRAME_H):
        camera.release()
        raise ValueError(
            f"Camera produced {actual_width}x{actual_height}, "
            f"but config expects {config.FRAME_W}x{config.FRAME_H}"
        )

    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print(f"Reference directory: {reference_directory.resolve()}")
    print("S: save reference    Q/Esc: quit")

    try:
        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                print("Possible camera failure.")
                break

            display_frame = frame.copy()

            cv2.putText(display_frame, object_vision_spec_id.name, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, object_vision_spec_id.name, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(display_frame, "S: save    Q/Esc: quit", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, "S: save    Q/Esc: quit", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                output_path = getNextReferencePath(reference_directory)

                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not save reference image: {output_path}")

                print(f"Saved {output_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()