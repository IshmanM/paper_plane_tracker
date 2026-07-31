import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from src.primary.config import FRAME_W, FRAME_H
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType
from src.primary.detection import DetectionDebug, findSingleObjectUsingBestTriangleGroup


WINDOW_NAME = "Detection stages"
DEFAULT_REFERENCE_IMAGE_PATH = Path("images/primary_detection_references/reference_1.png")
DEFAULT_SAVE_DIRECTORY = Path("images/primary_detection_debug")


def fitImageForDisplay(image: np.ndarray, maximum_width: int = 1400, maximum_height: int = 850) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    scale = min(maximum_width / image_width, maximum_height / image_height, 1.0)

    if scale >= 1.0:
        return image.copy()

    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def createStageDisplayImage(
    stage_name: str, stage_image: np.ndarray, stage_index: int, stage_count: int
) -> np.ndarray:
    if stage_image.ndim == 2:
        display_image = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = stage_image.copy()

    display_image = fitImageForDisplay(display_image)
    display_image = cv2.copyMakeBorder(
        display_image, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )

    cv2.putText(
        display_image, f"{stage_index + 1}/{stage_count}: {stage_name}", (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        display_image, "A/P: previous    D/Space: next    Q/Esc: quit", (15, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA
    )

    return display_image


def showDetectionStages(stages: list[tuple[str, np.ndarray]]) -> None:
    if not stages:
        print("No debug stages were recorded.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    stage_index = 0

    while True:
        stage_name, stage_image = stages[stage_index]
        display_image = createStageDisplayImage(stage_name, stage_image, stage_index, len(stages))

        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("a"), ord("p")):
            stage_index = max(0, stage_index - 1)
        elif key in (ord("d"), ord("n"), ord(" "), 13):
            stage_index = min(len(stages) - 1, stage_index + 1)

    cv2.destroyAllWindows()


def saveDetectionStages(stages: list[tuple[str, np.ndarray]], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for stage_index, (stage_name, stage_image) in enumerate(stages):
        safe_stage_name = re.sub(r"[^A-Za-z0-9_-]+", "_", stage_name).strip("_")
        output_path = output_directory / f"{stage_index:02d}_{safe_stage_name}.png"

        if not cv2.imwrite(str(output_path), stage_image):
            raise RuntimeError(f"Could not save debug image: {output_path}")

    print(f"Saved {len(stages)} stages to {output_directory.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection on one image and inspect each processing stage."
    )
    parser.add_argument(
        "image_path", type=Path, nargs="?", default=DEFAULT_REFERENCE_IMAGE_PATH,
        help=f"Input image path. Default: {DEFAULT_REFERENCE_IMAGE_PATH}"
    )
    parser.add_argument(
        "--save-dir", type=Path, default=DEFAULT_SAVE_DIRECTORY,
        help=f"Directory for saved stages. Default: {DEFAULT_SAVE_DIRECTORY}"
    )
    parser.add_argument("--no-gui", action="store_true", help="Do not open the OpenCV stage viewer.")

    args = parser.parse_args()

    if not args.image_path.is_file():
        parser.error(f"Image does not exist: {args.image_path}")

    frame = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"OpenCV could not read: {args.image_path}")

    frame_height, frame_width = frame.shape[:2]

    if frame_width != FRAME_W or frame_height != FRAME_H:
        raise ValueError(
            f"Reference image is {frame_width}x{frame_height}, but config expects "
            f"{FRAME_W}x{FRAME_H}."
        )

    debug = DetectionDebug()

    object_vision_spec = OBJECT_VISION_SPECS[ObjectType.PAPER_PLANE_TRIANGLES]
    detection = findSingleObjectUsingBestTriangleGroup(frame, object_vision_spec, debug)

    print(f"Input image: {args.image_path.resolve()}")
    print(f"Image size: {frame_width} x {frame_height}")
    print(f"Recorded stages: {len(debug.stages)}")

    if detection is None:
        print("Detection result: no object detected")
    else:
        print("Detection result:")
        print(f"  u: {detection.u:.2f} px")
        print(f"  v: {detection.v:.2f} px")
        print(f"  width: {detection.px_w:.2f} px")
        print(f"  height: {detection.px_h:.2f} px")
        print(f"  triangles: {len(detection.triangles)}")

    saveDetectionStages(debug.stages, args.save_dir)

    if not args.no_gui:
        showDetectionStages(debug.stages)


if __name__ == "__main__":
    main()