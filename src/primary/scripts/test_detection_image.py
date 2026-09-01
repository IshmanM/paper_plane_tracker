import argparse
import math
import re
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.geometry import estimateObjectWorldPosition
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpecId
from src.primary.detection import (
    DetectionDebug, Measurement,
    findSingleObjectUsingBestShapeGroup, findSingleObjectSphere, detectArucoMarkerV2, detectSingleObject,
    createMeasurementUsingShapeGroup, drawDetection, drawModelOrigin,
)


WINDOW_NAME = "Detection stages"

DEFAULT_REFERENCE_DIRECTORY = Path("images/primary_detection_references")
DEFAULT_SAVE_DIRECTORY = Path("images/primary_detection_debug")

DEFAULT_DISPLAY_COLUMNS = 3
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 900

STATUS_BAR_HEIGHT = 58
ZOOM_STEP = 1.25
MAX_ZOOM = 10.0

MAIN_BENCHMARK_WARMUP_RUNS = 3
MAIN_BENCHMARK_TIMED_RUNS = 20

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# cv2.waitKeyEx arrow-key codes used by common Windows/Linux backends.
LEFT_ARROW_KEYS = {2424832, 65361}
RIGHT_ARROW_KEYS = {2555904, 65363}


class DetectionAlgorithm(Enum):
    SHAPE_GROUP = auto()
    SPHERE = auto()
    ARUCO_MARKER = auto()


ALGORITHM_OBJECT_TYPES = {
    DetectionAlgorithm.SHAPE_GROUP: {ObjectType.PAPER_PLANE_SHAPES},
    DetectionAlgorithm.SPHERE: {ObjectType.TENNIS_BALL},
    DetectionAlgorithm.ARUCO_MARKER: {ObjectType.ARUCO_MARKER},
}


def chooseOption(title: str, options: list) -> object:
    if not options:
        raise ValueError(f"No options available for {title}")

    print(title)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option.name}")

    while True:
        choice = input("Select: ").strip()

        try:
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        except ValueError:
            pass

        choice = choice.upper()
        for option in options:
            if option.name == choice:
                return option

        print("Enter one of the listed numbers or names.")


def naturalPathKey(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def getReferenceImages(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []

    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=naturalPathKey,
    )


def createStageGridImage(stages: list[tuple[str, np.ndarray]], columns: int) -> np.ndarray:
    columns = min(max(1, columns), len(stages))
    rows = math.ceil(len(stages)/columns)
    gap = 8
    title_height = 38

    display_stages = []

    for stage_name, stage_image in stages:
        if stage_image.ndim == 2:
            stage_image = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
        elif stage_image.shape[2] == 4:
            stage_image = cv2.cvtColor(stage_image, cv2.COLOR_BGRA2BGR)
        else:
            stage_image = stage_image.copy()

        display_stages.append((stage_name, stage_image))

    image_width = max(image.shape[1] for _, image in display_stages)
    image_height = max(image.shape[0] for _, image in display_stages)
    tile_width = image_width
    tile_height = image_height + title_height

    grid_width = columns*tile_width + (columns + 1)*gap
    grid_height = rows*tile_height + (rows + 1)*gap
    grid_image = np.full((grid_height, grid_width, 3), 25, dtype=np.uint8)

    for stage_index, (stage_name, stage_image) in enumerate(display_stages):
        row = stage_index//columns
        column = stage_index%columns
        tile_x = gap + column*(tile_width + gap)
        tile_y = gap + row*(tile_height + gap)

        stage_height, stage_width = stage_image.shape[:2]
        image_x = tile_x + (image_width - stage_width)//2
        image_y = tile_y + title_height + (image_height - stage_height)//2

        grid_image[image_y:image_y + stage_height, image_x:image_x + stage_width] = stage_image

        cv2.rectangle(
            grid_image,
            (tile_x, tile_y),
            (tile_x + tile_width - 1, tile_y + tile_height - 1),
            (70, 70, 70), 1,
        )

        cv2.putText(
            grid_image,
            f"{stage_index + 1}. {stage_name}",
            (tile_x + 10, tile_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    return grid_image


def saveDetectionStages(stages: list[tuple[str, np.ndarray]], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for old_image_path in output_directory.glob("*.png"):
        old_image_path.unlink()

    for stage_index, (stage_name, stage_image) in enumerate(stages):
        safe_stage_name = re.sub(r"[^A-Za-z0-9_-]+", "_", stage_name).strip("_")
        output_path = output_directory/f"{stage_index:02d}_{safe_stage_name}.png"

        if not cv2.imwrite(str(output_path), stage_image):
            raise RuntimeError(f"Could not save debug image: {output_path}")

    print(f"Saved {len(stages)} stages to {output_directory.resolve()}")


def showDetectionStages(
    stages: list[tuple[str, np.ndarray]], columns: int, output_directory: Path,
    image_name: str, image_index: int, image_count: int,
) -> int:
    """Return -1 for previous image, +1 for next image, 0 to quit."""
    if not stages:
        print("No debug stages were recorded.")
        return 0

    canvas = createStageGridImage(stages, columns)
    canvas_height, canvas_width = canvas.shape[:2]

    state = {
        "window_width": DEFAULT_WINDOW_WIDTH, "window_height": DEFAULT_WINDOW_HEIGHT,
        "zoom": 1.0, "offset_x": 0.0, "offset_y": 0.0,
        "dragging": False, "last_mouse_x": 0, "last_mouse_y": 0,
    }

    def getViewportHeight() -> int:
        return max(1, state["window_height"] - STATUS_BAR_HEIGHT)

    def clampView() -> None:
        viewport_height = getViewportHeight()
        scaled_width, scaled_height = canvas_width*state["zoom"], canvas_height*state["zoom"]

        if scaled_width <= state["window_width"]:
            state["offset_x"] = (state["window_width"] - scaled_width)/2
        else:
            state["offset_x"] = min(0.0, max(state["window_width"] - scaled_width, state["offset_x"]))

        if scaled_height <= viewport_height:
            state["offset_y"] = (viewport_height - scaled_height)/2
        else:
            state["offset_y"] = min(0.0, max(viewport_height - scaled_height, state["offset_y"]))

    def resetView() -> None:
        viewport_height = getViewportHeight()
        state["zoom"] = min(state["window_width"]/canvas_width, viewport_height/canvas_height, 1.0)
        state["offset_x"] = (state["window_width"] - canvas_width*state["zoom"])/2
        state["offset_y"] = (viewport_height - canvas_height*state["zoom"])/2

    def zoomAt(mouse_x: int, mouse_y: int, zoom_factor: float) -> None:
        viewport_height = getViewportHeight()
        mouse_x = min(max(mouse_x, 0), state["window_width"])
        mouse_y = min(max(mouse_y, 0), viewport_height)

        minimum_zoom = min(state["window_width"]/canvas_width, viewport_height/canvas_height, 1.0)
        old_zoom = state["zoom"]
        new_zoom = min(MAX_ZOOM, max(minimum_zoom, old_zoom*zoom_factor))

        canvas_x = (mouse_x - state["offset_x"])/old_zoom
        canvas_y = (mouse_y - state["offset_y"])/old_zoom

        state["zoom"] = new_zoom
        state["offset_x"] = mouse_x - canvas_x*new_zoom
        state["offset_y"] = mouse_y - canvas_y*new_zoom
        clampView()

    def mouseCallback(event: int, x: int, y: int, flags: int, _) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            zoomAt(x, y, ZOOM_STEP if flags > 0 else 1.0/ZOOM_STEP)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            resetView()
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"], state["last_mouse_x"], state["last_mouse_y"] = True, x, y
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["offset_x"] += x - state["last_mouse_x"]
            state["offset_y"] += y - state["last_mouse_y"]
            state["last_mouse_x"], state["last_mouse_y"] = x, y
            clampView()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback)
    resetView()

    last_window_size = (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        _, _, window_width, window_height = cv2.getWindowImageRect(WINDOW_NAME)
        window_width = max(1, window_width)
        window_height = max(STATUS_BAR_HEIGHT + 1, window_height)

        if (window_width, window_height) != last_window_size:
            old_viewport_height = getViewportHeight()
            center_canvas_x = (state["window_width"]/2 - state["offset_x"])/state["zoom"]
            center_canvas_y = (old_viewport_height/2 - state["offset_y"])/state["zoom"]

            state["window_width"], state["window_height"] = window_width, window_height
            state["offset_x"] = window_width/2 - center_canvas_x*state["zoom"]
            state["offset_y"] = getViewportHeight()/2 - center_canvas_y*state["zoom"]

            clampView()
            last_window_size = (window_width, window_height)

        viewport_height = getViewportHeight()
        transform = np.float32([
            [state["zoom"], 0.0, state["offset_x"]],
            [0.0, state["zoom"], state["offset_y"]],
        ])

        display_image = np.full((window_height, window_width, 3), 25, dtype=np.uint8)
        display_image[:viewport_height] = cv2.warpAffine(
            canvas, transform, (window_width, viewport_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(25, 25, 25),
        )

        cv2.rectangle(display_image, (0, viewport_height), (window_width, window_height), (35, 35, 35), -1)

        cv2.putText(
            display_image,
            f"Image {image_index + 1}/{image_count}: {image_name}",
            (10, viewport_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display_image,
            f"A/Left: previous    D/Right: next    Wheel or +/-: zoom    Drag: pan    R: reset    S: save    Q/Esc: quit    Zoom: {state['zoom']:.2f}x",
            (10, viewport_height + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (225, 225, 225), 1, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKeyEx(16)

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(WINDOW_NAME)
            return 0
        elif key in (ord("a"), ord("A"), ord("p"), ord("P")) or key in LEFT_ARROW_KEYS:
            cv2.destroyWindow(WINDOW_NAME)
            return -1
        elif key in (ord("d"), ord("D"), ord("n"), ord("N")) or key in RIGHT_ARROW_KEYS:
            cv2.destroyWindow(WINDOW_NAME)
            return 1
        elif key in (ord("r"), ord("R"), ord("0")):
            resetView()
        elif key in (ord("+"), ord("=")):
            zoomAt(state["window_width"]//2, getViewportHeight()//2, ZOOM_STEP)
        elif key in (ord("-"), ord("_")):
            zoomAt(state["window_width"]//2, getViewportHeight()//2, 1.0/ZOOM_STEP)
        elif key in (ord("s"), ord("S")):
            saveDetectionStages(stages, output_directory)

    cv2.destroyWindow(WINDOW_NAME)
    return 0



def createMainEquivalentBenchmarkStage(
    frame: np.ndarray, object_vision_spec_id: ObjectVisionSpecId,
    camera_calibration: CameraCalibration, debug: DetectionDebug,
) -> np.ndarray:
    """Benchmark the exact public detection call used by main.py on this static frame."""
    for _ in range(MAIN_BENCHMARK_WARMUP_RUNS):
        detectSingleObject(frame, object_vision_spec_id, camera_calibration)

    elapsed_times_ms = []

    for _ in range(MAIN_BENCHMARK_TIMED_RUNS):
        start_s = time.perf_counter()
        detectSingleObject(frame, object_vision_spec_id, camera_calibration)
        elapsed_times_ms.append((time.perf_counter() - start_s)*1000.0)

    elapsed_times_ms = np.asarray(elapsed_times_ms, dtype=np.float64)
    median_ms = float(np.median(elapsed_times_ms))
    mean_ms = float(np.mean(elapsed_times_ms))
    minimum_ms = float(np.min(elapsed_times_ms))
    maximum_ms = float(np.max(elapsed_times_ms))

    internal_total_ms = debug.timings_ms.get("TOTAL vision")
    delta_ms = None if internal_total_ms is None else median_ms - internal_total_ms

    print()
    print("MAIN-equivalent detectSingleObject benchmark:")
    print(f"  warm-up runs: {MAIN_BENCHMARK_WARMUP_RUNS}")
    print(f"  timed runs: {MAIN_BENCHMARK_TIMED_RUNS}")
    print(f"  median: {median_ms:.2f} ms")
    print(f"  mean:   {mean_ms:.2f} ms")
    print(f"  min:    {minimum_ms:.2f} ms")
    print(f"  max:    {maximum_ms:.2f} ms")

    if internal_total_ms is not None:
        print(f"  internal staged total: {internal_total_ms:.2f} ms")
        print(f"  median - internal:     {delta_ms:+.2f} ms")

    benchmark_image = np.full_like(frame, 25)
    lines = [
        "MAIN-EQUIVALENT detectSingleObject BENCHMARK",
        f"Warm-up: {MAIN_BENCHMARK_WARMUP_RUNS} calls | Timed: {MAIN_BENCHMARK_TIMED_RUNS} calls",
        "",
        f"Median: {median_ms:.2f} ms",
        f"Mean:   {mean_ms:.2f} ms",
        f"Min:    {minimum_ms:.2f} ms",
        f"Max:    {maximum_ms:.2f} ms",
    ]

    if internal_total_ms is not None:
        lines.extend([
            "",
            f"Internal staged total: {internal_total_ms:.2f} ms",
            f"Median - internal:     {delta_ms:+.2f} ms",
        ])

    lines.extend([
        "",
        "This calls the same detectSingleObject(...) path as main.py.",
        "Static repeated image: useful sanity check, not live-motion timing.",
    ])

    y = 34
    for line_index, line in enumerate(lines):
        if not line:
            y += 14
            continue

        font_scale = 0.62 if line_index == 0 else 0.54
        thickness = 2 if line_index == 0 else 1

        cv2.putText(
            benchmark_image, line, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (245, 245, 245), thickness, cv2.LINE_AA,
        )
        y += 28

    return benchmark_image

def runDetectionOnImage(
    image_path: Path, algorithm: DetectionAlgorithm, object_vision_spec_id: ObjectVisionSpecId,
    object_vision_spec, camera_calibration: CameraCalibration,
) -> tuple[list[tuple[str, np.ndarray]], object, object]:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")

    frame_height, frame_width = frame.shape[:2]
    if frame_width != config.FRAME_W or frame_height != config.FRAME_H:
        raise ValueError(
            f"Reference image is {frame_width}x{frame_height}, "
            f"but config expects {config.FRAME_W}x{config.FRAME_H}."
        )

    debug = DetectionDebug()
    detection = None
    measurement = None

    if algorithm == DetectionAlgorithm.SHAPE_GROUP:
        detection = findSingleObjectUsingBestShapeGroup(frame, object_vision_spec, debug)
        if detection is not None:
            measurement = createMeasurementUsingShapeGroup(detection, object_vision_spec, camera_calibration)

    elif algorithm == DetectionAlgorithm.SPHERE:
        detection = findSingleObjectSphere(frame, object_vision_spec, debug)
        if detection is not None:
            x, y, z = estimateObjectWorldPosition(
                detection.u, detection.v, detection.px_w, detection.px_h,
                object_vision_spec.width, camera_calibration,
            )
            measurement = Measurement(x, y, z)

    elif algorithm == DetectionAlgorithm.ARUCO_MARKER:
        _, detection, measurement = detectArucoMarkerV2(
            frame, object_vision_spec, camera_calibration, debug,
        )

    else:
        raise ValueError(f"Unsupported detection algorithm: {algorithm}")

    print()
    print(f"Input image: {image_path.resolve()}")
    print(f"Image size: {frame_width} x {frame_height}")

    detection_available = detection is not None and detection.u is not None

    if not detection_available:
        print("Detection result: no object detected")
        print("Measurement result: unavailable")
    else:
        print("Detection result:")
        print(f"  u: {detection.u:.2f} px")
        print(f"  v: {detection.v:.2f} px")
        print(f"  width: {detection.px_w:.2f} px")
        print(f"  height: {detection.px_h:.2f} px")
        print(f"  shapes: {len(detection.shapes)}")

        final_frame = frame.copy()
        drawDetection(final_frame, detection)

        measurement_available = (
            measurement is not None
            and measurement.x is not None
            and measurement.y is not None
            and measurement.z is not None
        )

        if measurement_available:
            print("Measurement result:")
            print(f"  x: {measurement.x:.4f} m")
            print(f"  y: {measurement.y:.4f} m")
            print(f"  z: {measurement.z:.4f} m")

            measurement_lines = [
                f"x: {measurement.x:.4f} m",
                f"y: {measurement.y:.4f} m",
                f"z: {measurement.z:.4f} m",
            ]

            if algorithm == DetectionAlgorithm.SHAPE_GROUP:
                drawModelOrigin(final_frame, measurement, camera_calibration)
                print("  yellow X: projected model origin")
                measurement_lines.append("yellow X: model origin")
        else:
            print("Measurement result: unavailable")
            measurement_lines = ["Measurement unavailable"]

        for line_index, line in enumerate(measurement_lines):
            line_y = 35 + line_index*28
            cv2.putText(final_frame, line, (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(final_frame, line, (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        debug.addStage("Final detection + measurement", final_frame)

    benchmark_stage = createMainEquivalentBenchmarkStage(
        frame, object_vision_spec_id, camera_calibration, debug,
    )
    debug.addStage("Main-equivalent benchmark", benchmark_stage)

    print(f"Recorded stages: {len(debug.stages)}")
    return debug.stages, detection, measurement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a selected detection algorithm on reference images and inspect its processing stages."
    )

    parser.add_argument("image_path", type=Path, nargs="?", default=None,
                        help="Optional initial image. If omitted, reference_1.png (or the first image) in the selected spec folder is used.")

    parser.add_argument("--algorithm", choices=[algorithm.name for algorithm in DetectionAlgorithm],
                        help="Detection algorithm. If omitted, you will be required to select one.")

    parser.add_argument("--spec", choices=[spec_id.name for spec_id in OBJECT_VISION_SPECS],
                        help="Registered ObjectVisionSpecId. If omitted, you will be required to select a compatible one.")

    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIRECTORY,
                        help=f"Base reference-image directory. Default: {DEFAULT_REFERENCE_DIRECTORY}")

    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIRECTORY,
                        help=f"Base debug-image directory. Default: {DEFAULT_SAVE_DIRECTORY}")

    parser.add_argument("--columns", type=int, default=DEFAULT_DISPLAY_COLUMNS,
                        help=f"Number of stage-grid columns. Default: {DEFAULT_DISPLAY_COLUMNS}")

    parser.add_argument("--no-gui", action="store_true", help="Do not open the OpenCV stage viewer.")

    args = parser.parse_args()

    if args.columns < 1:
        parser.error("--columns must be at least 1.")

    object_vision_spec_id = ObjectVisionSpecId[args.spec] if args.spec is not None else None

    if args.algorithm is not None:
        algorithm = DetectionAlgorithm[args.algorithm]
    else:
        available_algorithms = list(DetectionAlgorithm)

        if object_vision_spec_id is not None:
            object_type = OBJECT_VISION_SPECS[object_vision_spec_id].object_type
            available_algorithms = [
                candidate for candidate in DetectionAlgorithm
                if object_type in ALGORITHM_OBJECT_TYPES[candidate]
            ]

        algorithm = chooseOption("Available detection algorithms:", available_algorithms)

    compatible_spec_ids = [
        spec_id for spec_id, spec in OBJECT_VISION_SPECS.items()
        if spec.object_type in ALGORITHM_OBJECT_TYPES[algorithm]
    ]

    if object_vision_spec_id is None:
        object_vision_spec_id = chooseOption(
            f"ObjectVisionSpecIds compatible with {algorithm.name}:",
            compatible_spec_ids,
        )
    elif object_vision_spec_id not in compatible_spec_ids:
        parser.error(
            f"{object_vision_spec_id.name} is not compatible with detection algorithm {algorithm.name}. "
            f"Compatible specs: {', '.join(spec_id.name for spec_id in compatible_spec_ids)}"
        )

    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]
    reference_directory = args.reference_dir/object_vision_spec_id.name.lower()
    output_directory = args.save_dir/object_vision_spec_id.name.lower()

    # If an explicit image is supplied, navigate among other images in that image's folder.
    # Otherwise navigate the selected ObjectVisionSpecId's normal reference folder.
    if args.image_path is not None:
        if not args.image_path.is_file():
            parser.error(f"Image does not exist: {args.image_path}")
        image_paths = getReferenceImages(args.image_path.parent)
        if args.image_path not in image_paths:
            image_paths.append(args.image_path)
            image_paths.sort(key=naturalPathKey)
        image_index = image_paths.index(args.image_path)
    else:
        image_paths = getReferenceImages(reference_directory)
        if not image_paths:
            parser.error(f"No reference images found in: {reference_directory}")

        preferred_path = reference_directory/"reference_1.png"
        image_index = image_paths.index(preferred_path) if preferred_path in image_paths else 0

    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)

    print(f"Algorithm: {algorithm.name}")
    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print(f"Reference images: {len(image_paths)}")
    print(f"Debug save directory: {output_directory.resolve()}")

    if args.no_gui:
        runDetectionOnImage(
            image_paths[image_index], algorithm, object_vision_spec_id, object_vision_spec, camera_calibration,
        )
        return

    print("Viewer: A/Left/P = previous | D/Right/N = next | S = save stages | Q/Esc = quit")

    while True:
        image_path = image_paths[image_index]
        stages, _, _ = runDetectionOnImage(
            image_path, algorithm, object_vision_spec_id, object_vision_spec, camera_calibration,
        )

        navigation = showDetectionStages(
            stages, args.columns, output_directory,
            image_path.name, image_index, len(image_paths),
        )

        if navigation == 0:
            break

        image_index = (image_index + navigation) % len(image_paths)


if __name__ == "__main__":
    main()
