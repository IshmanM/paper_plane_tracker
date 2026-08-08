import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np

from src.primary.config import FRAME_W, FRAME_H
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpecId
from src.primary.detection import DetectionDebug, findSingleObjectUsingBestShapeGroup, drawDetection, createMeasurementUsingShapeGroup


WINDOW_NAME = "Detection stages"
DEFAULT_REFERENCE_IMAGE_PATH = Path("images/primary_detection_references/reference_1.png")
DEFAULT_SAVE_DIRECTORY = Path("images/primary_detection_debug")
DEFAULT_DISPLAY_COLUMNS = 3
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 900

STATUS_BAR_HEIGHT = 32
ZOOM_STEP = 1.25
MAX_ZOOM = 10.0

DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.PAPER_PLANE_SHAPES_1
PAPER_PLANE_SHAPE_SPEC_IDS = [
    spec_id for spec_id, spec in OBJECT_VISION_SPECS.items()
    if spec.object_type == ObjectType.PAPER_PLANE_SHAPES
]


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
        cv2.rectangle(grid_image, (tile_x, tile_y), (tile_x + tile_width - 1, tile_y + tile_height - 1),
                      (70, 70, 70), 1)
        cv2.putText(grid_image, f"{stage_index + 1}. {stage_name}", (tile_x + 10, tile_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return grid_image


def showDetectionStages(stages: list[tuple[str, np.ndarray]], columns: int) -> None:
    if not stages:
        print("No debug stages were recorded.")
        return

    canvas = createStageGridImage(stages, columns)
    canvas_height, canvas_width = canvas.shape[:2]

    state = {
        "window_width": DEFAULT_WINDOW_WIDTH,
        "window_height": DEFAULT_WINDOW_HEIGHT,
        "zoom": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "dragging": False,
        "last_mouse_x": 0,
        "last_mouse_y": 0,
    }

    def getViewportHeight() -> int:
        return max(1, state["window_height"] - STATUS_BAR_HEIGHT)

    def clampView() -> None:
        viewport_height = getViewportHeight()
        scaled_width = canvas_width*state["zoom"]
        scaled_height = canvas_height*state["zoom"]

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
            state["dragging"] = True
            state["last_mouse_x"] = x
            state["last_mouse_y"] = y
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["offset_x"] += x - state["last_mouse_x"]
            state["offset_y"] += y - state["last_mouse_y"]
            state["last_mouse_x"] = x
            state["last_mouse_y"] = y
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

            state["window_width"] = window_width
            state["window_height"] = window_height
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
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25)
        )

        cv2.rectangle(display_image, (0, viewport_height), (window_width, window_height), (35, 35, 35), -1)
        cv2.putText(
            display_image,
            f"Wheel or +/-: zoom    Drag: pan    R/double-click: reset    Q/Esc: quit    Zoom: {state['zoom']:.2f}x",
            (10, viewport_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("r"), ord("0")):
            resetView()
        elif key in (ord("+"), ord("=")):
            zoomAt(state["window_width"]//2, getViewportHeight()//2, ZOOM_STEP)
        elif key in (ord("-"), ord("_")):
            zoomAt(state["window_width"]//2, getViewportHeight()//2, 1.0/ZOOM_STEP)

    cv2.destroyAllWindows()


def saveDetectionStages(stages: list[tuple[str, np.ndarray]], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for stage_index, (stage_name, stage_image) in enumerate(stages):
        safe_stage_name = re.sub(r"[^A-Za-z0-9_-]+", "_", stage_name).strip("_")
        output_path = output_directory/f"{stage_index:02d}_{safe_stage_name}.png"

        if not cv2.imwrite(str(output_path), stage_image):
            raise RuntimeError(f"Could not save debug image: {output_path}")

    print(f"Saved {len(stages)} stages to {output_directory.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection on one image and inspect its processing stages.")
    parser.add_argument("image_path", type=Path, nargs="?", default=DEFAULT_REFERENCE_IMAGE_PATH,
                        help=f"Input image path. Default: {DEFAULT_REFERENCE_IMAGE_PATH}")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIRECTORY,
                        help=f"Directory for saved stages. Default: {DEFAULT_SAVE_DIRECTORY}")
    parser.add_argument("--columns", type=int, default=DEFAULT_DISPLAY_COLUMNS,
                        help=f"Number of stage-grid columns. Default: {DEFAULT_DISPLAY_COLUMNS}")
    parser.add_argument("--no-gui", action="store_true", help="Do not open the OpenCV stage viewer.")
    parser.add_argument(
        "--spec", default=DEFAULT_OBJECT_VISION_SPEC_ID.name,
        choices=[spec_id.name for spec_id in PAPER_PLANE_SHAPE_SPEC_IDS],
        help=f"Registered paper-plane ObjectVisionSpecId. Default: {DEFAULT_OBJECT_VISION_SPEC_ID.name}",
    )
    args = parser.parse_args()

    if not args.image_path.is_file():
        parser.error(f"Image does not exist: {args.image_path}")

    if args.columns < 1:
        parser.error("--columns must be at least 1.")

    frame = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"OpenCV could not read: {args.image_path}")

    frame_height, frame_width = frame.shape[:2]

    if frame_width != FRAME_W or frame_height != FRAME_H:
        raise ValueError(f"Reference image is {frame_width}x{frame_height}, but config expects {FRAME_W}x{FRAME_H}.")

    debug = DetectionDebug()
    object_vision_spec_id = ObjectVisionSpecId[args.spec]
    object_vision_spec = OBJECT_VISION_SPECS[object_vision_spec_id]
    detection = findSingleObjectUsingBestShapeGroup(frame, object_vision_spec, debug)

    print(f"Input image: {args.image_path.resolve()}")
    print(f"ObjectVisionSpecId: {object_vision_spec_id.name}")
    print(f"Image size: {frame_width} x {frame_height}")

    if detection is None:
        print("Detection result: no object detected")
        print("Measurement result: unavailable")
    else:
        print("Detection result:")
        print(f"  u: {detection.u:.2f} px")
        print(f"  v: {detection.v:.2f} px")
        print(f"  width: {detection.px_w:.2f} px")
        print(f"  height: {detection.px_h:.2f} px")
        print(f"  shapes: {len(detection.shapes)}")

        measurement = createMeasurementUsingShapeGroup(detection, object_vision_spec)
        measurement_frame = frame.copy()
        drawDetection(measurement_frame, detection)

        if measurement.x is None or measurement.y is None or measurement.z is None:
            print("Measurement result: unavailable")
            measurement_lines = ["Measurement unavailable"]
        else:
            print("Measurement result:")
            print(f"  x: {measurement.x:.4f} m")
            print(f"  y: {measurement.y:.4f} m")
            print(f"  z: {measurement.z:.4f} m")

            measurement_lines = [
                f"x: {measurement.x:.4f} m",
                f"y: {measurement.y:.4f} m",
                f"z: {measurement.z:.4f} m",
            ]

        # Add the world-space measurement as the final displayed and saved stage.
        text_x, text_y = 20, 35

        for line_index, line in enumerate(measurement_lines):
            line_y = text_y + line_index*28
            cv2.putText(measurement_frame, line, (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(measurement_frame, line, (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        debug.addStage("World-space measurement", measurement_frame)

    print(f"Recorded stages: {len(debug.stages)}")

    saveDetectionStages(debug.stages, args.save_dir)

    if not args.no_gui:
        showDetectionStages(debug.stages, args.columns)

if __name__ == "__main__":
    main()