from enum import Enum, auto
import numpy as np


class ColorId(Enum):
    TENNIS_GREEN = auto()
    GREEN = auto()
    CYAN = auto()
    MAGENTA = auto()


class ColorSpec:
    def __init__(
        self,
        hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
        draw_bgr: tuple[int, int, int],
    ):
        self.hsv_ranges = hsv_ranges
        self.draw_bgr = draw_bgr


COLOR_SPECS = {
    ColorId.TENNIS_GREEN: ColorSpec(
        hsv_ranges=[
            (
                np.array([23, 35, 110], dtype=np.uint8),
                np.array([40, 220, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
    ),

    ColorId.GREEN: ColorSpec(
        hsv_ranges=[
            (
                np.array([40, 25, 100], dtype=np.uint8),
                np.array([80, 255, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
    ),

    ColorId.CYAN: ColorSpec(
        hsv_ranges=[
            (
                np.array([82, 40, 100], dtype=np.uint8),
                np.array([105, 255, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(255, 255, 0),
    ),

    ColorId.MAGENTA: ColorSpec(
        hsv_ranges=[
            (
                np.array([135, 120, 100], dtype=np.uint8),
                np.array([170, 255, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(255, 0, 255),
    ),
}