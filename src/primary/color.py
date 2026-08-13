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


# keep the SRICT spec here, margins will be added by detection functions for looser spec
COLOR_SPECS = {
    ColorId.TENNIS_GREEN: ColorSpec(
        hsv_ranges=[
            # (
            #     np.array([27, 35, 80], dtype=np.uint8),
            #     np.array([38, 255, 255], dtype=np.uint8),
            # ),
              
            (
                np.array([27, 75, 80], dtype=np.uint8),
                np.array([38, 255, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
    ),

    ColorId.GREEN: ColorSpec(
        hsv_ranges=[
            (
                np.array([60, 50, 65], dtype=np.uint8),
                np.array([82, 180, 235], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
    ),

    ColorId.CYAN: ColorSpec(
        hsv_ranges=[
            (
                np.array([90, 80, 90], dtype=np.uint8),
                np.array([106, 220, 185], dtype=np.uint8),
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