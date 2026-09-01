from enum import Enum, auto
import numpy as np


class ColorId(Enum):
    TENNIS_GREEN = auto()
    GREEN = auto()
    CYAN = auto()
    MAGENTA = auto()
    FERN = auto()
    ORANGE_CIRCUIT = auto()
    PINK = auto()


class ColorSpec:
    def __init__(
        self,
        hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
        draw_bgr: tuple[int, int, int],
        lab_value: np.ndarray | None = None,
    ):
        self.hsv_ranges = hsv_ranges
        self.draw_bgr = draw_bgr
        self.lab_value = lab_value


# keep the SRICT spec here, margins will be added by detection functions for looser spec
COLOR_SPECS = {
    ColorId.TENNIS_GREEN: ColorSpec(
        hsv_ranges=[
            # (
            #     np.array([27, 35, 80], dtype=np.uint8),
            #     np.array([38, 255, 255], dtype=np.uint8),
            # ),   
            # (
            #     np.array([30, 80, 80], dtype=np.uint8),
            #     np.array([42, 255, 255], dtype=np.uint8),
            # ),
            (
                np.array([27, 80, 80], dtype=np.uint8),
                np.array([40, 255, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
        lab_value=np.array([242, 95, 201], dtype=np.uint8),
    ),

    ColorId.GREEN: ColorSpec(
        hsv_ranges=[
            (
                np.array([56, 55, 90], dtype=np.uint8),
                np.array([84, 155, 225], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
        lab_value=np.array([150, 108, 137], dtype=np.uint8),
    ),

    ColorId.CYAN: ColorSpec(
        hsv_ranges=[
            (
                np.array([88, 85, 160], dtype=np.uint8),
                np.array([106, 255, 255], dtype=np.uint8),
            ),
        ],
        lab_value=np.array([174, 113, 103], dtype=np.uint8),
        draw_bgr=(255, 255, 0),
    ),

    ColorId.MAGENTA: ColorSpec(
        hsv_ranges=[
            (
                np.array([135, 120, 100], dtype=np.uint8),
                np.array([170, 255, 255], dtype=np.uint8),
            ),
        ],
        lab_value=np.array([114, 193, 91], dtype=np.uint8),
        draw_bgr=(255, 0, 255),
    ),

    ColorId.FERN: ColorSpec(
        hsv_ranges=[
            (
                np.array([45, 35, 55], dtype=np.uint8),
                np.array([72, 165, 235], dtype=np.uint8),
            ),
        ],
        lab_value=np.array([100, 111, 144], dtype=np.uint8),
        draw_bgr=(0, 255, 0),
    ),

    ColorId.ORANGE_CIRCUIT: ColorSpec(
        hsv_ranges=[
            (
                np.array([8, 65, 85], dtype=np.uint8),
                np.array([23, 220, 220], dtype=np.uint8),
            ),
        ],
        lab_value=np.array([120, 141, 156], dtype=np.uint8),
        draw_bgr=(0, 165, 255),
    ),

    ColorId.PINK: ColorSpec(
        hsv_ranges=[
            (
                np.array([168, 85, 175], dtype=np.uint8),
                np.array([179, 135, 245], dtype=np.uint8),
            ),
        ],
        draw_bgr=(180, 105, 255),
        lab_value=np.array([176, 155, 131], dtype=np.uint8),
    ),
}