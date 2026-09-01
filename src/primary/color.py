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
            # Normal / shadowed green, including weak-saturation edges.
            (
                np.array([62, 55, 85], dtype=np.uint8),
                np.array([84, 155, 195], dtype=np.uint8),
            ),

            # Bright / washed-out green.
            (
                np.array([68, 85, 180], dtype=np.uint8),
                np.array([86, 155, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(0, 255, 0),
        lab_value=np.array([180, 105, 137], dtype=np.uint8),
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
                np.array([162, 75, 170], dtype=np.uint8),
                np.array([179, 150, 255], dtype=np.uint8),
            ),
        ],
        draw_bgr=(180, 105, 255),
        lab_value=np.array([217, 150, 124], dtype=np.uint8),
    ),
}