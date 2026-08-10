from pathlib import Path
import json

import numpy as np


class CameraCalibration:
    def __init__(self, json_path: str | Path, image_width_px: int, image_height_px: int):
        self.json_path = Path(json_path)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if "calibrations" not in data or not isinstance(data["calibrations"], dict):
            raise ValueError(f"Camera calibration JSON must contain a 'calibrations' object: {self.json_path}")

        calibration_key = self.getCalibrationKey(self.image_width_px, self.image_height_px)
        if calibration_key not in data["calibrations"]:
            available = ", ".join(sorted(data["calibrations"].keys())) or "none"
            raise KeyError(f"No camera calibration for {calibration_key} in {self.json_path}. Available: {available}")

        calibration_data = data["calibrations"][calibration_key]
        self.camera_matrix = np.asarray(calibration_data["camera_matrix"], dtype=np.float64)
        self.distortion_coefficients = np.asarray(calibration_data["distortion_coefficients"], dtype=np.float64).reshape(1, -1)
        self.rms_reprojection_error_px = float(calibration_data["rms_reprojection_error_px"])
        self._validate()

    @staticmethod
    def getCalibrationKey(image_width_px: int, image_height_px: int) -> str:
        return f"{int(image_width_px)}x{int(image_height_px)}"

    @classmethod
    def fromValues(
        cls, image_width_px: int, image_height_px: int, camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray, rms_reprojection_error_px: float,
    ) -> "CameraCalibration":
        calibration = cls.__new__(cls)
        calibration.json_path = None
        calibration.image_width_px = int(image_width_px)
        calibration.image_height_px = int(image_height_px)
        calibration.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        calibration.distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64).reshape(1, -1)
        calibration.rms_reprojection_error_px = float(rms_reprojection_error_px)
        calibration._validate()
        return calibration

    def _validate(self) -> None:
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must have shape (3, 3), got {self.camera_matrix.shape}")
        if self.distortion_coefficients.size < 4:
            raise ValueError("distortion_coefficients must contain at least 4 values")

    def save(self, json_path: str | Path | None = None) -> None:
        output_path = Path(json_path) if json_path is not None else self.json_path
        if output_path is None:
            raise ValueError("json_path is required when saving a calibration created with fromValues()")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if "calibrations" not in data or not isinstance(data["calibrations"], dict):
                raise ValueError(f"Camera calibration JSON must contain a 'calibrations' object: {output_path}")
        else:
            data = {"calibrations": {}}

        calibration_key = self.getCalibrationKey(self.image_width_px, self.image_height_px)
        data["calibrations"][calibration_key] = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.reshape(-1).tolist(),
            "rms_reprojection_error_px": self.rms_reprojection_error_px,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        self.json_path = output_path