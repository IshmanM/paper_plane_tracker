import argparse
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera.camera_calibration import CameraCalibration


IMAGE_DIRECTORY = Path("images/camera_calibration")

CHECKERBOARD_SIZE = (9, 6)  # number of inner corners: columns, rows
SQUARE_SIZE_M = 0.0228      # physical checkerboard square width in metres


def main(save: bool = True, output_path: str | Path = config.CAMERA_CALIBRATION_PATH) -> CameraCalibration:
    # Create the known 3D positions of every checkerboard corner.
    # The checkerboard is treated as flat, so all z coordinates are 0.
    object_points_template = np.zeros((CHECKERBOARD_SIZE[0]*CHECKERBOARD_SIZE[1], 3), dtype=np.float32)
    object_points_template[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points_template *= SQUARE_SIZE_M

    object_points: list[np.ndarray] = []  # known 3D checkerboard points for each accepted image
    image_points: list[np.ndarray] = []   # detected 2D pixel positions for each accepted image
    image_size = None
    image_paths = sorted(IMAGE_DIRECTORY.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No PNG calibration images found in {IMAGE_DIRECTORY.resolve()}")

    # Detect the checkerboard corners in every calibration image.
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # All calibration images must use the same camera resolution.
        if image_size is None:
            image_size = gray.shape[::-1]
        elif gray.shape[::-1] != image_size:
            raise ValueError(f"{image_path.name} has a different resolution from the other calibration images.")

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

        if not found:
            print(f"Rejected: {image_path.name}")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        object_points.append(object_points_template.copy())
        image_points.append(corners)
        print(f"Accepted: {image_path.name}")

    if image_size is None:
        raise RuntimeError("No readable calibration images found.")
    if len(object_points) < 10:
        raise RuntimeError(f"Only {len(object_points)} valid checkerboard images found; use at least 10.")

    rms_error, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
    )
    calibration = CameraCalibration.fromValues(
        image_width_px=image_size[0], image_height_px=image_size[1],
        camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients,
        rms_reprojection_error_px=rms_error,
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print("\n--- CAMERA CALIBRATION RESULTS ---")
    print(f"Valid images: {len(object_points)}/{len(image_paths)}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error: {rms_error:.4f} px")
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion_coefficients.ravel())
    print("\nIndividual parameters:")
    print(f"fx = {fx:.6f} px")
    print(f"fy = {fy:.6f} px")
    print(f"cx = {cx:.6f} px")
    print(f"cy = {cy:.6f} px")

    if save:
        calibration.save(output_path)
        print(f"\nSaved calibration to {Path(output_path).resolve()}")
    else:
        print("\nCalibration not saved.")

    return calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true", help="Compute and print calibration without saving it.")
    parser.add_argument("--output", type=Path, default=config.CAMERA_CALIBRATION_PATH, help="JSON output path; defaults to config.CAMERA_CALIBRATION_PATH.")
    args = parser.parse_args()
    main(save=not args.no_save, output_path=args.output)