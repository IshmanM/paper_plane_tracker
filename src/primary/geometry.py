import numpy as np
import cv2

from src.primary.camera_calibration import CameraCalibration
from src.primary.camera_to_platform_calibration import CameraToPlatformCalibration



#TODO: ensure the -1 determinant is preserved when validating camera_to_platform_calibration 

# # todo: getting rid of this once replaced
# # need to tune platform rotation and translation according to design...
# CAMERA_TO_PLATFORM_R = np.array([
#     [1.0,  0.0, 0.0],
#     [0.0, -1.0, 0.0], # -1.0 since +y is up
#     [0.0,  0.0, 1.0],
# ])

# # todo: getting rid of this once replaced
# # meters camera x, y, z relative to platform origin
# CAMERA_ORIGIN_IN_PLATFORM = np.array([-0.5, 0.15, 0.0], dtype=float) 



# image frame to world frame relative to camera lens

def estimateObjectWorldPosition(u, v, px_w, px_h, object_w, camera_calibration: CameraCalibration):
    camera_matrix = camera_calibration.camera_matrix
    distortion_coefficients = camera_calibration.distortion_coefficients
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]

    # Estimate depth using whichever detected dimension occupies more normalized image space.
    z = object_w/max(px_w/fx, px_h/fy)

    # Undistort the detected center and convert it to normalized camera coordinates.
    center_px = np.array([[[u, v]]], dtype=np.float64)
    normalized_center = cv2.undistortPoints(center_px, camera_matrix, distortion_coefficients)
    normalized_x, normalized_y = normalized_center[0, 0]

    x = normalized_x*z
    y = normalized_y*z

    return x, y, z


# world frame to image frame relative to camera lens

def estimateObjectImagePosition(x, y, z, camera_calibration: CameraCalibration):
    if z <= 0:
        return 0, 0

    object_point = np.array([[[x, y, z]]], dtype=np.float64)
    image_point, _ = cv2.projectPoints(
        object_point, np.zeros(3), np.zeros(3),
        camera_calibration.camera_matrix, camera_calibration.distortion_coefficients,
    )

    u, v = image_point[0, 0]
    return float(u), float(v)


# world frame to platform frame. Assuming platform position = servo position.
# todo: replace CAMERA_TO_PLATFORM_R use with the new camera_to_platform_calibration stuff
def estimateObjectPlatformPosition(world_position: np.ndarray, camera_to_platform_calibration: CameraToPlatformCalibration) -> np.ndarray:
    """
    Convert object position from camera/world frame to platform frame.

    Returns:
        np.ndarray shape (3,), object position relative to the platform origin.

    Convention:
        p_P = R_PC @ p_C + t_PC
    """

    p_C = np.asarray(world_position, dtype=float).reshape(-1)

    if p_C.shape != (3,) or not np.all(np.isfinite(p_C)):
        raise ValueError(f"world_position must be a finite shape-(3,) array, got {p_C.shape}")

    # Vision/OpenCV (+y down) -> camera-relative robot convention (+y up).
    p_C[1] *= -1.0

    return camera_to_platform_calibration.transformPosition(p_C)


# For use in platform.py and calibration
def rotationPlatformFromPanTilt(platform_yaw_rad: float, platform_theta_rad: float) -> np.ndarray:
    """
    Rotation caused by the pan/tilt joints.
    +yaw points right and +theta points up.
    """

    cy, sy = np.cos(platform_yaw_rad), np.sin(platform_yaw_rad)
    ct, st = np.cos(platform_theta_rad), np.sin(platform_theta_rad)

    rotation_yaw = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])

    rotation_theta = np.array([
        [1.0, 0.0, 0.0],
        [0.0, ct, st],
        [0.0, -st, ct],
    ])

    return rotation_yaw@rotation_theta