import src.primary.config as config
import numpy as np
import cv2


# need to tune platform rotation and translation according to design...
CAMERA_TO_PLATFORM_R = np.array([
    [1.0,  0.0, 0.0],
    [0.0, -1.0, 0.0], # -1.0 since +y is up
    [0.0,  0.0, 1.0],
])

# meters camera x, y, z relative to platform origin
CAMERA_ORIGIN_IN_PLATFORM = np.array([-0.5, 0.15, 0.0], dtype=float) 



# image frame to world frame relative to camera lens

def estimateObjectWorldPosition(u, v, px_w, px_h, object_w):
    fx, fy = config.CAMERA_MATRIX[0, 0], config.CAMERA_MATRIX[1, 1]

    # Estimate depth using whichever detected dimension occupies more normalized image space.
    z = object_w/max(px_w/fx, px_h/fy)

    # Undistort the detected center and convert it to normalized camera coordinates.
    center_px = np.array([[[u, v]]], dtype=np.float64)
    normalized_center = cv2.undistortPoints(center_px, config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS)
    normalized_x, normalized_y = normalized_center[0, 0]

    x = normalized_x*z
    y = normalized_y*z

    return x, y, z


# world frame to image frame relative to camera lens

def estimateObjectImagePosition(x, y, z):
    if z <= 0:
        return 0, 0

    object_point = np.array([[[x, y, z]]], dtype=np.float64)
    image_point, _ = cv2.projectPoints(
        object_point, np.zeros(3), np.zeros(3),
        config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS,
    )

    u, v = image_point[0, 0]
    return float(u), float(v)



# world frame to platform frame. Assuming platform position = servo position.

def estimateObjectPlatformPosition(world_position: np.ndarray) -> np.ndarray:
    """
    Convert object position from camera/world frame to platform frame.
    
    Returns:
        np.ndarray shape (3,), object position expressed relative to the platform origin.
    
    Convention:
        p_L = R_LC @ p_C + t_LC

        p_C  = object position in camera frame
        p_L  = object position in platform frame
        R_LC = rotation mapping camera-frame vectors into platform-frame vectors
        t_LC = camera origin position expressed in platform frame
    """
    p_C = np.asarray(world_position, dtype=float).reshape(-1)

    if p_C.shape != (3,):
        raise ValueError(f"world_position must have shape (3,), got {p_C.shape}")
    
    R_LC = np.asarray(CAMERA_TO_PLATFORM_R, dtype=float)
    t_LC = np.asarray(CAMERA_ORIGIN_IN_PLATFORM, dtype=float).reshape(-1)

    if R_LC.shape != (3, 3):
        raise ValueError(f"CAMERA_TO_PLATFORM_R must have shape (3, 3), got {R_LC.shape}")

    if t_LC.shape != (3,):
        raise ValueError(f"CAMERA_ORIGIN_IN_PLATFORM must have shape (3,), got {t_LC.shape}")

    p_L = R_LC @ p_C + t_LC

    return p_L
