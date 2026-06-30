import src.primary.config as config
import numpy as np


# need to tune platform rotation and translation according to design...
CAMERA_TO_PLATFORM_R = np.array([
    [1.0,  0.0, 0.0],
    [0.0, -1.0, 0.0], # -1.0 since +y is up
    [0.0,  0.0, 1.0],
])

# meters camera x, y, z relative to platform origin
CAMERA_ORIGIN_IN_PLATFORM = np.array([-0.2, 0.15, 0.0], dtype=float) 



# image frame to world frame relative to camera lens

def estimateTargetWorldPosition(u, v, px_w, px_h):
    # Depth estimator based on emperical calibration & Pose Converter
    z = config.PX_FOCAL_LENGTH*config.W/max(px_w, px_h)
    # Position estimate based on depth estimate
    x = (u - config.FRAME_W/2)*z/config.PX_FOCAL_LENGTH 
    y = (v - config.FRAME_H/2)*z/config.PX_FOCAL_LENGTH 
    return x, y, z


# world frame to image frame relative to camera lens

def estimateTargetImagePosition(x, y, z):
    if z <= 0:
        return 0, 0 
    u = (x * config.PX_FOCAL_LENGTH / z) + config.FRAME_W / 2
    v = (y * config.PX_FOCAL_LENGTH / z) + config.FRAME_H / 2
    return u, v



# world frame to platform frame. Assuming platform position = servo position.

def estimateTargetPlatformPosition(world_position: np.ndarray) -> np.ndarray:
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
