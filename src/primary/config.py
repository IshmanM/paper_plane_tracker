from pathlib import Path

import numpy as np

FRAME_W = 640
FRAME_H = 480
FPS = 60

# todo: adjust calibration as needed

# PX_FOCAL_LENGTH = 500  # depends on frame width/height and = to the average of a few (reference_pixel_width * reference_distance / reference_width).

# CAMERA_MATRIX = np.array([
#     [633.737631, 0.0, 308.887194],
#     [0.0, 633.992320, 236.067349],
#     [0.0, 0.0, 1.0],
# ], dtype=np.float64)

# DISTORTION_COEFFICIENTS = np.array([
#     [0.09131862, -0.47045404, 0.00119751, 0.00289575, 0.76219894]
# ], dtype=np.float64)

# CAMERA_MATRIX = np.array([
#     [640.820092, 0.0, 298.963328],
#     [0.0, 639.770129, 225.482995],
#     [0.0, 0.0, 1.0],
# ], dtype=np.float64)

# DISTORTION_COEFFICIENTS = np.array([
#     [0.09141723, -0.39690385, -0.00317970, -0.00211780, 0.48441084]
# ], dtype=np.float64)

CAMERA_CALIBRATION_NAME = "asus_laptop_webcam"
CAMERA_CALIBRATION_PATH = Path(f"src/primary/calibration_data/camera/{CAMERA_CALIBRATION_NAME}_results.json")

CAMERA_TO_PLATFORM_CALIBRATION_PATH = Path("src/primary/calibration_data/camera_to_platform/results.json")

CAMERA_INDEX = 0

# Best for Tennis ball: 
# CAMERA_AUTO_EXPOSURE = False
# CAMERA_EXPOSURE = -5.0 
# CAMERA_GAIN = 0.0 
# CAMERA_AUTO_WHITE_BALANCE = False
# CAMERA_WHITE_BALANCE_TEMPERATURE = 4600.0
# CAMERA_AUTOFOCUS = False
# CAMERA_FOCUS = 0  # tune

# Best for moving ArUco:
CAMERA_AUTO_EXPOSURE = False
CAMERA_EXPOSURE = -7.5       # start here; try -10 if blur remains
CAMERA_GAIN = 45.0           # instead of 0; tune upward if image is dark
CAMERA_AUTO_WHITE_BALANCE = False
CAMERA_WHITE_BALANCE_TEMPERATURE = 4600.0
CAMERA_AUTOFOCUS = False
CAMERA_FOCUS = 0             # tune specifically for ~3-5 m


CMD_FREQUENCY_HZ = 120.0 # hz
CMD_THREAD_MAX_DELAY = 1.0/CMD_FREQUENCY_HZ

UDP_TX_DELAY = 0.0005 # seconds
ENDPOINT_CMD_MAX_DELAY = 1.0/120.0 # 8.33 ms. change if endpoint slows


CMD_SMOOTHING_TAU = 0.005 # seconds. used in comm_buffer.py


SERVO_NAMES = ("pan", "tilt")
NUM_SERVOS = len(SERVO_NAMES)
SERVO_IDX = {
    name: i for i, name in enumerate(SERVO_NAMES)
}

DEFAULT_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
DEFAULT_SERVO_ANGLES[SERVO_IDX["pan"]] = 90.0 # degrees
DEFAULT_SERVO_ANGLES[SERVO_IDX["tilt"]] = 75.0 # degrees

FORWARD_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
FORWARD_SERVO_ANGLES[SERVO_IDX["pan"]] = 90.0 # degrees
FORWARD_SERVO_ANGLES[SERVO_IDX["tilt"]] = 90.0 # degrees

MIN_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
MIN_SERVO_ANGLES[SERVO_IDX["pan"]] = 5.0 # degrees
MIN_SERVO_ANGLES[SERVO_IDX["tilt"]] = 30.0 # degrees

MAX_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
MAX_SERVO_ANGLES[SERVO_IDX["pan"]] = 175.0 # degrees
MAX_SERVO_ANGLES[SERVO_IDX["tilt"]] = 115.0 # degrees

SERVO_DEADBAND = np.zeros(NUM_SERVOS, dtype=float)
SERVO_DEADBAND[SERVO_IDX["pan"]] = 0.1 # degrees
SERVO_DEADBAND[SERVO_IDX["tilt"]] = 0.1 # degrees

MAX_SERVO_SPEEDS = np.zeros(NUM_SERVOS, dtype=float)
MAX_SERVO_SPEEDS[SERVO_IDX["pan"]] = 390.0 # degrees/s. TODO: slow this if too fast
MAX_SERVO_SPEEDS[SERVO_IDX["tilt"]] = 390.0 # degrees/s TODO: slow this if too fast

# # Calibration biases after testing. <--Todo: maybe remove. this might be duplicate of the trim offset oon the endpoint side.
# SERVO_BIASES = np.zeros(NUM_SERVOS, dtype=float)
# SERVO_BIASES[SERVO_IDX["pan"]] = 0.0 # degrees
# SERVO_BIASES[SERVO_IDX["tilt"]] = 0.0 # degrees

# Sign depends on physical servo mounting.
SERVO_SIGNS = np.zeros(NUM_SERVOS, dtype=float)
SERVO_SIGNS[SERVO_IDX["pan"]] = 1.0 # +yaw is physical left; positive yaw increases this servo angle.
SERVO_SIGNS[SERVO_IDX["tilt"]] = -1.0 # +theta is physical up; positive theta decreases this servo angle.