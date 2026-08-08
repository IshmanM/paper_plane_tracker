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

CAMERA_MATRIX = np.array([
    [640.820092, 0.0, 298.963328],
    [0.0, 639.770129, 225.482995],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

DISTORTION_COEFFICIENTS = np.array([
    [0.09141723, -0.39690385, -0.00317970, -0.00211780, 0.48441084]
], dtype=np.float64)

CAMERA_INDEX = 0
CAMERA_AUTO_EXPOSURE = False
CAMERA_EXPOSURE = -5.0 #originally -6.0
CAMERA_GAIN = 0.0
CAMERA_AUTO_WHITE_BALANCE = False
CAMERA_WHITE_BALANCE_TEMPERATURE = 4600.0
CAMERA_AUTOFOCUS = False
CAMERA_FOCUS = 0  # tune


UDP_TX_DELAY = 0.001 # seconds

SERVO_NAMES = ("pan", "tilt")
NUM_SERVOS = len(SERVO_NAMES)
SERVO_IDX = {
    name: i for i, name in enumerate(SERVO_NAMES)
}

DEFAULT_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
DEFAULT_SERVO_ANGLES[SERVO_IDX["pan"]] = 90.0 # degrees
DEFAULT_SERVO_ANGLES[SERVO_IDX["tilt"]] = 60.0 # degrees

FORWARD_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
FORWARD_SERVO_ANGLES[SERVO_IDX["pan"]] = 90.0 # degrees
FORWARD_SERVO_ANGLES[SERVO_IDX["tilt"]] = 90.0 # degrees

MIN_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
MIN_SERVO_ANGLES[SERVO_IDX["pan"]] = 0.0 # degrees
MIN_SERVO_ANGLES[SERVO_IDX["tilt"]] = 0.0 # degrees

MAX_SERVO_ANGLES = np.zeros(NUM_SERVOS, dtype=float)
MAX_SERVO_ANGLES[SERVO_IDX["pan"]] = 175.0 # degrees
MAX_SERVO_ANGLES[SERVO_IDX["tilt"]] = 115.0 # degrees

SERVO_DEADBAND = np.zeros(NUM_SERVOS, dtype=float)
SERVO_DEADBAND[SERVO_IDX["pan"]] = 0.5 # degrees
SERVO_DEADBAND[SERVO_IDX["tilt"]] = 0.5 # degrees

MAX_SERVO_SPEEDS = np.zeros(NUM_SERVOS, dtype=float)
MAX_SERVO_SPEEDS[SERVO_IDX["pan"]] = 120.0 # degrees/s
MAX_SERVO_SPEEDS[SERVO_IDX["tilt"]] = 90.0 # degrees/s

# Calibration biases after testing. <--Todo: maybe remove. this might be duplicate of the trim offset oon the endpoint side.
SERVO_BIASES = np.zeros(NUM_SERVOS, dtype=float)
SERVO_BIASES[SERVO_IDX["pan"]] = 0.0 # degrees
SERVO_BIASES[SERVO_IDX["tilt"]] = 0.0 # degrees

# Sign depends on your physical servo mounting.
SERVO_SIGNS = np.zeros(NUM_SERVOS, dtype=float)
SERVO_SIGNS[SERVO_IDX["pan"]] = -1.0 # If aiming right makes the servo move left, flip to -1.0.
SERVO_SIGNS[SERVO_IDX["tilt"]] = -1.0 # If aiming up makes the servo move down, flip to -1.0.