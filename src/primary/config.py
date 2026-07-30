import numpy as np

FRAME_W = 640
FRAME_H = 480
FPS = 60
PX_FOCAL_LENGTH = 500  # depends on frame width/height and = to the average of a few (reference_pixel_width * reference_distance / reference_width).

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