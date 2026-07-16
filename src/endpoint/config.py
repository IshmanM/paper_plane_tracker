import numpy as np

from src.endpoint.drivers.servo_calibration import ServoCalibration

PCA9685_FREQUENCY_HZ = 50.0
PCA9685_NUM_CHANNELS = 16

PAN_CHANNEL = 0
TILT_CHANNEL = 1
FOAM_CHANNEL = 2



DEFAULT_PAN_ANGLE = 90.0
DEFAULT_TILT_ANGLE = 45.0


TILT_SERVO_OFFSET_DEG = 10.0


DEFAULT_SERVO_CALIBRATION = ServoCalibration(
    min_angle_deg=0.0,
    max_angle_deg=180.0,
    min_pulse_us=500.0,
    max_pulse_us=2500.0,
    angle_trim_deg=0
)

# Todo: tune these
SERVO_CALIBRATIONS = {
    PAN_CHANNEL: ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=5
    ),

    TILT_CHANNEL: ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=10
    ),

    FOAM_CHANNEL: ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=0
    ),
}

# Todo: calibrate these...
FOAM_RESET_ANGLE_DEG = 180
FOAM_RESET_HOLD_DELAY = 0.2 # seconds
FOAM_TRIGGER_ANGLE_DEG = 85
FOAM_TRIGGER_HOLD_DELAY = 0.25 # seconds