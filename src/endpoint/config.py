import numpy as np

from src.endpoint.drivers.servo_calibration import ServoCalibration

PCA9685_FREQUENCY_HZ = 50.0
PCA9685_NUM_CHANNELS = 16

# PCA9685 oscillator calibration measured with debug_pca9685_clock at 3.3 V logic VCC.
# Changing PCA VCC can shift the oscillator frequency, so remeasure if that changes.
PCA9685_REFERENCE_CLOCK_FREQUENCY_HZ = 24_516_000


PAN_CHANNEL = 0
TILT_CHANNEL = 1
FOAM_CHANNEL = 2


DEFAULT_PAN_ANGLE = 90.0
DEFAULT_TILT_ANGLE = 60.0


DEFAULT_SERVO_CALIBRATION = ServoCalibration(
    min_angle_deg=0.0,
    max_angle_deg=180.0,
    min_pulse_us=500.0,
    max_pulse_us=2500.0,
    angle_trim_deg=0
)

# Trim shifts the INPUT to the servo's pulse model. Ideally the same trim would be
# obtained from any accurately known angle (e.g. 30 deg or 90 deg); if not, the
# nonlinear model is imperfect. Pan/tilt trim is still measured at forward pose by convention.
# Todo: tune these
SERVO_CALIBRATIONS = {
    PAN_CHANNEL: ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=0,
        # Candidate pulse-width quadratic reconstructed from pan_20260821_152514. OLD RUN: rerun with
        # endpoint USE_SERVO_CALIBRATION=False after the final driver changes before enabling this.
        # x = servo_angle_deg - 110; pulse_us = c2*x^2 + c1*x + c0
        # pulse_polynomial_coefficients_descending=(-0.0237015538, 9.79857658, 1720.90106),
        # pulse_polynomial_reference_deg=110.0, pulse_polynomial_valid_angle_range_deg=(88.51, 134.27),
        # For a generated lookup table, endpoint-linear extrapolation can be allowed
        # from the tested table endpoints out to the servo calibration limits:
        # pulse_lookup_extrapolation_angle_range_deg=(0.0, 180.0),
    ),

    TILT_CHANNEL: ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=7.0
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
FOAM_RESET_ANGLE_DEG = 165
FOAM_RESET_HOLD_DELAY = 0.20 # seconds
FOAM_TRIGGER_ANGLE_DEG = 53
FOAM_TRIGGER_HOLD_DELAY = 0.20 # seconds
FOAM_MOTOR_SPEED_MAGNITUDE = 0.75 # MAX MAGNITUDE IS 1.0

# Time required for the flywheels DC motors to reach required speed.
FOAM_MOTOR_SPINUP_DELAY = 3.5