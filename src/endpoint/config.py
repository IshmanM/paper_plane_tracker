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
        # endpoint --no-servo-calibration after the final driver changes before enabling this.
        # x = servo_angle_deg - 110; pulse_us = c2*x^2 + c1*x + c0
        pulse_polynomial_coefficients_descending=(0.000134301131282, 2.40649407868e-05, 9.68513977329, 1500.0,),
        pulse_polynomial_reference_deg=90.0,
        pulse_polynomial_valid_angle_range_deg=(16.043, 164.340),

        # Same OLD pan run represented as a lookup table instead of a polynomial.
        # pulse_lookup_table=(
        #     (16.043, 722.2),
        #     (24.446, 833.3),
        #     (34.377, 944.4),
        #     (45.073, 1055.6),
        #     (56.387, 1166.7),
        #     (67.846, 1277.8),
        #     (78.540, 1388.9),
        #     (90.000, 1500.0),
        #     (100.695, 1611.1),
        #     (111.391, 1722.2),
        #     (123.614, 1833.3),
        #     (134.928, 1944.4),
        #     (146.005, 2055.6),
        #     (154.790, 2166.7),
        #     (164.340, 2277.8),
        # ),
        # pulse_lookup_extrapolation_angle_range_deg=(0.0, 180.0),
    ),

    TILT_CHANNEL: ServoCalibration(
        min_angle_deg=20.0,
        max_angle_deg=160.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        angle_trim_deg=7.0,

        # x = servo_angle_deg - 78.025; pulse_us = c3*x^3 + c2*x^2 + c1*x + c0
        # pulse_polynomial_coefficients_descending=(0.000348793116, 0.000390340314, 9.13116963, 1389.04346),
        # pulse_polynomial_reference_deg=78.025,
        # pulse_polynomial_valid_angle_range_deg=(32.95, 123.10),

        pulse_lookup_table=(
            (32.95, 944.444444),
            (42.80, 1055.555556),
            (54.05, 1166.666667),
            (66.10, 1277.777778),
            (78.30, 1388.888889),
            (90.00, 1500.000000),
            (101.50, 1611.111111),
            (113.00, 1722.222222),
            (123.10, 1833.333333),
        ),
        pulse_lookup_extrapolation_angle_range_deg=(20.0, 160.0),
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
FOAM_MOTOR_SPEED_MAGNITUDE = 0.85 # MAX MAGNITUDE IS 1.0

# Time required for the flywheels DC motors to reach required speed.
FOAM_MOTOR_SPINUP_DELAY = 3.5