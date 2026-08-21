import bisect
import math


class ServoCalibration:
    """
    Converts a requested mechanism angle into the PWM pulse width required by one servo.

    angle_trim_deg shifts the command into this servo's calibration-angle coordinate:
        servo_angle_deg = cmd_angle_deg + angle_trim_deg

    In a correct nonlinear calibration, the same trim should be obtained from any
    accurately known physical reference pose (for example 30 deg or 90 deg). If the
    required trim changes with the reference pose, the polynomial/lookup model is
    imperfect. For pan/tilt we still define and measure trim at the physical forward
    pose by convention so its meaning is unambiguous.

    Optional pulse models are evaluated AFTER that input shift:
        - polynomial: pulse_us = f(servo_angle_deg)
        - lookup:     pulse_us = interpolate(servo_angle_deg)
        - neither:    normal linear min_angle/max_angle -> min_pulse/max_pulse mapping

    Polynomial coefficients use descending powers around pulse_polynomial_reference_deg:
        x = servo_angle_deg - pulse_polynomial_reference_deg
        pulse_us = cN*x**N + ... + c1*x + c0

    Lookup rows are:
        ((servo_angle_deg, pulse_us), ...)

    Inside the lookup range, pulse width is linearly interpolated. Outside the
    tested lookup range, endpoint-linear extrapolation can optionally be enabled
    up to pulse_lookup_extrapolation_angle_range_deg. Below the table, the slope
    of the first two points is continued; above it, the slope of the last two
    points is continued. None means no extrapolation.

    This is servo calibration, not platform-level motion safety.
    """
    def __init__(self, min_angle_deg: float=0.0, max_angle_deg: float=180.0, min_pulse_us: float=500.0, max_pulse_us: float=2500.0,
                 angle_trim_deg: float=0.0, pulse_polynomial_coefficients_descending: tuple[float, ...] | None=None,
                 pulse_polynomial_reference_deg: float=0.0, pulse_polynomial_valid_angle_range_deg: tuple[float, float] | None=None,
                 pulse_lookup_table: tuple[tuple[float, float], ...] | None=None,
                 pulse_lookup_extrapolation_angle_range_deg: tuple[float, float] | None=None):
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)
        self.min_pulse_us = float(min_pulse_us)
        self.max_pulse_us = float(max_pulse_us)
        self.angle_trim_deg = float(angle_trim_deg)
        self.pulse_polynomial_reference_deg = float(pulse_polynomial_reference_deg)
        self.pulse_polynomial_coefficients_descending = None if pulse_polynomial_coefficients_descending is None else tuple(float(x) for x in pulse_polynomial_coefficients_descending)
        self.pulse_polynomial_valid_angle_range_deg = None if pulse_polynomial_valid_angle_range_deg is None else tuple(float(x) for x in pulse_polynomial_valid_angle_range_deg)
        self.pulse_lookup_table = None if pulse_lookup_table is None else tuple((float(angle), float(pulse)) for angle, pulse in pulse_lookup_table)
        self.pulse_lookup_extrapolation_angle_range_deg = None if pulse_lookup_extrapolation_angle_range_deg is None else tuple(float(x) for x in pulse_lookup_extrapolation_angle_range_deg)
        self.validate()

    def validate(self) -> None:
        values = [self.min_angle_deg, self.max_angle_deg, self.min_pulse_us, self.max_pulse_us, self.angle_trim_deg, self.pulse_polynomial_reference_deg]
        if not all(math.isfinite(x) for x in values): raise ValueError("ServoCalibration values must be finite")
        if self.max_angle_deg <= self.min_angle_deg: raise ValueError("max_angle_deg must be greater than min_angle_deg")
        if self.max_pulse_us <= self.min_pulse_us: raise ValueError("max_pulse_us must be greater than min_pulse_us")
        if self.pulse_polynomial_coefficients_descending is not None and self.pulse_lookup_table is not None:
            raise ValueError("Use either pulse_polynomial_coefficients_descending or pulse_lookup_table, not both")

        if self.pulse_polynomial_coefficients_descending is not None:
            if len(self.pulse_polynomial_coefficients_descending) < 2:
                raise ValueError("Pulse polynomial needs at least slope + constant; use None for the normal linear mapping")
            if not all(math.isfinite(x) for x in self.pulse_polynomial_coefficients_descending):
                raise ValueError("Pulse polynomial coefficients must be finite")

        if self.pulse_polynomial_coefficients_descending is not None and self.pulse_polynomial_valid_angle_range_deg is None:
            raise ValueError("pulse_polynomial_valid_angle_range_deg is required when a pulse polynomial is used")
        if self.pulse_polynomial_coefficients_descending is None and self.pulse_polynomial_valid_angle_range_deg is not None:
            raise ValueError("pulse_polynomial_valid_angle_range_deg is only valid with a pulse polynomial")

        if self.pulse_polynomial_valid_angle_range_deg is not None:
            lo, hi = self.pulse_polynomial_valid_angle_range_deg
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                raise ValueError("pulse_polynomial_valid_angle_range_deg must be (min, max) with min < max")
            if lo < self.min_angle_deg or hi > self.max_angle_deg:
                raise ValueError("Pulse polynomial valid range must lie inside min_angle_deg/max_angle_deg")

        if self.pulse_lookup_table is None and self.pulse_lookup_extrapolation_angle_range_deg is not None:
            raise ValueError("pulse_lookup_extrapolation_angle_range_deg is only valid with a pulse lookup table")

        if self.pulse_lookup_table is not None:
            if len(self.pulse_lookup_table) < 2: raise ValueError("pulse_lookup_table must contain at least two points")
            angles = [angle for angle, _ in self.pulse_lookup_table]
            pulses = [pulse for _, pulse in self.pulse_lookup_table]
            if not all(math.isfinite(x) for x in angles + pulses): raise ValueError("Lookup-table values must be finite")
            if any(b <= a for a, b in zip(angles[:-1], angles[1:])): raise ValueError("Lookup-table servo angles must be strictly increasing")
            if angles[0] < self.min_angle_deg or angles[-1] > self.max_angle_deg:
                raise ValueError("Lookup-table angle range must lie inside min_angle_deg/max_angle_deg")
            if any(pulse < self.min_pulse_us or pulse > self.max_pulse_us for pulse in pulses):
                raise ValueError("Lookup-table pulse widths must lie inside min_pulse_us/max_pulse_us")

            if self.pulse_lookup_extrapolation_angle_range_deg is not None:
                extrap_lo, extrap_hi = self.pulse_lookup_extrapolation_angle_range_deg
                if not math.isfinite(extrap_lo) or not math.isfinite(extrap_hi) or extrap_hi <= extrap_lo:
                    raise ValueError("pulse_lookup_extrapolation_angle_range_deg must be (min, max) with min < max")
                if extrap_lo < self.min_angle_deg or extrap_hi > self.max_angle_deg:
                    raise ValueError("Lookup extrapolation range must lie inside min_angle_deg/max_angle_deg")
                if extrap_lo > angles[0] or extrap_hi < angles[-1]:
                    raise ValueError("Lookup extrapolation range must contain the entire lookup-table angle range")

    def cmd_to_servo_angle_deg(self, cmd_angle_deg: float) -> float:
        cmd_angle_deg = float(cmd_angle_deg)
        if not math.isfinite(cmd_angle_deg): raise ValueError("cmd_angle_deg must be finite")
        return cmd_angle_deg + self.angle_trim_deg

    def cmd_to_pulse_us(self, cmd_angle_deg: float) -> float:
        return self.servo_angle_to_pulse_us(self.cmd_to_servo_angle_deg(cmd_angle_deg))

    def servo_angle_to_pulse_us(self, servo_angle_deg: float) -> float:
        servo_angle_deg = self._validate_servo_angle(servo_angle_deg)

        if self.pulse_polynomial_coefficients_descending is not None:
            if self.pulse_polynomial_valid_angle_range_deg is not None:
                lo, hi = self.pulse_polynomial_valid_angle_range_deg
                if servo_angle_deg < lo or servo_angle_deg > hi:
                    raise ValueError(f"servo_angle_deg={servo_angle_deg} is outside polynomial calibration range [{lo}, {hi}]")
            pulse_us = self._polynomial_pulse_us(servo_angle_deg)
        elif self.pulse_lookup_table is not None:
            pulse_us = self._lookup_pulse_us(servo_angle_deg)
        else:
            pulse_us = self.baseline_angle_to_pulse_us(servo_angle_deg)

        if pulse_us < self.min_pulse_us or pulse_us > self.max_pulse_us:
            raise ValueError(f"Calibrated pulse_us={pulse_us} is outside allowed range [{self.min_pulse_us}, {self.max_pulse_us}]")
        return pulse_us

    def baseline_angle_to_pulse_us(self, angle_deg: float) -> float:
        """Linear angle->pulse mapping only: no trim, polynomial, or lookup table."""
        angle_deg = self._validate_servo_angle(angle_deg)
        alpha = (angle_deg - self.min_angle_deg)/(self.max_angle_deg - self.min_angle_deg)
        return self.min_pulse_us + alpha*(self.max_pulse_us - self.min_pulse_us)

    def _polynomial_pulse_us(self, servo_angle_deg: float) -> float:
        x = servo_angle_deg - self.pulse_polynomial_reference_deg
        pulse_us = 0.0
        for coefficient in self.pulse_polynomial_coefficients_descending:
            pulse_us = pulse_us*x + coefficient
        return pulse_us

    def _lookup_pulse_us(self, servo_angle_deg: float) -> float:
        table = self.pulse_lookup_table
        angles = [row[0] for row in table]

        # Outside the tested range, optionally continue only the nearest endpoint
        # segment's slope. This is deliberately linear extrapolation, never a
        # higher-order continuation of the whole lookup table.
        if servo_angle_deg < angles[0]:
            if self.pulse_lookup_extrapolation_angle_range_deg is None:
                raise ValueError(f"servo_angle_deg={servo_angle_deg} is below pulse lookup-table range [{angles[0]}, {angles[-1]}] and lookup extrapolation is disabled")
            extrap_lo, _ = self.pulse_lookup_extrapolation_angle_range_deg
            if servo_angle_deg < extrap_lo:
                raise ValueError(f"servo_angle_deg={servo_angle_deg} is below lookup extrapolation minimum {extrap_lo}")
            angle_0, pulse_0 = table[0]
            angle_1, pulse_1 = table[1]
            slope_us_per_deg = (pulse_1 - pulse_0)/(angle_1 - angle_0)
            return pulse_0 + slope_us_per_deg*(servo_angle_deg - angle_0)

        if servo_angle_deg > angles[-1]:
            if self.pulse_lookup_extrapolation_angle_range_deg is None:
                raise ValueError(f"servo_angle_deg={servo_angle_deg} is above pulse lookup-table range [{angles[0]}, {angles[-1]}] and lookup extrapolation is disabled")
            _, extrap_hi = self.pulse_lookup_extrapolation_angle_range_deg
            if servo_angle_deg > extrap_hi:
                raise ValueError(f"servo_angle_deg={servo_angle_deg} is above lookup extrapolation maximum {extrap_hi}")
            angle_0, pulse_0 = table[-2]
            angle_1, pulse_1 = table[-1]
            slope_us_per_deg = (pulse_1 - pulse_0)/(angle_1 - angle_0)
            return pulse_1 + slope_us_per_deg*(servo_angle_deg - angle_1)

        index = bisect.bisect_left(angles, servo_angle_deg)
        if index < len(table) and angles[index] == servo_angle_deg: return table[index][1]

        angle_0, pulse_0 = table[index - 1]
        angle_1, pulse_1 = table[index]
        alpha = (servo_angle_deg - angle_0)/(angle_1 - angle_0)
        return pulse_0 + alpha*(pulse_1 - pulse_0)

    def _validate_servo_angle(self, angle_deg: float) -> float:
        angle_deg = float(angle_deg)
        if not math.isfinite(angle_deg): raise ValueError("Servo angle must be finite")
        if not self.min_angle_deg <= angle_deg <= self.max_angle_deg:
            raise ValueError(f"servo_angle_deg={angle_deg} is outside calibration range [{self.min_angle_deg}, {self.max_angle_deg}]")
        return angle_deg