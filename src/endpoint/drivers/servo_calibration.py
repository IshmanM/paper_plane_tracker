import math

class ServoCalibration:
    """
    Describes how to convert a requested servo angle into a PWM pulse width.

    This is NOT the platform safety limit.

    Example:
        min_angle_deg = 0
        max_angle_deg = 180
        min_pulse_us = 500
        max_pulse_us = 2500

    means:
        0 deg   -> 500 us pulse
        90 deg  -> 1500 us pulse
        180 deg -> 2500 us pulse

    Different servos may need different pulse ranges. For example, a servo may
    buzz or hit a hard stop at 500/2500 us, so you might calibrate it to
    700/2300 us instead.

    Platform-level angle safety should happen above this class, probably in
    EndpointController or mechanism code.
    """

    def __init__(
        self,
        min_angle_deg: float = 0.0,
        max_angle_deg: float = 180.0,
        min_pulse_us: float = 500.0,
        max_pulse_us: float = 2500.0,
        angle_trim_deg: float = 0.0
    ):
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)
        self.min_pulse_us = float(min_pulse_us)
        self.max_pulse_us = float(max_pulse_us)
         # raw_servo_angle = commanded_angle + angle_trim_deg
        self.angle_trim_deg = float(angle_trim_deg)

        #note that all scaling/polynomials are applied AFTER trim deg is applied

        self.validate()


    def validate(self) -> None:
        values = [
            self.min_angle_deg,
            self.max_angle_deg,
            self.min_pulse_us,
            self.max_pulse_us,
            self.angle_trim_deg
        ]

        if not all(math.isfinite(x) for x in values):
            raise ValueError("ServoCalibration values must be finite")

        if self.max_angle_deg <= self.min_angle_deg:
            raise ValueError("max_angle_deg must be greater than min_angle_deg")

        if self.max_pulse_us <= self.min_pulse_us:
            raise ValueError("max_pulse_us must be greater than min_pulse_us")
        
    
    def cmd_to_adjusted_angle_deg(self, cmd_angle_deg: float) -> float:
        cmd_angle_deg = float(cmd_angle_deg)

        if not math.isfinite(cmd_angle_deg):
            raise ValueError("cmd_angle_deg must be finite")

        return cmd_angle_deg + self.angle_trim_deg