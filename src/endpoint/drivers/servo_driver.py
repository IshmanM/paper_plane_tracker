import math
from adafruit_pca9685 import PCA9685

from src.endpoint.drivers.servo_calibration import ServoCalibration


class ServoDriver:
    """
    Low-level servo driver for the endpoint.

    This class owns the Adafruit PCA9685 object.

    Responsibilities:
        - create PCA9685 using i2c
        - set PCA9685 PWM frequency
        - convert servo angle degrees into PWM pulse width
        - convert pulse width into PCA9685 duty_cycle
        - command arbitrary PCA9685 servo channels

    Not responsible for:
        - UDP messages
        - endpoint safe mode
        - command timeout
        - pan/tilt meaning
        - triggering meaning
        - platform state machine
        - platform-level min/max angle safety
    """
    def __init__(self, i2c, frequency_hz: float = 50.0, num_channels: int = 16, default_calibration: ServoCalibration | None = None):

        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")

        if num_channels <= 0:
            raise ValueError("num_channels must be positive")

        if default_calibration is None:
            self.default_calibration = ServoCalibration()
        else:
            self.default_calibration = default_calibration
            self.default_calibration.validate()

        self.frequency_hz = float(frequency_hz)
        self.num_channels = int(num_channels)

        # Optional per-channel calibration.
        # Example:
        #     channel 0 might use 500-2500 us
        #     channel 1 might use 700-2300 us
        #
        # If a channel is not listed here, default_calibration is used.
        self.channel_calibrations: dict[int, ServoCalibration] = {}

        # Last commanded values are only what we commanded, not measured servo position.
        self.last_angle_deg: dict[int, float] = {}
        self.last_pulse_us: dict[int, float] = {}

        self.i2c = i2c
        self.pca = PCA9685(self.i2c)

        # PCA9685 has one shared PWM frequency for all channels.
        self.pca.frequency = self.frequency_hz


    def set_channel_calibration(self, channel: int, calibration: ServoCalibration) -> None:
        """
        Set a custom PWM conversion calibration for one servo channel.
        """
        self._validate_channel(channel)
        calibration.validate()
        self.channel_calibrations[channel] = calibration


    def set_angle_deg(self, channel: int, angle_deg: float, clamp_to_calibration: bool=False) -> float:
        """
        Command one servo channel by angle.

        clamp_to_calibration=True:
            angles outside the calibration range are clipped.

        clamp_to_calibration=False:
            angles outside the calibration range raise an error.

        Returns:
            the actual angle used after optional calibration-range clamping.
        """
        self._validate_channel(channel)

        if not math.isfinite(angle_deg):
            raise ValueError("angle_deg must be finite")

        calibration = self.get_calibration(channel)

        used_angle_deg = calibration.cmd_to_adjusted_angle_deg(float(angle_deg))


        if clamp_to_calibration:
            used_angle_deg = self._clip(
                used_angle_deg,
                calibration.min_angle_deg,
                calibration.max_angle_deg,
            )
        else:
            if not (
                calibration.min_angle_deg
                <= used_angle_deg
                <= calibration.max_angle_deg
            ):
                raise ValueError(
                    f"angle_deg={used_angle_deg} is outside calibration range "
                    f"[{calibration.min_angle_deg}, {calibration.max_angle_deg}]"
                )

        pulse_us = self._angle_to_pulse_us(used_angle_deg, calibration)
        self.set_pulse_us(channel, pulse_us, clamp_to_calibration=clamp_to_calibration)

        self.last_angle_deg[channel] = used_angle_deg

        return used_angle_deg


    def set_pulse_us(self, channel: int, pulse_us: float, clamp_to_calibration: bool=False) -> float:
        """
        Command one servo channel by raw pulse width.

        This is mostly for testing/calibration. Normal platform code should
        usually call set_angle_deg().
        """

        self._validate_channel(channel)

        if not math.isfinite(pulse_us):
            raise ValueError("pulse_us must be finite")

        calibration = self.get_calibration(channel)

        used_pulse_us = float(pulse_us)

        if clamp_to_calibration:
            used_pulse_us = self._clip(
                used_pulse_us,
                calibration.min_pulse_us,
                calibration.max_pulse_us,
            )
        else:
            if not (
                calibration.min_pulse_us
                <= used_pulse_us
                <= calibration.max_pulse_us
            ):
                raise ValueError(
                    f"pulse_us={used_pulse_us} is outside calibration range "
                    f"[{calibration.min_pulse_us}, {calibration.max_pulse_us}]"
                )

        duty_cycle = self._pulse_us_to_duty_cycle(used_pulse_us)

        self.pca.channels[channel].duty_cycle = duty_cycle
        self.last_pulse_us[channel] = used_pulse_us

        return used_pulse_us


    def release_channel(self, channel: int) -> None:
        """
        Stop sending PWM on one channel.

        For positional servos, this usually means the servo stops actively
        holding torque. Do not use this if you need the platform to hold pose.
        """

        self._validate_channel(channel)

        self.pca.channels[channel].duty_cycle = 0

        self.last_angle_deg.pop(channel, None)
        self.last_pulse_us.pop(channel, None)
    

    def release_all(self) -> None:
        for channel in range(self.num_channels):
            self.release_channel(channel)


    def close(self, release: bool = False) -> None:
        """
        Clean up the servo driver.

        release=False:
            leave PWM outputs at their last commanded values.

        release=True:
            set all duty cycles to zero before closing.
        """

        if release:
            self.release_all()

        if hasattr(self.pca, "deinit"):
            self.pca.deinit()


    def get_last_angle_deg(self, channel: int) -> float | None:
        self._validate_channel(channel)
        return self.last_angle_deg.get(channel)


    def get_last_pulse_us(self, channel: int) -> float | None:
        self._validate_channel(channel)
        return self.last_pulse_us.get(channel)


    def get_calibration(self, channel: int) -> ServoCalibration:
        return self.channel_calibrations.get(channel, self.default_calibration)


    def _angle_to_pulse_us(self, angle_deg: float, calibration: ServoCalibration) -> float:
        """
        Linearly map angle to pulse width.

        Example with default calibration:
            0 deg   -> 500 us
            90 deg  -> 1500 us
            180 deg -> 2500 us
        """

        angle_span = calibration.max_angle_deg - calibration.min_angle_deg
        pulse_span = calibration.max_pulse_us - calibration.min_pulse_us

        alpha = (angle_deg - calibration.min_angle_deg) / angle_span

        return calibration.min_pulse_us + alpha * pulse_span


    def _pulse_us_to_duty_cycle(self, pulse_us: float) -> int:
        """
        Convert pulse width into Adafruit PCA9685 duty_cycle value.

        At 50 Hz:
            period = 20,000 us

        Example:
            pulse_us = 1500
            duty fraction = 1500 / 20000 = 0.075
            duty_cycle = 0.075 * 65535 ~= 4915
        """

        period_us = 1_000_000.0 / self.frequency_hz
        duty_fraction = pulse_us / period_us

        duty_cycle = int(round(duty_fraction * 0xFFFF))

        return int(self._clip(duty_cycle, 0, 0xFFFF))


    def _validate_channel(self, channel: int) -> None:
        if not isinstance(channel, int):
            raise TypeError("channel must be an int")

        if channel < 0 or channel >= self.num_channels:
            raise ValueError(
                f"channel must be in [0, {self.num_channels - 1}], got {channel}"
            )
        

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return min(max(x, lo), hi)



