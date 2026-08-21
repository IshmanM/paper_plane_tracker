import math
from adafruit_pca9685 import PCA9685

from src.endpoint.drivers.servo_calibration import ServoCalibration


class ServoDriver:
    """
    Low-level PCA9685 servo driver.

    ServoCalibration owns command-angle -> pulse-width calibration. This class owns
    PCA9685 timing and converts pulse width into the board's duty-cycle representation.

    use_calibration=True is normal operation.

    use_calibration=False is ONLY for empirical servo characterization. It bypasses
    angle trim plus polynomial/lookup calibration and sends the baseline linear
    angle->pulse mapping. Endpoint main.py should only set it False while collecting
    data with the servo calibration sweep.
    """
    def __init__(self, i2c, frequency_hz: float=50.0, num_channels: int=16, default_calibration: ServoCalibration | None=None,
                 pca_reference_clock_frequency_hz: int=25_000_000, use_calibration: bool=True):
        if not math.isfinite(float(frequency_hz)) or frequency_hz <= 0.0: raise ValueError("frequency_hz must be positive and finite")
        if num_channels <= 0: raise ValueError("num_channels must be positive")
        if isinstance(pca_reference_clock_frequency_hz, bool) or not isinstance(pca_reference_clock_frequency_hz, int) or pca_reference_clock_frequency_hz <= 0:
            raise ValueError("pca_reference_clock_frequency_hz must be a positive integer")
        if not isinstance(use_calibration, bool): raise TypeError("use_calibration must be bool")

        self.default_calibration = ServoCalibration() if default_calibration is None else default_calibration
        self.default_calibration.validate()
        self.num_channels = int(num_channels)
        self.use_calibration = use_calibration
        self.channel_calibrations: dict[int, ServoCalibration] = {}
        self.last_angle_deg: dict[int, float] = {}
        self.last_pulse_us: dict[int, float] = {}

        self.i2c = i2c
        self.pca_reference_clock_frequency_hz = int(pca_reference_clock_frequency_hz)
        self.requested_pwm_frequency_hz = float(frequency_hz)
        self.pca = PCA9685(i2c_bus=self.i2c, reference_clock_speed=self.pca_reference_clock_frequency_hz)
        self.pca.frequency = self.requested_pwm_frequency_hz

        # PCA9685 frequency is quantized by its integer prescaler. Use the actual
        # configured frequency for pulse-width -> duty-cycle conversion.
        self.actual_pwm_frequency_hz = float(self.pca.frequency)
        self.pwm_period_us = 1_000_000.0/self.actual_pwm_frequency_hz

    def set_channel_calibration(self, channel: int, calibration: ServoCalibration) -> None:
        self._validate_channel(channel)
        calibration.validate()
        self.channel_calibrations[channel] = calibration

    def set_angle_deg(self, channel: int, angle_deg: float, clamp_to_calibration: bool=False) -> float:
        self._validate_channel(channel)
        if not math.isfinite(float(angle_deg)): raise ValueError("angle_deg must be finite")
        calibration = self.get_calibration(channel)
        used_angle_deg = float(angle_deg)

        if clamp_to_calibration:
            used_angle_deg = self._clip(used_angle_deg, calibration.min_angle_deg, calibration.max_angle_deg)
        elif not calibration.min_angle_deg <= used_angle_deg <= calibration.max_angle_deg:
            raise ValueError(f"angle_deg={used_angle_deg} is outside calibration range [{calibration.min_angle_deg}, {calibration.max_angle_deg}]")

        # Characterization mode intentionally bypasses BOTH trim and nonlinear
        # calibration so the test directly measures baseline pulse width -> motion.
        pulse_us = calibration.cmd_to_pulse_us(used_angle_deg) if self.use_calibration else calibration.baseline_angle_to_pulse_us(used_angle_deg)
        self.set_pulse_us(channel, pulse_us, clamp_to_calibration=clamp_to_calibration)
        self.last_angle_deg[channel] = used_angle_deg
        return used_angle_deg

    def set_pulse_us(self, channel: int, pulse_us: float, clamp_to_calibration: bool=False) -> float:
        """Command one channel by raw pulse width; always bypasses angle calibration."""
        self._validate_channel(channel)
        if not math.isfinite(float(pulse_us)): raise ValueError("pulse_us must be finite")
        calibration = self.get_calibration(channel)
        used_pulse_us = float(pulse_us)

        if clamp_to_calibration:
            used_pulse_us = self._clip(used_pulse_us, calibration.min_pulse_us, calibration.max_pulse_us)
        elif not calibration.min_pulse_us <= used_pulse_us <= calibration.max_pulse_us:
            raise ValueError(f"pulse_us={used_pulse_us} is outside calibration range [{calibration.min_pulse_us}, {calibration.max_pulse_us}]")

        self.pca.channels[channel].duty_cycle = self._pulse_us_to_duty_cycle(used_pulse_us)
        self.last_pulse_us[channel] = used_pulse_us
        return used_pulse_us

    def release_channel(self, channel: int) -> None:
        self._validate_channel(channel)
        self.pca.channels[channel].duty_cycle = 0
        self.last_angle_deg.pop(channel, None)
        self.last_pulse_us.pop(channel, None)

    def release_all(self) -> None:
        for channel in range(self.num_channels): self.release_channel(channel)

    def close(self, release: bool=False) -> None:
        if release: self.release_all()
        if hasattr(self.pca, "deinit"): self.pca.deinit()

    def get_last_angle_deg(self, channel: int) -> float | None:
        self._validate_channel(channel)
        return self.last_angle_deg.get(channel)

    def get_last_pulse_us(self, channel: int) -> float | None:
        self._validate_channel(channel)
        return self.last_pulse_us.get(channel)

    def get_calibration(self, channel: int) -> ServoCalibration:
        return self.channel_calibrations.get(channel, self.default_calibration)

    def _pulse_us_to_duty_cycle(self, pulse_us: float) -> int:
        duty_cycle = int(round((pulse_us/self.pwm_period_us)*0xFFFF))
        return int(self._clip(duty_cycle, 0, 0xFFFF))

    def _validate_channel(self, channel: int) -> None:
        if not isinstance(channel, int): raise TypeError("channel must be an int")
        if channel < 0 or channel >= self.num_channels: raise ValueError(f"channel must be in [0, {self.num_channels - 1}], got {channel}")

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return min(max(x, lo), hi)