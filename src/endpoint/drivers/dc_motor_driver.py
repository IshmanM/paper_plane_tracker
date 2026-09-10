import errno
import math
from pathlib import Path
import threading
import time

from gpiozero import DigitalOutputDevice


class _LinuxSysfsPWM:
    """
    Minimal userspace wrapper for the Linux kernel PWM sysfs interface.

    The interface is documented by the Linux kernel at:
    https://docs.kernel.org/driver-api/pwm.html

    This class is intentionally small and project-local so the motor driver
    does not depend on a separate Raspberry Pi PWM package.
    """

    _SYSFS_ROOT = Path("/sys/class/pwm")
    _EXPORT_TIMEOUT_S = 0.25
    _POLL_INTERVAL_S = 0.005

    def __init__(
        self,
        pwm_channel: int,
        hz: int,
        chip: int = 0,
    ):
        if isinstance(pwm_channel, bool) or not isinstance(pwm_channel, int):
            raise ValueError("pwm_channel must be an integer")
        if pwm_channel < 0:
            raise ValueError("pwm_channel must be non-negative")

        if isinstance(chip, bool) or not isinstance(chip, int):
            raise ValueError("chip must be an integer")
        if chip < 0:
            raise ValueError("chip must be non-negative")

        if isinstance(hz, bool) or not isinstance(hz, int) or hz <= 0:
            raise ValueError("hz must be a positive integer")

        self._channel = pwm_channel
        self._chip = chip
        self._frequency_hz = hz
        self._period_ns = round(1_000_000_000 / hz)

        if self._period_ns <= 0:
            raise ValueError("PWM frequency is too high")

        self._chip_path = self._SYSFS_ROOT / f"pwmchip{chip}"
        self._pwm_path = self._chip_path / f"pwm{pwm_channel}"
        self._exported_by_this_instance = False
        self._started = False
        self._closed = False

        self._validate_pwm_chip()

    def start(self, duty_cycle_percent: float = 0.0) -> None:
        """Export, configure, and enable the PWM channel."""
        self._ensure_not_closed()
        duty_cycle_percent = self._validate_duty_cycle(duty_cycle_percent)

        self._export_if_needed()

        # Reconfigure from a known inactive state. Setting duty to zero before
        # changing the period avoids an invalid duty > period intermediate state.
        self._write_pwm_value("enable", 0)
        self._write_pwm_value("duty_cycle", 0)
        self._write_pwm_value("period", self._period_ns)
        self._write_pwm_value(
            "duty_cycle",
            self._duty_cycle_ns(duty_cycle_percent),
        )
        self._write_pwm_value("enable", 1)
        self._started = True

    def change_duty_cycle(self, duty_cycle_percent: float) -> None:
        """Change duty cycle while preserving the configured frequency."""
        self._ensure_not_closed()
        if not self._started:
            raise RuntimeError("PWM must be started before changing duty cycle")

        duty_cycle_percent = self._validate_duty_cycle(duty_cycle_percent)
        self._write_pwm_value(
            "duty_cycle",
            self._duty_cycle_ns(duty_cycle_percent),
        )

    def stop(self) -> None:
        """Drive the PWM inactive, disable it, and release it when appropriate."""
        if self._closed:
            return

        first_error: Exception | None = None

        if self._pwm_path.exists():
            try:
                # The kernel documentation notes that a disabled PWM is not
                # guaranteed to hold a particular output state. Set zero duty
                # first so the signal is inactive before disabling it.
                self._write_pwm_value("duty_cycle", 0)
            except Exception as exc:
                first_error = exc

            try:
                self._write_pwm_value("enable", 0)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self._exported_by_this_instance:
            try:
                self._write_chip_value("unexport", self._channel)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        self._started = False
        self._closed = True

        if first_error is not None:
            raise first_error

    def _validate_pwm_chip(self) -> None:
        if not self._chip_path.is_dir():
            raise RuntimeError(
                f"Linux PWM chip not found at {self._chip_path}. "
                "Ensure the Raspberry Pi PWM device-tree overlay is enabled "
                "and that the expected pwmchip number is correct."
            )

        npwm_path = self._chip_path / "npwm"
        try:
            num_channels = int(npwm_path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read PWM channel count from {npwm_path}"
            ) from exc

        if self._channel >= num_channels:
            raise ValueError(
                f"PWM channel {self._channel} is unavailable on pwmchip{self._chip}; "
                f"the chip exposes {num_channels} channel(s)"
            )

    def _export_if_needed(self) -> None:
        if self._pwm_path.is_dir():
            return

        try:
            self._write_chip_value("export", self._channel)
            self._exported_by_this_instance = True
        except OSError as exc:
            # Another process may have exported the channel between our path
            # check and our write. Treat EBUSY as success if the channel appears.
            if exc.errno != errno.EBUSY:
                raise

        deadline = time.monotonic() + self._EXPORT_TIMEOUT_S
        required_files = ("period", "duty_cycle", "enable")

        while time.monotonic() < deadline:
            if self._pwm_path.is_dir() and all(
                (self._pwm_path / name).exists()
                for name in required_files
            ):
                return
            time.sleep(self._POLL_INTERVAL_S)

        raise RuntimeError(
            f"PWM channel {self._channel} was exported, but {self._pwm_path} "
            "did not become ready in time"
        )

    def _write_chip_value(self, filename: str, value: int) -> None:
        path = self._chip_path / filename
        self._write_sysfs_value(path, value)

    def _write_pwm_value(self, filename: str, value: int) -> None:
        path = self._pwm_path / filename
        self._write_sysfs_value(path, value)

    @staticmethod
    def _write_sysfs_value(path: Path, value: int) -> None:
        try:
            path.write_text(str(value), encoding="ascii")
        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied writing {path}. Configure permissions for "
                "the Linux PWM sysfs interface before running the endpoint."
            ) from exc
        except OSError as exc:
            raise OSError(
                exc.errno,
                f"Could not write {value} to Linux PWM control {path}: {exc}",
            ) from exc

    def _duty_cycle_ns(self, duty_cycle_percent: float) -> int:
        duty_ns = round(
            self._period_ns * duty_cycle_percent / 100.0
        )
        return min(max(duty_ns, 0), self._period_ns)

    @staticmethod
    def _validate_duty_cycle(duty_cycle_percent: float) -> float:
        if (
            isinstance(duty_cycle_percent, bool)
            or not isinstance(duty_cycle_percent, (int, float))
        ):
            raise ValueError("duty cycle must be numeric")

        duty_cycle_percent = float(duty_cycle_percent)

        if not math.isfinite(duty_cycle_percent):
            raise ValueError("duty cycle must be finite")

        if not 0.0 <= duty_cycle_percent <= 100.0:
            raise ValueError("duty cycle must be between 0 and 100 percent")

        return duty_cycle_percent

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot use PWM after it has been stopped")


class DCMotorDriver:
    """
    Controls two DC motors through a DRV8833-style dual H-bridge using
    Raspberry Pi kernel hardware PWM.

    Expected Raspberry Pi 4 device-tree mapping:

        PWM channel 0 -> BCM GPIO12
        PWM channel 1 -> BCM GPIO13

    Each motor GPIO tuple is:

        (pwm_gpio, direction_gpio)

    Example wiring:

        Motor 1:
            GPIO12 -> AIN1
            GPIO18 -> AIN2

        Motor 2:
            GPIO13 -> BIN1
            GPIO19 -> BIN2

    The optional sleep_gpio may be connected to nSLEEP. If sleep_gpio
    is None, the driver assumes nSLEEP is permanently pulled high.

    Speed range:

        -1.0 = full reverse
         0.0 = coast
        +1.0 = full forward

    Use BCM GPIO integers such as 12 and 18.
    Do not pass board.D12 or other board pin objects.
    """

    # Raspberry Pi 4 kernel PWM controller.
    _PWM_CHIP = 0

    # DRV8833 can require up to 1 ms after nSLEEP is raised.
    _WAKE_DELAY_S = 0.001

    _PWM_CHANNEL_BY_GPIO = {
        12: 0,
        13: 1,
    }

    def __init__(
        self,
        motor_1_gpio_pins: tuple[int, int],
        motor_2_gpio_pins: tuple[int, int],
        pwm_frequency_hz: int = 20_000,
        sleep_gpio: int | None = None,
    ):
        self._validate_gpio_pair(
            gpio_pins=motor_1_gpio_pins,
            name="motor_1_gpio_pins",
        )
        self._validate_gpio_pair(
            gpio_pins=motor_2_gpio_pins,
            name="motor_2_gpio_pins",
        )

        if sleep_gpio is not None:
            self._validate_gpio(
                gpio=sleep_gpio,
                name="sleep_gpio",
            )

        if (
            isinstance(pwm_frequency_hz, bool)
            or not isinstance(pwm_frequency_hz, int)
            or pwm_frequency_hz <= 0
        ):
            raise ValueError(
                "pwm_frequency_hz must be a positive integer"
            )

        all_gpios = (
            motor_1_gpio_pins
            + motor_2_gpio_pins
        )

        if sleep_gpio is not None:
            all_gpios += (sleep_gpio,)

        if len(set(all_gpios)) != len(all_gpios):
            raise ValueError(
                "Every PWM, direction, and optional sleep signal "
                "must use a different GPIO"
            )

        motor_1_channel = self._get_pwm_channel(
            motor_1_gpio_pins[0]
        )
        motor_2_channel = self._get_pwm_channel(
            motor_2_gpio_pins[0]
        )

        if motor_1_channel == motor_2_channel:
            raise ValueError(
                "The two motors must use different PWM channels"
            )

        self._motor_gpio_pairs = (
            motor_1_gpio_pins,
            motor_2_gpio_pins,
        )
        self._pwm_frequency_hz = pwm_frequency_hz

        self._lock = threading.Lock()
        self._closed = False
        self._current_speeds = [0.0, 0.0]

        self._sleep: DigitalOutputDevice | None = None
        self._directions: list[DigitalOutputDevice] = []
        self._pwms: list[_LinuxSysfsPWM] = []

        try:
            if sleep_gpio is not None:
                # Start with the H-bridge disabled while its control
                # signals are configured.
                self._sleep = DigitalOutputDevice(
                    sleep_gpio,
                    active_high=True,
                    initial_value=False,
                )

            for _, direction_gpio in self._motor_gpio_pairs:
                self._directions.append(
                    DigitalOutputDevice(
                        direction_gpio,
                        active_high=True,
                        initial_value=False,
                    )
                )

            for pwm_gpio, _ in self._motor_gpio_pairs:
                self._pwms.append(
                    _LinuxSysfsPWM(
                        pwm_channel=self._get_pwm_channel(
                            pwm_gpio
                        ),
                        hz=self._pwm_frequency_hz,
                        chip=self._PWM_CHIP,
                    )
                )

            # Export and enable both PWM channels at 0% duty cycle.
            for pwm in self._pwms:
                pwm.start(0.0)

            self._wake_bridge_if_available()

        except Exception:
            self._cleanup_resources()
            raise

    def set_speeds(
        self,
        motor_1_speed: float,
        motor_2_speed: float,
    ) -> None:
        """
        Set both motor speeds from -1.0 to 1.0.
        """
        new_speeds = [
            self._validate_speed(
                speed=motor_1_speed,
                name="motor_1_speed",
            ),
            self._validate_speed(
                speed=motor_2_speed,
                name="motor_2_speed",
            ),
        ]

        with self._lock:
            self._ensure_open()

            polarity_changes = [
                (old_speed < 0.0) != (new_speed < 0.0)
                for old_speed, new_speed in zip(
                    self._current_speeds,
                    new_speeds,
                )
            ]

            direction_changed = any(polarity_changes)

            if direction_changed:
                self._sleep_bridge_if_available()

                # When nSLEEP is unavailable, first move any motor
                # changing polarity toward its coast state.
                for pwm, direction, changed in zip(
                    self._pwms,
                    self._directions,
                    polarity_changes,
                ):
                    if changed:
                        pwm.change_duty_cycle(0.0)
                        direction.off()

            try:
                for pwm, direction, speed in zip(
                    self._pwms,
                    self._directions,
                    new_speeds,
                ):
                    self._set_motor_speed(
                        pwm=pwm,
                        direction=direction,
                        speed=speed,
                    )

                self._current_speeds = new_speeds

                if direction_changed:
                    self._wake_bridge_if_available()

            except Exception:
                self._sleep_bridge_if_available()

                try:
                    self._zero_control_signals()
                except Exception:
                    pass

                self._current_speeds = [0.0, 0.0]
                raise

    def stop_all(self) -> None:
        """
        Coast both motors to a stop.
        """
        with self._lock:
            self._ensure_open()

            self._sleep_bridge_if_available()

            try:
                self._zero_control_signals()
                self._current_speeds = [0.0, 0.0]
            finally:
                self._wake_bridge_if_available()

    def close(self) -> None:
        """
        Stop both motors and release PWM and GPIO resources.
        """
        with self._lock:
            if self._closed:
                return

            first_error: Exception | None = None

            try:
                self._sleep_bridge_if_available()
            except Exception as exc:
                first_error = exc

            try:
                self._zero_control_signals()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

            for pwm in self._pwms:
                try:
                    pwm.stop()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

            for direction in self._directions:
                try:
                    direction.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

            if self._sleep is not None:
                try:
                    self._sleep.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

            self._closed = True

            if first_error is not None:
                raise first_error

    @staticmethod
    def _set_motor_speed(
        pwm: _LinuxSysfsPWM,
        direction: DigitalOutputDevice,
        speed: float,
    ) -> None:
        if speed == 0.0:
            # PWM LOW and direction LOW -> coast.
            pwm.change_duty_cycle(0.0)
            direction.off()
            return

        if speed > 0.0:
            # xIN1 = PWM
            # xIN2 = LOW
            #
            # Forward PWM with fast decay.
            direction.off()
            duty_cycle_percent = speed * 100.0

        else:
            # xIN1 = PWM
            # xIN2 = HIGH
            #
            # Reverse PWM with slow decay.
            #
            # With xIN2 held high:
            #   xIN1 LOW  -> reverse drive
            #   xIN1 HIGH -> brake
            #
            # Therefore the hardware PWM high-time is inverted.
            direction.on()
            duty_cycle_percent = (
                1.0 - abs(speed)
            ) * 100.0

        pwm.change_duty_cycle(duty_cycle_percent)

    def _zero_control_signals(self) -> None:
        first_error: Exception | None = None

        for pwm in self._pwms:
            try:
                pwm.change_duty_cycle(0.0)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        for direction in self._directions:
            try:
                direction.off()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    def _sleep_bridge_if_available(self) -> None:
        if self._sleep is not None:
            self._sleep.off()

    def _wake_bridge_if_available(self) -> None:
        if self._sleep is not None:
            self._sleep.on()
            time.sleep(self._WAKE_DELAY_S)

    def _cleanup_resources(self) -> None:
        try:
            self._sleep_bridge_if_available()
        except Exception:
            pass

        for pwm in self._pwms:
            try:
                pwm.stop()
            except Exception:
                pass

        for direction in self._directions:
            try:
                direction.close()
            except Exception:
                pass

        if self._sleep is not None:
            try:
                self._sleep.close()
            except Exception:
                pass

        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Cannot use DC motor driver after it has been closed"
            )

    @classmethod
    def _validate_gpio_pair(
        cls,
        gpio_pins: tuple[int, int],
        name: str,
    ) -> None:
        if not isinstance(gpio_pins, tuple) or len(gpio_pins) != 2:
            raise ValueError(
                f"{name} must be a tuple containing "
                "(PWM GPIO, direction GPIO)"
            )

        pwm_gpio, direction_gpio = gpio_pins

        cls._validate_gpio(
            gpio=pwm_gpio,
            name=f"{name}[0]",
        )
        cls._validate_gpio(
            gpio=direction_gpio,
            name=f"{name}[1]",
        )

        if pwm_gpio not in cls._PWM_CHANNEL_BY_GPIO:
            raise ValueError(
                f"{name}[0] must be BCM GPIO12 or GPIO13"
            )

        if direction_gpio in cls._PWM_CHANNEL_BY_GPIO:
            raise ValueError(
                f"{name}[1] must be an ordinary GPIO, not GPIO12 "
                "or GPIO13"
            )

    @staticmethod
    def _validate_gpio(
        gpio: int,
        name: str,
    ) -> None:
        if (
            isinstance(gpio, bool)
            or not isinstance(gpio, int)
            or not 0 <= gpio <= 27
        ):
            raise ValueError(
                f"{name} must be a BCM GPIO number from 0 to 27"
            )

    @classmethod
    def _get_pwm_channel(cls, pwm_gpio: int) -> int:
        return cls._PWM_CHANNEL_BY_GPIO[pwm_gpio]

    @staticmethod
    def _validate_speed(
        speed: float,
        name: str,
    ) -> float:
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
        ):
            raise ValueError(f"{name} must be numeric")

        speed = float(speed)

        if not math.isfinite(speed):
            raise ValueError(f"{name} must be finite")

        if not -1.0 <= speed <= 1.0:
            raise ValueError(
                f"{name} must be between -1.0 and 1.0, "
                f"got {speed}"
            )

        return speed
