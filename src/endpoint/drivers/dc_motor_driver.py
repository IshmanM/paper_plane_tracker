import math
import threading
import time

from gpiozero import DigitalOutputDevice
from rpi_hardware_pwm import HardwarePWM


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
        self._pwms: list[HardwarePWM] = []

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
                    HardwarePWM(
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
        pwm: HardwarePWM,
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

# import math
# import threading

# import pwmio


# class DCMotorDriver:
#     """
#     Controls two DC motors through a dual H-bridge such as the DRV8833.

#     Each motor GPIO pair is:
#         (input_1_pin, input_2_pin)

#     Speeds range from -1.0 to 1.0:
#         positive: input 1 receives PWM
#         negative: input 2 receives PWM
#         zero:     both inputs are low
#     """

#     _MAX_DUTY_CYCLE = 65535

#     def __init__(
#         self,
#         motor_1_gpio_pins: tuple[object, object],
#         motor_2_gpio_pins: tuple[object, object],
#         pwm_frequency_hz: int = 20000,
#     ):
#         self._validate_gpio_pair(
#             gpio_pins=motor_1_gpio_pins,
#             name="motor_1_gpio_pins",
#         )
#         self._validate_gpio_pair(
#             gpio_pins=motor_2_gpio_pins,
#             name="motor_2_gpio_pins",
#         )

#         if isinstance(pwm_frequency_hz, bool) or not isinstance(
#             pwm_frequency_hz,
#             int,
#         ):
#             raise ValueError("pwm_frequency_hz must be an integer")

#         if pwm_frequency_hz <= 0:
#             raise ValueError("pwm_frequency_hz must be greater than zero")

#         all_pins = motor_1_gpio_pins + motor_2_gpio_pins

#         if self._contains_duplicate_pins(all_pins):
#             raise ValueError(
#                 "Each motor-driver input must use a different GPIO pin"
#             )

#         motor_1_in1_pin, motor_1_in2_pin = motor_1_gpio_pins
#         motor_2_in1_pin, motor_2_in2_pin = motor_2_gpio_pins

#         self._lock = threading.Lock()
#         self._closed = False

#         self._motor_1_in1 = None
#         self._motor_1_in2 = None
#         self._motor_2_in1 = None
#         self._motor_2_in2 = None

#         try:
#             self._motor_1_in1 = pwmio.PWMOut(
#                 motor_1_in1_pin,
#                 frequency=pwm_frequency_hz,
#                 duty_cycle=0,
#             )
#             self._motor_1_in2 = pwmio.PWMOut(
#                 motor_1_in2_pin,
#                 frequency=pwm_frequency_hz,
#                 duty_cycle=0,
#             )
#             self._motor_2_in1 = pwmio.PWMOut(
#                 motor_2_in1_pin,
#                 frequency=pwm_frequency_hz,
#                 duty_cycle=0,
#             )
#             self._motor_2_in2 = pwmio.PWMOut(
#                 motor_2_in2_pin,
#                 frequency=pwm_frequency_hz,
#                 duty_cycle=0,
#             )
#         except Exception:
#             # Do not let cleanup failures hide the original setup error.
#             self._deinit_outputs(suppress_errors=True)
#             raise

#     def set_speeds(
#         self,
#         motor_1_speed: float,
#         motor_2_speed: float,
#     ) -> None:
#         """
#         Set both motor speeds from -1.0 to 1.0.
#         """
#         motor_1_speed = self._validate_speed(
#             speed=motor_1_speed,
#             name="motor_1_speed",
#         )
#         motor_2_speed = self._validate_speed(
#             speed=motor_2_speed,
#             name="motor_2_speed",
#         )

#         with self._lock:
#             self._ensure_open()

#             try:
#                 self._set_motor_speed(
#                     in1=self._motor_1_in1,
#                     in2=self._motor_1_in2,
#                     speed=motor_1_speed,
#                 )
#                 self._set_motor_speed(
#                     in1=self._motor_2_in1,
#                     in2=self._motor_2_in2,
#                     speed=motor_2_speed,
#                 )
#             except Exception:
#                 # Do not leave one motor running if the other update fails.
#                 try:
#                     self._stop_all_unlocked()
#                 except Exception:
#                     pass
#                 raise

#     def stop_all(self) -> None:
#         """
#         Coast both motors to a stop.
#         """
#         with self._lock:
#             self._ensure_open()
#             self._stop_all_unlocked()

#     def close(self) -> None:
#         """
#         Stop both motors and release the GPIO resources.
#         """
#         with self._lock:
#             if self._closed:
#                 return

#             first_error = None

#             try:
#                 self._stop_all_unlocked()
#             except Exception as exc:
#                 first_error = exc

#             try:
#                 self._deinit_outputs()
#             except Exception as exc:
#                 if first_error is None:
#                     first_error = exc
#             finally:
#                 self._closed = True

#             if first_error is not None:
#                 raise first_error

#     def _stop_all_unlocked(self) -> None:
#         first_error = None

#         for output in self._outputs():
#             if output is None:
#                 continue

#             try:
#                 output.duty_cycle = 0
#             except Exception as exc:
#                 if first_error is None:
#                     first_error = exc

#         if first_error is not None:
#             raise first_error

#     @classmethod
#     def _set_motor_speed(
#         cls,
#         in1: pwmio.PWMOut,
#         in2: pwmio.PWMOut,
#         speed: float,
#     ) -> None:
#         # Remove drive before changing direction.
#         in1.duty_cycle = 0
#         in2.duty_cycle = 0

#         duty_cycle = round(abs(speed) * cls._MAX_DUTY_CYCLE)

#         if speed > 0.0:
#             in1.duty_cycle = duty_cycle
#         elif speed < 0.0:
#             in2.duty_cycle = duty_cycle

#     def _deinit_outputs(self, suppress_errors: bool = False) -> None:
#         first_error = None

#         output_attributes = (
#             "_motor_1_in1",
#             "_motor_1_in2",
#             "_motor_2_in1",
#             "_motor_2_in2",
#         )

#         for attribute_name in output_attributes:
#             output = getattr(self, attribute_name)

#             try:
#                 if output is not None:
#                     output.deinit()
#             except Exception as exc:
#                 if first_error is None:
#                     first_error = exc
#             finally:
#                 setattr(self, attribute_name, None)

#         if first_error is not None and not suppress_errors:
#             raise first_error

#     def _outputs(self) -> tuple[object, object, object, object]:
#         return (
#             self._motor_1_in1,
#             self._motor_1_in2,
#             self._motor_2_in1,
#             self._motor_2_in2,
#         )

#     def _ensure_open(self) -> None:
#         if self._closed:
#             raise RuntimeError(
#                 "Cannot use DC motor driver after it has been closed"
#             )

#     @staticmethod
#     def _contains_duplicate_pins(pins: tuple[object, ...]) -> bool:
#         return any(
#             pin == other_pin
#             for index, pin in enumerate(pins)
#             for other_pin in pins[index + 1:]
#         )

#     @staticmethod
#     def _validate_gpio_pair(
#         gpio_pins: tuple[object, object],
#         name: str,
#     ) -> None:
#         if not isinstance(gpio_pins, tuple) or len(gpio_pins) != 2:
#             raise ValueError(
#                 f"{name} must be a tuple containing two board GPIO pins"
#             )

#         if gpio_pins[0] == gpio_pins[1]:
#             raise ValueError(
#                 f"{name} must contain two different GPIO pins"
#             )

#     @staticmethod
#     def _validate_speed(speed: float, name: str) -> float:
#         if isinstance(speed, bool):
#             raise ValueError(f"{name} must be numeric, not bool")

#         if not isinstance(speed, int | float):
#             raise ValueError(
#                 f"{name} must be numeric, "
#                 f"got {type(speed).__name__}"
#             )

#         speed = float(speed)

#         if not math.isfinite(speed):
#             raise ValueError(f"{name} must be finite")

#         if not -1.0 <= speed <= 1.0:
#             raise ValueError(
#                 f"{name} must be between -1.0 and 1.0, got {speed}"
#             )

#         return speed

