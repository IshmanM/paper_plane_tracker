import math
import threading

import pwmio


class DCMotorDriver:
    """
    Controls two DC motors through a dual H-bridge such as the DRV8833.

    Each motor GPIO pair is:
        (input_1_pin, input_2_pin)

    Speeds range from -1.0 to 1.0:
        positive: input 1 receives PWM
        negative: input 2 receives PWM
        zero:     both inputs are low
    """

    _MAX_DUTY_CYCLE = 65535

    def __init__(
        self,
        motor_1_gpio_pins: tuple[object, object],
        motor_2_gpio_pins: tuple[object, object],
        pwm_frequency_hz: int = 20000,
    ):
        self._validate_gpio_pair(
            gpio_pins=motor_1_gpio_pins,
            name="motor_1_gpio_pins",
        )
        self._validate_gpio_pair(
            gpio_pins=motor_2_gpio_pins,
            name="motor_2_gpio_pins",
        )

        if isinstance(pwm_frequency_hz, bool) or not isinstance(
            pwm_frequency_hz,
            int,
        ):
            raise ValueError("pwm_frequency_hz must be an integer")

        if pwm_frequency_hz <= 0:
            raise ValueError("pwm_frequency_hz must be greater than zero")

        all_pins = motor_1_gpio_pins + motor_2_gpio_pins

        if self._contains_duplicate_pins(all_pins):
            raise ValueError(
                "Each motor-driver input must use a different GPIO pin"
            )

        motor_1_in1_pin, motor_1_in2_pin = motor_1_gpio_pins
        motor_2_in1_pin, motor_2_in2_pin = motor_2_gpio_pins

        self._lock = threading.Lock()
        self._closed = False

        self._motor_1_in1 = None
        self._motor_1_in2 = None
        self._motor_2_in1 = None
        self._motor_2_in2 = None

        try:
            self._motor_1_in1 = pwmio.PWMOut(
                motor_1_in1_pin,
                frequency=pwm_frequency_hz,
                duty_cycle=0,
            )
            self._motor_1_in2 = pwmio.PWMOut(
                motor_1_in2_pin,
                frequency=pwm_frequency_hz,
                duty_cycle=0,
            )
            self._motor_2_in1 = pwmio.PWMOut(
                motor_2_in1_pin,
                frequency=pwm_frequency_hz,
                duty_cycle=0,
            )
            self._motor_2_in2 = pwmio.PWMOut(
                motor_2_in2_pin,
                frequency=pwm_frequency_hz,
                duty_cycle=0,
            )
        except Exception:
            # Do not let cleanup failures hide the original setup error.
            self._deinit_outputs(suppress_errors=True)
            raise

    def set_speeds(
        self,
        motor_1_speed: float,
        motor_2_speed: float,
    ) -> None:
        """
        Set both motor speeds from -1.0 to 1.0.
        """
        motor_1_speed = self._validate_speed(
            speed=motor_1_speed,
            name="motor_1_speed",
        )
        motor_2_speed = self._validate_speed(
            speed=motor_2_speed,
            name="motor_2_speed",
        )

        with self._lock:
            self._ensure_open()

            try:
                self._set_motor_speed(
                    in1=self._motor_1_in1,
                    in2=self._motor_1_in2,
                    speed=motor_1_speed,
                )
                self._set_motor_speed(
                    in1=self._motor_2_in1,
                    in2=self._motor_2_in2,
                    speed=motor_2_speed,
                )
            except Exception:
                # Do not leave one motor running if the other update fails.
                try:
                    self._stop_all_unlocked()
                except Exception:
                    pass
                raise

    def stop_all(self) -> None:
        """
        Coast both motors to a stop.
        """
        with self._lock:
            self._ensure_open()
            self._stop_all_unlocked()

    def close(self) -> None:
        """
        Stop both motors and release the GPIO resources.
        """
        with self._lock:
            if self._closed:
                return

            first_error = None

            try:
                self._stop_all_unlocked()
            except Exception as exc:
                first_error = exc

            try:
                self._deinit_outputs()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._closed = True

            if first_error is not None:
                raise first_error

    def _stop_all_unlocked(self) -> None:
        first_error = None

        for output in self._outputs():
            if output is None:
                continue

            try:
                output.duty_cycle = 0
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    @classmethod
    def _set_motor_speed(
        cls,
        in1: pwmio.PWMOut,
        in2: pwmio.PWMOut,
        speed: float,
    ) -> None:
        # Remove drive before changing direction.
        in1.duty_cycle = 0
        in2.duty_cycle = 0

        duty_cycle = round(abs(speed) * cls._MAX_DUTY_CYCLE)

        if speed > 0.0:
            in1.duty_cycle = duty_cycle
        elif speed < 0.0:
            in2.duty_cycle = duty_cycle

    def _deinit_outputs(self, suppress_errors: bool = False) -> None:
        first_error = None

        output_attributes = (
            "_motor_1_in1",
            "_motor_1_in2",
            "_motor_2_in1",
            "_motor_2_in2",
        )

        for attribute_name in output_attributes:
            output = getattr(self, attribute_name)

            try:
                if output is not None:
                    output.deinit()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                setattr(self, attribute_name, None)

        if first_error is not None and not suppress_errors:
            raise first_error

    def _outputs(self) -> tuple[object, object, object, object]:
        return (
            self._motor_1_in1,
            self._motor_1_in2,
            self._motor_2_in1,
            self._motor_2_in2,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Cannot use DC motor driver after it has been closed"
            )

    @staticmethod
    def _contains_duplicate_pins(pins: tuple[object, ...]) -> bool:
        return any(
            pin == other_pin
            for index, pin in enumerate(pins)
            for other_pin in pins[index + 1:]
        )

    @staticmethod
    def _validate_gpio_pair(
        gpio_pins: tuple[object, object],
        name: str,
    ) -> None:
        if not isinstance(gpio_pins, tuple) or len(gpio_pins) != 2:
            raise ValueError(
                f"{name} must be a tuple containing two board GPIO pins"
            )

        if gpio_pins[0] == gpio_pins[1]:
            raise ValueError(
                f"{name} must contain two different GPIO pins"
            )

    @staticmethod
    def _validate_speed(speed: float, name: str) -> float:
        if isinstance(speed, bool):
            raise ValueError(f"{name} must be numeric, not bool")

        if not isinstance(speed, int | float):
            raise ValueError(
                f"{name} must be numeric, "
                f"got {type(speed).__name__}"
            )

        speed = float(speed)

        if not math.isfinite(speed):
            raise ValueError(f"{name} must be finite")

        if not -1.0 <= speed <= 1.0:
            raise ValueError(
                f"{name} must be between -1.0 and 1.0, got {speed}"
            )

        return speed



# import math
# import threading

# import pigpio


# class DCMotorDriver:
#     """
#     Controls two DC motors through a DRV8833 using Raspberry Pi hardware PWM.

#     Each GPIO tuple is:
#         (hardware_pwm_gpio, direction_gpio)

#     Positive speed:
#         PWM input controls forward power.
#         Direction input is held low.

#     Negative speed:
#         PWM input controls reverse power.
#         Direction input is held high.
#     """

#     _PWM_CHANNEL_BY_GPIO = {
#         12: 0,
#         18: 0,
#         13: 1,
#         19: 1,
#     }

#     _PWM_DUTY_MAX = 1_000_000

#     def __init__(
#         self,
#         motor_1_gpio_pins: tuple[int, int],
#         motor_2_gpio_pins: tuple[int, int],
#         pwm_frequency_hz: int = 20000,
#     ):
#         self._validate_gpio_pair(
#             motor_1_gpio_pins,
#             "motor_1_gpio_pins",
#         )
#         self._validate_gpio_pair(
#             motor_2_gpio_pins,
#             "motor_2_gpio_pins",
#         )

#         motor_1_pwm_gpio, motor_1_direction_gpio = motor_1_gpio_pins
#         motor_2_pwm_gpio, motor_2_direction_gpio = motor_2_gpio_pins

#         all_gpios = (
#             motor_1_pwm_gpio,
#             motor_1_direction_gpio,
#             motor_2_pwm_gpio,
#             motor_2_direction_gpio,
#         )

#         if len(set(all_gpios)) != len(all_gpios):
#             raise ValueError(
#                 "Each DRV8833 input must use a different GPIO"
#             )

#         motor_1_pwm_channel = self._PWM_CHANNEL_BY_GPIO[
#             motor_1_pwm_gpio
#         ]
#         motor_2_pwm_channel = self._PWM_CHANNEL_BY_GPIO[
#             motor_2_pwm_gpio
#         ]

#         if motor_1_pwm_channel == motor_2_pwm_channel:
#             raise ValueError(
#                 "The two PWM GPIOs must use different hardware PWM channels"
#             )

#         if isinstance(pwm_frequency_hz, bool):
#             raise ValueError("pwm_frequency_hz must be an integer")

#         if not isinstance(pwm_frequency_hz, int):
#             raise ValueError("pwm_frequency_hz must be an integer")

#         if pwm_frequency_hz <= 0:
#             raise ValueError(
#                 "pwm_frequency_hz must be greater than zero"
#             )

#         self._motor_1_pwm_gpio = motor_1_pwm_gpio
#         self._motor_1_direction_gpio = motor_1_direction_gpio

#         self._motor_2_pwm_gpio = motor_2_pwm_gpio
#         self._motor_2_direction_gpio = motor_2_direction_gpio

#         self._pwm_frequency_hz = pwm_frequency_hz

#         self._lock = threading.Lock()
#         self._closed = False

#         # Connect to the pigpio daemon.
#         self._pi = pigpio.pi()

#         if not self._pi.connected:
#             raise RuntimeError(
#                 "Could not connect to pigpio. Ensure pigpiod is running."
#             )

#         try:
#             self._initialize_motor_outputs(
#                 pwm_gpio=self._motor_1_pwm_gpio,
#                 direction_gpio=self._motor_1_direction_gpio,
#             )

#             self._initialize_motor_outputs(
#                 pwm_gpio=self._motor_2_pwm_gpio,
#                 direction_gpio=self._motor_2_direction_gpio,
#             )

#         except Exception:
#             self._pi.stop()
#             raise

#     def set_speeds(
#         self,
#         motor_1_speed: float,
#         motor_2_speed: float,
#     ) -> None:
#         """
#         Set both motor speeds.

#         Speed range:
#             1.0: full-speed forward
#             0.0: coast
#            -1.0: full-speed reverse
#         """
#         motor_1_speed = self._validate_speed(
#             motor_1_speed,
#             "motor_1_speed",
#         )
#         motor_2_speed = self._validate_speed(
#             motor_2_speed,
#             "motor_2_speed",
#         )

#         with self._lock:
#             self._ensure_open()

#             try:
#                 self._set_motor_speed(
#                     pwm_gpio=self._motor_1_pwm_gpio,
#                     direction_gpio=self._motor_1_direction_gpio,
#                     speed=motor_1_speed,
#                 )

#                 self._set_motor_speed(
#                     pwm_gpio=self._motor_2_pwm_gpio,
#                     direction_gpio=self._motor_2_direction_gpio,
#                     speed=motor_2_speed,
#                 )

#             except Exception:
#                 # Leave both motors stopped if either command fails.
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
#         Stop both motors and release the pigpio connection.
#         """
#         with self._lock:
#             if self._closed:
#                 return

#             try:
#                 # Hardware PWM must be disabled before disconnecting.
#                 self._stop_all_unlocked()
#             finally:
#                 self._pi.stop()
#                 self._closed = True

#     def _initialize_motor_outputs(
#         self,
#         pwm_gpio: int,
#         direction_gpio: int,
#     ) -> None:
#         # Begin with both DRV8833 inputs low.
#         self._check_status(
#             self._pi.set_mode(direction_gpio, pigpio.OUTPUT),
#             f"Configure GPIO {direction_gpio} as an output",
#         )

#         self._check_status(
#             self._pi.write(direction_gpio, 0),
#             f"Set GPIO {direction_gpio} low",
#         )

#         self._disable_hardware_pwm(pwm_gpio)

#     def _set_motor_speed(
#         self,
#         pwm_gpio: int,
#         direction_gpio: int,
#         speed: float,
#     ) -> None:
#         # Stop the existing signal before changing direction.
#         self._disable_hardware_pwm(pwm_gpio)

#         if speed == 0.0:
#             self._check_status(
#                 self._pi.write(direction_gpio, 0),
#                 f"Stop motor direction GPIO {direction_gpio}",
#             )
#             return

#         if speed > 0.0:
#             # PWM/LOW gives forward drive with fast decay.
#             self._check_status(
#                 self._pi.write(direction_gpio, 0),
#                 f"Set motor direction GPIO {direction_gpio} forward",
#             )

#             pwm_duty = round(speed * self._PWM_DUTY_MAX)

#         else:
#             # PWM/HIGH gives reverse drive with slow decay.
#             self._check_status(
#                 self._pi.write(direction_gpio, 1),
#                 f"Set motor direction GPIO {direction_gpio} reverse",
#             )

#             # LOW is reverse drive and HIGH is braking in this mode.
#             pwm_duty = round(
#                 (1.0 - abs(speed)) * self._PWM_DUTY_MAX
#             )

#         self._check_status(
#             self._pi.hardware_PWM(
#                 pwm_gpio,
#                 self._pwm_frequency_hz,
#                 pwm_duty,
#             ),
#             f"Start hardware PWM on GPIO {pwm_gpio}",
#         )

#     def _stop_all_unlocked(self) -> None:
#         self._stop_motor_unlocked(
#             pwm_gpio=self._motor_1_pwm_gpio,
#             direction_gpio=self._motor_1_direction_gpio,
#         )

#         self._stop_motor_unlocked(
#             pwm_gpio=self._motor_2_pwm_gpio,
#             direction_gpio=self._motor_2_direction_gpio,
#         )

#     def _stop_motor_unlocked(
#         self,
#         pwm_gpio: int,
#         direction_gpio: int,
#     ) -> None:
#         self._disable_hardware_pwm(pwm_gpio)

#         self._check_status(
#             self._pi.write(direction_gpio, 0),
#             f"Set motor direction GPIO {direction_gpio} low",
#         )

#     def _disable_hardware_pwm(self, pwm_gpio: int) -> None:
#         # A zero frequency disables pigpio hardware PWM.
#         self._check_status(
#             self._pi.hardware_PWM(pwm_gpio, 0, 0),
#             f"Disable hardware PWM on GPIO {pwm_gpio}",
#         )

#         # Explicitly leave the input low after disabling PWM.
#         self._check_status(
#             self._pi.set_mode(pwm_gpio, pigpio.OUTPUT),
#             f"Configure GPIO {pwm_gpio} as an output",
#         )

#         self._check_status(
#             self._pi.write(pwm_gpio, 0),
#             f"Set GPIO {pwm_gpio} low",
#         )

#     def _ensure_open(self) -> None:
#         if self._closed:
#             raise RuntimeError(
#                 "Cannot use DC motor driver after it has been closed"
#             )

#     @classmethod
#     def _validate_gpio_pair(
#         cls,
#         gpio_pins: tuple[int, int],
#         name: str,
#     ) -> None:
#         if not isinstance(gpio_pins, tuple) or len(gpio_pins) != 2:
#             raise ValueError(
#                 f"{name} must be a tuple containing "
#                 "(hardware_pwm_gpio, direction_gpio)"
#             )

#         pwm_gpio, direction_gpio = gpio_pins

#         for gpio in gpio_pins:
#             if isinstance(gpio, bool) or not isinstance(gpio, int):
#                 raise ValueError(
#                     f"{name} must contain integer BCM GPIO numbers"
#                 )

#         if pwm_gpio not in cls._PWM_CHANNEL_BY_GPIO:
#             raise ValueError(
#                 f"{name}[0] must be a hardware PWM GPIO: "
#                 "12, 13, 18, or 19"
#             )

#         if pwm_gpio == direction_gpio:
#             raise ValueError(
#                 f"{name} must contain two different GPIOs"
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
#                 f"{name} must be between -1.0 and 1.0, "
#                 f"got {speed}"
#             )

#         return speed

#     @staticmethod
#     def _check_status(status: int, operation: str) -> None:
#         if status < 0:
#             raise RuntimeError(
#                 f"{operation} failed with pigpio status {status}"
#             )