import threading
import time

from src.endpoint.config import (
    FOAM_RESET_ANGLE_DEG,
    FOAM_RESET_HOLD_DELAY,
    FOAM_TRIGGER_ANGLE_DEG,
    FOAM_TRIGGER_HOLD_DELAY,
    FOAM_MOTOR_SPINUP_DELAY,
)
from src.endpoint.drivers.servo_driver import ServoDriver
from src.endpoint.drivers.dc_motor_driver import DCMotorDriver


class FoamMechanismError(Exception):
    """
    Raised when the foam mechanism rejects or fails to apply a command.
    """
    pass


class FoamMechanism:
    def __init__(
        self,
        servo_driver: ServoDriver,
        dc_motor_driver: DCMotorDriver,
        foam_channel: int,
        motor_1_speed: float,
        motor_2_speed: float,
    ):
        self.servo_driver = servo_driver
        self.dc_motor_driver = dc_motor_driver

        self.foam_channel = foam_channel
        self.motor_1_speed = motor_1_speed
        self.motor_2_speed = motor_2_speed

        # Serializes trigger, halt, and stop commands.
        self._command_lock = threading.RLock()

        # Protects state shared with the worker thread.
        self._state_lock = threading.Lock()

        # Wakes the worker for a new trigger sequence.
        self._trigger_event = threading.Event()

        # Interrupts the triggering hold and begins servo reset.
        self._interrupt_event = threading.Event()

        # Permanently shuts down the worker.
        self._stop_event = threading.Event()

        # Set whenever no trigger/reset sequence is running.
        self._sequence_done_event = threading.Event()
        self._sequence_done_event.set()

        self._trigger_in_progress = False
        self._trigger_halted = True

        self._dc_motors_started_time: float | None = None # None if motors are stopped

        # Initialize both motors and the trigger servo safely.
        try:
            self._stop_dc_motors()
            self._return_to_reset()
        except Exception as exc:
            raise FoamMechanismError(
                f"Failed to initialize foam mechanism: {exc}"
            ) from exc

        self._worker_thread = threading.Thread(
            target=self._worker,
            name="FoamMechanismWorker",
            daemon=True,
        )
        self._worker_thread.start()


    def arm(self) -> None:
        """
        Enable triggering and start the flywheel motors.

        Calling arm repeatedly while already armed has no effect.
        """
        with self._command_lock:
            with self._state_lock:
                if self._stop_event.is_set():
                    raise FoamMechanismError(
                        "Cannot arm foam mechanism after it has stopped."
                    )

                if not self._trigger_halted:
                    return

            try:
                self._start_dc_motors()
            except Exception as exc:
                raise FoamMechanismError(
                    f"Failed to start foam DC motors: {exc}"
                ) from exc

            with self._state_lock:
                self._trigger_halted = False


    def trigger(self) -> None:
        """
        Start one asynchronous trigger sequence.

        The mechanism must already be armed. The worker waits for any
        remaining flywheel spin-up time before moving the trigger servo.
        """
        with self._command_lock:
            with self._state_lock:
                if self._stop_event.is_set():
                    raise FoamMechanismError(
                        "Cannot trigger foam mechanism after it has stopped."
                    )

                if self._trigger_halted:
                    raise FoamMechanismError(
                        "Cannot trigger foam mechanism while triggering is halted."
                    )

                if self._trigger_in_progress:
                    raise FoamMechanismError(
                        "Foam trigger sequence is already in progress."
                    )

                self._trigger_in_progress = True

                self._interrupt_event.clear()
                self._sequence_done_event.clear()

            self._trigger_event.set()


    def halt_trigger(self) -> None:
        """
        Halt triggering and return the servo to its reset position.

        An active triggering hold is interrupted. Both DC motors are stopped
        only after the trigger servo has completed its reset.
        """
        with self._command_lock:
            with self._state_lock:
                if (self._trigger_halted and not self._trigger_in_progress and self._dc_motors_started_time is None):
                    # return early if already halted to avoid extra delay
                    return
     
                self._trigger_halted = True
                trigger_in_progress = self._trigger_in_progress

            # Tell an active sequence to reset immediately.
            self._interrupt_event.set()

            if trigger_in_progress:
                # The worker performs the reset.
                self._sequence_done_event.wait()
            else:
                # Enforce the reset position even while idle.
                try:
                    self._return_to_reset()
                except Exception as exc:
                    raise FoamMechanismError(
                        f"Failed to reset foam trigger servo: {exc}"
                    ) from exc

            try:
                # Stop both DC motor outputs after servo reset.
                self._stop_dc_motors()
            except Exception as exc:
                raise FoamMechanismError(
                    f"Failed to stop foam DC motors: {exc}"
                ) from exc


    def stop(self) -> None:
        """
        Safely and permanently stop the foam mechanism.
        """
        with self._command_lock:
            if self._stop_event.is_set():
                if self._worker_thread.is_alive():
                    self._worker_thread.join()
                return

            # Prevent any later trigger requests.
            self._stop_event.set()

            halt_error: FoamMechanismError | None = None

            try:
                # Reset the servo before stopping both motors.
                self.halt_trigger()
            except FoamMechanismError as exc:
                halt_error = exc
            finally:
                # Wake an idle worker so it can exit.
                self._trigger_event.set()
                self._worker_thread.join()

            if halt_error is not None:
                raise halt_error


    def trigger_in_progress(self) -> bool:
        with self._state_lock:
            return self._trigger_in_progress


    def trigger_is_halted(self) -> bool:
        with self._state_lock:
            return self._trigger_halted


    def _worker(self) -> None:
        while True:
            # Wait for a trigger request or shutdown.
            self._trigger_event.wait()
            self._trigger_event.clear()

            with self._state_lock:
                trigger_pending = self._trigger_in_progress
                trigger_cancelled = (
                    self._trigger_halted or self._stop_event.is_set()
                )

            if not trigger_pending:
                if self._stop_event.is_set():
                    break

                continue

            try:
                if trigger_cancelled:
                    self._return_to_reset()
                else:
                    with self._state_lock:
                        motors_started_time = self._dc_motors_started_time

                    if motors_started_time is None:
                        raise FoamMechanismError("Cannot trigger because the DC motors are not running.")

                    motor_run_time = time.perf_counter() - motors_started_time
                    remaining_spinup_time = max(0.0, FOAM_MOTOR_SPINUP_DELAY - motor_run_time)

                    interrupted = self._interrupt_event.wait(remaining_spinup_time)

                    if interrupted:
                        self._return_to_reset()
                    else:
                        self._perform_trigger_sequence()

            except Exception as exc:
                # The original trigger() call has already returned.
                print(
                    FoamMechanismError(
                        f"Foam trigger sequence failed: {exc}"
                    ),
                    flush=True,
                )

            finally:
                with self._state_lock:
                    self._trigger_in_progress = False

                # Allow halt or stop to continue.
                self._sequence_done_event.set()

            if self._stop_event.is_set():
                break


    def _perform_trigger_sequence(self) -> None:
        try:
            with self._state_lock:
                trigger_cancelled = (self._trigger_halted or self._stop_event.is_set())

            if trigger_cancelled or self._interrupt_event.is_set():
                return
            
            # Move the trigger servo into the triggering position.
            self.servo_driver.set_angle_deg(
                channel=self.foam_channel,
                angle_deg=FOAM_TRIGGER_ANGLE_DEG,
            )

            # Wait normally, or reset early after halt/stop.
            self._interrupt_event.wait(FOAM_TRIGGER_HOLD_DELAY)

        finally:
            # Always return the servo to its resting position.
            self._return_to_reset()


    def _return_to_reset(self) -> None:
        self.servo_driver.set_angle_deg(
            channel=self.foam_channel,
            angle_deg=FOAM_RESET_ANGLE_DEG,
        )

        # Reset completion must not be interrupted.
        time.sleep(FOAM_RESET_HOLD_DELAY)


    def _start_dc_motors(self) -> None:
        # Motor signs may differ because the flywheels face opposite ways.
        self.dc_motor_driver.set_speeds(
            motor_1_speed=self.motor_1_speed,
            motor_2_speed=self.motor_2_speed,
        )

        with self._state_lock:
            self._dc_motors_started_time = time.perf_counter()


    def _stop_dc_motors(self) -> None:
        # Stop both DC motor outputs.
        self.dc_motor_driver.stop_all()

        with self._state_lock:
            self._dc_motors_started_time = None