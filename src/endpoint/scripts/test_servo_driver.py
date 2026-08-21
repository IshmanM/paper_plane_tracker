import argparse
import time

import board

import src.endpoint.config as config
from src.endpoint.drivers.servo_driver import ServoDriver, ServoCalibration


DEFAULT_CHANNEL = 15

# Conservative test range.
# Do not start by slamming a new servo from 0 to 180.
DEFAULT_MIN_TEST_ANGLE_DEG = 60.0
DEFAULT_MAX_TEST_ANGLE_DEG = 120.0
DEFAULT_CENTER_ANGLE_DEG = 90.0

DEFAULT_STEP_DEG = 10.0
DEFAULT_HOLD_S = 0.5

DEFAULT_PWM_HZ = 50.0

# Common starting calibration for hobby servos.
# This is the angle -> PWM conversion range, not platform safety limits.
DEFAULT_MIN_PULSE_US = 500.0
DEFAULT_MAX_PULSE_US = 2500.0


def move_and_pause(
    servo_driver: ServoDriver,
    channel: int,
    angle_deg: float,
    hold_s: float,
) -> None:
    used_angle = servo_driver.set_angle_deg(
        channel=channel,
        angle_deg=angle_deg,
    )

    used_pulse = servo_driver.get_last_pulse_us(channel)

    print(
        f"channel={channel}, "
        f"angle_cmd={angle_deg:.1f} deg, "
        f"angle_used={used_angle:.1f} deg, "
        f"pulse={used_pulse:.1f} us"
    )

    time.sleep(hold_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cautious PCA9685 servo test using ServoDriver."
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help="PCA9685 channel to test.",
    )

    parser.add_argument(
        "--center-angle",
        type=float,
        default=DEFAULT_CENTER_ANGLE_DEG,
        help="Center/default test angle in degrees.",
    )

    parser.add_argument(
        "--min-angle",
        type=float,
        default=DEFAULT_MIN_TEST_ANGLE_DEG,
        help="Minimum angle used during sweep test.",
    )

    parser.add_argument(
        "--max-angle",
        type=float,
        default=DEFAULT_MAX_TEST_ANGLE_DEG,
        help="Maximum angle used during sweep test.",
    )

    parser.add_argument(
        "--step-deg",
        type=float,
        default=DEFAULT_STEP_DEG,
        help="Sweep step size in degrees.",
    )

    parser.add_argument(
        "--hold-s",
        type=float,
        default=DEFAULT_HOLD_S,
        help="Time to hold each angle.",
    )

    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=DEFAULT_PWM_HZ,
        help="Servo PWM frequency.",
    )

    parser.add_argument(
        "--reference-clock-frequency-hz",
        type=int,
        default=config.PCA9685_REFERENCE_CLOCK_FREQUENCY_HZ,
        help="PCA9685 reference clock frequency. Defaults to endpoint config.",
    )

    parser.add_argument(
        "--min-pulse-us",
        type=float,
        default=DEFAULT_MIN_PULSE_US,
        help="Pulse width corresponding to min calibration angle.",
    )

    parser.add_argument(
        "--max-pulse-us",
        type=float,
        default=DEFAULT_MAX_PULSE_US,
        help="Pulse width corresponding to max calibration angle.",
    )

    parser.add_argument(
        "--release",
        action="store_true",
        help="Release PWM on the test channel at the end.",
    )

    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Run without interactive confirmation.",
    )

    args = parser.parse_args()

    if args.channel < 0 or args.channel > 15:
        raise ValueError("--channel must be between 0 and 15")

    if args.min_angle >= args.max_angle:
        raise ValueError("--min-angle must be less than --max-angle")

    if args.step_deg <= 0.0:
        raise ValueError("--step-deg must be positive")

    print()
    print("ServoDriver hardware test")
    print("-------------------------")
    print(f"Test channel: {args.channel}")
    print(f"PWM frequency: {args.frequency_hz} Hz")
    print(f"PCA reference clock: {args.reference_clock_frequency_hz} Hz")
    print(f"Calibration: 0-180 deg -> {args.min_pulse_us}-{args.max_pulse_us} us")
    print(f"Test sweep: {args.min_angle} deg to {args.max_angle} deg")
    print()
    print("Before continuing:")
    print("  1. Servo signal wire is connected to the selected PCA9685 channel.")
    print("  2. Servo power is from an external 5-6 V supply.")
    print("  3. RPi/PCA9685 ground and servo supply ground are common.")
    print("  4. The servo horn is not mounted in a way that can bind or hit a hard stop.")
    print()

    if not args.skip_confirm:
        answer = input("Type 'yes' to move the servo: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    calibration = ServoCalibration(
        min_angle_deg=0.0,
        max_angle_deg=180.0,
        min_pulse_us=args.min_pulse_us,
        max_pulse_us=args.max_pulse_us,
    )

    i2c = board.I2C()

    servo_driver = ServoDriver(
        i2c,
        frequency_hz=args.frequency_hz,
        num_channels=16,
        default_calibration=calibration,
        pca_reference_clock_speed=args.reference_clock_frequency_hz,
    )

    try:
        print()
        print("Moving to center...")
        move_and_pause(
            servo_driver=servo_driver,
            channel=args.channel,
            angle_deg=args.center_angle,
            hold_s=1.0,
        )

        print()
        print("Sweeping upward...")
        angle = args.center_angle
        while angle <= args.max_angle + 1e-9:
            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=angle,
                hold_s=args.hold_s,
            )
            angle += args.step_deg

        print()
        print("Returning to center...")
        move_and_pause(
            servo_driver=servo_driver,
            channel=args.channel,
            angle_deg=args.center_angle,
            hold_s=1.0,
        )

        print()
        print("Sweeping downward...")
        angle = args.center_angle
        while angle >= args.min_angle - 1e-9:
            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=angle,
                hold_s=args.hold_s,
            )
            angle -= args.step_deg

        print()
        print("Returning to center...")
        move_and_pause(
            servo_driver=servo_driver,
            channel=args.channel,
            angle_deg=args.center_angle,
            hold_s=1.0,
        )

        print()
        print("Servo test complete.")

    finally:
        if args.release:
            print("Releasing PWM on test channel.")
            servo_driver.release_channel(args.channel)

        servo_driver.close(release=False)


if __name__ == "__main__":
    main()