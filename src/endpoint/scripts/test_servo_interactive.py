import argparse
import time

import board

import src.endpoint.config as config
from src.endpoint.drivers.servo_driver import ServoDriver, ServoCalibration


DEFAULT_CHANNEL = 15

DEFAULT_HOME_ANGLE_DEG = 90.0
DEFAULT_HOLD_S = 0.25
DEFAULT_PWM_HZ = 50.0

# Common hobby servo calibration.
# This maps angle commands to PWM pulse width.
DEFAULT_CAL_MIN_ANGLE_DEG = 0.0
DEFAULT_CAL_MAX_ANGLE_DEG = 180.0
DEFAULT_MIN_PULSE_US = 500.0
DEFAULT_MAX_PULSE_US = 2500.0
DEFAULT_ANGLE_TRIM_DEG = 0.0


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


def parse_angle_input(
    user_input: str,
    home_angle: float,
) -> float | None:
    text = user_input.strip().lower()

    if text == "":
        return None

    if text in ("h", "home", "c", "center"):
        return home_angle

    return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive PCA9685 servo angle test using ServoDriver."
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help="PCA9685 channel to test.",
    )

    parser.add_argument(
        "--home-angle",
        type=float,
        default=DEFAULT_HOME_ANGLE_DEG,
        help="Home/default angle in degrees.",
    )

    parser.add_argument(
        "--hold-s",
        type=float,
        default=DEFAULT_HOLD_S,
        help="Small pause after each commanded move.",
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
        "--cal-min-angle",
        type=float,
        default=DEFAULT_CAL_MIN_ANGLE_DEG,
        help="Minimum calibration angle.",
    )

    parser.add_argument(
        "--cal-max-angle",
        type=float,
        default=DEFAULT_CAL_MAX_ANGLE_DEG,
        help="Maximum calibration angle.",
    )

    parser.add_argument(
        "--min-pulse-us",
        type=float,
        default=DEFAULT_MIN_PULSE_US,
        help="Pulse width corresponding to calibration minimum angle.",
    )

    parser.add_argument(
        "--max-pulse-us",
        type=float,
        default=DEFAULT_MAX_PULSE_US,
        help="Pulse width corresponding to calibration maximum angle.",
    )

    parser.add_argument(
        "--angle-trim-deg",
        type=float,
        default=DEFAULT_ANGLE_TRIM_DEG,
        help="Angle trim added to commanded angle before PWM conversion.",
    )

    parser.add_argument(
        "--move-home-on-start",
        action="store_true",
        help="Move to home angle before entering interactive mode.",
    )

    parser.add_argument(
        "--return-home",
        action="store_true",
        help="Return to home angle before exiting.",
    )

    parser.add_argument(
        "--release",
        action="store_true",
        help="Release PWM on the test channel at the end.",
    )

    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Run without interactive safety confirmation.",
    )

    args = parser.parse_args()

    if args.channel < 0 or args.channel > 15:
        raise ValueError("--channel must be between 0 and 15")

    if args.cal_min_angle >= args.cal_max_angle:
        raise ValueError("--cal-min-angle must be less than --cal-max-angle")

    if args.min_pulse_us >= args.max_pulse_us:
        raise ValueError("--min-pulse-us must be less than --max-pulse-us")

    print()
    print("Interactive ServoDriver Hardware Test")
    print("-------------------------------------")
    print(f"Test channel: {args.channel}")
    print(f"PWM frequency: {args.frequency_hz} Hz")
    print(f"PCA reference clock: {args.reference_clock_frequency_hz} Hz")
    print(
        f"Calibration: {args.cal_min_angle}-{args.cal_max_angle} deg "
        f"-> {args.min_pulse_us}-{args.max_pulse_us} us"
    )
    print(f"Angle trim: {args.angle_trim_deg} deg")
    print(f"Home angle: {args.home_angle} deg")
    print()
    print("Before continuing:")
    print("  1. Servo signal wire is connected to the selected PCA9685 channel.")
    print("  2. Servo power is from an external 5-6 V supply.")
    print("  3. RPi/PCA9685 ground and servo supply ground are common.")
    print("  4. The mechanism is not mounted in a way that can bind or hit a hard stop.")
    print("  5. Be ready to cut power if the servo moves the wrong way.")
    print()
    print("This script does not enforce a separate test min/max angle.")
    print("Your ServoDriver/calibration may still clamp the command internally.")
    print()

    if not args.skip_confirm:
        answer = input("Type 'yes' to move the servo: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    calibration = ServoCalibration(
        min_angle_deg=args.cal_min_angle,
        max_angle_deg=args.cal_max_angle,
        min_pulse_us=args.min_pulse_us,
        max_pulse_us=args.max_pulse_us,
        angle_trim_deg=args.angle_trim_deg,
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
        if args.move_home_on_start:
            print()
            print("Moving to home...")
            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=args.home_angle,
                hold_s=1.0,
            )

        print()
        print("Interactive mode started.")
        print("Enter any angle in degrees.")
        print("Commands:")
        print("  h / home     -> move to home angle")
        print("  c / center   -> move to home angle")
        print("  q / quit     -> exit")
        print("  blank input  -> do nothing")
        print()

        while True:
            raw_input_text = input("New servo angle deg > ").strip()

            if raw_input_text.lower() in ("q", "quit", "exit"):
                print("Exiting interactive mode.")
                break

            try:
                requested_angle = parse_angle_input(
                    user_input=raw_input_text,
                    home_angle=args.home_angle,
                )
            except ValueError:
                print("Invalid input. Enter a number, 'home', or 'q'.")
                continue

            if requested_angle is None:
                continue

            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=requested_angle,
                hold_s=args.hold_s,
            )

        if args.return_home:
            print()
            print("Returning to home...")
            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=args.home_angle,
                hold_s=1.0,
            )

    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")

        if args.return_home:
            print("Returning to home...")
            move_and_pause(
                servo_driver=servo_driver,
                channel=args.channel,
                angle_deg=args.home_angle,
                hold_s=1.0,
            )

    finally:
        if args.release:
            print("Releasing PWM on test channel.")
            servo_driver.release_channel(args.channel)

        servo_driver.close(release=False)


if __name__ == "__main__":
    main()