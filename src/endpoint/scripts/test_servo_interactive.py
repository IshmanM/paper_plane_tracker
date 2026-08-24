import argparse
import time

import board

import src.endpoint.config as config
from src.endpoint.drivers.servo_driver import ServoDriver
from src.endpoint.drivers.servo_calibration import ServoCalibration


DEFAULT_CHANNEL = 15

DEFAULT_HOME_ANGLE_DEG = 90.0
DEFAULT_HOLD_S = 0.25
DEFAULT_PWM_HZ = 50.0

DEFAULT_CAL_MIN_ANGLE_DEG = 0.0
DEFAULT_CAL_MAX_ANGLE_DEG = 180.0
DEFAULT_MIN_PULSE_US = 500.0
DEFAULT_MAX_PULSE_US = 2500.0
DEFAULT_ANGLE_TRIM_DEG = 0.0


def move_and_pause(servo_driver: ServoDriver, channel: int, angle_deg: float, hold_s: float) -> None:
    used_angle = servo_driver.set_angle_deg(channel=channel, angle_deg=angle_deg)
    used_pulse = servo_driver.get_last_pulse_us(channel)
    print(f"channel={channel}, angle_cmd={angle_deg:.1f} deg, angle_used={used_angle:.1f} deg, pulse={used_pulse:.1f} us")
    time.sleep(hold_s)


def parse_angle_input(user_input: str, home_angle: float) -> float | None:
    text = user_input.strip().lower()
    if text == "": return None
    if text in ("h", "home", "c", "center"): return home_angle
    return float(text)


def _promptFloat(prompt: str, default: float | None=None) -> float:
    text = input(f"{prompt}{'' if default is None else f' [{default:g}]'}: ").strip()
    if not text:
        if default is None: raise ValueError(f"{prompt} is required")
        return float(default)
    return float(text)


def _promptManualCalibrationModel(base_kwargs: dict) -> ServoCalibration:
    print("\nOptional nonlinear calibration model")
    print("------------------------------------")
    model = input("Model: none | polynomial | lookup [none]: ").strip().lower() or "none"

    if model in ("none", "n"):
        return ServoCalibration(**base_kwargs)

    if model in ("polynomial", "poly", "p"):
        coeff_text = input("Polynomial coefficients, descending powers (e.g. -0.02,9.8,1721): ").strip()
        coefficients = tuple(float(x.strip()) for x in coeff_text.split(",") if x.strip())
        if len(coefficients) < 2: raise ValueError("Polynomial requires at least two coefficients")
        reference_deg = _promptFloat("Polynomial reference angle deg", 90.0)
        valid_min_deg = _promptFloat("Polynomial valid minimum servo angle deg", base_kwargs["min_angle_deg"])
        valid_max_deg = _promptFloat("Polynomial valid maximum servo angle deg", base_kwargs["max_angle_deg"])
        return ServoCalibration(
            **base_kwargs,
            pulse_polynomial_coefficients_descending=coefficients,
            pulse_polynomial_reference_deg=reference_deg,
            pulse_polynomial_valid_angle_range_deg=(valid_min_deg, valid_max_deg),
        )

    if model in ("lookup", "table", "l"):
        print("Enter rows as servo_angle:pulse_us pairs separated by commas.")
        print("Example: 90:1500, 100:1611.1, 110:1722.2")
        table_text = input("Lookup table: ").strip()
        table = tuple((float(pair.split(":")[0].strip()), float(pair.split(":")[1].strip())) for pair in table_text.split(",") if pair.strip())
        if len(table) < 2: raise ValueError("Lookup table requires at least two points")

        print("Outside the lookup table, endpoint-linear extrapolation is always used within the servo min/max angle limits.")
        record_range = input("Record intended/evaluated extrapolation range as metadata? [y/N]: ").strip().lower() in ("y", "yes")
        extrapolation_range = None
        if record_range:
            extrap_min_deg = _promptFloat("Metadata extrapolation minimum servo angle deg", base_kwargs["min_angle_deg"])
            extrap_max_deg = _promptFloat("Metadata extrapolation maximum servo angle deg", base_kwargs["max_angle_deg"])
            extrapolation_range = (extrap_min_deg, extrap_max_deg)

        return ServoCalibration(**base_kwargs, pulse_lookup_table=table, pulse_lookup_extrapolation_angle_range_deg=extrapolation_range)

    raise ValueError("Model must be none, polynomial, or lookup")


def _describeCalibration(calibration: ServoCalibration) -> str:
    if calibration.pulse_polynomial_coefficients_descending is not None:
        return f"polynomial degree {len(calibration.pulse_polynomial_coefficients_descending) - 1}"
    if calibration.pulse_lookup_table is not None:
        return f"lookup table ({len(calibration.pulse_lookup_table)} points)"
    return "baseline linear"


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive PCA9685 servo angle test using ServoDriver.")

    parser.add_argument(
        "--test-calibration",
        choices=("pan", "tilt"),
        default=None,
        help="Use the configured PAN or TILT calibration and automatically select its PCA9685 channel and default/home angle.",
    )

    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL, help="PCA9685 channel to test in manual mode.")
    parser.add_argument("--home-angle", type=float, default=DEFAULT_HOME_ANGLE_DEG, help="Home/default angle in manual mode.")
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S, help="Small pause after each commanded move.")
    parser.add_argument("--frequency-hz", type=float, default=DEFAULT_PWM_HZ, help="Servo PWM frequency.")
    parser.add_argument("--reference-clock-frequency-hz", type=int, default=config.PCA9685_REFERENCE_CLOCK_FREQUENCY_HZ, help="PCA9685 reference clock frequency. Defaults to endpoint config.")
    parser.add_argument("--cal-min-angle", type=float, default=DEFAULT_CAL_MIN_ANGLE_DEG, help="Minimum calibration angle in manual mode.")
    parser.add_argument("--cal-max-angle", type=float, default=DEFAULT_CAL_MAX_ANGLE_DEG, help="Maximum calibration angle in manual mode.")
    parser.add_argument("--min-pulse-us", type=float, default=DEFAULT_MIN_PULSE_US, help="Pulse width corresponding to calibration minimum angle in manual mode.")
    parser.add_argument("--max-pulse-us", type=float, default=DEFAULT_MAX_PULSE_US, help="Pulse width corresponding to calibration maximum angle in manual mode.")
    parser.add_argument("--angle-trim-deg", type=float, default=DEFAULT_ANGLE_TRIM_DEG, help="Angle trim in manual mode.")
    parser.add_argument("--prompt-calibration-model", action="store_true", help="Manual mode only: prompt for polynomial or lookup-table calibration parameters.")
    parser.add_argument("--move-home-on-start", action="store_true", help="Move to home angle before entering interactive mode.")
    parser.add_argument("--return-home", action="store_true", help="Return to home angle before exiting.")
    parser.add_argument("--release", action="store_true", help="Release PWM on the test channel at the end.")
    parser.add_argument("--skip-confirm", action="store_true", help="Run without interactive safety confirmation.")
    args = parser.parse_args()

    if args.test_calibration is not None and args.prompt_calibration_model:
        raise ValueError("--prompt-calibration-model is manual-mode only")

    if args.test_calibration is not None:
        channel = config.PAN_CHANNEL if args.test_calibration == "pan" else config.TILT_CHANNEL
        home_angle = float(config.DEFAULT_PAN_ANGLE if args.test_calibration == "pan" else config.DEFAULT_TILT_ANGLE)
        calibration = config.SERVO_CALIBRATIONS.get(channel, config.DEFAULT_SERVO_CALIBRATION)
        calibration.validate()
        mode_text = f"configured {args.test_calibration.upper()} calibration"
    else:
        channel = args.channel
        home_angle = args.home_angle
        if channel < 0 or channel > 15: raise ValueError("--channel must be between 0 and 15")
        if args.cal_min_angle >= args.cal_max_angle: raise ValueError("--cal-min-angle must be less than --cal-max-angle")
        if args.min_pulse_us >= args.max_pulse_us: raise ValueError("--min-pulse-us must be less than --max-pulse-us")

        base_kwargs = dict(
            min_angle_deg=args.cal_min_angle,
            max_angle_deg=args.cal_max_angle,
            min_pulse_us=args.min_pulse_us,
            max_pulse_us=args.max_pulse_us,
            angle_trim_deg=args.angle_trim_deg,
        )
        calibration = _promptManualCalibrationModel(base_kwargs) if args.prompt_calibration_model else ServoCalibration(**base_kwargs)
        mode_text = "manual calibration"

    print()
    print("Interactive ServoDriver Hardware Test")
    print("-------------------------------------")
    print(f"Mode: {mode_text}")
    print(f"Test channel: {channel}")
    print(f"PWM frequency: {args.frequency_hz} Hz")
    print(f"PCA reference clock: {args.reference_clock_frequency_hz} Hz")
    print(f"Calibration range: {calibration.min_angle_deg:g}-{calibration.max_angle_deg:g} deg -> {calibration.min_pulse_us:g}-{calibration.max_pulse_us:g} us")
    print(f"Angle trim: {calibration.angle_trim_deg:g} deg")
    print(f"Pulse model: {_describeCalibration(calibration)}")
    if calibration.pulse_polynomial_coefficients_descending is not None:
        print(f"Polynomial coefficients descending: {calibration.pulse_polynomial_coefficients_descending}")
        print(f"Polynomial reference: {calibration.pulse_polynomial_reference_deg:g} deg")
        print(f"Polynomial valid range: {calibration.pulse_polynomial_valid_angle_range_deg}")
    elif calibration.pulse_lookup_table is not None:
        print(f"Lookup table: {calibration.pulse_lookup_table}")
        print(f"Lookup extrapolation range: {calibration.pulse_lookup_extrapolation_angle_range_deg}")
    print(f"Home angle: {home_angle:g} deg")
    print()
    print("Before continuing:")
    print("  1. Servo signal wire is connected to the selected PCA9685 channel.")
    print("  2. Servo power is from an external 5-6 V supply.")
    print("  3. RPi/PCA9685 ground and servo supply ground are common.")
    print("  4. The mechanism is not mounted in a way that can bind or hit a hard stop.")
    print("  5. Be ready to cut power if the servo moves the wrong way.")
    print()
    print("This script does not enforce a separate test min/max angle.")
    print("ServoDriver/ServoCalibration limits still apply.")
    print()

    if not args.skip_confirm:
        if input("Type 'yes' to move the servo: ").strip().lower() != "yes":
            print("Aborted.")
            return

    i2c = board.I2C()
    servo_driver = ServoDriver(
        i2c,
        frequency_hz=args.frequency_hz,
        num_channels=16,
        default_calibration=calibration,
        pca_reference_clock_frequency_hz=args.reference_clock_frequency_hz,
        use_calibration=True,
    )

    try:
        if args.move_home_on_start:
            print("\nMoving to home...")
            move_and_pause(servo_driver, channel, home_angle, 1.0)

        print("\nInteractive mode started.")
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
                requested_angle = parse_angle_input(raw_input_text, home_angle)
            except ValueError:
                print("Invalid input. Enter a number, 'home', or 'q'.")
                continue

            if requested_angle is not None:
                try:
                    move_and_pause(servo_driver, channel, requested_angle, args.hold_s)
                except ValueError as e:
                    print(f"Rejected: {e}")

        if args.return_home:
            print("\nReturning to home...")
            move_and_pause(servo_driver, channel, home_angle, 1.0)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if args.return_home:
            print("Returning to home...")
            move_and_pause(servo_driver, channel, home_angle, 1.0)

    finally:
        if args.release:
            print("Releasing PWM on test channel.")
            servo_driver.release_channel(channel)
        servo_driver.close(release=False)


if __name__ == "__main__":
    main()