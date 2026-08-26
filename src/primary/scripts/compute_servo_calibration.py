import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
import src.endpoint.config as endpoint_config
from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.object_vision_spec import ObjectVisionSpecId
from src.primary.camera_to_platform_calibration import servoAnglesToPlatformYawElevation
from src.comm.link import UdpLink
from src.comm.protocol import CMD_PLATFORM_CONTROL, next_msg_id
from src.comm.network_config import PRIMARY_IP, ENDPOINT_IP, UDP_PORT, PRIMARY_NODE_ID, ENDPOINT_NODE_ID, DEFAULT_MAX_PACKET_BYTES


CALIBRATION_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.ARUCO_MARKER_1

PRIMARY_DIR = Path(__file__).resolve().parents[1]
SERVO_CALIBRATION_DATA_DIR = PRIMARY_DIR/"calibration_data"/"servos"

WINDOW_NAME = "Servo calibration sweep"

POSITION_AVERAGING_WINDOW = 10
CMD_FREQUENCY_HZ = 30.0
DISPLAY_SCALES = (1.0, 1.5, 2.0)

DEFAULT_STEP_DEG = 10.0
DEFAULT_NUM_ROUND_TRIPS = 2
DEFAULT_POLYNOMIAL_DEGREE = 3

MIN_RAY_POINT_SEPARATION_M = 0.30
ANGLE_MATCH_TOLERANCE_DEG = 0.05


def cmd_thread_servo_test(servo_angles: np.ndarray, servo_angles_lock: threading.Lock, stop_event: threading.Event, link: UdpLink, cmd_frequency_hz: float = CMD_FREQUENCY_HZ) -> None:
    cmd_period = 1.0/cmd_frequency_hz
    last_msg_id = None

    try:
        while not stop_event.is_set():
            iter_start = time.perf_counter()

            with servo_angles_lock:
                q = servo_angles.copy()

            last_msg_id = next_msg_id(last_msg_id)
            link.send_cmd(
                msg_id=last_msg_id,
                sender_time=iter_start,
                cmd_name=CMD_PLATFORM_CONTROL,
                cmd_payload={
                    "pan_deg": float(q[config.SERVO_IDX["pan"]]),
                    "tilt_deg": float(q[config.SERVO_IDX["tilt"]]),
                    "triggering_halted": True,
                    "trigger": False,
                },
            )

            link.recv_telemetry_available()
            for error_msg in link.recv_errors_available():
                print(f"Endpoint error: {error_msg}")

            stop_event.wait(max(0.0, cmd_period - (time.perf_counter() - iter_start)))

    finally:
        q_default = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float)

        for _ in range(3):
            try:
                now = time.perf_counter()
                last_msg_id = next_msg_id(last_msg_id)
                link.send_cmd(
                    msg_id=last_msg_id,
                    sender_time=now,
                    cmd_name=CMD_PLATFORM_CONTROL,
                    cmd_payload={
                        "pan_deg": float(q_default[config.SERVO_IDX["pan"]]),
                        "tilt_deg": float(q_default[config.SERVO_IDX["tilt"]]),
                        "triggering_halted": True,
                        "trigger": False,
                    },
                )
                time.sleep(0.02)
            except Exception:
                break


def _drawOverlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], display_scale: float = 1.0) -> None:
    x, y = int(round(10*display_scale)), int(round(22*display_scale))
    y_step = int(round(20*display_scale))
    font_scale = 0.48*display_scale
    outline_thickness = max(1, int(round(3*display_scale)))
    text_thickness = max(1, int(round(display_scale)))

    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
        y += y_step


def _promptFloat(prompt: str, default: float) -> float:
    text = input(f"{prompt} [{default:g}]: ").strip()
    return default if not text else float(text)


def _promptInt(prompt: str, default: int) -> int:
    text = input(f"{prompt} [{default}]: ").strip()
    return default if not text else int(text)


def _promptYesNo(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter yes or no.")


def _buildSweep(axis: str, angle_a_deg: float, angle_b_deg: float, step_deg: float, num_round_trips: int) -> list[dict]:
    if step_deg <= 0.0 or num_round_trips < 1 or np.isclose(angle_a_deg, angle_b_deg):
        raise ValueError("Step must be > 0, round trips >= 1, and A/B must differ.")

    sign = 1.0 if angle_b_deg > angle_a_deg else -1.0
    step = sign*step_deg
    forward = list(np.arange(angle_a_deg, angle_b_deg, step))
    if not forward or not np.isclose(forward[0], angle_a_deg):
        forward.insert(0, angle_a_deg)
    if not np.isclose(forward[-1], angle_b_deg):
        forward.append(angle_b_deg)

    reverse = forward[-2::-1]
    poses = []

    for cycle in range(1, num_round_trips + 1):
        forward_cycle = forward if cycle == 1 else forward[1:]
        poses += [{"axis": axis, "angle_deg": float(a), "cycle": cycle, "direction": "forward"} for a in forward_cycle]
        poses += [{"axis": axis, "angle_deg": float(a), "cycle": cycle, "direction": "reverse"} for a in reverse]

    return poses


def _angleBetweenDirectionsDeg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _saveRaw(path: Path, test_config: dict, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_samples = []

    for sample in samples:
        serializable_samples.append({
            **{k: v for k, v in sample.items() if k not in ("near_position_camera_m", "far_position_camera_m", "direction_camera")},
            "near_position_camera_m": sample["near_position_camera_m"].tolist(),
            "far_position_camera_m": sample["far_position_camera_m"].tolist(),
            "direction_camera": sample["direction_camera"].tolist(),
        })

    path.write_text(json.dumps({"test_config": test_config, "samples": serializable_samples}, indent=4) + "\n", encoding="utf-8")


def _fitPhysicalCoordinates(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray([sample["direction_camera"] for sample in samples], dtype=float)

    # With one joint moving and the other fixed, laser directions rotate about one
    # fixed axis in camera coordinates. Fit that axis from the direction-vector arc.
    centered = directions - np.mean(directions, axis=0)
    _, _, Vt = np.linalg.svd(centered)
    rotation_axis_camera = Vt[-1]
    rotation_axis_camera /= np.linalg.norm(rotation_axis_camera)

    first_projection = directions[0] - rotation_axis_camera*np.dot(rotation_axis_camera, directions[0])
    first_projection /= np.linalg.norm(first_projection)
    second_basis = np.cross(rotation_axis_camera, first_projection)
    second_basis /= np.linalg.norm(second_basis)

    phase_rad = np.unwrap(np.arctan2(directions@second_basis, directions@first_projection))
    commands = np.asarray([sample["test_angle_deg"] for sample in samples], dtype=float)

    if np.corrcoef(commands, phase_rad)[0, 1] < 0.0:
        phase_rad = -phase_rad
        rotation_axis_camera = -rotation_axis_camera

    phase_deg = np.rad2deg(phase_rad)

    # This experiment determines relative servo-angle shape, not absolute mechanical zero.
    # Choose one constant coordinate alignment near the command values. The final endpoint
    # trim later shifts the INPUT to the polynomial/lookup model to establish absolute zero.
    physical_angle_deg = phase_deg + float(np.median(commands - phase_deg))
    return physical_angle_deg, rotation_axis_camera


def _buildPulseLookupTable(samples: list[dict]) -> list[dict]:
    rows = []
    unique_commands = sorted(set(round(float(sample["test_angle_deg"]), 8) for sample in samples))

    for command in unique_commands:
        group = [sample for sample in samples if abs(sample["test_angle_deg"] - command) <= ANGLE_MATCH_TOLERANCE_DEG]
        physical_values = np.asarray([sample["estimated_physical_angle_deg"] for sample in group], dtype=float)
        pulse_values = np.asarray([sample["pulse_us"] for sample in group], dtype=float)
        rows.append({
            "source_command_angle_deg": float(command),
            "servo_angle_deg": float(np.median(physical_values)),
            "pulse_us": float(np.median(pulse_values)),
            "num_samples": len(group),
            "physical_angle_std_deg": float(np.std(physical_values, ddof=1)) if len(group) >= 2 else 0.0,
        })

    rows.sort(key=lambda row: row["servo_angle_deg"])
    if any(b["servo_angle_deg"] <= a["servo_angle_deg"] for a, b in zip(rows[:-1], rows[1:])):
        raise RuntimeError("Reconstructed servo angles are not strictly increasing; cannot build a monotonic lookup table.")
    return rows



def _extrapolateLookupEndpointLinear(lookup_table: list[list[float]], angle_deg: float) -> float:
    if len(lookup_table) < 2: raise ValueError("Lookup table needs at least two points")
    if angle_deg < lookup_table[0][0]:
        angle_0, pulse_0 = lookup_table[0]
        angle_1, pulse_1 = lookup_table[1]
        return float(pulse_0 + (pulse_1 - pulse_0)/(angle_1 - angle_0)*(angle_deg - angle_0))
    if angle_deg > lookup_table[-1][0]:
        angle_0, pulse_0 = lookup_table[-2]
        angle_1, pulse_1 = lookup_table[-1]
        return float(pulse_1 + (pulse_1 - pulse_0)/(angle_1 - angle_0)*(angle_deg - angle_1))

    angles = [row[0] for row in lookup_table]
    index = int(np.searchsorted(angles, angle_deg))
    if index < len(lookup_table) and np.isclose(angles[index], angle_deg): return float(lookup_table[index][1])
    angle_0, pulse_0 = lookup_table[index - 1]
    angle_1, pulse_1 = lookup_table[index]
    return float(pulse_0 + (pulse_1 - pulse_0)/(angle_1 - angle_0)*(angle_deg - angle_0))

def _analyze(samples: list[dict], test_config: dict, summary_path: Path) -> dict | None:
    if len(samples) < 4:
        print("Need at least 4 completed rays before analysis.")
        return None

    physical_angle_deg, rotation_axis_camera = _fitPhysicalCoordinates(samples)
    commands = np.asarray([sample["test_angle_deg"] for sample in samples], dtype=float)

    for sample, physical in zip(samples, physical_angle_deg):
        sample["estimated_physical_angle_deg"] = float(physical)

    # Keep the old scale/nonlinearity/backlash diagnostics because they are useful
    # descriptions of the hardware even though the endpoint model is now angle->pulse.
    linear_coeff = np.polyfit(commands, physical_angle_deg, 1)
    overall_scale = float(linear_coeff[0])
    linear_prediction = np.polyval(linear_coeff, commands)
    nonlinearity_residual = physical_angle_deg - linear_prediction

    interval_rows = []
    for previous, current in zip(samples[:-1], samples[1:]):
        command_delta = current["test_angle_deg"] - previous["test_angle_deg"]
        if abs(command_delta) <= ANGLE_MATCH_TOLERANCE_DEG: continue
        measured_delta = current["estimated_physical_angle_deg"] - previous["estimated_physical_angle_deg"]
        interval_rows.append({
            "from_command_deg": previous["test_angle_deg"], "to_command_deg": current["test_angle_deg"],
            "cycle": current["cycle"], "direction": current["direction"],
            "command_delta_deg": float(command_delta), "measured_delta_deg": float(measured_delta),
            "physical_per_command_scale": float(abs(measured_delta/command_delta)),
        })

    unique_commands = sorted(set(round(float(q), 8) for q in commands))
    backlash_values = []
    for command in unique_commands:
        forward_values = [sample["estimated_physical_angle_deg"] for sample in samples if abs(sample["test_angle_deg"] - command) <= ANGLE_MATCH_TOLERANCE_DEG and sample["direction"] == "forward"]
        reverse_values = [sample["estimated_physical_angle_deg"] for sample in samples if abs(sample["test_angle_deg"] - command) <= ANGLE_MATCH_TOLERANCE_DEG and sample["direction"] == "reverse"]
        if forward_values and reverse_values: backlash_values.append(abs(float(np.mean(forward_values) - np.mean(reverse_values))))

    repeatability_values = []
    for command in unique_commands:
        for direction in ("forward", "reverse"):
            values = [sample["estimated_physical_angle_deg"] for sample in samples if abs(sample["test_angle_deg"] - command) <= ANGLE_MATCH_TOLERANCE_DEG and sample["direction"] == direction]
            if len(values) >= 2: repeatability_values.append(float(np.std(values, ddof=1)))

    local_scales = np.asarray([row["physical_per_command_scale"] for row in interval_rows], dtype=float)
    backlash_values = np.asarray(backlash_values, dtype=float)
    repeatability_values = np.asarray(repeatability_values, dtype=float)

    # Collapse repeated forward/reverse/cycle measurements at each test pulse into one
    # robust lookup point. This table directly means (servo calibration angle, pulse_us).
    lookup_rows = _buildPulseLookupTable(samples)
    lookup_angles = np.asarray([row["servo_angle_deg"] for row in lookup_rows], dtype=float)
    lookup_pulses = np.asarray([row["pulse_us"] for row in lookup_rows], dtype=float)
    pulse_lookup_table = [[float(angle), float(pulse)] for angle, pulse in zip(lookup_angles, lookup_pulses)]
    lookup_extrapolation_angle_range_deg = [
        float(test_config["baseline_min_angle_deg"]),
        float(test_config["baseline_max_angle_deg"]),
    ]
    lookup_extrapolated_min_pulse_us = _extrapolateLookupEndpointLinear(pulse_lookup_table, lookup_extrapolation_angle_range_deg[0])
    lookup_extrapolated_max_pulse_us = _extrapolateLookupEndpointLinear(pulse_lookup_table, lookup_extrapolation_angle_range_deg[1])
    lookup_extrapolation_within_allowed_pulse_range = (
        test_config["baseline_min_pulse_us"] <= lookup_extrapolated_min_pulse_us <= test_config["baseline_max_pulse_us"]
        and test_config["baseline_min_pulse_us"] <= lookup_extrapolated_max_pulse_us <= test_config["baseline_max_pulse_us"]
    )

    max_polynomial_degree = min(test_config["polynomial_degree"], len(lookup_rows) - 1)
    reference_deg = float(0.5*(lookup_angles[0] + lookup_angles[-1]))
    polynomial_fits = {}
    raw_physical = np.asarray([sample["estimated_physical_angle_deg"] for sample in samples], dtype=float)
    raw_pulses = np.asarray([sample["pulse_us"] for sample in samples], dtype=float)

    for degree in range(1, max_polynomial_degree + 1):
        coefficients = np.polyfit(lookup_angles - reference_deg, lookup_pulses, degree)
        lookup_prediction = np.polyval(coefficients, lookup_angles - reference_deg)
        lookup_residual = lookup_pulses - lookup_prediction
        raw_prediction = np.polyval(coefficients, raw_physical - reference_deg)
        raw_residual = raw_pulses - raw_prediction

        polynomial_fits[str(degree)] = {
            "degree": degree,
            "pulse_polynomial_coefficients_descending": coefficients.tolist(),
            "pulse_polynomial_reference_deg": reference_deg,
            "pulse_polynomial_valid_angle_range_deg": [float(lookup_angles[0]), float(lookup_angles[-1])],
            "lookup_point_fit_rms_us": float(np.sqrt(np.mean(lookup_residual**2))),
            "lookup_point_fit_max_abs_us": float(np.max(np.abs(lookup_residual))),
            "raw_sample_fit_rms_us": float(np.sqrt(np.mean(raw_residual**2))),
            "raw_sample_fit_max_abs_us": float(np.max(np.abs(raw_residual))),
        }

    summary = {
        "test_config": test_config,
        "num_ray_samples": len(samples),
        "estimated_rotation_axis_camera": rotation_axis_camera.tolist(),
        "relative_angle_coordinate_note": "Camera sweep determines servo-angle shape up to one constant offset. Endpoint trim later shifts the model INPUT; for pan/tilt trim is conventionally established at the physical forward pose.",
        "overall_physical_per_command_scale": overall_scale,
        "inverse_global_scale_correction": None if abs(overall_scale) < 1e-12 else 1.0/overall_scale,
        "local_scale_mean": float(np.mean(local_scales)) if local_scales.size else None,
        "local_scale_min": float(np.min(local_scales)) if local_scales.size else None,
        "local_scale_max": float(np.max(local_scales)) if local_scales.size else None,
        "nonlinearity_rms_deg_from_best_linear": float(np.sqrt(np.mean(nonlinearity_residual**2))),
        "nonlinearity_max_abs_deg_from_best_linear": float(np.max(np.abs(nonlinearity_residual))),
        "backlash_mean_deg": float(np.mean(backlash_values)) if backlash_values.size else None,
        "backlash_max_deg": float(np.max(backlash_values)) if backlash_values.size else None,
        "repeatability_same_direction_std_mean_deg": float(np.mean(repeatability_values)) if repeatability_values.size else None,
        "repeatability_same_direction_std_max_deg": float(np.max(repeatability_values)) if repeatability_values.size else None,
        "pulse_lookup_points": lookup_rows,
        "pulse_lookup_table": pulse_lookup_table,
        "pulse_lookup_extrapolation_mode": "endpoint_linear",
        "pulse_lookup_extrapolation_angle_range_deg": lookup_extrapolation_angle_range_deg,
        "lookup_extrapolated_min_pulse_us": lookup_extrapolated_min_pulse_us,
        "lookup_extrapolated_max_pulse_us": lookup_extrapolated_max_pulse_us,
        "lookup_extrapolation_within_allowed_pulse_range": lookup_extrapolation_within_allowed_pulse_range,
        "maximum_polynomial_degree": max_polynomial_degree,
        "pulse_polynomial_reference_deg": reference_deg,
        "polynomial_fits": polynomial_fits,
        "intervals": interval_rows,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=4) + "\n", encoding="utf-8")

    print("\nSERVO CALIBRATION SUMMARY")
    print("-------------------------")
    print(f"Axis: {test_config['axis'].upper()} | samples: {len(samples)} | round trips: {test_config['num_round_trips']}")
    print(f"Overall physical/command scale: {overall_scale:.6f}")
    if local_scales.size: print(f"Local scale: mean {np.mean(local_scales):.6f} | min {np.min(local_scales):.6f} | max {np.max(local_scales):.6f}")
    print(f"Nonlinearity vs best line: RMS {summary['nonlinearity_rms_deg_from_best_linear']:.4f} deg | max {summary['nonlinearity_max_abs_deg_from_best_linear']:.4f} deg")
    print(f"Backlash diagnostic only: mean {summary['backlash_mean_deg']:.4f} deg | max {summary['backlash_max_deg']:.4f} deg" if summary["backlash_mean_deg"] is not None else "Backlash: insufficient forward/reverse overlap")
    print(f"Repeatability same-direction std: mean {summary['repeatability_same_direction_std_mean_deg']:.4f} deg | max {summary['repeatability_same_direction_std_max_deg']:.4f} deg" if summary["repeatability_same_direction_std_mean_deg"] is not None else "Repeatability: insufficient repeated same-direction samples")

    print("\nPulse lookup table: (servo calibration angle deg, pulse_us)")
    print("pulse_lookup_table=(")
    for angle, pulse in pulse_lookup_table:
        print(f"    ({angle:.9f}, {pulse:.6f}),")
    print(")")
    print(f"pulse_lookup_extrapolation_angle_range_deg=({lookup_extrapolation_angle_range_deg[0]:.1f}, {lookup_extrapolation_angle_range_deg[1]:.1f})")
    print("Outside the tested table range: continue the first/last lookup segment linearly.")
    print(f"Extrapolated pulse at {lookup_extrapolation_angle_range_deg[0]:g} deg: {lookup_extrapolated_min_pulse_us:.3f} us")
    print(f"Extrapolated pulse at {lookup_extrapolation_angle_range_deg[1]:g} deg: {lookup_extrapolated_max_pulse_us:.3f} us")
    if not lookup_extrapolation_within_allowed_pulse_range:
        print("WARNING: full-range endpoint extrapolation leaves the configured min/max pulse range; runtime will reject those out-of-range pulses.")

    print(f"\nPulse polynomial reference: {reference_deg:.9f} deg")
    for degree in range(1, max_polynomial_degree + 1):
        fit = polynomial_fits[str(degree)]
        print(f"  Degree {degree}: lookup RMS {fit['lookup_point_fit_rms_us']:.4f} us | raw RMS {fit['raw_sample_fit_rms_us']:.4f} us")
        print(f"    coefficients descending: {fit['pulse_polynomial_coefficients_descending']}")
        print(f"    valid angle range: {fit['pulse_polynomial_valid_angle_range_deg']}")

    print("\nTrim note: the fitted table/polynomial is NOT shifted when trim changes.")
    print("Normal endpoint evaluation is pulse_model(command_angle + angle_trim_deg).")
    print("Ideally a trim measured at 30 deg or 90 deg would be identical; pan/tilt trim is still defined at forward pose by convention.")
    print(f"Saved summary: {summary_path}")
    return summary


def main() -> None:
    print("SERVO CHARACTERIZATION MODE")
    print("---------------------------")
    print("Endpoint must currently be running with:")
    print("  python -m src.endpoint.main --no-servo-calibration")
    print()
    print("In that mode:")
    print("  trim:               BYPASSED")
    print("  pulse polynomial:   BYPASSED")
    print("  pulse lookup table: BYPASSED")
    print("  output:              baseline linear angle->pulse mapping only")
    print("NOTE: this primary checkout's src.endpoint.config must match the config running on the endpoint.")
    print()

    if not _promptYesNo("Is the endpoint currently running with --no-servo-calibration?"):
        print("Test cancelled. Restart the endpoint with --no-servo-calibration, then run this script again.")
        return

    axis_text = input("Test axis [pan]: ").strip().lower()
    axis = "pan" if not axis_text else axis_text
    if axis not in ("pan", "tilt"):
        raise ValueError("Axis must be 'pan' or 'tilt'.")

    axis_idx = config.SERVO_IDX[axis]
    other_axis = "tilt" if axis == "pan" else "pan"
    other_idx = config.SERVO_IDX[other_axis]
    forward_angle = float(config.FORWARD_SERVO_ANGLES[axis_idx])

    endpoint_channel = endpoint_config.PAN_CHANNEL if axis == "pan" else endpoint_config.TILT_CHANNEL
    endpoint_servo_calibration = endpoint_config.SERVO_CALIBRATIONS.get(endpoint_channel, endpoint_config.DEFAULT_SERVO_CALIBRATION)
    print(
        f"Endpoint baseline for {axis.upper()} channel {endpoint_channel}: "
        f"{endpoint_servo_calibration.min_angle_deg:g}-{endpoint_servo_calibration.max_angle_deg:g} deg -> "
        f"{endpoint_servo_calibration.min_pulse_us:g}-{endpoint_servo_calibration.max_pulse_us:g} us"
    )

    default_a = max(float(config.MIN_SERVO_ANGLES[axis_idx]), forward_angle - 30.0)
    default_b = min(float(config.MAX_SERVO_ANGLES[axis_idx]), forward_angle + 30.0)
    angle_a_deg = _promptFloat("A/start angle deg", default_a)
    angle_b_deg = _promptFloat("B/end angle deg", default_b)
    step_deg = _promptFloat("Step deg", DEFAULT_STEP_DEG)
    fixed_other_angle_deg = _promptFloat(f"Fixed {other_axis} angle deg", float(config.FORWARD_SERVO_ANGLES[other_idx]))
    num_round_trips = _promptInt("Number of round trips", DEFAULT_NUM_ROUND_TRIPS)
    polynomial_degree = _promptInt("Maximum polynomial degree", DEFAULT_POLYNOMIAL_DEGREE)

    q_check = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
    q_check[axis_idx] = angle_a_deg
    q_check[other_idx] = fixed_other_angle_deg
    if np.any(q_check < config.MIN_SERVO_ANGLES) or np.any(q_check > config.MAX_SERVO_ANGLES):
        raise ValueError("A/fixed angle is outside configured servo limits.")
    q_check[axis_idx] = angle_b_deg
    if np.any(q_check < config.MIN_SERVO_ANGLES) or np.any(q_check > config.MAX_SERVO_ANGLES):
        raise ValueError("B/fixed angle is outside configured servo limits.")

    sweep = _buildSweep(axis, angle_a_deg, angle_b_deg, step_deg, num_round_trips)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = SERVO_CALIBRATION_DATA_DIR/f"{axis}_{timestamp}_raw.json"
    summary_path = SERVO_CALIBRATION_DATA_DIR/f"{axis}_{timestamp}_summary.json"

    test_config = {
        "axis": axis,
        "other_axis": other_axis,
        "angle_a_deg": angle_a_deg,
        "angle_b_deg": angle_b_deg,
        "step_deg": step_deg,
        "fixed_other_angle_deg": fixed_other_angle_deg,
        "num_round_trips": num_round_trips,
        "polynomial_degree": polynomial_degree,
        "polynomial_degree_meaning": "maximum pulse-polynomial degree; analysis reports every degree from 1 through this value",
        "endpoint_use_servo_calibration_required": False,
        "baseline_min_angle_deg": endpoint_servo_calibration.min_angle_deg,
        "baseline_max_angle_deg": endpoint_servo_calibration.max_angle_deg,
        "baseline_min_pulse_us": endpoint_servo_calibration.min_pulse_us,
        "baseline_max_pulse_us": endpoint_servo_calibration.max_pulse_us,
        "endpoint_pca_requested_pwm_frequency_hz": endpoint_config.PCA9685_FREQUENCY_HZ,
        "endpoint_pca_reference_clock_frequency_hz": endpoint_config.PCA9685_REFERENCE_CLOCK_FREQUENCY_HZ,
        "trim_note": "Trim is bypassed while collecting data. After fitting, trim shifts the INPUT to the pulse polynomial/lookup; the model itself is not shifted.",
    }

    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, config.FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if config.CAMERA_AUTO_EXPOSURE else 0.25)

    if not config.CAMERA_AUTO_EXPOSURE:
        camera.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE)
        camera.set(cv2.CAP_PROP_GAIN, config.CAMERA_GAIN)

    camera.set(cv2.CAP_PROP_AUTO_WB, 1 if config.CAMERA_AUTO_WHITE_BALANCE else 0)
    if not config.CAMERA_AUTO_WHITE_BALANCE:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, config.CAMERA_WHITE_BALANCE_TEMPERATURE)

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if config.CAMERA_AUTOFOCUS else 0)
    if not config.CAMERA_AUTOFOCUS:
        camera.set(cv2.CAP_PROP_FOCUS, config.CAMERA_FOCUS)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    link = UdpLink(
        local_ip=PRIMARY_IP,
        remote_ip=ENDPOINT_IP,
        port=UDP_PORT,
        local_node_id=PRIMARY_NODE_ID,
        remote_node_id=ENDPOINT_NODE_ID,
        max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
        check_remote_ip=True,
    )

    servo_angles = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
    servo_angles_lock = threading.Lock()
    stop_event = threading.Event()
    cmd_thread = threading.Thread(target=cmd_thread_servo_test, args=(servo_angles, servo_angles_lock, stop_event, link), daemon=True)
    cmd_thread.start()

    measurement_buffer = deque(maxlen=POSITION_AVERAGING_WINDOW)
    samples: list[dict] = []
    near_position_camera_m = None
    current_pose_commanded = False
    video_frozen = False
    last_vision_frame = None
    display_scale_index = 0

    print("\nServo calibration sweep")
    print(f"{axis.upper()}: {angle_a_deg:g} -> {angle_b_deg:g} -> {angle_a_deg:g}, {num_round_trips} round trips, step {step_deg:g} deg")
    print(f"Fixed {other_axis}: {fixed_other_angle_deg:g} deg")
    print(f"Raw output: {raw_path}")
    print(f"Summary:    {summary_path}")
    print("SPACE step | N near+lock | B far+record | C cancel near | X erase last | O analyze/save summary | P freeze | Tab resize | Q/Esc quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.FRAME_W, config.FRAME_H)

    try:
        while camera.isOpened():
            mean_position_camera_m = None
            position_rms_m = None

            if not video_frozen:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read camera frame.")

                object_detected, detection, measurement = detectSingleObject(frame, CALIBRATION_OBJECT_VISION_SPEC_ID, camera_calibration)

                if object_detected:
                    position_camera_m = np.array([measurement.x, measurement.y, measurement.z], dtype=float)
                    if np.all(np.isfinite(position_camera_m)):
                        measurement_buffer.append(position_camera_m)
                        drawDetection(frame, detection)
                    else:
                        measurement_buffer.clear()
                else:
                    measurement_buffer.clear()

                if measurement_buffer:
                    positions = np.asarray(measurement_buffer, dtype=float)
                    mean_position_camera_m = np.mean(positions, axis=0)
                    position_rms_m = float(np.sqrt(np.mean(np.sum((positions - mean_position_camera_m)**2, axis=1))))

                last_vision_frame = frame.copy()
            else:
                if last_vision_frame is None:
                    continue
                frame = last_vision_frame.copy()
                if measurement_buffer:
                    positions = np.asarray(measurement_buffer, dtype=float)
                    mean_position_camera_m = np.mean(positions, axis=0)
                    position_rms_m = float(np.sqrt(np.mean(np.sum((positions - mean_position_camera_m)**2, axis=1))))

            with servo_angles_lock:
                q = servo_angles.copy()

            pan_deg = float(q[config.SERVO_IDX["pan"]])
            tilt_deg = float(q[config.SERVO_IDX["tilt"]])
            yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)

            sweep_complete = len(samples) == len(sweep)

            if sweep_complete:
                next_action = "SWEEP COMPLETE - press O to analyze/save summary"
            elif not current_pose_commanded:
                next_pose = sweep[len(samples)]
                next_action = f"NEXT: press SPACE to command {axis} {next_pose['angle_deg']:.2f} deg"
            elif near_position_camera_m is None:
                next_action = "NEXT: move ArUco NEAR onto laser center, then press N"
            else:
                next_action = "NEXT: move SAME ArUco FARTHER along laser, then press B"

            lines = [
                (f"{axis.upper()} sweep {angle_a_deg:g}->{angle_b_deg:g}->{angle_a_deg:g} | step {step_deg:g} | round trips {num_round_trips} | rays {len(samples)}/{len(sweep)}", (255, 255, 255)),
                (f"Pan {pan_deg:.2f} | tilt {tilt_deg:.2f} | yaw {yaw_deg:+.2f} | elev {elevation_deg:+.2f}", (255, 255, 255)),
                (next_action, (0, 255, 255)),
            ]

            if mean_position_camera_m is not None:
                lines.append((
                    f"ArUco avg {len(measurement_buffer)}/{POSITION_AVERAGING_WINDOW}: "
                    f"[{mean_position_camera_m[0]:+.3f}, {mean_position_camera_m[1]:+.3f}, {mean_position_camera_m[2]:+.3f}] m | RMS {position_rms_m:.4f} m",
                    (0, 255, 0),
                ))
            else:
                lines.append((f"ArUco avg 0/{POSITION_AVERAGING_WINDOW}: no stable measurement", (0, 0, 255)))

            if not sweep_complete and current_pose_commanded:
                pose = sweep[len(samples)]
                lines.append((
                    f"Current scheduled pose: cycle {pose['cycle']}/{num_round_trips} | {pose['direction']} | "
                    f"{axis} {pose['angle_deg']:.2f} deg | {other_axis} {fixed_other_angle_deg:.2f} deg",
                    (255, 0, 255),
                ))

            display_scale = DISPLAY_SCALES[display_scale_index]
            lines += [
                ("SPACE next step | N near+lock | B far+record | C cancel | X erase last", (255, 255, 255)),
                (f"O analyze | P freeze | Tab resize ({display_scale:.1f}x) | Q/Esc quit", (255, 255, 255)),
            ]

            try:
                _, _, display_w, display_h = cv2.getWindowImageRect(WINDOW_NAME)
            except cv2.error:
                display_w, display_h = config.FRAME_W, config.FRAME_H

            display_w, display_h = max(1, display_w), max(1, display_h)
            display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            display_frame = cv2.flip(display_frame, 1)
            display_scale = min(display_w/config.FRAME_W, display_h/config.FRAME_H)
            _drawOverlay(display_frame, lines, display_scale)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Quitting...")
                break

            elif key == 9:
                display_scale_index = (display_scale_index + 1)%len(DISPLAY_SCALES)
                preset_scale = DISPLAY_SCALES[display_scale_index]
                cv2.resizeWindow(WINDOW_NAME, int(round(config.FRAME_W*preset_scale)), int(round(config.FRAME_H*preset_scale)))
                print(f"Display preset: {preset_scale:.1f}x")

            elif key == ord("p"):
                video_frozen = not video_frozen
                if not video_frozen:
                    measurement_buffer.clear()
                print(f"Video {'frozen' if video_frozen else 'live'}.")

            elif key == 32:
                if sweep_complete:
                    print("Sweep already complete. Press O to analyze.")
                elif near_position_camera_m is not None:
                    print("Finish the current near/far ray with B or cancel it with C before stepping.")
                elif current_pose_commanded:
                    print("Record the current ray with N then B before stepping.")
                else:
                    pose = sweep[len(samples)]
                    q_command = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
                    q_command[axis_idx] = pose["angle_deg"]
                    q_command[other_idx] = fixed_other_angle_deg

                    with servo_angles_lock:
                        servo_angles[:] = q_command

                    current_pose_commanded = True
                    measurement_buffer.clear()
                    print(
                        f"\nSTEP {len(samples) + 1}/{len(sweep)}: cycle {pose['cycle']} {pose['direction']} | "
                        f"command {axis}={pose['angle_deg']:.3f} deg, {other_axis}={fixed_other_angle_deg:.3f} deg"
                    )
                    print("  Next: center ArUco NEAR on laser, then press N.")

            elif key == ord("n"):
                if not current_pose_commanded or sweep_complete:
                    print("Press SPACE to command the scheduled servo pose first.")
                elif near_position_camera_m is not None:
                    print("Near point already captured. Move the ArUco farther and press B.")
                elif len(measurement_buffer) < POSITION_AVERAGING_WINDOW or mean_position_camera_m is None:
                    print(f"Need {POSITION_AVERAGING_WINDOW} consecutive valid measurements.")
                else:
                    near_position_camera_m = mean_position_camera_m.copy()
                    measurement_buffer.clear()
                    print(f"  Near captured: {near_position_camera_m}")
                    print("  Servo command is now logically locked. Move the SAME ArUco farther along the laser and press B.")

            elif key == ord("b"):
                if near_position_camera_m is None:
                    print("Press N first to capture the near point.")
                elif len(measurement_buffer) < POSITION_AVERAGING_WINDOW or mean_position_camera_m is None:
                    print(f"Need {POSITION_AVERAGING_WINDOW} consecutive valid measurements.")
                else:
                    far_position_camera_m = mean_position_camera_m.copy()
                    ray_vector = far_position_camera_m - near_position_camera_m
                    baseline_m = float(np.linalg.norm(ray_vector))

                    if baseline_m < MIN_RAY_POINT_SEPARATION_M:
                        print(f"Rejected: near/far separation {baseline_m:.3f} m < {MIN_RAY_POINT_SEPARATION_M:.2f} m.")
                    else:
                        pose = sweep[len(samples)]
                        direction_camera = ray_vector/baseline_m
                        sample = {
                            "index": len(samples) + 1,
                            "cycle": pose["cycle"],
                            "direction": pose["direction"],
                            "test_axis": axis,
                            "test_angle_deg": pose["angle_deg"],
                            "pulse_us": float(endpoint_servo_calibration.baseline_angle_to_pulse_us(pose["angle_deg"])),
                            "fixed_axis": other_axis,
                            "fixed_angle_deg": fixed_other_angle_deg,
                            "pan_deg": float(q[config.SERVO_IDX["pan"]]),
                            "tilt_deg": float(q[config.SERVO_IDX["tilt"]]),
                            "near_position_camera_m": near_position_camera_m.copy(),
                            "far_position_camera_m": far_position_camera_m.copy(),
                            "baseline_m": baseline_m,
                            "direction_camera": direction_camera,
                        }
                        samples.append(sample)
                        _saveRaw(raw_path, test_config, samples)

                        print(
                            f"  RAW ray #{sample['index']}: cycle={sample['cycle']} {sample['direction']} | "
                            f"{axis}={sample['test_angle_deg']:.3f} deg | pulse={sample['pulse_us']:.3f} us | baseline={baseline_m:.3f} m | "
                            f"ray_C=[{direction_camera[0]:+.6f}, {direction_camera[1]:+.6f}, {direction_camera[2]:+.6f}]"
                        )

                        if len(samples) >= 2:
                            previous = samples[-2]
                            raw_change = _angleBetweenDirectionsDeg(previous["direction_camera"], direction_camera)
                            command_change = sample["test_angle_deg"] - previous["test_angle_deg"]
                            print(f"  Raw previous-ray change: command {command_change:+.3f} deg | laser-direction separation {raw_change:.4f} deg")

                        near_position_camera_m = None
                        measurement_buffer.clear()
                        current_pose_commanded = False

                        if len(samples) == len(sweep):
                            print("\nSWEEP COMPLETE. Press O for scale/nonlinearity/backlash/repeatability analysis.")
                        else:
                            next_pose = sweep[len(samples)]
                            print(
                                f"  Next: press SPACE when ready to command {axis}={next_pose['angle_deg']:.3f} deg "
                                f"(cycle {next_pose['cycle']} {next_pose['direction']})."
                            )

            elif key == ord("c"):
                if near_position_camera_m is None:
                    print("No near-point capture to cancel.")
                else:
                    near_position_camera_m = None
                    measurement_buffer.clear()
                    print("Near-point capture cancelled. Servo remains at the current scheduled pose.")

            elif key == ord("x"):
                if near_position_camera_m is not None:
                    print("Cancel the current near-point capture with C first.")
                elif not samples:
                    print("No completed rays to erase.")
                else:
                    removed = samples.pop()
                    _saveRaw(raw_path, test_config, samples)
                    current_pose_commanded = False
                    print(f"Erased ray #{removed['index']}. Press SPACE to repeat that scheduled pose.")

            elif key == ord("o"):
                if _analyze(samples, test_config, summary_path) is not None:
                    _saveRaw(raw_path, test_config, samples)

    finally:
        if samples:
            _saveRaw(raw_path, test_config, samples)

        stop_event.set()
        if cmd_thread.is_alive():
            cmd_thread.join(timeout=1.0)

        try:
            link.close()
        except Exception as e:
            print(f"Failed to close UDP link: {e}")

        camera.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()