import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera_calibration import CameraCalibration
from src.primary.camera_to_platform_calibration import (
    CameraToPlatformCalibration,
    loadCameraToPlatformSamples,
    saveCameraToPlatformSamples,
    servoAnglesToLaserRay,
    servoAnglesToPlatformYawElevation,
    solveCameraToPlatformCalibration,
)
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.object_vision_spec import ObjectVisionSpecId
from src.primary.platform_geometry_spec import PlatformGeometrySpecId, PLATFORM_GEOMETRY_SPECS
from src.comm.link import UdpLink
from src.comm.protocol import CMD_PLATFORM_CONTROL, next_msg_id
from src.comm.network_config import PRIMARY_IP, ENDPOINT_IP, UDP_PORT, PRIMARY_NODE_ID, ENDPOINT_NODE_ID, DEFAULT_MAX_PACKET_BYTES


CALIBRATION_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.TENNIS_BALL_DEFAULT
CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID = PlatformGeometrySpecId.PLATFORM_1

PRIMARY_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_DATA_DIR = PRIMARY_DIR/"calibration_data"/"camera_to_platform"
SAMPLES_PATH = CALIBRATION_DATA_DIR/"samples.json"
RESULTS_PATH = CALIBRATION_DATA_DIR/"results.json"

WINDOW_NAME = "Camera to platform calibration"

POSITION_AVERAGING_WINDOW = 10
COARSE_SERVO_STEP_DEG = 5.0
FINE_SERVO_STEP_DEG = 0.5
CMD_FREQUENCY_HZ = 30.0


def cmd_thread_calibrate_camera_to_platform(servo_angles: np.ndarray, servo_angles_lock: threading.Lock, stop_event: threading.Event, link: UdpLink, cmd_frequency_hz: float = CMD_FREQUENCY_HZ) -> None:
    cmd_period = 1.0/cmd_frequency_hz
    last_msg_id = None

    try:
        while not stop_event.is_set():
            iter_start = time.perf_counter()

            with servo_angles_lock:
                q = servo_angles.copy()

            msg_id = next_msg_id(last_msg_id)
            link.send_cmd(
                msg_id=msg_id,
                sender_time=iter_start,
                cmd_name=CMD_PLATFORM_CONTROL,
                cmd_payload={
                    "pan_deg": float(q[config.SERVO_IDX["pan"]]),
                    "tilt_deg": float(q[config.SERVO_IDX["tilt"]]),
                    "triggering_halted": True,
                    "trigger": False,
                },
            )
            last_msg_id = msg_id

            # Drain endpoint traffic so telemetry/errors do not accumulate.
            link.recv_telemetry_available()
            for error_msg in link.recv_errors_available():
                print(f"Endpoint error: {error_msg}")

            stop_event.wait(max(0.0, cmd_period - (time.perf_counter() - iter_start)))

    finally:
        # Finish at the normal default pose with triggering halted.
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


def _drawOverlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 22

    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 20


def main() -> None:
    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)
    platform_geometry_spec = PLATFORM_GEOMETRY_SPECS[CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID]
    samples = loadCameraToPlatformSamples(SAMPLES_PATH)

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

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (actual_width, actual_height) != (config.FRAME_W, config.FRAME_H):
        camera.release()
        raise ValueError(
            f"Camera produced {actual_width}x{actual_height}, "
            f"but config expects {config.FRAME_W}x{config.FRAME_H}"
        )

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

    cmd_thread = threading.Thread(
        target=cmd_thread_calibrate_camera_to_platform,
        args=(servo_angles, servo_angles_lock, stop_event, link),
        daemon=True,
    )
    cmd_thread.start()

    measurement_buffer = deque(maxlen=POSITION_AVERAGING_WINDOW)
    latched_position_camera_m = None
    latched_position_rms_m = None

    candidate_calibration: CameraToPlatformCalibration | None = None
    candidate_diagnostics = None

    video_frozen = False
    fine_step = False
    frame_convention_confirmed = False
    last_vision_frame = None

    print("Camera-to-platform calibration")
    print(f"Platform geometry used by laser model: {CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID.name}")
    print(f"Samples: {SAMPLES_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print("Stored camera samples: +x RIGHT, +y DOWN, +z FORWARD (raw OpenCV).")
    print("Platform convention (FLU): +x FORWARD, +y LEFT, +z UP.")
    print("Positive yaw = LEFT / ANTICLOCKWISE viewed from above.")
    print("A = LEFT / ANTICLOCKWISE, D = RIGHT / CLOCKWISE, W = UP, S = DOWN.")
    print("Latch a target with T, physically verify these controls, then press C to confirm the convention.")
    print(f"Loaded {len(samples)} existing samples.")

    try:
        while camera.isOpened():
            mean_position_camera_m = None
            position_rms_m = None

            if not video_frozen:
                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                object_detected, detection, measurement = detectSingleObject(
                    frame,
                    CALIBRATION_OBJECT_VISION_SPEC_ID,
                    camera_calibration,
                )

                if object_detected:
                    # Store raw OpenCV camera coordinates: +x right, +y down, +z forward.
                    position_camera_m = np.array([
                        measurement.x,
                        measurement.y,
                        measurement.z,
                    ], dtype=float)

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
                    position_rms_m = float(
                        np.sqrt(np.mean(np.sum((positions - mean_position_camera_m)**2, axis=1)))
                    )

                last_vision_frame = frame.copy()

            else:
                if last_vision_frame is None:
                    continue

                frame = last_vision_frame.copy()

                if measurement_buffer:
                    positions = np.asarray(measurement_buffer, dtype=float)
                    mean_position_camera_m = np.mean(positions, axis=0)
                    position_rms_m = float(
                        np.sqrt(np.mean(np.sum((positions - mean_position_camera_m)**2, axis=1)))
                    )

            with servo_angles_lock:
                q = servo_angles.copy()

            pan_deg = float(q[config.SERVO_IDX["pan"]])
            tilt_deg = float(q[config.SERVO_IDX["tilt"]])

            yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)
            laser_origin_platform, laser_direction_platform = servoAnglesToLaserRay(
                pan_deg,
                tilt_deg,
                platform_geometry_spec,
            )

            lines = [
                (
                    f"Target: {CALIBRATION_OBJECT_VISION_SPEC_ID.name} | "
                    f"geometry: {CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID.name} | samples: {len(samples)}",
                    (255, 255, 255),
                ),
                (
                    f"Pan {pan_deg:.2f} deg | tilt {tilt_deg:.2f} deg | "
                    f"step {'FINE' if fine_step else 'COARSE'} "
                    f"({FINE_SERVO_STEP_DEG if fine_step else COARSE_SERVO_STEP_DEG:.2f} deg)",
                    (255, 255, 255),
                ),
                (
                    f"Video: {'FROZEN' if video_frozen else 'LIVE'} | "
                    f"target: {'LATCHED' if latched_position_camera_m is not None else 'UNLATCHED'}",
                    (0, 255, 255),
                ),
            ]

            if mean_position_camera_m is not None:
                lines.append((
                    f"avg {len(measurement_buffer)}/{POSITION_AVERAGING_WINDOW}: "
                    f"[{mean_position_camera_m[0]:+.3f}, {mean_position_camera_m[1]:+.3f}, "
                    f"{mean_position_camera_m[2]:+.3f}] m | RMS {position_rms_m:.4f} m",
                    (0, 255, 0),
                ))
            else:
                lines.append((
                    f"avg 0/{POSITION_AVERAGING_WINDOW}: no stable target measurement",
                    (0, 0, 255),
                ))

            if latched_position_camera_m is not None:
                lines.append((
                    f"latched: [{latched_position_camera_m[0]:+.3f}, "
                    f"{latched_position_camera_m[1]:+.3f}, "
                    f"{latched_position_camera_m[2]:+.3f}] m | RMS {latched_position_rms_m:.4f} m",
                    (255, 255, 0),
                ))

            if not frame_convention_confirmed:
                lines += [
                    ("FRAME CHECK - verify physically before recording:", (0, 165, 255)),
                    ("A LEFT / ANTICLOCKWISE above | D RIGHT / CLOCKWISE above | W UP | S DOWN", (0, 165, 255)),
                    (
                        f"FLU +x FORWARD, +y LEFT, +z UP | yaw {yaw_deg:+.2f} deg | "
                        f"elev {elevation_deg:+.2f} deg",
                        (0, 165, 255),
                    ),
                    (
                        f"laser origin [{laser_origin_platform[0]:+.3f}, "
                        f"{laser_origin_platform[1]:+.3f}, {laser_origin_platform[2]:+.3f}] | "
                        f"ray [{laser_direction_platform[0]:+.3f}, "
                        f"{laser_direction_platform[1]:+.3f}, {laser_direction_platform[2]:+.3f}]",
                        (0, 165, 255),
                    ),
                    ("C = confirm convention", (0, 165, 255)),
                ]
            else:
                lines.append(("Frame convention: CONFIRMED | C = re-check", (0, 255, 0)))

            if candidate_diagnostics is not None:
                lines.append((
                    f"candidate: RMS {candidate_diagnostics['fit_rms_ray_error_m']:.4f} m | "
                    f"max {candidate_diagnostics['fit_max_ray_error_m']:.4f} m | V = save result",
                    (255, 0, 255),
                ))

            lines.append((
                "T latch | P freeze | WASD aim | F fine/coarse | G exact | "
                "R record | X undo | O solve | V save | Q/Esc quit",
                (255, 255, 255),
            ))

            _drawOverlay(frame, lines)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Quitting...")
                break

            elif key == ord("p"):
                video_frozen = not video_frozen

                if not video_frozen and latched_position_camera_m is None:
                    measurement_buffer.clear()

                print(f"Video {'frozen' if video_frozen else 'live'}.")

            elif key == ord("t"):
                if latched_position_camera_m is None:
                    if len(measurement_buffer) < POSITION_AVERAGING_WINDOW or mean_position_camera_m is None:
                        print(f"Need {POSITION_AVERAGING_WINDOW} consecutive valid measurements before latching.")
                    else:
                        latched_position_camera_m = mean_position_camera_m.copy()
                        latched_position_rms_m = position_rms_m
                        print(
                            f"Latched target at {latched_position_camera_m}, "
                            f"RMS={latched_position_rms_m:.4f} m"
                        )

                else:
                    latched_position_camera_m = None
                    latched_position_rms_m = None
                    measurement_buffer.clear()
                    print("Target unlatched.")

            elif key == ord("f"):
                fine_step = not fine_step
                print(f"Servo step: {'fine' if fine_step else 'coarse'}.")

            elif key == ord("c"):
                frame_convention_confirmed = not frame_convention_confirmed
                print(
                    f"Frame convention "
                    f"{'CONFIRMED' if frame_convention_confirmed else 'set back to UNCONFIRMED/check mode'}."
                )

            elif key in (ord("w"), ord("a"), ord("s"), ord("d")):
                if latched_position_camera_m is None:
                    print("Latch the target with T before moving the platform.")
                    continue

                step = FINE_SERVO_STEP_DEG if fine_step else COARSE_SERVO_STEP_DEG
                pan_idx = config.SERVO_IDX["pan"]
                tilt_idx = config.SERVO_IDX["tilt"]
                pan_sign = config.SERVO_SIGNS[pan_idx]
                tilt_sign = config.SERVO_SIGNS[tilt_idx]

                with servo_angles_lock:
                    if key == ord("a"):      # physical LEFT / positive yaw / anticlockwise viewed from above
                        servo_angles[pan_idx] += pan_sign*step
                    elif key == ord("d"):    # physical RIGHT / negative yaw / clockwise viewed from above
                        servo_angles[pan_idx] -= pan_sign*step
                    elif key == ord("w"):    # physical UP
                        servo_angles[tilt_idx] += tilt_sign*step
                    elif key == ord("s"):    # physical DOWN
                        servo_angles[tilt_idx] -= tilt_sign*step

                    servo_angles[:] = np.clip(
                        servo_angles,
                        config.MIN_SERVO_ANGLES,
                        config.MAX_SERVO_ANGLES,
                    )

            elif key == ord("g"):
                if latched_position_camera_m is None:
                    print("Latch the target with T before commanding exact angles.")
                    continue

                try:
                    text = input("Enter pan tilt degrees (example: 90 75): ").strip().replace(",", " ")
                    values = text.split()

                    if len(values) != 2:
                        raise ValueError

                    pan_deg_input, tilt_deg_input = map(float, values)
                    q_input = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
                    q_input[config.SERVO_IDX["pan"]] = pan_deg_input
                    q_input[config.SERVO_IDX["tilt"]] = tilt_deg_input

                    if np.any(q_input < config.MIN_SERVO_ANGLES) or np.any(q_input > config.MAX_SERVO_ANGLES):
                        print(
                            f"Rejected: angles must be within "
                            f"{config.MIN_SERVO_ANGLES} to {config.MAX_SERVO_ANGLES}."
                        )
                    else:
                        with servo_angles_lock:
                            servo_angles[:] = q_input

                except ValueError:
                    print("Invalid input. Enter exactly two numbers: pan tilt")

            elif key == ord("r"):
                if not frame_convention_confirmed:
                    print("Confirm the physical frame/control convention with C before recording samples.")
                    continue

                if latched_position_camera_m is None:
                    print("Latch the target with T before recording.")
                    continue

                with servo_angles_lock:
                    q_record = servo_angles.copy()

                sample = {
                    # position_camera_m is raw OpenCV camera coordinates.
                    "position_camera_m": latched_position_camera_m.tolist(),
                    "pan_deg": float(q_record[config.SERVO_IDX["pan"]]),
                    "tilt_deg": float(q_record[config.SERVO_IDX["tilt"]]),
                    "position_rms_m": float(latched_position_rms_m),
                    "num_measurements": POSITION_AVERAGING_WINDOW,
                }

                samples.append(sample)
                saveCameraToPlatformSamples(samples, SAMPLES_PATH)

                candidate_calibration = None
                candidate_diagnostics = None

                print(f"Recorded sample {len(samples)}: {sample}")

                # Force a fresh target measurement before the next sample.
                latched_position_camera_m = None
                latched_position_rms_m = None
                measurement_buffer.clear()

            elif key == ord("x"):
                if samples:
                    removed = samples.pop()
                    saveCameraToPlatformSamples(samples, SAMPLES_PATH)

                    candidate_calibration = None
                    candidate_diagnostics = None

                    print(f"Removed sample {len(samples) + 1}: {removed}")

                else:
                    print("No samples to remove.")

            elif key == ord("o"):
                try:
                    samples = loadCameraToPlatformSamples(SAMPLES_PATH)

                    candidate_calibration, candidate_diagnostics = solveCameraToPlatformCalibration(
                        samples,
                        platform_geometry_spec,
                    )

                    print("\nCandidate camera-to-platform calibration:")
                    print("R_platform_from_camera =")
                    print(candidate_calibration.rotation_platform_from_camera)
                    print("t_platform_from_camera_m =")
                    print(candidate_calibration.translation_platform_from_camera_m)
                    print(f"det(R):         {np.linalg.det(candidate_calibration.rotation_platform_from_camera):.6f}")
                    print(f"RMS ray error:  {candidate_diagnostics['fit_rms_ray_error_m']:.6f} m")
                    print(f"Mean ray error: {candidate_diagnostics['fit_mean_ray_error_m']:.6f} m")
                    print(f"Max ray error:  {candidate_diagnostics['fit_max_ray_error_m']:.6f} m")
                    print("Press V to save this candidate.")

                except Exception as e:
                    candidate_calibration = None
                    candidate_diagnostics = None
                    print(f"Calibration solve failed: {e}")

            elif key == ord("v"):
                if candidate_calibration is None or candidate_diagnostics is None:
                    print("No candidate calibration. Press O to solve first.")

                else:
                    candidate_calibration.save(
                        RESULTS_PATH,
                        diagnostics=candidate_diagnostics,
                    )
                    print(f"Saved calibration result to {RESULTS_PATH}")

    finally:
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