import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import src.primary.config as config
from src.primary.camera.camera_calibration import CameraCalibration
from src.primary.camera_to_platform_calibration import (
    CameraToPlatformCalibration,
    loadCameraToPlatformSamples,
    saveCameraToPlatformSamples,
    servoAnglesToLaserRay,
    servoAnglesToPlatformYawElevation,
    solveCameraToPlatformCalibration,
)
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.geometry import rotationPlatformFromPanTilt
from src.primary.object_vision_spec import ObjectVisionSpecId
from src.primary.platform_geometry_spec import PlatformGeometrySpecId, PLATFORM_GEOMETRY_SPECS
from src.comm.link import UdpLink
from src.comm.protocol import CMD_PLATFORM_CONTROL, next_msg_id
from src.comm.network_config import PRIMARY_IP, ENDPOINT_IP, UDP_PORT, PRIMARY_NODE_ID, ENDPOINT_NODE_ID, DEFAULT_MAX_PACKET_BYTES

# CALIBRATION_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.TENNIS_BALL_DEFAULT
CALIBRATION_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.ARUCO_MARKER_1
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

DISPLAY_SCALES = (1.0, 1.5, 2.0)

TEST_AIM_FINITE_DIFF_DEG = 0.05
TEST_AIM_MAX_STEP_DEG = 3.0
TEST_AIM_MAX_ITERATIONS = 20
TEST_AIM_TOLERANCE_DEG = 0.02


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


def _promptYesNo(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter yes or no.")


def _servoAnglesToFoamRay(pan_deg: float, tilt_deg: float, platform_geometry_spec) -> tuple[np.ndarray, np.ndarray]:
    yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)
    R_joint = rotationPlatformFromPanTilt(np.deg2rad(yaw_deg), np.deg2rad(elevation_deg))
    R_foam = R_joint@platform_geometry_spec.rotation_platform_from_foam_mechanism_at_forward
    origin = R_joint@platform_geometry_spec.foam_mechanism_origin_offset_m
    direction = R_foam@np.array([1.0, 0.0, 0.0], dtype=float)
    return origin, direction/np.linalg.norm(direction)


def _servoAnglesToEstimatedLaserRay(
    pan_deg: float,
    tilt_deg: float,
    platform_geometry_spec,
    laser_yaw_offset_rad: float,
    laser_elevation_offset_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)
    R_joint = rotationPlatformFromPanTilt(np.deg2rad(yaw_deg), np.deg2rad(elevation_deg))
    R_foam = R_joint@platform_geometry_spec.rotation_platform_from_foam_mechanism_at_forward
    foam_origin_platform = R_joint@platform_geometry_spec.foam_mechanism_origin_offset_m
    laser_origin_platform = foam_origin_platform + R_foam@platform_geometry_spec.laser_origin_offset_foam_mechanism_m

    cy, sy = np.cos(laser_yaw_offset_rad), np.sin(laser_yaw_offset_rad)
    ce, se = np.cos(laser_elevation_offset_rad), np.sin(laser_elevation_offset_rad)
    laser_direction_foam = np.array([ce*cy, ce*sy, se], dtype=float)
    laser_direction_platform = R_foam@laser_direction_foam
    return laser_origin_platform, laser_direction_platform/np.linalg.norm(laser_direction_platform)


def _solveServoAnglesToPoint(
    target_platform_m: np.ndarray,
    q_initial: np.ndarray,
    platform_geometry_spec,
    aim_with_laser: bool,
    laser_yaw_offset_rad: float = 0.0,
    laser_elevation_offset_rad: float = 0.0,
) -> tuple[bool, np.ndarray, float]:
    target_platform_m = np.asarray(target_platform_m, dtype=float)
    q = np.clip(np.asarray(q_initial, dtype=float).copy(), config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)
    pan_idx, tilt_idx = config.SERVO_IDX["pan"], config.SERVO_IDX["tilt"]
    controlled_indices = (pan_idx, tilt_idx)

    def residual(q_test: np.ndarray) -> tuple[np.ndarray | None, float]:
        pan_deg, tilt_deg = float(q_test[pan_idx]), float(q_test[tilt_idx])
        origin, direction = (
            _servoAnglesToEstimatedLaserRay(
                pan_deg,
                tilt_deg,
                platform_geometry_spec,
                laser_yaw_offset_rad,
                laser_elevation_offset_rad,
            )
            if aim_with_laser else
            _servoAnglesToFoamRay(pan_deg, tilt_deg, platform_geometry_spec)
        )
        to_target = target_platform_m - origin
        distance = float(np.linalg.norm(to_target))
        if distance <= 1e-9:
            return None, float("inf")
        target_direction = to_target/distance
        dot = float(np.clip(np.dot(direction, target_direction), -1.0, 1.0))
        return target_direction - direction, float(np.rad2deg(np.arccos(dot)))

    for _ in range(TEST_AIM_MAX_ITERATIONS):
        r, error_deg = residual(q)
        if r is None:
            return False, q, error_deg
        if error_deg <= TEST_AIM_TOLERANCE_DEG:
            return True, q, error_deg

        J = np.zeros((3, 2), dtype=float)
        for column, servo_idx in enumerate(controlled_indices):
            step = TEST_AIM_FINITE_DIFF_DEG
            if q[servo_idx] + step > config.MAX_SERVO_ANGLES[servo_idx]:
                step = -step
            q_step = q.copy()
            q_step[servo_idx] += step
            r_step, _ = residual(q_step)
            if r_step is None:
                return False, q, error_deg
            J[:, column] = (r_step - r)/step

        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        delta = np.clip(delta, -TEST_AIM_MAX_STEP_DEG, TEST_AIM_MAX_STEP_DEG)
        q[pan_idx] += delta[0]
        q[tilt_idx] += delta[1]
        q[:] = np.clip(q, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)

    _, error_deg = residual(q)
    return np.isfinite(error_deg) and error_deg <= 0.10, q, error_deg


def main() -> None:
    camera_calibration = CameraCalibration(config.CAMERA_CALIBRATION_PATH, config.FRAME_W, config.FRAME_H)
    platform_geometry_spec = PLATFORM_GEOMETRY_SPECS[CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID]
    samples = loadCameraToPlatformSamples(SAMPLES_PATH)

    if samples:
        print(f"Found {len(samples)} existing calibration samples in {SAMPLES_PATH}.")
        if _promptYesNo("Erase all existing calibration samples before starting?"):
            samples.clear()
            saveCameraToPlatformSamples(samples, SAMPLES_PATH)
            print("Erased all existing calibration samples.")
        else:
            print("Keeping existing calibration samples.")

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
    saved_calibration = None

    if RESULTS_PATH.exists():
        try:
            saved_calibration = CameraToPlatformCalibration(RESULTS_PATH)
            print(f"Loaded saved calibration for test mode: {RESULTS_PATH}")
        except Exception as e:
            print(f"Could not load saved calibration for test mode: {e}")

    video_frozen = False
    fine_step = False
    test_mode = False
    test_aim_with_laser = True
    test_status = "OFF"
    last_vision_frame = None
    display_scale_index = 0

    print("Camera-to-platform calibration")
    print(f"Platform geometry used by laser model: {CALIBRATION_PLATFORM_GEOMETRY_SPEC_ID.name}")
    print(f"Samples: {SAMPLES_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print("Stored camera samples: +x RIGHT, +y DOWN, +z FORWARD (raw OpenCV).")
    print("Platform convention (FLU): +x FORWARD, +y LEFT, +z UP.")
    print("Positive yaw = LEFT / ANTICLOCKWISE viewed from above.")
    print("A = LEFT / ANTICLOCKWISE, D = RIGHT / CLOCKWISE, W = UP, S = DOWN.")
    print("Physically verify the FLU controls before recording samples.")
    print(f"Loaded {len(samples)} existing samples.")
    print("The OpenCV window is freely resizable. Tab cycles 1.0x -> 1.5x -> 2.0x presets.")
    print("WASD/G can move the platform before or after latching; T only latches the target measurement for recording.")
    print("M toggles test mode; K swaps test aiming between LASER and FOAM AXIS.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.FRAME_W, config.FRAME_H)

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

            if test_mode:
                test_calibration = candidate_calibration if candidate_calibration is not None else saved_calibration
                test_source = "candidate" if candidate_calibration is not None else "saved"

                if test_calibration is None:
                    test_status = "NO CALIBRATION"
                elif len(measurement_buffer) < POSITION_AVERAGING_WINDOW or mean_position_camera_m is None:
                    test_status = f"{test_source}: waiting for stable ArUco"
                else:
                    target_platform_m = test_calibration.transformPosition(mean_position_camera_m)
                    with servo_angles_lock:
                        q_start = servo_angles.copy()

                    laser_yaw_offset_rad = float(getattr(test_calibration, "laser_yaw_offset_rad", 0.0))
                    laser_elevation_offset_rad = float(getattr(test_calibration, "laser_elevation_offset_rad", 0.0))

                    aim_valid, q_test, aim_error_deg = _solveServoAnglesToPoint(
                        target_platform_m,
                        q_start,
                        platform_geometry_spec,
                        test_aim_with_laser,
                        laser_yaw_offset_rad,
                        laser_elevation_offset_rad,
                    )

                    if aim_valid:
                        with servo_angles_lock:
                            servo_angles[:] = q_test
                        test_status = f"{test_source}: aim error {aim_error_deg:.3f} deg"
                    else:
                        test_status = f"{test_source}: aim solve failed ({aim_error_deg:.3f} deg)"
            else:
                test_status = "OFF"

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
                (
                    f"Test: {'ON' if test_mode else 'OFF'} | aim: {'LASER' if test_aim_with_laser else 'FOAM AXIS'} | {test_status}",
                    (0, 255, 255) if test_mode else (180, 180, 180),
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

            lines += [
                (
                    f"FLU: A left | D right | W up | S down | yaw {yaw_deg:+.2f} deg | elev {elevation_deg:+.2f} deg",
                    (0, 165, 255),
                ),
                (
                    f"laser origin [{laser_origin_platform[0]:+.3f}, {laser_origin_platform[1]:+.3f}, "
                    f"{laser_origin_platform[2]:+.3f}] | ray [{laser_direction_platform[0]:+.3f}, "
                    f"{laser_direction_platform[1]:+.3f}, {laser_direction_platform[2]:+.3f}]",
                    (0, 165, 255),
                ),
            ]

            if candidate_diagnostics is not None:
                lines.append((
                    f"candidate: RMS {candidate_diagnostics['fit_rms_ray_error_m']:.4f} m | "
                    f"max {candidate_diagnostics['fit_max_ray_error_m']:.4f} m | V = save result",
                    (255, 0, 255),
                ))

            display_scale = DISPLAY_SCALES[display_scale_index]
            lines += [
                ("T latch | P freeze | WASD aim | F fine/coarse | G exact", (255, 255, 255)),
                ("R record | X erase last | E erase all | O solve | V save", (255, 255, 255)),
                (f"M test | K laser/foam | Tab resize ({display_scale:.1f}x) | Q/Esc quit", (255, 255, 255)),
            ]

            # Render to the current user-selected window size. Detection, measurement,
            # averaging, and calibration always use the original camera-resolution frame.
            try:
                _, _, display_w, display_h = cv2.getWindowImageRect(WINDOW_NAME)
            except cv2.error:
                display_w, display_h = config.FRAME_W, config.FRAME_H

            display_w = max(1, display_w)
            display_h = max(1, display_h)
            display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            # Mirror only the user-facing view. Detection, measurement, averaging,
            # and calibration continue to use the original unflipped camera frame.
            display_frame = cv2.flip(display_frame, 1)

            display_scale = min(display_w/config.FRAME_W, display_h/config.FRAME_H)
            _drawOverlay(display_frame, lines, display_scale)

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Quitting...")
                break

            elif key == 9:  # Tab
                display_scale_index = (display_scale_index + 1)%len(DISPLAY_SCALES)
                preset_scale = DISPLAY_SCALES[display_scale_index]
                cv2.resizeWindow(WINDOW_NAME, int(round(config.FRAME_W*preset_scale)), int(round(config.FRAME_H*preset_scale)))
                print(f"Display preset: {preset_scale:.1f}x")

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

            elif key == ord("m"):
                if test_mode:
                    test_mode = False
                    print("Test mode OFF.")
                elif candidate_calibration is None and saved_calibration is None:
                    print("No calibration available for test mode. Press O to solve a candidate or save/load results first.")
                else:
                    test_mode = True
                    print(f"Test mode ON: aiming {'LASER' if test_aim_with_laser else 'FOAM AXIS'} at live ArUco center.")

            elif key == ord("k"):
                test_aim_with_laser = not test_aim_with_laser
                print(f"Test aim switched to {'LASER' if test_aim_with_laser else 'FOAM AXIS'}.")

            elif key in (ord("w"), ord("a"), ord("s"), ord("d")):
                if test_mode:
                    print("WASD disabled while test mode is ON. Press M to exit test mode.")
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
                if test_mode:
                    print("G disabled while test mode is ON. Press M to exit test mode.")
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
                if test_mode:
                    print("Recording disabled while test mode is ON. Press M to exit test mode.")
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
                if not samples:
                    print("No samples to erase.")
                elif _promptYesNo(f"Are you sure you want to erase the last sample ({len(samples)})?"):
                    removed = samples.pop()
                    saveCameraToPlatformSamples(samples, SAMPLES_PATH)
                    candidate_calibration = None
                    candidate_diagnostics = None
                    print(f"Erased sample {len(samples) + 1}: {removed}")
                else:
                    print("Erase cancelled.")

            elif key == ord("e"):
                if not samples:
                    print("No samples to erase.")
                elif _promptYesNo(f"Are you sure you want to erase ALL {len(samples)} samples?"):
                    samples.clear()
                    saveCameraToPlatformSamples(samples, SAMPLES_PATH)
                    candidate_calibration = None
                    candidate_diagnostics = None
                    print("Erased all calibration samples.")
                else:
                    print("Erase cancelled.")

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
                    if hasattr(candidate_calibration, "laser_yaw_offset_rad"):
                        print(
                            f"Test-mode laser model: yaw "
                            f"{np.rad2deg(candidate_calibration.laser_yaw_offset_rad):+.3f} deg, elevation "
                            f"{np.rad2deg(candidate_calibration.laser_elevation_offset_rad):+.3f} deg"
                        )
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
                    saved_calibration = candidate_calibration
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