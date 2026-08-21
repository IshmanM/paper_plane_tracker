from pathlib import Path
import json

import cv2
import numpy as np

import src.primary.config as config
from src.primary.geometry import rotationPlatformFromPanTilt
from src.primary.platform_geometry_spec import PlatformGeometrySpec


MIN_CALIBRATION_SAMPLES = 6
MAX_RAY_FIT_ITERATIONS = 100

# Final calibration objective is angular rather than meter distance-to-ray.
# Huber weighting prevents a poorly centered sample from dominating the fit.
ANGULAR_FIT_HUBER_DELTA_DEG = 0.35
MAX_ANGULAR_FIT_ITERATIONS = 50
ANGULAR_FIT_ROTATION_FINITE_DIFF_RAD = 1e-5
ANGULAR_FIT_TRANSLATION_FINITE_DIFF_M = 1e-5
MAX_ANGULAR_FIT_ROTATION_STEP_DEG = 2.0
MAX_ANGULAR_FIT_TRANSLATION_STEP_M = 0.05

# Jointly estimate a small fixed laser-axis angular offset relative to foam +x.
# These angles are used only while solving and are printed, not saved.
LASER_ANGLE_INITIAL_YAW_DEG = 0.0
LASER_ANGLE_INITIAL_ELEVATION_DEG = 0.0
MAX_ABS_LASER_ANGLE_DEG = 10.0
LASER_ANGLE_FINITE_DIFF_RAD = 1e-5
MAX_LASER_ANGLE_STEP_DEG = 1.0

# Platform FLU -> OpenCV-like axes for the PnP initializer only:
# [forward, left, up] -> [right, down, forward].
OPENCV_LIKE_FROM_PLATFORM = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)


class CameraToPlatformCalibration:
    """
    Raw OpenCV camera coordinates -> platform FLU coordinates.

    Camera:   +x right, +y down, +z forward
    Platform: +x forward, +y left, +z up

    p_platform = R_platform_from_camera @ p_camera + t_platform_from_camera
    """

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.rotation_platform_from_camera = np.asarray(data["rotation_platform_from_camera"], dtype=np.float64)
        self.translation_platform_from_camera_m = np.asarray(data["translation_platform_from_camera_m"], dtype=np.float64)
        self._validate()

    @classmethod
    def fromValues(cls, rotation_platform_from_camera: np.ndarray, translation_platform_from_camera_m: np.ndarray) -> "CameraToPlatformCalibration":
        calibration = cls.__new__(cls)
        calibration.json_path = None
        calibration.rotation_platform_from_camera = np.asarray(rotation_platform_from_camera, dtype=np.float64)
        calibration.translation_platform_from_camera_m = np.asarray(translation_platform_from_camera_m, dtype=np.float64)
        calibration._validate()
        return calibration

    def _validate(self) -> None:
        R = self.rotation_platform_from_camera
        t = self.translation_platform_from_camera_m

        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            raise ValueError("rotation_platform_from_camera must be a finite 3x3 matrix")
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError("translation_platform_from_camera_m must be a finite length-3 vector")
        if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            raise ValueError("rotation_platform_from_camera must be a proper rotation matrix")

    def transformPosition(self, position_camera_m: np.ndarray) -> np.ndarray:
        """Transform a raw OpenCV camera-frame position into the platform FLU frame."""
        position_camera_m = np.asarray(position_camera_m, dtype=np.float64)

        if position_camera_m.shape != (3,) or not np.all(np.isfinite(position_camera_m)):
            raise ValueError("position_camera_m must be a finite length-3 vector")

        return self.rotation_platform_from_camera@position_camera_m + self.translation_platform_from_camera_m

    def save(self, json_path: str | Path | None = None, diagnostics: dict | None = None) -> Path:
        output_path = Path(json_path) if json_path is not None else self.json_path

        if output_path is None:
            raise ValueError("json_path is required when saving a calibration created with fromValues()")

        data = {
            "rotation_platform_from_camera": self.rotation_platform_from_camera.tolist(),
            "translation_platform_from_camera_m": self.translation_platform_from_camera_m.tolist(),
        }

        if diagnostics is not None:
            data.update(diagnostics)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        self.json_path = output_path
        return output_path


def servoAnglesToPlatformYawElevation(pan_deg: float, tilt_deg: float) -> tuple[float, float]:
    pan_idx = config.SERVO_IDX["pan"]
    tilt_idx = config.SERVO_IDX["tilt"]
    pan_sign = float(config.SERVO_SIGNS[pan_idx])
    tilt_sign = float(config.SERVO_SIGNS[tilt_idx])

    if abs(pan_sign) < 1e-12 or abs(tilt_sign) < 1e-12:
        raise ValueError("SERVO_SIGNS for pan and tilt must be nonzero")

    yaw_deg = (float(pan_deg) - float(config.FORWARD_SERVO_ANGLES[pan_idx]))/pan_sign
    elevation_deg = (float(tilt_deg) - float(config.FORWARD_SERVO_ANGLES[tilt_idx]))/tilt_sign
    return yaw_deg, elevation_deg


def servoAnglesToLaserRay(pan_deg: float, tilt_deg: float, platform_geometry_spec: PlatformGeometrySpec) -> tuple[np.ndarray, np.ndarray]:
    """
    Return laser origin and direction in the platform frame.
    Laser direction is assumed aligned with foam-mechanism +x (forward in FLU).
    """
    yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)
    R_joint = rotationPlatformFromPanTilt(np.deg2rad(yaw_deg), np.deg2rad(elevation_deg))

    # Current foam-mechanism pose in the platform frame.
    R_foam = R_joint@platform_geometry_spec.rotation_platform_from_foam_mechanism_at_forward
    foam_origin_platform = R_joint@platform_geometry_spec.foam_mechanism_origin_offset_m

    # Laser offset is expressed in the moving foam-mechanism frame.
    laser_origin_platform = foam_origin_platform + R_foam@platform_geometry_spec.laser_origin_offset_foam_mechanism_m
    laser_direction_platform = R_foam@np.array([1.0, 0.0, 0.0], dtype=np.float64)
    laser_direction_platform /= np.linalg.norm(laser_direction_platform)

    return laser_origin_platform, laser_direction_platform


def loadCameraToPlatformSamples(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError(f"{path} must contain an object with a 'samples' list")

    return data["samples"]


def saveCameraToPlatformSamples(samples: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": samples}, indent=4) + "\n", encoding="utf-8")
    return path


def _rayErrors(R: np.ndarray, t: np.ndarray, positions_camera_m: np.ndarray, ray_origins_platform: np.ndarray, directions_platform: np.ndarray) -> np.ndarray:
    positions_platform = (R@positions_camera_m.T).T + t
    from_ray_origin = positions_platform - ray_origins_platform
    along_ray_m = np.sum(from_ray_origin*directions_platform, axis=1)
    closest_points = ray_origins_platform + along_ray_m[:, None]*directions_platform
    return np.linalg.norm(positions_platform - closest_points, axis=1)


def _refineUsingAlternatingRayFit(
    R: np.ndarray,
    t: np.ndarray,
    positions_camera_m: np.ndarray,
    ray_origins_platform: np.ndarray,
    directions_platform: np.ndarray,
    max_iterations: int = MAX_RAY_FIT_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray]:

    R = R.copy()
    t = t.copy()

    for _ in range(max_iterations):
        positions_platform = (R@positions_camera_m.T).T + t

        # Project transformed camera points onto their corresponding laser rays.
        from_ray_origin = positions_platform - ray_origins_platform
        along_ray_m = np.sum(from_ray_origin*directions_platform, axis=1)
        target_points_platform = ray_origins_platform + along_ray_m[:, None]*directions_platform

        # Best rigid camera->platform transform to the current ray projections.
        camera_centroid = np.mean(positions_camera_m, axis=0)
        platform_centroid = np.mean(target_points_platform, axis=0)
        camera_centered = positions_camera_m - camera_centroid
        platform_centered = target_points_platform - platform_centroid

        U, _, Vt = np.linalg.svd(camera_centered.T@platform_centered)
        R_new = Vt.T@U.T

        # Camera and platform frames are both right-handed, so do not allow a reflection.
        if np.linalg.det(R_new) < 0.0:
            Vt[-1] *= -1.0
            R_new = Vt.T@U.T

        t_new = platform_centroid - R_new@camera_centroid

        if np.linalg.norm(R_new - R) < 1e-12 and np.linalg.norm(t_new - t) < 1e-12:
            R, t = R_new, t_new
            break

        R, t = R_new, t_new

    return R, t


def _laserDirectionFoam(laser_yaw_rad: float, laser_elevation_rad: float) -> np.ndarray:
    cy, sy = np.cos(laser_yaw_rad), np.sin(laser_yaw_rad)
    ce, se = np.cos(laser_elevation_rad), np.sin(laser_elevation_rad)
    return np.array([ce*cy, ce*sy, se], dtype=np.float64)


def _laserDirectionsPlatform(foam_rotations_platform: np.ndarray, laser_yaw_rad: float, laser_elevation_rad: float) -> np.ndarray:
    direction_foam = _laserDirectionFoam(laser_yaw_rad, laser_elevation_rad)
    directions = np.einsum("nij,j->ni", foam_rotations_platform, direction_foam)
    return directions/np.linalg.norm(directions, axis=1, keepdims=True)


def _angularErrors(R: np.ndarray, t: np.ndarray, positions_camera_m: np.ndarray, ray_origins_platform: np.ndarray, directions_platform: np.ndarray) -> np.ndarray:
    positions_platform = (R@positions_camera_m.T).T + t
    to_targets = positions_platform - ray_origins_platform
    distances = np.linalg.norm(to_targets, axis=1)

    if np.any(distances <= 1e-9):
        return np.full(len(positions_camera_m), np.inf, dtype=np.float64)

    target_directions = to_targets/distances[:, None]
    dots = np.sum(target_directions*directions_platform, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _angularResidualVectors(R: np.ndarray, t: np.ndarray, positions_camera_m: np.ndarray, ray_origins_platform: np.ndarray, directions_platform: np.ndarray) -> np.ndarray:
    positions_platform = (R@positions_camera_m.T).T + t
    to_targets = positions_platform - ray_origins_platform
    distances = np.linalg.norm(to_targets, axis=1)

    if np.any(distances <= 1e-9):
        return np.full((len(positions_camera_m), 3), np.nan, dtype=np.float64)

    target_directions = to_targets/distances[:, None]
    return target_directions - directions_platform


def _robustAngularCost(errors_rad: np.ndarray) -> float:
    delta = np.deg2rad(ANGULAR_FIT_HUBER_DELTA_DEG)
    quadratic = np.minimum(errors_rad, delta)
    linear = errors_rad - quadratic
    return float(np.sum(0.5*quadratic*quadratic + delta*linear))


def _refineUsingRobustJointAngularFit(
    R: np.ndarray,
    t: np.ndarray,
    positions_camera_m: np.ndarray,
    ray_origins_platform: np.ndarray,
    foam_rotations_platform: np.ndarray,
    laser_yaw_rad: float = np.deg2rad(LASER_ANGLE_INITIAL_YAW_DEG),
    laser_elevation_rad: float = np.deg2rad(LASER_ANGLE_INITIAL_ELEVATION_DEG),
    max_iterations: int = MAX_ANGULAR_FIT_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, float, float]:

    R = R.copy()
    t = t.copy()
    max_rotation_step_rad = np.deg2rad(MAX_ANGULAR_FIT_ROTATION_STEP_DEG)
    max_laser_step_rad = np.deg2rad(MAX_LASER_ANGLE_STEP_DEG)
    max_abs_laser_angle_rad = np.deg2rad(MAX_ABS_LASER_ANGLE_DEG)
    delta_huber = np.deg2rad(ANGULAR_FIT_HUBER_DELTA_DEG)

    for _ in range(max_iterations):
        directions_platform = _laserDirectionsPlatform(foam_rotations_platform, laser_yaw_rad, laser_elevation_rad)
        errors_rad = _angularErrors(R, t, positions_camera_m, ray_origins_platform, directions_platform)
        residual_vectors = _angularResidualVectors(R, t, positions_camera_m, ray_origins_platform, directions_platform)

        if not np.all(np.isfinite(errors_rad)) or not np.all(np.isfinite(residual_vectors)):
            break

        weights = np.ones_like(errors_rad)
        large = errors_rad > delta_huber
        weights[large] = delta_huber/np.maximum(errors_rad[large], 1e-12)
        sqrt_weights = np.sqrt(weights)[:, None]
        residual = (sqrt_weights*residual_vectors).reshape(-1)

        # Parameters: camera rotation xyz, camera translation xyz,
        # laser yaw offset, laser elevation offset.
        J = np.zeros((residual.size, 8), dtype=np.float64)

        for axis in range(3):
            rotation_step = np.zeros(3, dtype=np.float64)
            rotation_step[axis] = ANGULAR_FIT_ROTATION_FINITE_DIFF_RAD
            dR, _ = cv2.Rodrigues(rotation_step)
            residual_step = _angularResidualVectors(dR@R, t, positions_camera_m, ray_origins_platform, directions_platform)
            J[:, axis] = ((sqrt_weights*(residual_step - residual_vectors))/ANGULAR_FIT_ROTATION_FINITE_DIFF_RAD).reshape(-1)

        for axis in range(3):
            t_step = t.copy()
            t_step[axis] += ANGULAR_FIT_TRANSLATION_FINITE_DIFF_M
            residual_step = _angularResidualVectors(R, t_step, positions_camera_m, ray_origins_platform, directions_platform)
            J[:, 3 + axis] = ((sqrt_weights*(residual_step - residual_vectors))/ANGULAR_FIT_TRANSLATION_FINITE_DIFF_M).reshape(-1)

        yaw_step_directions = _laserDirectionsPlatform(
            foam_rotations_platform, laser_yaw_rad + LASER_ANGLE_FINITE_DIFF_RAD, laser_elevation_rad
        )
        yaw_step_residual = _angularResidualVectors(R, t, positions_camera_m, ray_origins_platform, yaw_step_directions)
        J[:, 6] = ((sqrt_weights*(yaw_step_residual - residual_vectors))/LASER_ANGLE_FINITE_DIFF_RAD).reshape(-1)

        elevation_step_directions = _laserDirectionsPlatform(
            foam_rotations_platform, laser_yaw_rad, laser_elevation_rad + LASER_ANGLE_FINITE_DIFF_RAD
        )
        elevation_step_residual = _angularResidualVectors(R, t, positions_camera_m, ray_origins_platform, elevation_step_directions)
        J[:, 7] = ((sqrt_weights*(elevation_step_residual - residual_vectors))/LASER_ANGLE_FINITE_DIFF_RAD).reshape(-1)

        delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        rotation_delta, translation_delta, laser_delta = delta[:3], delta[3:6], delta[6:8]

        rotation_norm = float(np.linalg.norm(rotation_delta))
        if rotation_norm > max_rotation_step_rad:
            rotation_delta *= max_rotation_step_rad/rotation_norm

        translation_norm = float(np.linalg.norm(translation_delta))
        if translation_norm > MAX_ANGULAR_FIT_TRANSLATION_STEP_M:
            translation_delta *= MAX_ANGULAR_FIT_TRANSLATION_STEP_M/translation_norm

        laser_norm = float(np.linalg.norm(laser_delta))
        if laser_norm > max_laser_step_rad:
            laser_delta *= max_laser_step_rad/laser_norm

        old_cost = _robustAngularCost(errors_rad)
        accepted = False

        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            dR, _ = cv2.Rodrigues(scale*rotation_delta)
            R_candidate = dR@R
            t_candidate = t + scale*translation_delta
            yaw_candidate = float(np.clip(laser_yaw_rad + scale*laser_delta[0], -max_abs_laser_angle_rad, max_abs_laser_angle_rad))
            elevation_candidate = float(np.clip(laser_elevation_rad + scale*laser_delta[1], -max_abs_laser_angle_rad, max_abs_laser_angle_rad))
            candidate_directions = _laserDirectionsPlatform(foam_rotations_platform, yaw_candidate, elevation_candidate)
            candidate_errors = _angularErrors(R_candidate, t_candidate, positions_camera_m, ray_origins_platform, candidate_directions)

            if np.all(np.isfinite(candidate_errors)) and _robustAngularCost(candidate_errors) < old_cost:
                R, t = R_candidate, t_candidate
                laser_yaw_rad, laser_elevation_rad = yaw_candidate, elevation_candidate
                accepted = True
                break

        if not accepted:
            break

        if (
            np.linalg.norm(scale*rotation_delta) < 1e-9
            and np.linalg.norm(scale*translation_delta) < 1e-8
            and np.linalg.norm(scale*laser_delta) < 1e-9
        ):
            break

    return R, t, laser_yaw_rad, laser_elevation_rad


def solveCameraToPlatformCalibration(samples: list[dict], platform_geometry_spec: PlatformGeometrySpec) -> tuple[CameraToPlatformCalibration, dict]:
    if len(samples) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(f"Need at least {MIN_CALIBRATION_SAMPLES} calibration samples, got {len(samples)}")

    positions_camera_m = []
    ray_origins_platform = []
    foam_rotations_platform = []

    for sample_index, sample in enumerate(samples):
        try:
            # Samples are stored directly in raw OpenCV camera coordinates.
            position_camera_m = np.asarray(sample["position_camera_m"], dtype=np.float64)
            pan_deg = float(sample["pan_deg"])
            tilt_deg = float(sample["tilt_deg"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid calibration sample {sample_index}: {e}") from e

        if position_camera_m.shape != (3,) or not np.all(np.isfinite(position_camera_m)):
            raise ValueError(f"Sample {sample_index} position_camera_m must be a finite length-3 vector")

        yaw_deg, elevation_deg = servoAnglesToPlatformYawElevation(pan_deg, tilt_deg)
        R_joint = rotationPlatformFromPanTilt(np.deg2rad(yaw_deg), np.deg2rad(elevation_deg))
        R_foam = R_joint@platform_geometry_spec.rotation_platform_from_foam_mechanism_at_forward
        foam_origin_platform = R_joint@platform_geometry_spec.foam_mechanism_origin_offset_m
        ray_origin_platform = foam_origin_platform + R_foam@platform_geometry_spec.laser_origin_offset_foam_mechanism_m

        positions_camera_m.append(position_camera_m)
        ray_origins_platform.append(ray_origin_platform)
        foam_rotations_platform.append(R_foam)

    positions_camera_m = np.asarray(positions_camera_m, dtype=np.float64)
    ray_origins_platform = np.asarray(ray_origins_platform, dtype=np.float64)
    foam_rotations_platform = np.asarray(foam_rotations_platform, dtype=np.float64)
    directions_platform = _laserDirectionsPlatform(
        foam_rotations_platform,
        np.deg2rad(LASER_ANGLE_INITIAL_YAW_DEG),
        np.deg2rad(LASER_ANGLE_INITIAL_ELEVATION_DEG),
    )

    if np.linalg.matrix_rank(positions_camera_m - np.mean(positions_camera_m, axis=0)) < 2:
        raise ValueError("Calibration target positions are degenerate; collect samples spread across different directions/positions")

    if np.any(directions_platform[:, 0] <= 1e-6):
        raise ValueError("Calibration currently requires all laser rays to point into +x/forward; avoid exactly sideways/backward samples")

    initial_solutions = []

    # PnP uses a virtual OpenCV-like frame whose optical axis is forward. Convert
    # platform FLU ray directions into that frame for initialization only.
    directions_pnp = (OPENCV_LIKE_FROM_PLATFORM@directions_platform.T).T
    image_points = directions_pnp[:, :2]/directions_pnp[:, 2, None]

    solution_count, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
        positions_camera_m,
        image_points,
        np.eye(3, dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if solution_count:
        for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
            R_pnp, _ = cv2.Rodrigues(rotation_vector)
            t_pnp = translation_vector.reshape(3)

            # p_pnp = M @ p_platform, so convert the PnP transform back to FLU.
            initial_solutions.append((
                OPENCV_LIKE_FROM_PLATFORM.T@R_pnp,
                OPENCV_LIKE_FROM_PLATFORM.T@t_pnp,
            ))

    # Neutral physical-alignment fallback: camera and platform face the same way,
    # but their coordinate axes differ (OpenCV vs FLU).
    initial_solutions.append((OPENCV_LIKE_FROM_PLATFORM.T.copy(), np.zeros(3, dtype=np.float64)))

    best_R = None
    best_t = None
    best_laser_yaw_rad = None
    best_laser_elevation_rad = None
    best_angular_cost = float("inf")

    for R_initial, t_initial in initial_solutions:
        # Zero laser-angle error is only the initializer. The final fit jointly
        # estimates camera extrinsics plus fixed laser yaw/elevation relative to foam +x.
        R, t = _refineUsingAlternatingRayFit(
            R_initial,
            t_initial,
            positions_camera_m,
            ray_origins_platform,
            directions_platform,
        )
        R, t, laser_yaw_rad, laser_elevation_rad = _refineUsingRobustJointAngularFit(
            R,
            t,
            positions_camera_m,
            ray_origins_platform,
            foam_rotations_platform,
        )

        fitted_directions_platform = _laserDirectionsPlatform(
            foam_rotations_platform, laser_yaw_rad, laser_elevation_rad
        )
        positions_platform = (R@positions_camera_m.T).T + t
        along_ray_m = np.sum((positions_platform - ray_origins_platform)*fitted_directions_platform, axis=1)

        if np.any(along_ray_m <= 0.0):
            continue

        angular_errors_rad = _angularErrors(
            R, t, positions_camera_m, ray_origins_platform, fitted_directions_platform
        )
        angular_cost = _robustAngularCost(angular_errors_rad)

        if angular_cost < best_angular_cost:
            best_R = R
            best_t = t
            best_laser_yaw_rad = laser_yaw_rad
            best_laser_elevation_rad = laser_elevation_rad
            best_angular_cost = angular_cost

    if best_R is None:
        raise RuntimeError("Calibration solutions placed one or more targets behind their laser origins")

    fitted_directions_platform = _laserDirectionsPlatform(
        foam_rotations_platform, best_laser_yaw_rad, best_laser_elevation_rad
    )
    errors_m = _rayErrors(best_R, best_t, positions_camera_m, ray_origins_platform, fitted_directions_platform)
    angular_errors_deg = np.rad2deg(
        _angularErrors(best_R, best_t, positions_camera_m, ray_origins_platform, fitted_directions_platform)
    )
    calibration = CameraToPlatformCalibration.fromValues(best_R, best_t)

    # Transient solve-only laser-angle estimate. CameraToPlatformCalibration.save()
    # ignores these attributes, so results.json remains unchanged.
    calibration.laser_yaw_offset_rad = float(best_laser_yaw_rad)
    calibration.laser_elevation_offset_rad = float(best_laser_elevation_rad)

    laser_direction_foam = _laserDirectionFoam(best_laser_yaw_rad, best_laser_elevation_rad)
    print("\nEstimated laser angular offset relative to foam axis (NOT SAVED):")
    print(f"laser yaw offset:       {np.rad2deg(best_laser_yaw_rad):+.6f} deg  (+ = left)")
    print(f"laser elevation offset: {np.rad2deg(best_laser_elevation_rad):+.6f} deg  (+ = up)")
    print(f"laser direction foam:   [{laser_direction_foam[0]:+.8f}, {laser_direction_foam[1]:+.8f}, {laser_direction_foam[2]:+.8f}]")
    print(f"RMS angular residual:   {np.sqrt(np.mean(angular_errors_deg**2)):.6f} deg")
    print(f"Mean angular residual:  {np.mean(angular_errors_deg):.6f} deg")
    print(f"Max angular residual:   {np.max(angular_errors_deg):.6f} deg")

    # Keep results.json exactly compatible. Laser-angle values are intentionally
    # omitted from diagnostics, so CameraToPlatformCalibration.save() cannot save them.
    diagnostics = {
        "num_samples": len(samples),
        "fit_rms_ray_error_m": float(np.sqrt(np.mean(errors_m**2))),
        "fit_mean_ray_error_m": float(np.mean(errors_m)),
        "fit_max_ray_error_m": float(np.max(errors_m)),
    }

    return calibration, diagnostics