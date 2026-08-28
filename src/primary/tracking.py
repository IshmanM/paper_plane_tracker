import numpy as np
import cv2
from enum import Enum, auto

from src.primary.detection import Measurement
from src.primary.geometry import estimateObjectImagePosition
from src.primary.camera.camera_calibration import CameraCalibration


class TrackStatus(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DEAD = auto()


X, Y, Z, DX, DY, DZ = 0, 1, 2, 3, 4, 5

MAX_TRACK_ID = 2**32 - 1 # max uint32 number


class Track:

    def __init__(self, initial_measurement: Measurement, initial_time, min_hits: int, track_id=0,
                 sigma_x=0.1, sigma_y=0.1, sigma_z=0.2, sigma_dx=1.0, sigma_dy=1.0, sigma_dz=1.5,
                 horizontal_angular_uncertainty_deg=None, vertical_angular_uncertainty_deg=None,
                 angular_uncertainty_deg=None, range_uncertainty_m=None, innovation_mahalanobis=None):

        self.id = track_id # not useful until MOT

        # x, y, z, dx, dy, dz
        self.state = np.array([
            initial_measurement.x, initial_measurement.y, initial_measurement.z,
            0.0, 0.0, 0.0
        ], dtype=float)
        self.state_time = initial_time # time current KF state represents
        self.first_detection_time = initial_time # time first detection created this track

        # Need to tune starting covariance...
        # Note smaller initial sigma/covariance indicates higher initial position certainty
        self.covariance = np.diag([
            sigma_x**2,
            sigma_y**2,
            sigma_z**2,
            sigma_dx**2,
            sigma_dy**2,
            sigma_dz**2
        ]).astype(float)

        self.hit_streak = 1
        self.missed_streak = 0
        self.gate_rejected_streak = 0
        self.last_gate_rejected_measurement = None
        self.last_gate_rejected_time = None
        self.confirmed = False if min_hits > 1 else True

        self.last_hit_measurement = initial_measurement
        self.accepted_measurement_count = 1
        self.initial_velocity_initialized = False

        # Camera-space uncertainty/innovation diagnostics; useful to Platform later without inventing
        # a generic confidence score. The tracker fills these from the current KF covariance.
        self.horizontal_angular_uncertainty_deg = horizontal_angular_uncertainty_deg
        self.vertical_angular_uncertainty_deg = vertical_angular_uncertainty_deg
        self.angular_uncertainty_deg = angular_uncertainty_deg
        self.range_uncertainty_m = range_uncertainty_m
        self.innovation_mahalanobis = innovation_mahalanobis


    def mark_hit(self, measurement, min_hits: int):
        self.hit_streak += 1
        self.missed_streak = 0
        self.last_hit_measurement = measurement
        self.accepted_measurement_count += 1

        if self.hit_streak >= min_hits:
            self.confirmed = True


    def mark_missed(self):
        self.missed_streak += 1
        self.hit_streak = 0


    def is_dead(self, max_missed: int) -> bool:
        return self.missed_streak > max_missed


    @property
    def x(self):
        return self.state[X]

    @property
    def y(self):
        return self.state[Y]

    @property
    def z(self):
        return self.state[Z]

    @property
    def dx(self):
        return self.state[DX]

    @property
    def dy(self):
        return self.state[DY]

    @property
    def dz(self):
        return self.state[DZ]


def drawTrack(frame: np.ndarray, track, px_w: float, px_h: float, camera_calibration: CameraCalibration) -> None:
    if track.z <= 1e-6:
        return

    track_u, track_v = estimateObjectImagePosition(track.x, track.y, track.z, camera_calibration)
    if not np.isfinite(track_u) or not np.isfinite(track_v):
        return

    frame_h, frame_w = frame.shape[:2]
    if track_u < -2*frame_w or track_u > 3*frame_w or track_v < -2*frame_h or track_v > 3*frame_h:
        return # projected track is far outside the drawable image

    track_center = (int(round(track_u)), int(round(track_v)))

    cv2.rectangle(frame, (int(round(track_u - px_w / 2)), int(round(track_v - px_h / 2))),
                  (int(round(track_u + px_w / 2)), int(round(track_v + px_h / 2))), color=(0, 0, 255), thickness=2)

    cv2.circle(frame, track_center, radius=5, color=(0, 0, 255), thickness=-1)

    velocity_2d = np.array([track.dx, track.dy], dtype=float)
    velocity_norm = np.linalg.norm(velocity_2d)
    if velocity_norm > 1e-6:
        arrow_length_px = 40
        direction = velocity_2d / velocity_norm
        arrow_end = (int(round(track_u + arrow_length_px * direction[0])),
                     int(round(track_v + arrow_length_px * direction[1])))
        cv2.arrowedLine(frame, track_center, arrow_end, color=(0, 0, 255), thickness=2, tipLength=0.25)

# eventually need to make this tracker class derived from some base class,
# and use the base class in platform.py definitions


# Camera-aware gating compares bearing + range in normalized uncertainty units. This lets a stable
# track gate tightly while prediction uncertainty naturally grows after motion uncertainty or misses.
class SingleObjectTracker:
    def __init__(
        self,
        min_hits: int = 3,
        max_missed_on_confirmed: int = 5,
        max_missed_on_tentative: int = 1,
        sigma_accel: float = 3.0,
        sigma_meas_x: float = 0.05,
        sigma_meas_y: float = 0.05,
        sigma_meas_z: float = 0.20,
        tentative_gate_mahalanobis_sq_threshold: float = 16.27, # ~99.9% chi-square threshold, 3 DOF
        confirmed_gate_mahalanobis_sq_threshold: float = 11.34, # ~99% chi-square threshold, 3 DOF
        gate_measurement_angular_sigma_deg: float = 1.0,
        gate_measurement_min_range_sigma_m: float = 0.05,
        gate_measurement_range_sigma_fraction: float = 0.05,
        max_gate_angular_error_deg: float = 20.0, # hard sanity cap even if KF covariance grows very large
        tentative_min_gate_range_error_m: float = 0.30,
        tentative_gate_range_error_fraction: float = 0.25,
        confirmed_min_gate_range_error_m: float = 0.15,
        confirmed_gate_range_error_fraction: float = 0.15,
        max_initial_xy_speed_m_s: float = 8.0, # above expected ~6 m/s paper-plane speed
        max_initial_z_speed_m_s: float = 4.0, # tighter because monocular depth is much noisier
        max_gate_rejected_streak: int = 2, # reacquire after this many mutually consistent rejected detections
    ):
        self.track = None
        self.next_track_id = 0
        self.track_status = TrackStatus.DEAD
        self.min_hits = min_hits
        self.max_missed_on_confirmed = max_missed_on_confirmed
        self.max_missed_on_tentative = max_missed_on_tentative

        # Kalman tuning parameters
        self.sigma_accel = sigma_accel # expected unknown acceleration scale, m/s^2
        self.sigma_meas_x = sigma_meas_x # meters
        self.sigma_meas_y = sigma_meas_y # meters
        self.sigma_meas_z = sigma_meas_z # meters

        # Gate measurement noise is expressed directly in camera bearing/range coordinates rather
        # than inheriting the much noisier monocular-depth coupling in Cartesian x/y/z.
        self.tentative_gate_mahalanobis_sq_threshold = tentative_gate_mahalanobis_sq_threshold
        self.confirmed_gate_mahalanobis_sq_threshold = confirmed_gate_mahalanobis_sq_threshold
        self.gate_measurement_angular_sigma_deg = gate_measurement_angular_sigma_deg
        self.gate_measurement_min_range_sigma_m = gate_measurement_min_range_sigma_m
        self.gate_measurement_range_sigma_fraction = gate_measurement_range_sigma_fraction
        self.max_gate_angular_error_deg = max_gate_angular_error_deg
        self.tentative_min_gate_range_error_m = tentative_min_gate_range_error_m
        self.tentative_gate_range_error_fraction = tentative_gate_range_error_fraction
        self.confirmed_min_gate_range_error_m = confirmed_min_gate_range_error_m
        self.confirmed_gate_range_error_fraction = confirmed_gate_range_error_fraction

        # Initial velocity comes from the first two accepted detections. XY gets enough headroom for
        # a ~6 m/s plane; depth is clipped harder because apparent-size noise makes dz much less reliable.
        self.max_initial_xy_speed_m_s = max_initial_xy_speed_m_s
        self.max_initial_z_speed_m_s = max_initial_z_speed_m_s
        self.max_gate_rejected_streak = max_gate_rejected_streak


    @staticmethod
    def _camera_measurement_and_jacobian(position):
        # Camera-space measurement h=[horizontal bearing, vertical bearing, range].
        x, y, z = np.asarray(position, dtype=float)
        horizontal_range_sq = x*x + z*z
        horizontal_range = float(np.sqrt(horizontal_range_sq))
        range_sq = horizontal_range_sq + y*y
        object_range = float(np.sqrt(range_sq))
        if horizontal_range <= 1e-9 or object_range <= 1e-9:
            return None, None

        camera_measurement = np.array([
            np.arctan2(x, z),
            np.arctan2(y, horizontal_range),
            object_range
        ], dtype=float)

        # First-order projection of Cartesian position covariance into bearing/range covariance.
        J = np.array([
            [z/horizontal_range_sq, 0.0, -x/horizontal_range_sq],
            [-x*y/(horizontal_range*range_sq), horizontal_range/range_sq, -z*y/(horizontal_range*range_sq)],
            [x/object_range, y/object_range, z/object_range]
        ], dtype=float)
        return camera_measurement, J


    def _update_track_camera_uncertainty(self):
        if self.track is None:
            return

        _, J = self._camera_measurement_and_jacobian(self.track.state[:3])
        if J is None:
            self.track.horizontal_angular_uncertainty_deg = None
            self.track.vertical_angular_uncertainty_deg = None
            self.track.angular_uncertainty_deg = None
            self.track.range_uncertainty_m = None
            return

        camera_covariance = J @ self.track.covariance[:3, :3] @ J.T
        camera_covariance = 0.5*(camera_covariance + camera_covariance.T) # suppress numerical asymmetry

        horizontal_sigma_deg = float(np.rad2deg(np.sqrt(max(camera_covariance[0, 0], 0.0))))
        vertical_sigma_deg = float(np.rad2deg(np.sqrt(max(camera_covariance[1, 1], 0.0))))
        self.track.horizontal_angular_uncertainty_deg = horizontal_sigma_deg
        self.track.vertical_angular_uncertainty_deg = vertical_sigma_deg
        self.track.angular_uncertainty_deg = max(horizontal_sigma_deg, vertical_sigma_deg)
        self.track.range_uncertainty_m = float(np.sqrt(max(camera_covariance[2, 2], 0.0)))


    def _measurement_passes_gate(self, measurement: Measurement) -> bool:
        if self.track is None:
            raise ValueError("Cannot gate measurement without an active track.")

        predicted_position = np.asarray(self.track.state[:3], dtype=float)
        measurement_position = np.array([measurement.x, measurement.y, measurement.z], dtype=float)
        if not np.all(np.isfinite(predicted_position)) or not np.all(np.isfinite(measurement_position)):
            self.track.innovation_mahalanobis = np.inf
            return False

        predicted_camera_measurement, J = self._camera_measurement_and_jacobian(predicted_position)
        measured_camera_measurement, _ = self._camera_measurement_and_jacobian(measurement_position)
        if J is None or measured_camera_measurement is None:
            self.track.innovation_mahalanobis = np.inf
            return False

        predicted_range = predicted_camera_measurement[2]
        measurement_range = measured_camera_measurement[2]

        # Innovation is measured in two camera bearing angles plus radial range.
        innovation = measured_camera_measurement - predicted_camera_measurement
        innovation[:2] = (innovation[:2] + np.pi) % (2*np.pi) - np.pi

        predicted_camera_covariance = J @ self.track.covariance[:3, :3] @ J.T
        angular_sigma_rad = np.deg2rad(self.gate_measurement_angular_sigma_deg)
        range_sigma_m = max(self.gate_measurement_min_range_sigma_m,
                            self.gate_measurement_range_sigma_fraction*predicted_range)
        measurement_camera_covariance = np.diag([angular_sigma_rad**2, angular_sigma_rad**2, range_sigma_m**2])
        innovation_covariance = predicted_camera_covariance + measurement_camera_covariance
        innovation_covariance = 0.5*(innovation_covariance + innovation_covariance.T)

        try:
            mahalanobis_sq = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        except np.linalg.LinAlgError:
            self.track.innovation_mahalanobis = np.inf
            return False
        self.track.innovation_mahalanobis = float(np.sqrt(max(mahalanobis_sq, 0.0)))

        # Hard physical sanity caps stop a very uncertain/lost track from accepting arbitrary detections.
        cos_angular_error = float(np.dot(predicted_position, measurement_position)/(predicted_range*measurement_range))
        angular_error_deg = float(np.rad2deg(np.arccos(np.clip(cos_angular_error, -1.0, 1.0))))
        if self.track.confirmed:
            mahalanobis_sq_threshold = self.confirmed_gate_mahalanobis_sq_threshold
            min_range_error_m = self.confirmed_min_gate_range_error_m
            range_error_fraction = self.confirmed_gate_range_error_fraction
        else:
            mahalanobis_sq_threshold = self.tentative_gate_mahalanobis_sq_threshold
            min_range_error_m = self.tentative_min_gate_range_error_m
            range_error_fraction = self.tentative_gate_range_error_fraction

        range_error = abs(measurement_range - predicted_range)
        max_range_error = max(min_range_error_m, range_error_fraction*predicted_range)
        return (mahalanobis_sq <= mahalanobis_sq_threshold
                and angular_error_deg <= self.max_gate_angular_error_deg
                and range_error <= max_range_error)


    def _initialize_velocity_from_second_hit(self, measurement: Measurement, frame_time):
        if self.track is None or self.track.initial_velocity_initialized:
            return

        dt = frame_time - self.track.first_detection_time
        if dt <= 1e-6:
            return

        first = self.track.last_hit_measurement
        measured_velocity = np.array([
            (measurement.x - first.x)/dt,
            (measurement.y - first.y)/dt,
            (measurement.z - first.z)/dt
        ], dtype=float)

        # Clip XY by vector magnitude so diagonal motion cannot exceed the intended lateral speed cap.
        xy_speed = float(np.linalg.norm(measured_velocity[:2]))
        if xy_speed > self.max_initial_xy_speed_m_s:
            measured_velocity[:2] *= self.max_initial_xy_speed_m_s/xy_speed
        measured_velocity[2] = np.clip(measured_velocity[2], -self.max_initial_z_speed_m_s, self.max_initial_z_speed_m_s)

        # Fuse the two-frame velocity estimate with the KF velocity according to their uncertainties.
        # Shorter frame spacing makes the finite-difference velocity noisier through the 1/dt^2 term.
        measured_velocity_variance = 2.0*np.array([
            self.sigma_meas_x**2,
            self.sigma_meas_y**2,
            self.sigma_meas_z**2
        ], dtype=float)/(dt*dt)
        prior_velocity = self.track.state[DX:DZ + 1].copy()
        prior_velocity_variance = np.diag(self.track.covariance[DX:DZ + 1, DX:DZ + 1]).copy()
        velocity_gain = prior_velocity_variance/(prior_velocity_variance + measured_velocity_variance)

        self.track.state[DX:DZ + 1] = prior_velocity + velocity_gain*(measured_velocity - prior_velocity)
        # Do not reduce covariance: this velocity estimate reuses the same two position measurements.
        self.track.initial_velocity_initialized = True


    def _predict_state_and_covariance(self, dt):
        """Non-mutating CV prediction shared by the KF update and Platform prediction."""
        if self.track is None:
            raise ValueError("Cannot predict without an active track.")

        F_k = np.eye(6, dtype=float)
        F_k[X, DX] = dt
        F_k[Y, DY] = dt
        F_k[Z, DZ] = dt

        # Constant unknown acceleration over this timestep.
        accel_variance = self.sigma_accel**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q_k = np.zeros((6, 6), dtype=float)
        for position_index, velocity_index in ((X, DX), (Y, DY), (Z, DZ)):
            Q_k[position_index, position_index] = 0.25*dt4*accel_variance
            Q_k[position_index, velocity_index] = 0.5*dt3*accel_variance
            Q_k[velocity_index, position_index] = 0.5*dt3*accel_variance
            Q_k[velocity_index, velocity_index] = dt2*accel_variance

        predicted_state = F_k @ self.track.state
        predicted_covariance = F_k @ self.track.covariance @ F_k.T + Q_k
        return predicted_state, predicted_covariance


    def _prediction_update(self, dt):
        """Kalman prediction step; mean is CV and Q models unknown acceleration."""
        self.track.state, self.track.covariance = self._predict_state_and_covariance(dt)
        self._update_track_camera_uncertainty()


    def predict(self, dt, include_uncertainty=False):
        # Non-mutating prediction for Platform. Default return stays backward-compatible.
        predicted_state, predicted_covariance = self._predict_state_and_covariance(dt)
        if not include_uncertainty:
            return predicted_state

        camera_measurement, J = self._camera_measurement_and_jacobian(predicted_state[:3])
        if J is None:
            return predicted_state, None, None

        camera_covariance = J @ predicted_covariance[:3, :3] @ J.T
        camera_covariance = 0.5*(camera_covariance + camera_covariance.T)
        angular_sigma_rad = float(np.sqrt(max(camera_covariance[0, 0], camera_covariance[1, 1], 0.0)))
        transverse_uncertainty_m = float(camera_measurement[2]*angular_sigma_rad) # worst bearing 1-sigma, expressed transversely
        range_uncertainty_m = float(np.sqrt(max(camera_covariance[2, 2], 0.0)))
        return predicted_state, transverse_uncertainty_m, range_uncertainty_m


    def _measurement_update(self, measurement: Measurement):
        """
        Kalman measurement update/correction step.

        Theory:
            z_k = measurement
            y_k = innovation / residual
            S_k = innovation covariance
            K_k = Kalman gain
            x_k = corrected state estimate
            P_k = corrected covariance estimate

        State:
            [x, y, z, dx, dy, dz]

        Measurement:
            [x, y, z]
        """
        if self.track is None:
            raise ValueError("Cannot update without an active track.")

        z_k = np.array([measurement.x, measurement.y, measurement.z], dtype=float)
        x_k_pred = self.track.state
        P_k_pred = self.track.covariance

        # Maps state [x, y, z, dx, dy, dz] -> measurement [x, y, z]
        H_k = np.zeros((3, 6), dtype=float)
        H_k[0, X] = 1.0
        H_k[1, Y] = 1.0
        H_k[2, Z] = 1.0

        R_k = np.diag([
            self.sigma_meas_x**2,
            self.sigma_meas_y**2,
            self.sigma_meas_z**2
        ])

        y_k = z_k - H_k @ x_k_pred
        S_k = H_k @ P_k_pred @ H_k.T + R_k
        K_k = P_k_pred @ H_k.T @ np.linalg.inv(S_k)

        x_k = x_k_pred + K_k @ y_k
        I = np.eye(6, dtype=float)
        P_k = (I - K_k @ H_k) @ P_k_pred

        self.track.state = x_k
        self.track.covariance = P_k
        self._update_track_camera_uncertainty()


    def _create_track(self, measurement: Measurement, initial_time):
        self.track = Track(measurement, initial_time, self.min_hits, self.next_track_id)
        self._update_track_camera_uncertainty()
        self.next_track_id = self.next_track_id + 1 if self.next_track_id < MAX_TRACK_ID else 0


    def update(self, object_detected, measurement: Measurement, frame_time):

        # None track logic:
        #     any measurement should create a track.
        #     a missing measurement should do nothing.
        if self.track is None:
            if not object_detected:
                self.track_status = TrackStatus.DEAD # for sanity
                return self.track_status
            else:
                self._create_track(measurement, initial_time=frame_time)

        # TENTATIVE track logic:
        #     both a far & missing measurement should count as a miss, then check if dead.
        # CONFIRMED track logic:
        #     both a far & missing measurement should count as a miss, then check if dead.
        #     different max_missed
        #     they do not become TENTATIVE tracks just because of a miss
        else:
            dt = frame_time - self.track.state_time
            self._prediction_update(dt)

            predicted_state = self.track.state.copy() # FOR DEBUG ONLY
            if object_detected:
                measurement_passes_gate = self._measurement_passes_gate(measurement)

                if not measurement_passes_gate:
                    previous_rejected = self.track.last_gate_rejected_measurement
                    previous_rejected_time = self.track.last_gate_rejected_time
                    rejected_measurements_consistent = False

                    if previous_rejected is not None and previous_rejected_time is not None:
                        rejected_dt = frame_time - previous_rejected_time
                        if rejected_dt > 1e-6:
                            rejected_velocity = np.array([
                                (measurement.x - previous_rejected.x)/rejected_dt,
                                (measurement.y - previous_rejected.y)/rejected_dt,
                                (measurement.z - previous_rejected.z)/rejected_dt
                            ], dtype=float)
                            rejected_measurements_consistent = (
                                np.linalg.norm(rejected_velocity[:2]) <= self.max_initial_xy_speed_m_s
                                and abs(rejected_velocity[2]) <= self.max_initial_z_speed_m_s
                            )

                    self.track.gate_rejected_streak = self.track.gate_rejected_streak + 1 if rejected_measurements_consistent else 1
                    self.track.last_gate_rejected_measurement = measurement
                    self.track.last_gate_rejected_time = frame_time
                else:
                    self.track.gate_rejected_streak = 0
                    self.track.last_gate_rejected_measurement = None
                    self.track.last_gate_rejected_time = None
            else:
                self.track.innovation_mahalanobis = None
                self.track.gate_rejected_streak = 0
                self.track.last_gate_rejected_measurement = None
                self.track.last_gate_rejected_time = None
                measurement_passes_gate = False

            if object_detected and not measurement_passes_gate and self.track.gate_rejected_streak >= self.max_gate_rejected_streak:
                print(f"TRACK REACQUIRED | {self.track.gate_rejected_streak} mutually consistent gate rejections") # FOR DEBUG ONLY
                self._create_track(measurement, initial_time=frame_time)
            elif not measurement_passes_gate:
                self.track.mark_missed()
                max_missed = self.max_missed_on_confirmed if self.track.confirmed else self.max_missed_on_tentative

                if self.track.is_dead(max_missed):
                    if not object_detected:
                        self.track = None
                        self.track_status = TrackStatus.DEAD
                        return self.track_status
                    else:
                        self._create_track(measurement, initial_time=frame_time)
            else:
                was_confirmed = self.track.confirmed # FOR DEBUG ONLY
                self._measurement_update(measurement)
                self._initialize_velocity_from_second_hit(measurement, frame_time)
                self.track.mark_hit(measurement, self.min_hits)
                if not was_confirmed and self.track.confirmed:
                    print(f"TRACK CONFIRMED | first detection->confirmed {(frame_time - self.track.first_detection_time)*1000:.1f} ms") # FOR DEBUG ONLY

            if self.track is not None and self.track.confirmed: # FOR DEBUG ONLY
                measurement_text = (
                    f"[{measurement.x:.3f}, {measurement.y:.3f}, {measurement.z:.3f}]"
                    if object_detected else "NONE"
                )
                gate_text = (
                    f"maha_dist={self.track.innovation_mahalanobis:.2f} "
                    f"ang_sigma={self.track.angular_uncertainty_deg:.2f}deg "
                    f"range_sigma={self.track.range_uncertainty_m:.3f}m"
                    if self.track.innovation_mahalanobis is not None else "maha_dist=NONE"
                )
                state_correction = self.track.state - predicted_state
                future_position_correction_300ms = state_correction[:3] + 0.300*state_correction[3:]
                correction_text = (
                    f"dpos={state_correction[:3]} ({np.linalg.norm(state_correction[:3]):.3f}m) "
                    f"dvel={state_correction[3:]} ({np.linalg.norm(state_correction[3:]):.3f}m/s) "
                    f"future300={future_position_correction_300ms} ({np.linalg.norm(future_position_correction_300ms):.3f}m)"
                    if measurement_passes_gate else "dpos=NONE dvel=NONE future300=NONE"
                )
                print(
                    f"KF | dt={dt*1000:.1f} ms | "
                    f"pred pos={predicted_state[:3]} vel={predicted_state[3:]} | "
                    f"meas={measurement_text} {'ACCEPT' if measurement_passes_gate else 'REJECT'} | "
                    f"{gate_text} | corr pos={self.track.state[:3]} vel={self.track.state[3:]} | {correction_text}"
                ) # FOR DEBUG ONLY

            self.track.state_time = frame_time

        self.track_status = TrackStatus.CONFIRMED if self.track.confirmed else TrackStatus.TENTATIVE
        return self.track_status


# for future
# class MultiObjectTracker:
#     def __init__(self):
#         self.tracks = []
#         # etc...
#     def update(self, detections, dt):
#         # 1. Predict every existing track forward
#         # 2. Match detections to tracks
#         # 3. Update matched tracks
#         # 4. Mark unmatched tracks as missed
#         # 5. Create new tracks for unmatched detections
#         # 6. Delete dead tracks
#         # 7. Return confirmed active tracks