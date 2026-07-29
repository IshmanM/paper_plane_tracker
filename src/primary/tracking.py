import numpy as np
from enum import Enum, auto
from src.primary.detection import Measurement
from src.primary.geometry import estimateTargetImagePosition 
import cv2

class TrackStatus(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DEAD = auto()

X, Y, Z, DX, DY, DZ = 0,1,2,3,4,5

MAX_TRACK_ID = 2**32 - 1 # max uint32 number

class Track:
    
    def __init__ (self, initial_measurement: Measurement, initial_time, min_hits: int, track_id=0,
                  sigma_x=0.1, sigma_y=0.1, sigma_z=0.2, sigma_dx=1.0, sigma_dy=1.0, sigma_dz=1.0):
    
        self.id = track_id # not useful until MOT

        # x, y, z, dx, dy, dz
        self.state = np.array([
            initial_measurement.x, initial_measurement.y, initial_measurement.z,
            0.0, 0.0, 0.0
        ], dtype=float)
        self.state_time = initial_time # time current KF state represents
    
        # Need to tune starting covariance...
        #   Note smaller initial sigma/covariance indicates higher initial position certainty 
        self.covariance = np.diag([
            sigma_x ** 2,
            sigma_y ** 2,
            sigma_z ** 2,
            sigma_dx ** 2,
            sigma_dy ** 2,
            sigma_dy ** 2
        ]).astype(float)

        self.hit_streak = 1       # consecutive hits   
        self.missed_streak = 0    # consecutive misses
        self.confirmed = False if min_hits > 1 else True

        self.last_hit_measurement = initial_measurement


    def mark_hit(self, measurement, min_hits: int):
        self.hit_streak += 1
        self.missed_streak = 0
        self.last_hit_measurement = measurement

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


def drawTrack( frame: np.ndarray, track, px_w: float, px_h: float, ) -> None:
    track_u, track_v = estimateTargetImagePosition(track.x, track.y, track.z)
    track_center = (int(round(track_u)), int(round(track_v)))

    cv2.rectangle( frame, (int(round(track_u - px_w / 2)), int(round(track_v - px_h / 2))), (int(round(track_u + px_w / 2)), int(round(track_v + px_h / 2))), color=(0, 0, 255), thickness=2, )

    cv2.circle( frame, track_center, radius=5, color=(0, 0, 255), thickness=-1, )

    velocity_2d = np.array([track.dx, track.dy], dtype=float)
    velocity_norm = np.linalg.norm(velocity_2d)
    if velocity_norm > 1e-6:
        arrow_length_px = 40
        direction = velocity_2d / velocity_norm
        arrow_end = (int(round(track_u + arrow_length_px * direction[0])), int(round(track_v + arrow_length_px * direction[1])), )
        cv2.arrowedLine( frame, track_center, arrow_end, color=(0, 0, 255), thickness=2, tipLength=0.25, )


# eventially need to make this tracker class derived from some base class, and use the base class in platform.py definitions

class SingleObjectTracker:
    def __init__(
        self,
        min_hits: int = 3,
        max_missed_on_confirmed: int = 5,
        max_missed_on_tentative: int = 1,
        process_noise: float = 0.01,
        sigma_meas_x: float = 0.05,
        sigma_meas_y: float = 0.05,
        sigma_meas_z: float = 0.20,
        gate_threshold: float = 0.25,
    ):
        self.track = None
        self.next_track_id = 0 
        self.track_status = TrackStatus.DEAD 
        self.min_hits = min_hits
        self.max_missed_on_confirmed = max_missed_on_confirmed
        self.max_missed_on_tentative = max_missed_on_tentative
        
        # Kalman tuning parameters
        self.process_noise = process_noise
        self.sigma_meas_x = sigma_meas_x  # meters
        self.sigma_meas_y = sigma_meas_y  # meters
        self.sigma_meas_z = sigma_meas_z  # meters
        self.gate_threshold = gate_threshold

    
    def _gating_distance(self, measurement: Measurement):
        if self.track is None:
            raise ValueError("Cannot compute gating distance without an active track.")
        
        # Euclidian distance 
        measurement_position = np.array([measurement.x, measurement.y, measurement.z], dtype=float)  
        diff = measurement_position - self.track.state[:3]
        return float(diff @ diff)

    def _prediction_update(self, dt):
        """
        Kalman prediction step.

        Theory:
            x_k_pred = F_k @ x_k_prev
            P_k_pred = F_k @ P_k_prev @ F_k.T + Q_k

        In this tracker:
            state = [x, y, z, dx, dy, dz]
        """

        if self.track is None:
            raise ValueError("Cannot update without an active track.")
        
        x_k_prev = self.track.state
        P_k_prev = self.track.covariance
        F_k = np.eye(6, dtype=float) # State transition matrix
        F_k[X, DX] = dt
        F_k[Y, DY] = dt
        F_k[Z, DZ] = dt
        Q_k = np.eye(6, dtype=float) * self.process_noise
        
        x_k_pred = F_k @ x_k_prev
        P_k_pred = F_k @ P_k_prev @ F_k.T + Q_k

        self.track.state = x_k_pred
        self.track.covariance = P_k_pred
        return
    
    def predict(self, dt):
        # Not a Kalman update. Implemented for the Platform to call
        if self.track is None:
            raise ValueError("Cannot predict without an active track.")
        
        x_k_prev = self.track.state
        F_k = np.eye(6, dtype=float) # State transition matrix
        F_k[X, DX] = dt
        F_k[Y, DY] = dt
        F_k[Z, DZ] = dt

        x_k_pred = F_k @ x_k_prev
        return x_k_pred

       
    def _measurement_update(self, measurement: Measurement):
        """
        Kalman measurement update/correction step.

        Theory:
            z_k       = measurement
            y_k       = innovation / residual. Difference between actual measurement and predicted measurement
            S_k       = innovation covariance. Uncertainty of the innovation
            K_k       = Kalman gain
            x_k       = corrected state estimate
            P_k       = corrected covariance estimate

        In this tracker:
            state       = [x, y, z, dx, dy, dz]
            measurement = [x, y, z]
        """
        if self.track is None:
            raise ValueError("Cannot update without an active track.")
        
        z_k = np.array([measurement.x, measurement.y, measurement.z], dtype=float) 
        x_k_pred = self.track.state
        P_k_pred = self.track.covariance
        # H_k: Maps state [x, y, z, dx, dy, dz] -> measurement [x, y, z]
        H_k = np.zeros((3, 6), dtype=float)
        H_k[0, X] = 1.0
        H_k[1, Y] = 1.0
        H_k[2, Z] = 1.0
        # R_k: measurement noise covariance
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
        return


    def _create_track(self, measurement: Measurement, initial_time):
        self.track = Track(measurement, initial_time, self.min_hits, self.next_track_id)            
        self.next_track_id = self.next_track_id + 1 if self.next_track_id < MAX_TRACK_ID else 0 # circularly looping id


    def update(self, object_detected, measurement: Measurement, frame_time):
        
        # None track logic:
        #     any measurement should create a track.
        #     a missing measurement should do nothing.
        if self.track is None:
            if not object_detected:
                self.track_status = TrackStatus.DEAD #for sanity
                return self.track_status 
            else:
                self._create_track(measurement, initial_time=frame_time)

        # TENTATIVE track logic:
        #     both a far & missing measurement should count as a miss, then check if dead.
        # CONFIRMED track logic:
        #     both a far & missing measurement should count as a miss, then check if dead.
        #     defferent max_missed
        #     they do not become TENTATIVE tracks just because of a miss
        else:
            dt = frame_time - self.track.state_time
            self._prediction_update(dt) # Kalman Prediction update
            
            if (not object_detected) or (self._gating_distance(measurement) > self.gate_threshold):
                # miss 
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
                # hit
                self.track.mark_hit(measurement, self.min_hits)
                self._measurement_update(measurement) # Kalman Measurement update

            self.track.state_time = frame_time
        
        self.track_status = TrackStatus.CONFIRMED if self.track.confirmed else TrackStatus.TENTATIVE
        return self.track_status
    

    


    




# for future
# class MultiObjectTracker:
#     def __init__(self):
#         self.tracks = []
#         #etc...
#   def update(self, detections, dt):
        # 1. Predict every existing track forward
        # 2. Match detections to tracks
        # 3. Update matched tracks
        # 4. Mark unmatched tracks as missed
        # 5. Create new tracks for unmatched detections
        # 6. Delete dead tracks
        # 7. Return confirmed active tracks