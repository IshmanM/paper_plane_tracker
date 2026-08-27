import numpy as np
from enum import Enum, auto
import time



class PlanType(Enum):
    OFF = auto()
    SEARCH = auto()
    PRE_SLEW = auto()
    FIRST_INTERCEPT = auto()
    SUBSEQUENT_INTERCEPT = auto()

class Plan:
    def __init__(
        self, 
        track_id: int, #use -1 for SEARCHING
        plan_type: PlanType,
        raw_servo_angles: np.ndarray,
        estimate_time: float | None = None,
        ready_time: float | None = None, 
        intercept_position: np.ndarray | None = None, # in world/camera frame
        intercept_time: float | None = None,
        trigger_time: float | None = None
    ):
        self.track_id = track_id
        self.plan_type = plan_type

        self.intercept_position = None if intercept_position is None else np.asarray(intercept_position, dtype=float).copy()
        self.intercept_time = None if intercept_time is None else intercept_time

        # Desired planner angles
        self.raw_servo_angles = np.asarray(raw_servo_angles, dtype=float).copy()
        
        self.estimate_time = estimate_time # will use the track state_time
        self.created_time = time.perf_counter()    # is this necessary?
        self.ready_time = ready_time # expected time when the servos will be in position for triggering

        self.trigger_time = trigger_time


