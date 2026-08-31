from src.primary.tracking import Track, TrackStatus, SingleObjectTracker
import numpy as np
import time
import math
from src.primary.geometry import estimateObjectPlatformPosition
import src.primary.config as config
from src.primary.comm_buffer import CommBuffer
from src.primary.plan import Plan, PlanType
from src.primary.platform_mode import PlatformMode
from src.primary.geometry import rotationPlatformFromPanTilt
from src.primary.platform_geometry_spec import PlatformGeometrySpecId, PlatformGeometrySpec, PLATFORM_GEOMETRY_SPECS
from src.primary.camera_to_platform_calibration import CameraToPlatformCalibration



MAX_SERVO_SETTLING_TIME = 0.080 # seconds
SERVO_ANGLE_CHANGE_FOR_MAX_SETTLING = 30.0 # degrees

SERVO_ROTATION_TIME_MARGIN = 0.00 # seconds. an extra margin.


SEARCH_CENTER_PAN = 100.0 # degrees
SEARCH_PAN_AMPLITUDE = 25.0 # degrees
SEARCH_FREQUENCY = 0.2 # hz

PRE_SLEW_LOOKAHEAD = 0.100 # seconds. TODO: adjust this


TRIGGER_TIME_LOWER_THRESHOLD = 0.040 # seconds. allow up to this much time late
TRIGGER_TIME_UPPER_THRESHOLD = 0.020 # seconds. allow up to this much time early

FIRST_INTERCEPT_COARSE_NUM_CANDIDATES = 11
FIRST_INTERCEPT_FINE_NUM_CANDIDATES = 11
FIRST_INTERCEPT_MAX_LOOKAHEAD = 0.5 # seconds

FIRST_INTERCEPT_REFRESH_NUM_CANDIDATES = 6
FIRST_INTERCEPT_REFRESH_HALF_WINDOW = 0.010 # seconds, +/- around previous intercept time

SUBSEQUENT_INTERCEPT_MAX_NUM_CANDIDATES = 5 
SUBSEQUENT_INTERCEPT_MAX_LOOKAHEAD = 0.5 # seconds

# TODO: make sure this is > FOAM_TRIGGER_HOLD_DELAY + FOAM_RESET_HOLD_DELAY from the endpoint 
TRIGGER_DELAY = 0.050 # seconds

GRAVITY = 9.81  # m/s^2

# Increase as needed.
MAX_AIM_SOLVE_ITERATIONS = 5
MAX_TRAJECTORY_SOLVE_ITERATIONS = 8

# Tune experimentally.
DART_PROTRUSION_SPEED = 21.67 # m/s, actual protrusion speed v0
DART_DRAG_K = 0.040 # 1/m; k=rho*Cd*A/(2m) ~= 1.2*0.6*pi*(0.0065)^2/(2*0.001) ~= 0.048
DART_SIMULATION_DT = 0.001 # seconds, Euler used to precompute the trajectory table.
TRAJECTORY_HEIGHT_TOLERANCE = 0.0005 # m

DART_TRAJECTORY_TABLE_MAX_FORWARD_RANGE = 8.0 # m
DART_TRAJECTORY_TABLE_MIN_TARGET_UP = -3.0 # m
DART_TRAJECTORY_TABLE_MAX_TARGET_UP = 3.0 # m
DART_TRAJECTORY_TABLE_X_STEP = 0.01 # m
DART_TRAJECTORY_TABLE_MIN_TILT_DEG = -89.0
DART_TRAJECTORY_TABLE_MAX_TILT_DEG = 89.0
DART_TRAJECTORY_TABLE_TILT_STEP_DEG = 0.25
DART_TRAJECTORY_TABLE_MAX_GENERATION_TIME = 5.0 # s; safety cap for near-vertical trajectories.

# Do not aim at objects effectively behind / on top of platform.
MIN_FORWARD_RANGE = 0.02  # m

# Usually False for a turret. High arc is slower and less direct.
USE_HIGH_ARC = False

MIN_FIRST_INTERCEPT_READY_MARGIN = 0.010 # seconds. TODO: decraese if viable
MIN_SUBSEQUENT_INTERCEPT_READY_MARGIN = 0.005 # seconds

ACTIVE_PLAN_SERVO_ANGLE_TOLERANCES = np.zeros(config.NUM_SERVOS, dtype=float)
ACTIVE_PLAN_SERVO_ANGLE_TOLERANCES[config.SERVO_IDX["pan"]] = 1.25 # degrees
ACTIVE_PLAN_SERVO_ANGLE_TOLERANCES[config.SERVO_IDX["tilt"]] = 1.25 # degrees
ACTIVE_PLAN_FLIGHT_TIME_TOLERANCE = 0.010 # seconds
ACTIVE_PLAN_UNCERTAINTY_SIGMA_MULTIPLIER = 1.5

# Active-plan validity hysteresis. Moderate violations must persist; large violations invalidate immediately.
ACTIVE_PLAN_VALIDITY_RECOVER_RATIO = 0.80
ACTIVE_PLAN_VALIDITY_INVALID_RATIO = 1.00
ACTIVE_PLAN_VALIDITY_HARD_INVALID_RATIO = 2.00 # originally 2.0. TODO: maybe 2.5?
ACTIVE_PLAN_INVALID_STREAK_REQUIRED = 2

# Loose trigger-certainty limits. Planning/aiming continues when these are exceeded; only triggering is suppressed.
MAX_TRIGGER_TRANSVERSE_UNCERTAINTY_M = 0.10
MAX_TRIGGER_RANGE_UNCERTAINTY_M = 0.30


# If now is inside _close_to_trigger_time(...), treat time-to-trigger cost as ideal.
PLAN_COST_NOW_TRIGGER_TIME_COST = 0.0

# time_to_trigger_cost = time_to_trigger / PLAN_COST_TIME_SCALE
# So a candidate 0.5 s away has time cost ~= 1.
PLAN_COST_TIME_SCALE = 0.5 # seconds

# Extra ready-margin surplus of this amount cuts ready_margin_cost roughly in half.
# ready_margin_cost = 1 / (1 + surplus / scale)
PLAN_COST_READY_MARGIN_SCALE = 0.05 # seconds

# Normalize servo motion by each servo's full usable angular range.
PLAN_COST_SERVO_ANGLE_SCALES = (
    np.asarray(config.MAX_SERVO_ANGLES, dtype=float)
    - np.asarray(config.MIN_SERVO_ANGLES, dtype=float)
)

# Meters.
# Normalize intercept-position changes for continuity cost.
PLAN_COST_INTERCEPT_POSITION_SCALES = np.array([0.20, 0.20, 0.30], dtype=float)

# No uncertainty penalty while comfortably inside the trigger-certainty limits.
PLAN_COST_UNCERTAINTY_RISK_START_RATIO = 0.80

# Smoothstep reaches full uncertainty cost here. Initial value is based on current test data,
# where earliest-feasible first-plan uncertainty ratios were roughly 1.9-4.4 (median ~2.9).
# Tune this as more representative data is collected.
PLAN_COST_UNCERTAINTY_RISK_FULL_RATIO = 6.0


FIRST_INTERCEPT_PLAN_COST_WEIGHTS = {
    "time": 1.0, # originally 1.0
    "servo_motion": 0.25,
    "ready_margin": 0.15,
    "continuity": 0.0,
    "uncertainty_risk": 0.20,
}

SUBSEQUENT_INTERCEPT_PLAN_COST_WEIGHTS = {
    "time": 0.35,
    "servo_motion": 0.50,
    "ready_margin": 0.0,
    "continuity": 0.75,
    "uncertainty_risk": 0.25,
}



class Platform:
    def __init__(
        self,
        comm_buffer: CommBuffer,
        platform_geometry_spec_id: PlatformGeometrySpecId,
        camera_to_platform_calibration: CameraToPlatformCalibration
    ):
        self.mode = PlatformMode.OFF
        self.active_plan_invalid_streak = 0
        self.active_plan_last_accepted_measurement_count = None
        self.active_plan = self._make_off_plan(now=time.perf_counter())
        self.first_intercept_anchor_intercept_time  = None
        self.first_intercept_original_trigger_time = None

        self.triggering_halted = True # Forcing parameter
        self.track_certain = False

        self.comm_buffer = comm_buffer
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan, 
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )

        self.platform_geometry_spec = PLATFORM_GEOMETRY_SPECS[platform_geometry_spec_id]

        self.camera_to_platform_calibration = camera_to_platform_calibration
        self._generate_dart_trajectory_table()
        
        # extra Todos:
        # - if adding ACKs from the rpi: eg. self.last_ack_cmd_id, self.last_ack_time. <-- should use the buffer
        # - instead of guessing the communication speed/delay, an occasional ping can be used to estimate <-- cmd.py or comm script should figure out and put in buffer

    
    def _tracker_is_usable(self, tracker: SingleObjectTracker):
        if tracker is None:
            raise ValueError("None type Tracker passed to Platform")

        if tracker.track_status in {TrackStatus.TENTATIVE, TrackStatus.CONFIRMED} and (tracker.track is None or tracker.track.state_time is None or tracker.track.id is None):
            raise ValueError("Tracker with TENTATIVE/CONFIRMED track_status but None type Track or Track.state_time or Track.id passed to Platform")

        if tracker.track_status == TrackStatus.DEAD:
            return False

        return True


    def update(self, tracker: SingleObjectTracker):
        now = time.perf_counter()
        
        # Nothing updates if OFF
        if self.mode == PlatformMode.OFF:
            self.track_certain = False
            self.comm_buffer.set_platform_snapshot(
                active_plan=self.active_plan,
                platform_mode=self.mode,
                triggering_halted=self.triggering_halted
            )
            return
        
        ########## 0. UNIVERSAL TRACKER VALIDITY GUARD #############

        if not self._tracker_is_usable(tracker):
            self.track_certain = False
            self.active_plan = self._make_search_plan(now)
            self.active_plan_invalid_streak = 0
            self.active_plan_last_accepted_measurement_count = None
            self.first_intercept_anchor_intercept_time  = None
            self.first_intercept_original_trigger_time = None
            self.mode = PlatformMode.SEARCHING

            self.comm_buffer.set_platform_snapshot(
                active_plan=self.active_plan,
                platform_mode=self.mode,
                triggering_halted=self.triggering_halted
            )
            return
        
        _, transverse_uncertainty_m, range_uncertainty_m = tracker.predict(0.0, include_uncertainty=True)
        self.track_certain = (
            tracker.track_status == TrackStatus.CONFIRMED
            and transverse_uncertainty_m is not None and np.isfinite(transverse_uncertainty_m)
            and transverse_uncertainty_m <= MAX_TRIGGER_TRANSVERSE_UNCERTAINTY_M
            and range_uncertainty_m is not None and np.isfinite(range_uncertainty_m)
            and range_uncertainty_m <= MAX_TRIGGER_RANGE_UNCERTAINTY_M
        )

        ########## 1. TENTATIVE TRACK PRE-SLEW #####################

        if tracker.track_status == TrackStatus.TENTATIVE:
            valid_plan_computed, plan = self._make_pre_slew_plan(tracker, now)

            if valid_plan_computed:
                self.active_plan = plan
                self.active_plan_invalid_streak = 0
                self.active_plan_last_accepted_measurement_count = None
                self.mode = PlatformMode.PRE_SLEWING_TO_LEAD
            else:
                self.active_plan = self._make_search_plan(now)
                self.active_plan_invalid_streak = 0
                self.active_plan_last_accepted_measurement_count = None
                self.mode = PlatformMode.SEARCHING

            self.first_intercept_anchor_intercept_time  = None
            self.first_intercept_original_trigger_time = None

            self.comm_buffer.set_platform_snapshot(
                active_plan=self.active_plan,
                platform_mode=self.mode,
                triggering_halted=self.triggering_halted
            )
            return

        # Once the track is confirmed, immediately proceed into normal first-intercept planning.
        if self.mode == PlatformMode.PRE_SLEWING_TO_LEAD:
            self.mode = PlatformMode.SEARCHING

        ########## 2. TRACK ID SWITCH GUARD ########################

        if (self.mode != PlatformMode.SEARCHING and self.active_plan.track_id != tracker.track.id):
            self.active_plan = self._make_search_plan(now)
            self.active_plan_invalid_streak = 0
            self.active_plan_last_accepted_measurement_count = None
            self.first_intercept_anchor_intercept_time  = None
            self.first_intercept_original_trigger_time = None
            self.mode = PlatformMode.SEARCHING
            # Don't return. Let SEARCHING immediately try to acquire the new track
        
        ########## 3. MODE LOGIC ##################################

        # FOLLOWING_LEAD is trigger-capable in comm_buffer.should_trigger().
        # If certainty degrades, keep the manual triggering_halted state untouched and
        # temporarily return to the non-triggering SLEWING_TO_LEAD mode.
        if self.mode == PlatformMode.FOLLOWING_LEAD and not self.track_certain:
            self.mode = PlatformMode.SLEWING_TO_LEAD

        if self.mode == PlatformMode.SEARCHING:
            
            valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)
            
            if valid_plan_computed:
                self.active_plan = plan
                self.active_plan_invalid_streak = 0
                self.active_plan_last_accepted_measurement_count = None
                self.first_intercept_anchor_intercept_time  = plan.intercept_time
                self.first_intercept_original_trigger_time = plan.trigger_time
                self.mode = PlatformMode.SLEWING_TO_LEAD

                print(
                    f"FIRST PLAN | "
                    f"ready in {plan.ready_time - now:.3f}s | "
                    f"trigger in {plan.trigger_time - now:.3f}s | "
                    f"ready->trigger {plan.trigger_time - plan.ready_time:.3f}s"
                ) # FOR DEBUG ONLY

            else:
                self.active_plan = self._make_search_plan(now)
                self.active_plan_invalid_streak = 0
                self.active_plan_last_accepted_measurement_count = None
                self.first_intercept_anchor_intercept_time  = None
                self.first_intercept_original_trigger_time = None
                self.mode = PlatformMode.SEARCHING

        
            
        if self.mode == PlatformMode.SLEWING_TO_LEAD:
            
            if not self._active_plan_still_valid(tracker, now):
                valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now, fixed_intercept_time=self.first_intercept_anchor_intercept_time )

                if valid_plan_computed:
                    trigger_delay_added = plan.trigger_time - self.active_plan.trigger_time
                    total_trigger_delay = plan.trigger_time - self.first_intercept_original_trigger_time
                    self.active_plan = plan
                    self.active_plan_invalid_streak = 0
                    self.active_plan_last_accepted_measurement_count = None

                    print(
                        f"REPLACED FIRST PLAN, INTERCEPT WINDOW | "
                        f"trigger in {plan.trigger_time - now:.3f}s | "
                        f"delay added {trigger_delay_added*1000:.1f} ms | "
                        f"total delay {total_trigger_delay*1000:.1f} ms"
                    ) # FOR DEBUG ONLY

                else:
                    # Existing intercept time is no longer achievable. Find a new one.
                    valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)

                    if valid_plan_computed:
                        trigger_delay_added = plan.trigger_time - self.active_plan.trigger_time
                        total_trigger_delay = plan.trigger_time - self.first_intercept_original_trigger_time
                        self.active_plan = plan
                        self.active_plan_invalid_streak = 0
                        self.active_plan_last_accepted_measurement_count = None
                        self.first_intercept_anchor_intercept_time  = plan.intercept_time
                        print(
                            f"REPLACED FIRST PLAN, FROM SCRATCH | "
                            f"trigger in {plan.trigger_time - now:.3f}s | "
                            f"delay added {trigger_delay_added*1000:.1f} ms | "
                            f"total delay {total_trigger_delay*1000:.1f} ms"
                        ) # FOR DEBUG ONLY
                    else:
                        self.active_plan = self._make_search_plan(now)
                        self.active_plan_invalid_streak = 0
                        self.active_plan_last_accepted_measurement_count = None
                        self.first_intercept_anchor_intercept_time  = None
                        self.first_intercept_original_trigger_time = None
                        self.mode = PlatformMode.SEARCHING
            
            # if (and not elif) incase the new best first intercept plan somehow chooses a point that the platform is already pointed toward
            if (self.mode == PlatformMode.SLEWING_TO_LEAD and self._close_to_trigger_time(now) and self.track_certain):

                print(f"ENTER FOLLOWING | trigger error {(now - self.active_plan.trigger_time)*1000:.1f} ms") # FOR DEBUG ONLY

                self.mode = PlatformMode.FOLLOWING_LEAD

        # elif (and not if) because the we dont want to get into receding horizon stuff until first foam
        elif self.mode == PlatformMode.FOLLOWING_LEAD:
            valid_plan_computed, plan = self._make_best_valid_subsequent_intercept_plan(tracker, now)

            if valid_plan_computed:
                self.active_plan = plan
                self.active_plan_invalid_streak = 0
                self.active_plan_last_accepted_measurement_count = None

            else:
                valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)
                if valid_plan_computed: # tbh this case probably wont ever happen
                    self.active_plan = plan
                    self.active_plan_invalid_streak = 0
                    self.active_plan_last_accepted_measurement_count = None
                    self.first_intercept_anchor_intercept_time  = plan.intercept_time
                    self.first_intercept_original_trigger_time = plan.trigger_time
                    self.mode = PlatformMode.SLEWING_TO_LEAD
                else:
                    self.active_plan = self._make_search_plan(now)
                    self.active_plan_invalid_streak = 0
                    self.active_plan_last_accepted_measurement_count = None
                    self.first_intercept_anchor_intercept_time  = None
                    self.first_intercept_original_trigger_time = None
                    self.mode = PlatformMode.SEARCHING

    
        ########## 4. CMD OUTPUT ##############################
        
        # Output to the buffer
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan, 
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )
        # print(self.mode.name) # FOR DEBUG ONLY
        

    def turn_off(self):
        """
        Disable the platform immediately.

        Forces a safe/default plan and prevents update() from doing tracker logic or intercept planning.
        """
        now = time.perf_counter()

        self.active_plan = self._make_off_plan(now)
        self.active_plan_invalid_streak = 0
        self.active_plan_last_accepted_measurement_count = None
        self.first_intercept_anchor_intercept_time  = None
        self.first_intercept_original_trigger_time = None
        self.mode = PlatformMode.OFF
        self.triggering_halted = True
        self.track_certain = False

        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan,
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )
    

    def turn_on(self):
        """
        Enable the platform.

        Starts from SEARCHING mode. The platform can now acquire tracks and compute intercept plans.
        """
        now = time.perf_counter()

        self.active_plan = self._make_search_plan(now)
        self.active_plan_invalid_streak = 0
        self.active_plan_last_accepted_measurement_count = None
        self.first_intercept_anchor_intercept_time  = None
        self.first_intercept_original_trigger_time = None
        self.mode = PlatformMode.SEARCHING
        
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan,
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )


    def halt_triggering(self):
        self.triggering_halted = True
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan,
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )


    def allow_triggering(self):
        self.triggering_halted = False
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan,
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )


    def _make_off_plan(self, now) -> Plan:
        """
        Safe disabled plan.

        The platform should hold default servo angles and should not scan, track, or compute intercepts.
        """
        q = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
        q = np.clip(q, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)

        return Plan(
            track_id=-1,
            plan_type=PlanType.OFF,  # okay for now; later can add PlanType.OFF if desired
            raw_servo_angles=q,
            estimate_time=None,
            ready_time=None,
            intercept_position=None,
            intercept_time=None,
            trigger_time=None
        )


    def _make_search_plan(self, now) -> Plan:
        # oscillating search pattern with fixed tilt, rotating pan
        q = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
        q[config.SERVO_IDX["pan"]] = (SEARCH_CENTER_PAN + SEARCH_PAN_AMPLITUDE*np.sin(2.0*np.pi*SEARCH_FREQUENCY*now))
        q = np.clip(q, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)

        return Plan(
            track_id=-1, #-1 for search
            plan_type=PlanType.SEARCH,
            raw_servo_angles=q, # <--hopefully this doesnt conflict with the filter, make SEARCH_FREQUENCY slow enough
            estimate_time=None,
            ready_time=None,
            intercept_position=None,
            intercept_time=None,
            trigger_time=None
        )


    def _make_pre_slew_plan(self, tracker: SingleObjectTracker, now) -> tuple[bool, Plan | None]:
        """
        Aim a tentative track a short time into the future without creating a firing schedule.
        """

        prediction_time = now + PRE_SLEW_LOOKAHEAD
        dt = prediction_time - tracker.track.state_time
        if dt <= 0.0:
            return False, None

        object_position_world = np.asarray(tracker.predict(dt)[:3], dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(object_position_world)):
            return False, None

        object_position_platform = estimateObjectPlatformPosition(object_position_world, self.camera_to_platform_calibration)
        aim_valid, raw_servo_angles, _ = self._object_position_to_servo_angles_and_flight_time(object_position_platform)
        if not aim_valid:
            return False, None

        return True, Plan(
            track_id=tracker.track.id,
            plan_type=PlanType.PRE_SLEW, # PRE_SLEWING_TO_LEAD cannot trigger
            raw_servo_angles=raw_servo_angles,
            estimate_time=tracker.track.state_time,
            ready_time=None,
            intercept_position=object_position_world,
            intercept_time=None,
            trigger_time=None
        )


    def _close_to_trigger_time(self, now, trigger_time=None) -> bool:
        """
        Return True if now is inside the acceptable trigger-time window.

        dt > 0 means now is slightly before trigger time
        dt < 0 means now slightly after trigger time
        """

        # either specify a trigger time or use the current active_plan's trigger_time
        if trigger_time is None: 
            if self.active_plan is None or self.active_plan.trigger_time is None:
                raise ValueError("Cannot compare now to None trigger_time")

            trigger_time = self.active_plan.trigger_time
        
        dt = trigger_time - now
        return -TRIGGER_TIME_LOWER_THRESHOLD <= dt <= TRIGGER_TIME_UPPER_THRESHOLD


    def _planning_servo_angles(self) -> np.ndarray:
        """
        Best estimate of current servo angles for planning.

        For now, this uses the last cmd angles. 
        Later, will use Pi feedback/ACK or a better actuator-state estimate.
        """

        last_cmd_servo_angles = self.comm_buffer.get_last_cmd_servo_angles()
        if last_cmd_servo_angles is not None:
            return np.asarray(last_cmd_servo_angles, dtype=float).copy()

        # Fallback for startup.
        return np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()


    def _make_best_valid_subsequent_intercept_plan(self, tracker: SingleObjectTracker, now) -> tuple[bool, Plan | None]:
        """
        Build the best SUBSEQUENT_INTERCEPT plan. 
        Might have way different implementation than FIRST_INTERCEPT plans later.

        Returns:
        (True, plan)  if a feasible subsequent plan was found
        (False, None) otherwise

        Important timing meaning:
        intercept_time = when object and foam meet
        trigger_time      = when platform must trigger/release
        expected_ready_time = when pan/tilt is expected to be aimed and settled

        This function intentionally searches candidate intercept times instead of trying to solve everything analytically.
        """

        candidate_intercept_times = np.linspace(
            start=now, 
            stop=now + SUBSEQUENT_INTERCEPT_MAX_LOOKAHEAD, 
            num=SUBSEQUENT_INTERCEPT_MAX_NUM_CANDIDATES
        )

        cost_weights = SUBSEQUENT_INTERCEPT_PLAN_COST_WEIGHTS 

        warm_start_servo_angles = self.active_plan.raw_servo_angles if self.active_plan is not None else None
        return self._make_best_valid_intercept_plan_from_candidates(
            tracker, now, PlanType.SUBSEQUENT_INTERCEPT, candidate_intercept_times, cost_weights,
            min_ready_margin=MIN_SUBSEQUENT_INTERCEPT_READY_MARGIN, initial_aim_guess=warm_start_servo_angles
        )


    def _make_best_valid_first_intercept_plan(self, tracker: SingleObjectTracker, now, fixed_intercept_time=None) -> tuple[bool, Plan | None]:
        """
        Build the best FIRST_INTERCEPT plan.
        Might have way different implementation than SUBSEQUENT_INTERCEPT plans later.

        Returns:
        (True, plan)  if a feasible first plan was found
        (False, None) otherwise

        Important timing meaning:
        intercept_time = when object and foam meet
        trigger_time = when platform must trigger/release
        expected_ready_time = when pan/tilt is expected to be aimed and settled

        This function intentionally searches candidate intercept times instead of trying to solve everything analytically.
        """

        cost_weights = FIRST_INTERCEPT_PLAN_COST_WEIGHTS

        if fixed_intercept_time is not None:
            candidate_intercept_times = np.linspace(
                start=fixed_intercept_time - FIRST_INTERCEPT_REFRESH_HALF_WINDOW,
                stop=fixed_intercept_time + FIRST_INTERCEPT_REFRESH_HALF_WINDOW,
                num=FIRST_INTERCEPT_REFRESH_NUM_CANDIDATES
            )
            warm_start_servo_angles = self.active_plan.raw_servo_angles if self.active_plan is not None else None
            return self._make_best_valid_intercept_plan_from_candidates(
                tracker, now, PlanType.FIRST_INTERCEPT, candidate_intercept_times, cost_weights,
                min_ready_margin=MIN_FIRST_INTERCEPT_READY_MARGIN, debug_rejections=True, search_label="REFRESH",
                initial_aim_guess=warm_start_servo_angles
            )

        # Coarse-to-fine search: first find the useful time region, then recover roughly the old 10 ms resolution locally.
        coarse_times = np.linspace(now, now + FIRST_INTERCEPT_MAX_LOOKAHEAD, num=FIRST_INTERCEPT_COARSE_NUM_CANDIDATES)
        coarse_valid, coarse_plan = self._make_best_valid_intercept_plan_from_candidates(
            tracker, now, PlanType.FIRST_INTERCEPT, coarse_times, cost_weights,
            min_ready_margin=MIN_FIRST_INTERCEPT_READY_MARGIN, debug_rejections=True, search_label="COARSE"
        )
        if not coarse_valid:
            return False, None

        coarse_spacing = FIRST_INTERCEPT_MAX_LOOKAHEAD/(FIRST_INTERCEPT_COARSE_NUM_CANDIDATES - 1)
        fine_start = max(now, coarse_plan.intercept_time - coarse_spacing)
        fine_stop = min(now + FIRST_INTERCEPT_MAX_LOOKAHEAD, coarse_plan.intercept_time + coarse_spacing)
        fine_times = np.linspace(fine_start, fine_stop, num=FIRST_INTERCEPT_FINE_NUM_CANDIDATES)

        return self._make_best_valid_intercept_plan_from_candidates(
            tracker, now, PlanType.FIRST_INTERCEPT, fine_times, cost_weights,
            min_ready_margin=MIN_FIRST_INTERCEPT_READY_MARGIN, debug_rejections=False, search_label="FINE",
            initial_aim_guess=coarse_plan.raw_servo_angles
        )



    def _plan_cost(self, now, plan: Plan, weights: dict[str, float], min_ready_margin, uncertainty_ratio) -> float:
        """
            Generic plan cost.

            Lower cost is better.

            The caller controls behavior by passing different weights.
        """

        if plan is None:
            return np.inf
        
        if (not np.isfinite(plan.trigger_time)) or (not np.isfinite(plan.intercept_time)) or (not np.isfinite(plan.ready_time)):
            return np.inf

        raw_servo_angles = np.asarray(plan.raw_servo_angles, dtype=float).reshape(-1).copy()

        # 1. Time-to-trigger cost
        
        time_to_trigger_cost = np.inf
        time_to_trigger = plan.trigger_time - now

        if time_to_trigger <= 0.0:
            if not self._close_to_trigger_time(now, plan.trigger_time):
                return np.inf
            
            time_to_trigger_cost = PLAN_COST_NOW_TRIGGER_TIME_COST
        else:
            time_to_trigger_cost = time_to_trigger / PLAN_COST_TIME_SCALE
            
        
        # 2. Servo motion cost

        last_cmd_servo_angles = self.comm_buffer.get_last_cmd_servo_angles()
        
        if last_cmd_servo_angles is None:
            q_ref = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).reshape(-1).copy()
        else:
            q_ref = np.asarray(last_cmd_servo_angles, dtype=float).reshape(-1).copy()

        servo_scales = np.asarray(PLAN_COST_SERVO_ANGLE_SCALES, dtype=float).reshape(-1).copy()
        if np.any(servo_scales <= 0.0):
            raise ValueError("PLAN_COST_SERVO_ANGLE_SCALES must be > 0")

        servo_error = raw_servo_angles - q_ref
        servo_motion_cost = float(np.clip(np.linalg.norm(servo_error / servo_scales), 0.0, 1.0))

        # 3. Ready margin cost
        
        ready_margin_cost = np.inf
        ready_margin = plan.trigger_time - plan.ready_time
        
        if ready_margin < min_ready_margin:
            return np.inf

        ready_margin_surplus = ready_margin - min_ready_margin

        ready_margin_cost = 1.0 / (1.0 + ready_margin_surplus / PLAN_COST_READY_MARGIN_SCALE)

        # 4. Intercept continuity cost
        continuity_cost = 0.0

        if self.active_plan is not None:
            p_new = np.asarray(plan.intercept_position, dtype=float).reshape(-1).copy()
            p_old = np.asarray(self.active_plan.intercept_position, dtype=float).reshape(-1).copy()

            # If the new candidate has a bad intercept point, reject it.
            if not np.all(np.isfinite(p_new)):
                return np.inf
            
            # If the old active plan has a bad intercept point, don't punish the new plan.
            # Just skip the continuity cost.
            if np.all(np.isfinite(p_old)):
                position_scales = np.asarray(PLAN_COST_INTERCEPT_POSITION_SCALES, dtype=float).reshape(-1).copy()
                # TODO: once continuity-cost distributions are logged, replace this hard clip
                # with a smoother saturation whose scale is tuned from observed plan data.
                continuity_cost = float(np.clip(np.linalg.norm((p_new - p_old) / position_scales), 0.0, 1.0))

        # 5. Uncertainty risk cost. Smoothstep maps the tuned uncertainty-ratio interval to [0, 1]
        # and saturates outside it, so the weight controls the maximum contribution.
        if not np.isfinite(uncertainty_ratio):
            return np.inf
        if PLAN_COST_UNCERTAINTY_RISK_FULL_RATIO <= PLAN_COST_UNCERTAINTY_RISK_START_RATIO:
            raise ValueError("PLAN_COST_UNCERTAINTY_RISK_FULL_RATIO must be > PLAN_COST_UNCERTAINTY_RISK_START_RATIO")

        uncertainty_risk_x = np.clip(
            (uncertainty_ratio - PLAN_COST_UNCERTAINTY_RISK_START_RATIO)
            /(PLAN_COST_UNCERTAINTY_RISK_FULL_RATIO - PLAN_COST_UNCERTAINTY_RISK_START_RATIO),
            0.0, 1.0
        )
        uncertainty_risk_cost = float(uncertainty_risk_x**2 * (3.0 - 2.0*uncertainty_risk_x))

        # 6. Total weighted cost
        cost = float(
            weights.get("time", 0.0) * time_to_trigger_cost
            + weights.get("servo_motion", 0.0) * servo_motion_cost
            + weights.get("ready_margin", 0.0) * ready_margin_cost
            + weights.get("continuity", 0.0) * continuity_cost
            + weights.get("uncertainty_risk", 0.0) * uncertainty_risk_cost
        )

        if not np.isfinite(cost):
            return np.inf
        
        return cost


    def _make_best_valid_intercept_plan_from_candidates(self, tracker: SingleObjectTracker, now, plan_type: PlanType, candidate_intercept_times, cost_weights, min_ready_margin, debug_rejections=False, search_label=None, initial_aim_guess=None) -> tuple[bool, Plan | None]:
        search_start_time = time.perf_counter()

        debug_full_first_search = plan_type == PlanType.FIRST_INTERCEPT and not debug_rejections
        rejection_counts = {"backward": 0, "prediction": 0, "aim": 0, "missed_trigger": 0, "ready_margin": 0, "cost": 0}

        # First compute the expensive target/trajectory results. Do not continuously update "now" here:
        # all candidates should later be compared against one common decision time.
        computed_candidates = []
        warm_start_servo_angles = None if initial_aim_guess is None else np.asarray(initial_aim_guess, dtype=float).reshape(-1).copy()
        for intercept_time in candidate_intercept_times:
            dt = intercept_time - tracker.track.state_time
            if dt <= 0.0:
                rejection_counts["backward"] += 1
                continue

            predicted_state, transverse_uncertainty_m, range_uncertainty_m = tracker.predict(dt, include_uncertainty=True)
            object_position_world = predicted_state[:3].copy()
            if (not np.all(np.isfinite(object_position_world))
                    or transverse_uncertainty_m is None or not np.isfinite(transverse_uncertainty_m)
                    or range_uncertainty_m is None or not np.isfinite(range_uncertainty_m)):
                rejection_counts["prediction"] += 1
                continue

            uncertainty_ratio = max(
                transverse_uncertainty_m/MAX_TRIGGER_TRANSVERSE_UNCERTAINTY_M,
                range_uncertainty_m/MAX_TRIGGER_RANGE_UNCERTAINTY_M
            )

            object_position_platform = estimateObjectPlatformPosition(object_position_world, self.camera_to_platform_calibration)
            angles_valid, q_raw, foam_flight_time = self._object_position_to_servo_angles_and_flight_time(
                object_position_platform, initial_servo_angles=warm_start_servo_angles
            )
            if not angles_valid:
                rejection_counts["aim"] += 1
                continue

            # Adjacent intercept candidates are close in time/space, so reuse the last solved aim as the next initial guess.
            warm_start_servo_angles = q_raw.copy()
            computed_candidates.append((
                intercept_time,
                object_position_world,
                uncertainty_ratio,
                q_raw,
                foam_flight_time
            ))

        physics_done_time = time.perf_counter()

        # Take one fresh common timestamp after the expensive work. Candidates computed first and last are therefore
        # judged consistently, while any trigger opportunities that expired during computation are rejected here.
        decision_now = physics_done_time
        q_start = self._planning_servo_angles()

        best_plan = None
        best_cost = np.inf
        earliest_feasible_diag = None
        max_margin_diag = None
        best_diag = None
        best_failed_ready_margin = -np.inf

        for intercept_time, object_position_world, uncertainty_ratio, q_raw, foam_flight_time in computed_candidates:
            trigger_time = intercept_time - foam_flight_time - TRIGGER_DELAY
            if trigger_time < decision_now and not self._close_to_trigger_time(decision_now, trigger_time=trigger_time):
                rejection_counts["missed_trigger"] += 1
                continue

            servo_rotation_time = self._estimate_servo_rotation_time(q_from=q_start, q_to=q_raw)
            expected_ready_time = decision_now + servo_rotation_time + config.CMD_THREAD_MAX_DELAY + config.UDP_TX_DELAY + config.ENDPOINT_CMD_MAX_DELAY
            ready_margin = trigger_time - expected_ready_time

            if ready_margin < min_ready_margin:
                rejection_counts["ready_margin"] += 1
                best_failed_ready_margin = max(best_failed_ready_margin, ready_margin)
                continue

            plan = Plan(
                track_id=tracker.track.id,
                plan_type=plan_type,
                raw_servo_angles=q_raw,
                estimate_time=tracker.track.state_time,
                ready_time=expected_ready_time,
                intercept_position=object_position_world,
                intercept_time=intercept_time,
                trigger_time=trigger_time
            )

            cost = self._plan_cost(decision_now, plan, cost_weights, min_ready_margin, uncertainty_ratio)

            if debug_full_first_search and np.isfinite(cost):
                time_cost = PLAN_COST_NOW_TRIGGER_TIME_COST if trigger_time <= decision_now else (trigger_time - decision_now)/PLAN_COST_TIME_SCALE
                servo_cost = float(np.clip(np.linalg.norm((q_raw - q_start)/PLAN_COST_SERVO_ANGLE_SCALES), 0.0, 1.0))
                ready_margin_cost = 1.0/(1.0 + (ready_margin - min_ready_margin)/PLAN_COST_READY_MARGIN_SCALE)
                uncertainty_x = np.clip(
                    (uncertainty_ratio - PLAN_COST_UNCERTAINTY_RISK_START_RATIO)
                    /(PLAN_COST_UNCERTAINTY_RISK_FULL_RATIO - PLAN_COST_UNCERTAINTY_RISK_START_RATIO),
                    0.0, 1.0
                )
                uncertainty_cost = float(uncertainty_x**2 * (3.0 - 2.0*uncertainty_x))

                diag = (
                    (trigger_time - decision_now)*1000.0, ready_margin*1000.0, cost,
                    cost_weights.get("time", 0.0)*time_cost,
                    cost_weights.get("servo_motion", 0.0)*servo_cost,
                    cost_weights.get("ready_margin", 0.0)*ready_margin_cost,
                    cost_weights.get("uncertainty_risk", 0.0)*uncertainty_cost,
                    uncertainty_ratio
                )

                if earliest_feasible_diag is None:
                    earliest_feasible_diag = diag
                if max_margin_diag is None or diag[1] > max_margin_diag[1]:
                    max_margin_diag = diag

            if cost < best_cost:
                best_cost = cost
                best_plan = plan
                if debug_full_first_search:
                    best_diag = diag
            elif not np.isfinite(cost):
                rejection_counts["cost"] += 1

        search_done_time = time.perf_counter()
        label = f"{plan_type.name} {search_label}" if search_label else plan_type.name
        print(
            f"PLAN SEARCH TIME | {label} | "
            f"compute={(physics_done_time - search_start_time)*1000:.1f} ms | "
            f"total={(search_done_time - search_start_time)*1000:.1f} ms | "
            f"computed={len(computed_candidates)}/{len(candidate_intercept_times)}"
        ) # FOR DEBUG ONLY

        if debug_rejections and best_plan is None:
            ready_detail = ""
            if rejection_counts["ready_margin"] > 0 and np.isfinite(best_failed_ready_margin):
                ready_detail = f" | best ready margin {best_failed_ready_margin*1000:.1f} ms (need {min_ready_margin*1000:.1f})"

            print(
                f"INTERCEPT WINDOW FAILED | "
                f"backward={rejection_counts['backward']} | "
                f"prediction={rejection_counts['prediction']} | "
                f"aim={rejection_counts['aim']} | "
                f"missed_trigger={rejection_counts['missed_trigger']} | "
                f"ready_margin={rejection_counts['ready_margin']} | "
                f"cost={rejection_counts['cost']}"
                f"{ready_detail}"
            ) # FOR DEBUG ONLY

        if debug_full_first_search and best_diag is not None:
            print(
                f"FIRST SEARCH DIAG | ready_w={cost_weights.get('ready_margin', 0.0):.3f} | "
                f"EARLIEST trigger={earliest_feasible_diag[0]:.1f}ms margin={earliest_feasible_diag[1]:.1f}ms cost={earliest_feasible_diag[2]:.3f} "
                f"[time={earliest_feasible_diag[3]:.3f} servo={earliest_feasible_diag[4]:.3f} margin={earliest_feasible_diag[5]:.3f} risk={earliest_feasible_diag[6]:.3f} ratio={earliest_feasible_diag[7]:.2f}]"
            )
            print(
                f"FIRST SEARCH DIAG | WINNER   trigger={best_diag[0]:.1f}ms margin={best_diag[1]:.1f}ms cost={best_diag[2]:.3f} "
                f"[time={best_diag[3]:.3f} servo={best_diag[4]:.3f} margin={best_diag[5]:.3f} risk={best_diag[6]:.3f} ratio={best_diag[7]:.2f}]"
            )
            print(
                f"FIRST SEARCH DIAG | MAXMARGIN trigger={max_margin_diag[0]:.1f}ms margin={max_margin_diag[1]:.1f}ms cost={max_margin_diag[2]:.3f} "
                f"[time={max_margin_diag[3]:.3f} servo={max_margin_diag[4]:.3f} margin={max_margin_diag[5]:.3f} risk={max_margin_diag[6]:.3f} ratio={max_margin_diag[7]:.2f}]"
            )

        return best_plan is not None, best_plan


    def _generate_dart_trajectory_table(self):
        """Precompute z(x) and t(x) once for each launch tilt; live planning only interpolates this table."""
        start = time.perf_counter()
        v0, g, k, dt = float(DART_PROTRUSION_SPEED), float(GRAVITY), float(DART_DRAG_K), float(DART_SIMULATION_DT)
        if v0 <= 0.0 or g < 0.0 or k < 0.0 or dt <= 0.0:
            raise ValueError("Invalid dart simulation constants")

        x_step = float(DART_TRAJECTORY_TABLE_X_STEP)
        if x_step <= 0.0 or DART_TRAJECTORY_TABLE_MAX_FORWARD_RANGE <= 0.0 or DART_TRAJECTORY_TABLE_TILT_STEP_DEG <= 0.0:
            raise ValueError("Invalid dart trajectory table constants")

        self._dart_trajectory_table_tilts_rad = np.deg2rad(np.arange(
            DART_TRAJECTORY_TABLE_MIN_TILT_DEG,
            DART_TRAJECTORY_TABLE_MAX_TILT_DEG + 0.5*DART_TRAJECTORY_TABLE_TILT_STEP_DEG,
            DART_TRAJECTORY_TABLE_TILT_STEP_DEG,
            dtype=float
        ))
        num_x = int(round(DART_TRAJECTORY_TABLE_MAX_FORWARD_RANGE/x_step)) + 1
        shape = (self._dart_trajectory_table_tilts_rad.size, num_x)
        self._dart_trajectory_table_height_m = np.full(shape, np.nan, dtype=np.float32)
        self._dart_trajectory_table_time_s = np.full(shape, np.nan, dtype=np.float32)
        self._dart_trajectory_table_height_m[:, 0] = 0.0
        self._dart_trajectory_table_time_s[:, 0] = 0.0

        max_steps = int(math.ceil(DART_TRAJECTORY_TABLE_MAX_GENERATION_TIME/dt))
        for tilt_idx, theta_rad in enumerate(self._dart_trajectory_table_tilts_rad):
            x = z = t = 0.0
            vx, vz = v0*math.cos(theta_rad), v0*math.sin(theta_rad)
            next_x_idx = 1
            if vx <= 1e-9:
                continue

            for _ in range(max_steps):
                speed = math.hypot(vx, vz)
                ax, az = -k*speed*vx, -g - k*speed*vz
                next_x, next_z, next_t = x + vx*dt, z + vz*dt, t + dt

                if next_x > x:
                    while next_x_idx < num_x and next_x_idx*x_step <= next_x + 1e-12:
                        query_x = next_x_idx*x_step
                        alpha = min(max((query_x - x)/(next_x - x), 0.0), 1.0)
                        self._dart_trajectory_table_height_m[tilt_idx, next_x_idx] = z + alpha*(next_z - z)
                        self._dart_trajectory_table_time_s[tilt_idx, next_x_idx] = t + alpha*dt
                        next_x_idx += 1

                vx += ax*dt
                vz += az*dt
                x, z, t = next_x, next_z, next_t
                if next_x_idx >= num_x or vx <= 0.0 or (z < DART_TRAJECTORY_TABLE_MIN_TARGET_UP and vz < 0.0):
                    break

        # Per-range branch metadata makes the live inverse lookup scalar/logarithmic instead of scanning every tilt.
        valid = np.isfinite(self._dart_trajectory_table_height_m)
        self._dart_trajectory_table_first_valid_tilt_idx = np.argmax(valid, axis=0).astype(np.int16)
        self._dart_trajectory_table_last_valid_tilt_idx = (shape[0] - 1 - np.argmax(valid[::-1], axis=0)).astype(np.int16)
        self._dart_trajectory_table_peak_tilt_idx = np.argmax(np.where(valid, self._dart_trajectory_table_height_m, -np.inf), axis=0).astype(np.int16)

        memory_bytes = (
            self._dart_trajectory_table_height_m.nbytes + self._dart_trajectory_table_time_s.nbytes
            + self._dart_trajectory_table_first_valid_tilt_idx.nbytes + self._dart_trajectory_table_last_valid_tilt_idx.nbytes
            + self._dart_trajectory_table_peak_tilt_idx.nbytes
        )
        print(
            f"DART TRAJECTORY TABLE | tilts={shape[0]} | x={shape[1]} | "
            f"memory={memory_bytes/(1024.0*1024.0):.2f} MB | generation={(time.perf_counter() - start)*1000.0:.1f} ms"
        ) # FOR DEBUG ONLY


    def _solve_dart_tilt_and_flight_time(self, horizontal_range: float, target_up: float, initial_theta_rad=None) -> tuple[bool, float | None, float | None]:
        """Fast inverse lookup on the precomputed trajectory table; warm-neighbor probe with binary-search fallback."""
        if (
            horizontal_range <= 0.0 or not np.isfinite(horizontal_range) or not np.isfinite(target_up)
            or horizontal_range > DART_TRAJECTORY_TABLE_MAX_FORWARD_RANGE
            or target_up < DART_TRAJECTORY_TABLE_MIN_TARGET_UP or target_up > DART_TRAJECTORY_TABLE_MAX_TARGET_UP
        ):
            return False, None, None

        heights_table, times_table = self._dart_trajectory_table_height_m, self._dart_trajectory_table_time_s
        tilts = self._dart_trajectory_table_tilts_rad
        x_position = horizontal_range/DART_TRAJECTORY_TABLE_X_STEP
        x0_idx = int(math.floor(x_position))
        x1_idx = min(x0_idx + 1, heights_table.shape[1] - 1)
        x_alpha = x_position - x0_idx

        # A row is usable only if both neighboring x samples exist.
        first_idx = max(int(self._dart_trajectory_table_first_valid_tilt_idx[x0_idx]), int(self._dart_trajectory_table_first_valid_tilt_idx[x1_idx]))
        last_idx = min(int(self._dart_trajectory_table_last_valid_tilt_idx[x0_idx]), int(self._dart_trajectory_table_last_valid_tilt_idx[x1_idx]))
        if first_idx >= last_idx:
            return False, None, None

        def sample(idx: int):
            z0, t0 = float(heights_table[idx, x0_idx]), float(times_table[idx, x0_idx])
            if x1_idx == x0_idx:
                return (z0, t0) if math.isfinite(z0) and math.isfinite(t0) else None
            z1, t1 = float(heights_table[idx, x1_idx]), float(times_table[idx, x1_idx])
            if not (math.isfinite(z0) and math.isfinite(z1) and math.isfinite(t0) and math.isfinite(t1)):
                return None
            return z0 + x_alpha*(z1 - z0), t0 + x_alpha*(t1 - t0)

        # Adjacent x columns have nearly identical peak indices. Their interpolated peak is accurate enough to
        # bound a warm probe; the exact interpolated peak is only refined if that fast path does not bracket.
        peak_guess = int(round(
            (1.0 - x_alpha)*float(self._dart_trajectory_table_peak_tilt_idx[x0_idx])
            + x_alpha*float(self._dart_trajectory_table_peak_tilt_idx[x1_idx])
        ))
        peak_guess = min(max(peak_guess, first_idx), last_idx)

        if initial_theta_rad is not None and np.isfinite(initial_theta_rad):
            warm_lo, warm_hi = (peak_guess, last_idx) if USE_HIGH_ARC else (first_idx, peak_guess)
            tilt_step = tilts[1] - tilts[0]
            guess_idx = int(round((float(initial_theta_rad) - tilts[0])/tilt_step))
            guess_idx = min(max(guess_idx, warm_lo), warm_hi)
            guess_sample = sample(guess_idx)
            if guess_sample is not None:
                guess_error = guess_sample[0] - target_up
                if abs(guess_error) <= 1e-12:
                    return True, float(tilts[guess_idx]), float(guess_sample[1])
                direction = (-1 if guess_error > 0.0 else 1) if not USE_HIGH_ARC else (1 if guess_error > 0.0 else -1)
                neighbor_idx = guess_idx + direction
                if warm_lo <= neighbor_idx <= warm_hi:
                    neighbor_sample = sample(neighbor_idx)
                    if neighbor_sample is not None and guess_error*(neighbor_sample[0] - target_up) <= 0.0:
                        if guess_idx < neighbor_idx:
                            return self._interpolate_dart_trajectory_table_rows(guess_idx, neighbor_idx, guess_sample, neighbor_sample, target_up)
                        return self._interpolate_dart_trajectory_table_rows(neighbor_idx, guess_idx, neighbor_sample, guess_sample, target_up)

        # Robust fallback: hill-climb a few scalar samples to the exact peak of the x-interpolated trajectory family.
        peak_idx = peak_guess
        peak_sample = sample(peak_idx)
        if peak_sample is None:
            return False, None, None
        while peak_idx > first_idx:
            left_sample = sample(peak_idx - 1)
            if left_sample is None or left_sample[0] <= peak_sample[0]:
                break
            peak_idx, peak_sample = peak_idx - 1, left_sample
        while peak_idx < last_idx:
            right_sample = sample(peak_idx + 1)
            if right_sample is None or right_sample[0] <= peak_sample[0]:
                break
            peak_idx, peak_sample = peak_idx + 1, right_sample

        if target_up > peak_sample[0] + TRAJECTORY_HEIGHT_TOLERANCE:
            return False, None, None

        branch_lo, branch_hi = (peak_idx, last_idx) if USE_HIGH_ARC else (first_idx, peak_idx)
        lo_sample, hi_sample = sample(branch_lo), sample(branch_hi)
        if lo_sample is None or hi_sample is None:
            return False, None, None

        # Low branch rises with tilt; high branch falls. Reject targets outside the selected branch's height span.
        if USE_HIGH_ARC:
            if target_up < hi_sample[0] - TRAJECTORY_HEIGHT_TOLERANCE or target_up > lo_sample[0] + TRAJECTORY_HEIGHT_TOLERANCE:
                return False, None, None
        else:
            if target_up < lo_sample[0] - TRAJECTORY_HEIGHT_TOLERANCE or target_up > hi_sample[0] + TRAJECTORY_HEIGHT_TOLERANCE:
                return False, None, None

        # Warm-start: if the prior solution and one neighboring row already straddle the target, finish immediately.
        if initial_theta_rad is not None and np.isfinite(initial_theta_rad):
            tilt_step = tilts[1] - tilts[0]
            guess_idx = int(round((float(initial_theta_rad) - tilts[0])/tilt_step))
            guess_idx = min(max(guess_idx, branch_lo), branch_hi)
            guess_sample = sample(guess_idx)
            if guess_sample is not None:
                guess_error = guess_sample[0] - target_up
                if abs(guess_error) <= 1e-12:
                    return True, float(tilts[guess_idx]), float(guess_sample[1])
                direction = (-1 if guess_error > 0.0 else 1) if not USE_HIGH_ARC else (1 if guess_error > 0.0 else -1)
                neighbor_idx = guess_idx + direction
                if branch_lo <= neighbor_idx <= branch_hi:
                    neighbor_sample = sample(neighbor_idx)
                    if neighbor_sample is not None:
                        neighbor_error = neighbor_sample[0] - target_up
                        if guess_error*neighbor_error <= 0.0:
                            lower_idx, upper_idx = sorted((guess_idx, neighbor_idx))
                            lower_sample, upper_sample = sample(lower_idx), sample(upper_idx)
                            return self._interpolate_dart_trajectory_table_rows(lower_idx, upper_idx, lower_sample, upper_sample, target_up)

                # The warm row still shrinks the binary-search interval even when its immediate neighbor does not bracket.
                if not USE_HIGH_ARC:
                    if guess_error < 0.0:
                        branch_lo, lo_sample = guess_idx, guess_sample
                    else:
                        branch_hi, hi_sample = guess_idx, guess_sample
                else:
                    if guess_error > 0.0:
                        branch_lo, lo_sample = guess_idx, guess_sample
                    else:
                        branch_hi, hi_sample = guess_idx, guess_sample

        # Binary search the monotonic selected branch until two adjacent table tilts straddle the requested height.
        while branch_hi - branch_lo > 1:
            mid_idx = (branch_lo + branch_hi)//2
            mid_sample = sample(mid_idx)
            if mid_sample is None:
                return False, None, None
            mid_error = mid_sample[0] - target_up
            if abs(mid_error) <= 1e-12:
                return True, float(tilts[mid_idx]), float(mid_sample[1])
            if (mid_error < 0.0) != USE_HIGH_ARC:
                branch_lo, lo_sample = mid_idx, mid_sample
            else:
                branch_hi, hi_sample = mid_idx, mid_sample

        return self._interpolate_dart_trajectory_table_rows(branch_lo, branch_hi, lo_sample, hi_sample, target_up)


    def _interpolate_dart_trajectory_table_rows(self, idx0: int, idx1: int, sample0, sample1, target_up: float):
        """Interpolate tilt and flight time between two already-sampled neighboring trajectory rows."""
        if sample0 is None or sample1 is None:
            return False, None, None
        z0, t0 = sample0
        z1, t1 = sample1
        denom = z1 - z0
        alpha = 0.0 if abs(denom) <= 1e-12 else min(max((target_up - z0)/denom, 0.0), 1.0)
        theta = float(self._dart_trajectory_table_tilts_rad[idx0] + alpha*(self._dart_trajectory_table_tilts_rad[idx1] - self._dart_trajectory_table_tilts_rad[idx0]))
        flight_time = float(t0 + alpha*(t1 - t0))
        return (True, theta, flight_time) if math.isfinite(theta) and math.isfinite(flight_time) and flight_time > 0.0 else (False, None, None)


    def _solve_dart_tilt_and_flight_time_euler(self, horizontal_range: float, target_up: float, initial_theta_rad=None) -> tuple[bool, float | None, float | None]:
        """Numerical inverse solver retained for debug/validation; live planning uses the precomputed table."""

        if horizontal_range <= 0.0 or not np.isfinite(horizontal_range) or not np.isfinite(target_up):
            return False, None, None

        v0 = float(DART_PROTRUSION_SPEED)
        g = float(GRAVITY)
        k = float(DART_DRAG_K)
        dt = float(DART_SIMULATION_DT)
        if v0 <= 0.0 or g < 0.0 or k < 0.0 or dt <= 0.0:
            raise ValueError("Invalid dart simulation constants")

        # Planner lookahead already bounds any useful dart flight; longer flights cannot produce a feasible trigger.
        max_steps = int(math.ceil(max(FIRST_INTERCEPT_MAX_LOOKAHEAD, SUBSEQUENT_INTERCEPT_MAX_LOOKAHEAD)/dt))

        def simulate(theta_rad: float) -> tuple[float, float] | None:
            vx = v0*math.cos(theta_rad)
            vz = v0*math.sin(theta_rad)
            if vx <= 1e-9:
                return None

            x = z = t = 0.0
            for _ in range(max_steps):
                speed = math.hypot(vx, vz)
                ax = -k*speed*vx
                az = -g - k*speed*vz

                # Explicit Euler. Interpolate the target-range crossing so height/time are not quantized to dt.
                next_x = x + vx*dt
                next_z = z + vz*dt
                if next_x >= horizontal_range:
                    dx = next_x - x
                    if dx <= 1e-12:
                        return None
                    alpha = min(max((horizontal_range - x)/dx, 0.0), 1.0)
                    return z + alpha*(next_z - z), t + alpha*dt

                vx += ax*dt
                vz += az*dt
                x, z, t = next_x, next_z, t + dt
                if vx <= 0.0:
                    return None

            return None

        # Drag cannot make a point reachable if the same v0 cannot reach it without drag.
        A = g*horizontal_range*horizontal_range/(2.0*v0*v0)
        if A <= 1e-12:
            theta_low_no_drag = theta_high_no_drag = math.atan2(target_up, horizontal_range)
        else:
            discriminant = horizontal_range*horizontal_range - 4.0*A*(A + target_up)
            if discriminant < -1e-9:
                return False, None, None
            sqrt_disc = math.sqrt(max(discriminant, 0.0))
            theta_low_no_drag = math.atan((horizontal_range - sqrt_disc)/(2.0*A))
            theta_high_no_drag = math.atan((horizontal_range + sqrt_disc)/(2.0*A))

        if USE_HIGH_ARC:
            # High arc is intentionally slower: scan for both roots and select the later-angle crossing.
            theta = math.atan2(target_up, horizontal_range)
            prev_result = simulate(theta)
            if prev_result is None:
                return False, None, None
            prev_f = prev_result[0] - target_up
            brackets = []
            step = math.radians(0.5)
            while theta + step < math.radians(89.5):
                next_theta = theta + step
                result = simulate(next_theta)
                if result is None:
                    break
                f = result[0] - target_up
                if prev_f*f <= 0.0:
                    brackets.append((theta, prev_f, next_theta, f))
                theta, prev_f = next_theta, f
            if len(brackets) < 2:
                return False, None, None
            theta_lo, f_lo, theta_hi, f_hi = brackets[-1]
        else:
            theta_los = math.atan2(target_up, horizontal_range)
            max_theta = min(theta_high_no_drag, math.radians(89.0))
            bracket_found = False

            # Warm start from the previous nearby solution. Search outward only as far as needed, then fall back below.
            if initial_theta_rad is not None and np.isfinite(initial_theta_rad) and max_theta >= theta_los:
                theta_guess = float(np.clip(initial_theta_rad, theta_los, max_theta))
                result_guess = simulate(theta_guess)
                if result_guess is not None:
                    f_guess = result_guess[0] - target_up
                    if abs(f_guess) <= TRAJECTORY_HEIGHT_TOLERANCE:
                        return True, theta_guess, result_guess[1]

                    step = math.radians(0.25)
                    if f_guess < 0.0:
                        theta_lo, f_lo = theta_guess, f_guess
                        while theta_lo < max_theta - 1e-12:
                            theta_hi = min(theta_lo + step, max_theta)
                            result_hi = simulate(theta_hi)
                            if result_hi is None:
                                break
                            f_hi = result_hi[0] - target_up
                            if f_lo*f_hi <= 0.0:
                                bracket_found = True
                                break
                            theta_lo, f_lo = theta_hi, f_hi
                            step *= 2.0
                    else:
                        theta_hi, f_hi = theta_guess, f_guess
                        while theta_hi > theta_los + 1e-12:
                            theta_lo = max(theta_hi - step, theta_los)
                            result_lo = simulate(theta_lo)
                            if result_lo is None:
                                break
                            f_lo = result_lo[0] - target_up
                            if f_lo*f_hi <= 0.0:
                                bracket_found = True
                                break
                            theta_hi, f_hi = theta_lo, f_lo
                            step *= 2.0

            if not bracket_found:
                theta_lo = theta_los
                result_lo = simulate(theta_lo)
                if result_lo is None:
                    return False, None, None
                f_lo = result_lo[0] - target_up
                if abs(f_lo) <= TRAJECTORY_HEIGHT_TOLERANCE:
                    return True, theta_lo, result_lo[1]

                # The no-drag low solution is a close starting point; step upward until drag trajectory brackets the target.
                theta_hi = max(theta_low_no_drag, theta_lo) + math.radians(1.0)
                while theta_hi <= max_theta + 1e-12:
                    result_hi = simulate(theta_hi)
                    if result_hi is None:
                        return False, None, None
                    f_hi = result_hi[0] - target_up
                    if f_lo*f_hi <= 0.0:
                        break
                    theta_hi += math.radians(1.0)
                else:
                    return False, None, None

        # False position converges quickly for this smooth low-arc problem.
        result = None
        for _ in range(MAX_TRAJECTORY_SOLVE_ITERATIONS):
            denom = f_hi - f_lo
            if abs(denom) <= 1e-12:
                break
            theta = (theta_lo*f_hi - theta_hi*f_lo)/denom
            result = simulate(theta)
            if result is None:
                return False, None, None
            f = result[0] - target_up
            if abs(f) <= TRAJECTORY_HEIGHT_TOLERANCE:
                return True, theta, result[1]
            if f_lo*f <= 0.0:
                theta_hi, f_hi = theta, f
            else:
                theta_lo, f_lo = theta, f

        # Rare fallback: finish robustly with bisection if false-position convergence stalls.
        for _ in range(MAX_TRAJECTORY_SOLVE_ITERATIONS):
            theta = 0.5*(theta_lo + theta_hi)
            result = simulate(theta)
            if result is None:
                return False, None, None
            f = result[0] - target_up
            if abs(f) <= TRAJECTORY_HEIGHT_TOLERANCE:
                return True, theta, result[1]
            if f_lo*f <= 0.0:
                theta_hi, f_hi = theta, f
            else:
                theta_lo, f_lo = theta, f

        return (True, theta, result[1]) if result is not None else (False, None, None)


    def _object_position_to_servo_angles_and_flight_time(self, position: np.ndarray, initial_servo_angles=None) -> tuple[bool, np.ndarray | None, float | None]:
        """
        Convert platform-frame object position to pan/tilt angles.

        Platform frame is FLU:
            +x = forward
            +y = left
            +z = up

        Returns:
            success, servo_angles, flight_time

        success = False means this object position is not aimable/reachable.
        """

        def fail() -> tuple[bool, np.ndarray | None, float | None]:
            return False, None, None

        position = np.asarray(position, dtype=float).reshape(-1).copy()
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return fail()

        # Lightweight fixed-point iteration accounts for the foam-mechanism origin moving with pan/tilt.
        # TODO: If geometry becomes more complicated, replace this with full numerical IK over pan/tilt/time.
        foam_origin_forward = self.platform_geometry_spec.foam_mechanism_origin_offset_m
        R_foam_forward = self.platform_geometry_spec.rotation_platform_from_foam_mechanism_at_forward
        foam_direction_forward = R_foam_forward@np.array([1.0, 0.0, 0.0])

        # Use a nearby solved aim when available; otherwise start from the platform-origin line of sight.
        target_theta_guess_rad = None
        if initial_servo_angles is not None:
            initial_servo_angles = np.asarray(initial_servo_angles, dtype=float).reshape(-1).copy()
            pan_idx, tilt_idx = config.SERVO_IDX["pan"], config.SERVO_IDX["tilt"]
            pan_sign, tilt_sign = config.SERVO_SIGNS[pan_idx], config.SERVO_SIGNS[tilt_idx]
            if initial_servo_angles.shape == (config.NUM_SERVOS,) and np.all(np.isfinite(initial_servo_angles)) and abs(pan_sign) > 1e-12 and abs(tilt_sign) > 1e-12:
                platform_yaw_rad = np.deg2rad((initial_servo_angles[pan_idx] - config.FORWARD_SERVO_ANGLES[pan_idx])/pan_sign)
                platform_theta_rad = np.deg2rad((initial_servo_angles[tilt_idx] - config.FORWARD_SERVO_ANGLES[tilt_idx])/tilt_sign)
                initial_foam_direction = rotationPlatformFromPanTilt(platform_yaw_rad, platform_theta_rad)@foam_direction_forward
                target_theta_guess_rad = np.arctan2(initial_foam_direction[2], np.hypot(initial_foam_direction[0], initial_foam_direction[1]))
            else:
                initial_servo_angles = None

        if initial_servo_angles is None:
            platform_yaw_rad = np.arctan2(position[1], position[0])
            platform_theta_rad = np.arctan2(position[2], np.hypot(position[0], position[1]))

        for _ in range(MAX_AIM_SOLVE_ITERATIONS):
            R_joint = rotationPlatformFromPanTilt(platform_yaw_rad, platform_theta_rad)
            foam_origin_platform = R_joint@foam_origin_forward
            relative_position = position - foam_origin_platform
            forward, left, up = relative_position

            if forward <= MIN_FORWARD_RANGE:
                return fail()

            target_yaw_rad = np.arctan2(left, forward)
            horizontal_range = np.hypot(forward, left)
            if horizontal_range <= 1e-9:
                return fail()

            trajectory_valid, target_theta_rad, _ = self._solve_dart_tilt_and_flight_time(
                horizontal_range, up, initial_theta_rad=target_theta_guess_rad
            )
            if not trajectory_valid:
                return fail()
            target_theta_guess_rad = target_theta_rad

            # Convert required exit direction into pan/tilt joint angles, compensating for fixed mechanism rotation.
            axis_x, axis_y, axis_z = foam_direction_forward
            axis_xz = np.hypot(axis_x, axis_z)
            if axis_xz <= 1e-9:
                return fail()

            target_z = np.sin(target_theta_rad)
            if abs(target_z) > axis_xz + 1e-9:
                return fail()

            foam_theta_offset_rad = np.arctan2(axis_z, axis_x)
            new_platform_theta_rad = np.arcsin(np.clip(target_z/axis_xz, -1.0, 1.0)) - foam_theta_offset_rad

            rotated_axis_x = np.cos(new_platform_theta_rad)*axis_x - np.sin(new_platform_theta_rad)*axis_z
            foam_yaw_offset_rad = np.arctan2(axis_y, rotated_axis_x)
            new_platform_yaw_rad = target_yaw_rad - foam_yaw_offset_rad
            new_platform_yaw_rad = (new_platform_yaw_rad + np.pi)%(2.0*np.pi) - np.pi

            yaw_change = (new_platform_yaw_rad - platform_yaw_rad + np.pi)%(2.0*np.pi) - np.pi
            theta_change = new_platform_theta_rad - platform_theta_rad
            platform_yaw_rad = new_platform_yaw_rad
            platform_theta_rad = new_platform_theta_rad

            if np.hypot(yaw_change, theta_change) < 1e-6:
                break

        # Recalculate the trajectory once from the final moving foam-exit position for the final flight time.
        R_joint = rotationPlatformFromPanTilt(platform_yaw_rad, platform_theta_rad)
        foam_origin_platform = R_joint@foam_origin_forward
        relative_position = position - foam_origin_platform
        forward, left, up = relative_position
        if forward <= MIN_FORWARD_RANGE:
            return fail()

        horizontal_range = np.hypot(forward, left)
        if horizontal_range <= 1e-9:
            return fail()

        trajectory_valid, _, flight_time = self._solve_dart_tilt_and_flight_time(
            horizontal_range, up, initial_theta_rad=target_theta_guess_rad
        )
        if not trajectory_valid or flight_time is None or not np.isfinite(flight_time) or flight_time <= 0.0:
            return fail()

        # Convert platform yaw/elevation angles into servo commands.
        platform_yaw_deg = np.rad2deg(platform_yaw_rad)
        platform_theta_deg = np.rad2deg(platform_theta_rad)

        pan_angle = config.FORWARD_SERVO_ANGLES[config.SERVO_IDX["pan"]] + config.SERVO_SIGNS[config.SERVO_IDX["pan"]]*platform_yaw_deg
        tilt_angle = config.FORWARD_SERVO_ANGLES[config.SERVO_IDX["tilt"]] + config.SERVO_SIGNS[config.SERVO_IDX["tilt"]]*platform_theta_deg

        servo_angles = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
        servo_angles[config.SERVO_IDX["pan"]] = pan_angle
        servo_angles[config.SERVO_IDX["tilt"]] = tilt_angle

        if not np.all(np.isfinite(servo_angles)):
            return fail()

        min_angles = np.asarray(config.MIN_SERVO_ANGLES, dtype=float)
        max_angles = np.asarray(config.MAX_SERVO_ANGLES, dtype=float)
        outside_limits = (servo_angles < min_angles) | (servo_angles > max_angles)
        if np.any(outside_limits):
            return fail()

        return True, servo_angles, flight_time


    def _estimate_servo_rotation_time(self, q_from: np.ndarray, q_to: np.ndarray):
        """
        Conservative estimate of time until the servos are aimed and settled.

        Accounts approximately for command speed limiting, exponential command smoothing,
        and mechanical settling. Pan/tilt move simultaneously, so the slowest joint determines
        the total time.
        """

        if q_from.shape != q_to.shape or q_from.shape != (config.NUM_SERVOS,):
            raise ValueError("q_from or q_to shape does not match NUM_SERVOS")

        if not np.all(np.isfinite(q_from)) or not np.all(np.isfinite(q_to)):
            raise ValueError("q_from or q_to has non-finite values")
        
        q_from = np.asarray(q_from, dtype=float).reshape(-1).copy()
        q_to = np.asarray(q_to, dtype=float).reshape(-1).copy()
        dq = np.abs(q_to - q_from)

        # Motion inside the deadband will not produce a new servo command.
        deadband = np.asarray(config.SERVO_DEADBAND, dtype=float).copy()
        moving = dq > deadband
        if not np.any(moving):
            return 0.0

        # Minimum travel time imposed by the command-side servo speed limits.
        servo_speeds = np.asarray(config.MAX_SERVO_SPEEDS, dtype=float).copy()
        joint_times = dq/servo_speeds

        # Approximate additional time for the exponentially smoothed command to decay
        # from the initial error dq to the deadband: t = tau*ln(dq/deadband).
        # Adding this to the speed-limit time intentionally gives a conservative estimate.
        if config.CMD_SMOOTHING_TAU > 0.0:
            joint_times[moving] += config.CMD_SMOOTHING_TAU*np.log(dq[moving]/deadband[moving])

        # Joints move together, so wait for the slowest one, then allow mechanical settling.
        rotation_time = float(np.max(joint_times))

        # Estimate settling time as linear scaling up to a max
        max_angle_change = float(np.max(dq))
        settling_time = MAX_SERVO_SETTLING_TIME * min(max_angle_change / SERVO_ANGLE_CHANGE_FOR_MAX_SETTLING, 1.0)

        return rotation_time + settling_time + SERVO_ROTATION_TIME_MARGIN

    
    def _active_plan_still_valid(self, tracker: SingleObjectTracker, now) -> bool:
        """
        Still valid if the updated prediction at the planned intercept time does not
        materially change either the required servo angles or foam flight time.

        ToDo (not now, but maybe later): add additional validity checks
        """

        dt = self.active_plan.intercept_time - tracker.track.state_time
        if dt <= 0.0:
            return False

        if self.active_plan.trigger_time < now and not self._close_to_trigger_time(now):
            print(f"PLAN INVALID: missed trigger by {(now - self.active_plan.trigger_time)*1000:.1f} ms") # FOR DEBUG ONLY
            return False

        current_intercept_position_world = np.asarray(tracker.predict(dt)[:3], dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(current_intercept_position_world)):
            return False

        current_intercept_position_platform = estimateObjectPlatformPosition(
            current_intercept_position_world, self.camera_to_platform_calibration
        )

        aim_valid, current_servo_angles, current_foam_flight_time = self._object_position_to_servo_angles_and_flight_time(
            current_intercept_position_platform, initial_servo_angles=self.active_plan.raw_servo_angles
        )
        if not aim_valid:
            return False

        planned_servo_angles = np.asarray(self.active_plan.raw_servo_angles, dtype=float).reshape(-1).copy()
        servo_angle_tolerances = np.asarray(ACTIVE_PLAN_SERVO_ANGLE_TOLERANCES, dtype=float).reshape(-1).copy()

        angular_uncertainty = tracker.track.angular_uncertainty_deg
        if angular_uncertainty is not None and np.isfinite(angular_uncertainty):
            servo_angle_tolerances = np.maximum(
                servo_angle_tolerances,
                ACTIVE_PLAN_UNCERTAINTY_SIGMA_MULTIPLIER*angular_uncertainty
            )

        if np.any(servo_angle_tolerances <= 0.0):
            raise ValueError("ACTIVE_PLAN_SERVO_ANGLE_TOLERANCES must be > 0")

        servo_angle_error = current_servo_angles - planned_servo_angles

        planned_foam_flight_time = self.active_plan.intercept_time - self.active_plan.trigger_time - TRIGGER_DELAY
        foam_flight_time_error = current_foam_flight_time - planned_foam_flight_time

        flight_time_tolerance = ACTIVE_PLAN_FLIGHT_TIME_TOLERANCE
        range_uncertainty = tracker.track.range_uncertainty_m
        if range_uncertainty is not None and np.isfinite(range_uncertainty):
            range_uncertainty = min(range_uncertainty, MAX_TRIGGER_RANGE_UNCERTAINTY_M)
            # Approximate range-uncertainty -> time-uncertainty using current average dart speed.
            horizontal_range = np.hypot(current_intercept_position_platform[0], current_intercept_position_platform[1])
            average_dart_speed = horizontal_range/current_foam_flight_time if current_foam_flight_time > 0.0 else DART_PROTRUSION_SPEED
            flight_time_tolerance = max(
                flight_time_tolerance,
                ACTIVE_PLAN_UNCERTAINTY_SIGMA_MULTIPLIER*range_uncertainty/max(average_dart_speed, 1e-6)
            )

        servo_error_ratio = float(np.max(np.abs(servo_angle_error)/servo_angle_tolerances))
        flight_time_error_ratio = float(abs(foam_flight_time_error)/flight_time_tolerance)
        plan_error_ratio = max(servo_error_ratio, flight_time_error_ratio)

        if plan_error_ratio >= ACTIVE_PLAN_VALIDITY_HARD_INVALID_RATIO:
            print(
                f"PLAN INVALID HARD | ratio={plan_error_ratio:.2f} | "
                f"servo error={servo_angle_error} deg | "
                f"flight time error={foam_flight_time_error*1000:.1f} ms"
            ) # FOR DEBUG ONLY
            return False

        accepted_measurement_count = tracker.track.accepted_measurement_count
        new_accepted_measurement = accepted_measurement_count != self.active_plan_last_accepted_measurement_count
        self.active_plan_last_accepted_measurement_count = accepted_measurement_count

        if new_accepted_measurement:
            if plan_error_ratio > ACTIVE_PLAN_VALIDITY_INVALID_RATIO:
                self.active_plan_invalid_streak += 1
            elif plan_error_ratio < ACTIVE_PLAN_VALIDITY_RECOVER_RATIO:
                self.active_plan_invalid_streak = 0

        if self.active_plan_invalid_streak >= ACTIVE_PLAN_INVALID_STREAK_REQUIRED:
            print(
                f"PLAN INVALID | ratio={plan_error_ratio:.2f} | "
                f"streak={self.active_plan_invalid_streak}/{ACTIVE_PLAN_INVALID_STREAK_REQUIRED} | "
                f"servo error={servo_angle_error} deg | "
                f"flight time error={foam_flight_time_error*1000:.1f} ms"
            ) # FOR DEBUG ONLY
            return False

        if new_accepted_measurement and self.active_plan_invalid_streak > 0:
            print(
                f"PLAN HYSTERESIS HOLD | ratio={plan_error_ratio:.2f} | "
                f"streak={self.active_plan_invalid_streak}/{ACTIVE_PLAN_INVALID_STREAK_REQUIRED} | "
                f"servo error={servo_angle_error} deg | "
                f"flight time error={foam_flight_time_error*1000:.1f} ms"
            ) # FOR DEBUG ONLY

        return True
