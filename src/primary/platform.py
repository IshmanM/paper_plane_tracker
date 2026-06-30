
from src.primary.tracking import Track, TrackStatus, SingleObjectTracker
import numpy as np
import time
from src.primary.geometry import estimateTargetPlatformPosition
import src.primary.config as config
from src.primary.comm_buffer import CommBuffer
from src.primary.plan import Plan, PlanType
from src.primary.platform_mode import PlatformMode



SERVO_SETTLING_TIME = 0.1 # seconds
SERVO_ROTATION_TIME_MARGIN = 0.05 # seconds


SEARCH_CENTER_PAN = 90.0 # degrees
SEARCH_PAN_AMPLITUDE = 75.0 # degrees
SEARCH_FREQUENCY = 0.2 # hz


TRIGGER_TIME_LOWER_THRESHOLD = -0.25 # seconds
TRIGGER_TIME_UPPER_THRESHOLD = 0.50 # seconds

FIRST_INTERCEPT_MAX_NUM_CANDIDATES = 10 
FIRST_INTERCEPT_MAX_LOOKAHEAD = 2.5 # seconds

SUBSEQUENT_INTERCEPT_MAX_NUM_CANDIDATES = 5 
SUBSEQUENT_INTERCEPT_MAX_LOOKAHEAD = 0.5 # seconds

TRIGGER_DELAY = 0.1 # seconds

GRAVITY = 9.81  # m/s^2

# Tune experimentally.
DART_PROTRUSION_SPEED = 12.0  # m/s

# Do not aim at objects effectively behind / on top of platform.
MIN_FORWARD_RANGE = 0.02  # m

# Position of the dart protrusion relative to the platform origin.
# Keep zero initially unless I carefully measure the geometry.
PLATFORM_PROTRUSION_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)

# Usually False for a turret. High arc is slower and less direct.
USE_HIGH_ARC = False

MIN_FIRST_INTERCEPT_READY_MARGIN = 0.05 # seconds
MIN_SUBSEQUENT_INTERCEPT_READY_MARGIN = 0.025 # seconds

ACTIVE_PLAN_INTERCEPT_POSITION_TOLERANCES = np.array([0.03, 0.03, 0.06], dtype=float) # x, y, z meters


# If now is inside _close_to_trigger_time(...), treat time-to-trigger cost as ideal.
PLAN_COST_NOW_TRIGGER_TIME_COST = 0.0

# time_to_trigger_cost = time_to_trigger / PLAN_COST_TIME_SCALE
# So a candidate 0.5 s away has time cost ~= 1.
PLAN_COST_TIME_SCALE = 0.5 # seconds

# Extra ready-margin surplus of this amount cuts ready_margin_cost roughly in half.
# ready_margin_cost = 1 / (1 + surplus / scale)
PLAN_COST_READY_MARGIN_SCALE = 0.05 # seconds

# Normalize servo motion cost.
PLAN_COST_SERVO_ANGLE_SCALES = np.zeros(config.NUM_SERVOS, dtype=float)
PLAN_COST_SERVO_ANGLE_SCALES[config.SERVO_IDX["pan"]] = 45.0 # degrees
PLAN_COST_SERVO_ANGLE_SCALES[config.SERVO_IDX["tilt"]] = 30.0 # degrees

# Meters.
# Normalize intercept-position changes for continuity cost.
PLAN_COST_INTERCEPT_POSITION_SCALES = np.array([0.20, 0.20, 0.30], dtype=float)


FIRST_INTERCEPT_PLAN_COST_WEIGHTS = {
    "time": 1.0,
    "servo_motion": 0.25,
    "ready_margin": 0.75,
    "continuity": 0.0,
}

SUBSEQUENT_INTERCEPT_PLAN_COST_WEIGHTS = {
    "time": 0.35,
    "servo_motion": 0.50,
    "ready_margin": 0.75,
    "continuity": 0.75,
}



class Platform:
    def __init__(
        self,
        comm_buffer: CommBuffer
    ):
        self.mode = PlatformMode.OFF
        self.active_plan = self._make_off_plan(now=time.perf_counter())

        self.triggering_halted = True # Forcing parameter

        self.comm_buffer = comm_buffer
        self.comm_buffer.set_platform_snapshot(
            active_plan=self.active_plan, 
            platform_mode=self.mode,
            triggering_halted=self.triggering_halted
        )

        
        # extra Todos:
        # - if adding ACKs from the rpi: eg. self.last_ack_cmd_id, self.last_ack_time. <-- should use the buffer
        # - instead of guessing the communication speed/delay, an occasional ping can be used to estimate <-- cmd.py or comm script should figure out and put in buffer

    
    def _tracker_is_usable(self, tracker: SingleObjectTracker):
        if tracker is None:
            raise ValueError("None type Tracker passed to Platform")
        
        if tracker.track_status == TrackStatus.CONFIRMED and (tracker.track is None or tracker.track.state_time is None or tracker.track.id is None):
            raise ValueError("Tracker with CONFIRMED track_status but None type Track or Track.state_time or Track.id passed to Platform")

        if tracker.track_status in {TrackStatus.TENTATIVE, TrackStatus.DEAD}:
            return False
        
        return True


    def update(self, tracker: SingleObjectTracker):
        now = time.perf_counter()
        
        # Nothing updates if OFF
        if self.mode == PlatformMode.OFF:
            self.comm_buffer.set_platform_snapshot(
                active_plan=self.active_plan,
                platform_mode=self.mode,
                triggering_halted=self.triggering_halted
            )
            return
        
        ########## 0. UNIVERSAL TRACKER VALIDITY GUARD #############

        if not self._tracker_is_usable(tracker):
            self.active_plan = self._make_search_plan(now)
            self.mode = PlatformMode.SEARCHING

            self.comm_buffer.set_platform_snapshot(
                active_plan=self.active_plan,
                platform_mode=self.mode,
                triggering_halted=self.triggering_halted
            )
            return
        
        ########## 1. TRACK ID SWITCH GUARD ########################

        if (self.mode != PlatformMode.SEARCHING and self.active_plan.track_id != tracker.track.id):
            self.active_plan = self._make_search_plan(now)
            self.mode = PlatformMode.SEARCHING
            # Don't return. Let SEARCHING immediately try to acquire the new track
        
        ########## 2. MODE LOGIC ##################################

        if self.mode == PlatformMode.SEARCHING:
            
            valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)
            
            if valid_plan_computed:
                self.active_plan = plan
                self.mode = PlatformMode.SLEWING_TO_LEAD

                # print("made it to A") # FOR DEBUG ONLY
            else:
                self.active_plan = self._make_search_plan(now)
                self.mode = PlatformMode.SEARCHING

                # print("made it to B") # FOR DEBUG ONLY
        
            
        if self.mode == PlatformMode.SLEWING_TO_LEAD:
            
            if not self._active_plan_still_valid(tracker, now):
                valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)

                if valid_plan_computed:
                    self.active_plan = plan

                    # print("made it to C") # FOR DEBUG ONLY
                else:
                    self.active_plan = self._make_search_plan(now)
                    self.mode = PlatformMode.SEARCHING

                    # print("made it to D") # FOR DEBUG ONLY
            
            # if (and not elif) incase the new best first intercept plan somehow chooses a point that the platform is already pointed toward
            if (self.mode == PlatformMode.SLEWING_TO_LEAD and self._close_to_trigger_time(now)):
                self.mode = PlatformMode.FOLLOWING_LEAD

        # elif (and not if) because the we dont want to get into receding horizon stuff until first foam
        elif self.mode == PlatformMode.FOLLOWING_LEAD:
            valid_plan_computed, plan = self._make_best_valid_subsequent_intercept_plan(tracker, now)

            if valid_plan_computed:
                self.active_plan = plan

                # print("made it to E") # FOR DEBUG ONLY
            else:
                # print("made it to F") # FOR DEBUG ONLY
                valid_plan_computed, plan = self._make_best_valid_first_intercept_plan(tracker, now)
                if valid_plan_computed: # tbh this case probably wont ever happen
                    self.active_plan = plan
                    self.mode = PlatformMode.SLEWING_TO_LEAD
                else:
                    self.active_plan = self._make_search_plan(now)
                    self.mode = PlatformMode.SEARCHING

    
        ########## 3. CMD OUTPUT ##############################
        
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
        self.mode = PlatformMode.OFF

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
        return TRIGGER_TIME_LOWER_THRESHOLD <= dt <= TRIGGER_TIME_UPPER_THRESHOLD


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
        intercept_time = when object and projectile meet
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

        return self._make_best_valid_intercept_plan_from_candidates(
            tracker, 
            now, 
            PlanType.SUBSEQUENT_INTERCEPT, 
            candidate_intercept_times, 
            cost_weights, 
            min_ready_margin=MIN_SUBSEQUENT_INTERCEPT_READY_MARGIN
        )


    def _make_best_valid_first_intercept_plan(self, tracker: SingleObjectTracker, now) -> tuple[bool, Plan | None]:
        """
        Build the best FIRST_INTERCEPT plan. 
        Might have way different implementation than SUBSEQUENT_INTERCEPT plans later.

        Returns:
        (True, plan)  if a feasible first plan was found
        (False, None) otherwise

        Important timing meaning:
        intercept_time = when object and projectile meet
        trigger_time      = when platform must trigger/release
        expected_ready_time = when pan/tilt is expected to be aimed and settled

        This function intentionally searches candidate intercept times instead of trying to solve everything analytically.
        """

        candidate_intercept_times = np.linspace(
            start=now, 
            stop=now + FIRST_INTERCEPT_MAX_LOOKAHEAD, 
            num=FIRST_INTERCEPT_MAX_NUM_CANDIDATES
        )

        cost_weights = FIRST_INTERCEPT_PLAN_COST_WEIGHTS 

        return self._make_best_valid_intercept_plan_from_candidates(
            tracker, 
            now, 
            PlanType.FIRST_INTERCEPT, 
            candidate_intercept_times, 
            cost_weights, 
            min_ready_margin=MIN_FIRST_INTERCEPT_READY_MARGIN
        )


    def _plan_cost(self, now, plan: Plan, weights: dict[str, float], min_ready_margin) -> float:
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

        servo_error = raw_servo_angles - q_ref
        servo_motion_cost = float(np.linalg.norm(servo_error / servo_scales))

        # 3. Ready margin cost
        
        ready_margin_cost = np.inf
        ready_margin = plan.trigger_time - plan.ready_time
        
        if ready_margin <= min_ready_margin:
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
                continuity_cost = float(np.linalg.norm((p_new - p_old) / position_scales))

        # 5. Total weighted cost
        cost = float(
            weights.get("time", 0.0) * time_to_trigger_cost
            + weights.get("servo_motion", 0.0) * servo_motion_cost
            + weights.get("ready_margin", 0.0) * ready_margin_cost
            + weights.get("continuity", 0.0) * continuity_cost
        )

        if not np.isfinite(cost):
            return np.inf
        
        return cost


    def _make_best_valid_intercept_plan_from_candidates(self, tracker: SingleObjectTracker, now, plan_type: PlanType, candidate_intercept_times, cost_weights, min_ready_margin) -> tuple[bool, Plan | None]:
        
        # Current estimate of where the servos are starting from
        # should eventually come from feedback or Pi-side ACK
        q_start = self._planning_servo_angles()

        best_plan = None
        best_cost = np.inf # lower cost is better

        for intercept_time in candidate_intercept_times:
            
            # print("_make_best_valid_intercept_plan_from_candidates A") # FOR DEBUG ONLY

            # Cannot predict backwards.
            dt = intercept_time - tracker.track.state_time
            if dt <= 0.0:
                continue

            # 1. Predict object position at candidate intercept time.
            # Todo: might eventually want to add different behavior for position outside the visible range
            object_position_world = tracker.predict(dt)[:3].copy()
            if not np.all(np.isfinite(object_position_world)):
                continue
            

            # print("_make_best_valid_intercept_plan_from_candidates B") # FOR DEBUG ONLY
                

            # 2. Transform object position into platform frame.
            object_position_platform = estimateTargetPlatformPosition(object_position_world)

            # 3. Use platform-frame point to compute servo raw pan/tilt angles and projectile flight time
            angles_valid, q_raw, projectile_flight_time = self._object_position_to_servo_angles_and_flight_time(object_position_platform)
            if not angles_valid:
                continue


            # print("_make_best_valid_intercept_plan_from_candidates c") # FOR DEBUG ONLY

            # 4. Convert intercept time to trigger time.
            trigger_time = intercept_time - projectile_flight_time - TRIGGER_DELAY
            if trigger_time < now and not self._close_to_trigger_time(now, trigger_time=trigger_time):
                continue


            # print("_make_best_valid_intercept_plan_from_candidates D") # FOR DEBUG ONLY
            # 5. Estimate the ready time and whether its within margin

            servo_rotation_time = self._estimate_servo_rotation_time(q_from=q_start, q_to=q_raw)
            expected_ready_time = now + servo_rotation_time + config.UDP_TX_DELAY

            if (trigger_time - expected_ready_time) < min_ready_margin:
                continue

            # print("_make_best_valid_intercept_plan_from_candidates E") # FOR DEBUG ONLY
            # 6. Create the plan and compute its cost

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

            cost = self._plan_cost(now, plan, cost_weights, min_ready_margin) # Todo: determine what to do with infinite cost
    
            # 7. Replace the best plan if new one is better

            if cost < best_cost: # also ensures np.inf cost doesn't pass a plan
                
                # print("_make_best_valid_intercept_plan_from_candidates F") # FOR DEBUG ONLY

                best_cost = cost
                best_plan = plan

        
        return best_plan is not None, best_plan   


    def _object_position_to_servo_angles_and_flight_time(self, position: np.ndarray) -> tuple[bool, np.ndarray | None, float | None]:
        """
        Convert platform-frame object position to pan/tilt angles.
        
        Returns:
            success, servo_angles, projectile_flight_time
        
        success = False means this object position is not aimable/reachable.
        """

        def fail() -> tuple[bool, np.ndarray | None, float | None]:
            return False, None, None

        position = np.asarray(position, dtype=float).reshape(-1).copy()
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return fail()

        # print("made it to _object_position_to_servo_angles_and_flight_time A") # FOR DEBUG ONLY

        protrusion_offset = np.asarray(PLATFORM_PROTRUSION_OFFSET, dtype=float).reshape(-1).copy()
        if protrusion_offset.shape != (3,):
            raise ValueError("Invalid protrusion_offset ndarray shape")
        
        position = position - protrusion_offset

        x, y, z = position[0], position[1], position[2] # ToDo: orientation TBD...

        if z <= MIN_FORWARD_RANGE:
            return fail()

        # print("made it to _object_position_to_servo_angles_and_flight_time B") # FOR DEBUG ONLY


        # 1. Pan angle

        platform_yaw_rad = np.arctan2(x, z)
        platform_yaw_deg = np.rad2deg(platform_yaw_rad)

        pan_angle = (
            config.FORWARD_SERVO_ANGLES[config.SERVO_IDX["pan"]]
            + config.SERVO_SIGNS[config.SERVO_IDX["pan"]] * platform_yaw_deg
            + config.SERVO_BIASES[config.SERVO_IDX["pan"]]
        )

        # 2. Ballistic tilt angle
        # Projectile equation:
        # y = r tan(theta) - g r^2 / (2 v0^2 cos^2(theta))
        # Let u = tan(theta):
        # A u^2 - r u + (A + y) = 0
        # where: A = g r^2 / (2 v0^2)

        r = np.hypot(x, z)
        if r <= 1e-9: # sanity
            return fail()
        
        v0 = float(DART_PROTRUSION_SPEED)
        g = float(GRAVITY)
        tan_theta = None

        A = g*r*r/(2.0*v0*v0)
        
        if A <= 1e-12:
            tan_theta = y/r
        else:
            discriminant = r*r - 4.0*A*(A + y)
            # Allow tiny numerical roundoff.
            if discriminant < -1e-9:
                return fail()
            discriminant = max(discriminant, 0.0)
            sqrt_disc = np.sqrt(discriminant)

            tan_theta_low = (r - sqrt_disc) / (2.0 * A)
            tan_theta_high = (r + sqrt_disc) / (2.0 * A)

            tan_theta = tan_theta_high if USE_HIGH_ARC else tan_theta_low

        platform_theta_rad = np.arctan(tan_theta)
        platform_theta_deg = np.rad2deg(platform_theta_rad) 

        cos_theta = np.cos(platform_theta_rad)
        if cos_theta <= 1e-9:
            return fail()
        
        # print("made it to _object_position_to_servo_angles_and_flight_time C") # FOR DEBUG ONLY

        
        projectile_flight_time = r / (v0 * cos_theta)
        if not np.isfinite(projectile_flight_time) or projectile_flight_time <= 0.0:
            return fail()
        
        # print("made it to _object_position_to_servo_angles_and_flight_time D") # FOR DEBUG ONLY


        tilt_angle = (
            config.FORWARD_SERVO_ANGLES[config.SERVO_IDX["tilt"]]
            + config.SERVO_SIGNS[config.SERVO_IDX["tilt"]]  * platform_theta_deg
            + config.SERVO_BIASES[config.SERVO_IDX["tilt"]]
        ) 
  
        # 3. Build servo command

        servo_angles = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()
        servo_angles[config.SERVO_IDX["pan"]] = pan_angle
        servo_angles[config.SERVO_IDX["tilt"]] = tilt_angle 

        if not np.all(np.isfinite(servo_angles)):
            return fail()

        # print("made it to _object_position_to_servo_angles_and_flight_time E") # FOR DEBUG ONLY


        # 4. Limit check        

        min_angles = np.asarray(config.MIN_SERVO_ANGLES, dtype=float).copy()
        max_angles = np.asarray(config.MAX_SERVO_ANGLES, dtype=float).copy()

        outside_limits = (servo_angles < min_angles) | (servo_angles > max_angles)
        if np.any(outside_limits):
            # print("Servo angle outside limits") # FOR DEBUG ONLY
            # print(f"position_platform = {position}") # FOR DEBUG ONLY
            # print(f"servo_angles      = {servo_angles}") # FOR DEBUG ONLY
            # print(f"min_angles        = {min_angles}") # FOR DEBUG ONLY
            # print(f"max_angles        = {max_angles}") # FOR DEBUG ONLY
            # print(f"outside_limits    = {outside_limits}") # FOR DEBUG ONLY
            # print(f"pan_angle         = {pan_angle:.2f}") # FOR DEBUG ONLY
            # print(f"tilt_angle        = {tilt_angle:.2f}") # FOR DEBUG ONLY
            # print(f"platform_yaw_deg  = {platform_yaw_deg:.2f}") # FOR DEBUG ONLY
            # print(f"platform_theta_deg= {platform_theta_deg:.2f}") # FOR DEBUG ONLY
            return fail()


        # print("made it to _object_position_to_servo_angles_and_flight_time F") # FOR DEBUG ONLY
        return True, servo_angles, projectile_flight_time


    def _estimate_servo_rotation_time(self, q_from: np.ndarray, q_to: np.ndarray):
        """
        Simple conservative servo motion-time estimate.

        This does NOT exactly simulate _cmd_filter().
        Instead, it uses a reduced effective servo speed plus a fixed margin.

        Pan/tilt move simultaneously, so total rotation time is the max joint time.
        """
        
        if q_from.shape != q_to.shape or q_from.shape != (config.NUM_SERVOS,):
           raise ValueError("q_from or q_to shape does not match NUM_SERVOS") 

        if not np.all(np.isfinite(q_from)) or not np.all(np.isfinite(q_to)):
            raise ValueError("q_from or q_to has non-finite values")
        
        q_from = np.asarray(q_from, dtype=float).reshape(-1).copy()
        q_to = np.asarray(q_to, dtype=float).reshape(-1).copy()
        dq = np.abs(q_to - q_from)

        deadband = np.asarray(config.SERVO_DEADBAND, dtype=float).copy()
        if np.all(dq <= deadband):
            return 0.0
        
        servo_speeds = np.asarray(config.MAX_SERVO_SPEEDS, dtype=float).copy()
        rotation_time = float(np.max(dq/servo_speeds))
        
        return rotation_time + SERVO_SETTLING_TIME + SERVO_ROTATION_TIME_MARGIN

    
    def _active_plan_still_valid(self, tracker: SingleObjectTracker, now) -> bool:
        """
        Still valid if the anticipated intercept location at the intercept time is within the some threshold of the plan
       
        If now is still _close_to_trigger_time, still valid as long as the intercept point adds up
            (of course now can be any amount of time earlier than the trigger_time)
        
        ToDo (not now, but maybe later): add additional validity checks
        """


        dt = self.active_plan.intercept_time - tracker.track.state_time
        if dt <= 0.0:
            return False
        
        if self.active_plan.trigger_time < now and not self._close_to_trigger_time(now):
            return False
        
        current_intercept_prediction = np.asarray(tracker.predict(dt)[:3], dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(current_intercept_prediction)):
            return False
        
        planned_intercept_prediction = np.asarray(self.active_plan.intercept_position, dtype=float).reshape(-1).copy()

        tolerances = np.asarray(ACTIVE_PLAN_INTERCEPT_POSITION_TOLERANCES, dtype=float).reshape(-1).copy()
        if np.any(tolerances <= 0.0):
            raise ValueError("ACTIVE_PLAN_INTERCEPT_POSITION_TOLERANCES must be > 0")

        normalized_error = (current_intercept_prediction - planned_intercept_prediction) / tolerances
        return np.linalg.norm(normalized_error) <= 1.0
 
