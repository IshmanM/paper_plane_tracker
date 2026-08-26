import threading
import numpy as np
from src.primary.plan import Plan
import copy
import time
from src.primary.platform_mode import PlatformMode
import src.primary.config as config
from src.comm.link import UdpLink
from src.comm.protocol import (
    CMD_PLATFORM_CONTROL,
    next_msg_id,
)



class CommBuffer:
    def __init__(self):
        self._lock = threading.Lock()

        # Written by endpoint, read by primary
        self._last_cmd_servo_angles = None # Latest actual cmd sent after filtering / rate limiting
        self._last_cmd_time = None
        self._last_cmd_id = None # if i add send other types of messages, can either add seperate types of counters or integrate somehow

        self._latest_endpoint_telemetry = None

        # Written by primary, read by endpoint
        self._active_plan_snapshot = None
        self._platform_mode_snapshot = None
        self._triggering_halted = None

    # ----------------------------
    # endpoint -> primary
    # ----------------------------

    def set_last_cmd_servo_angles(self, last_cmd_servo_angles: np.ndarray | None):
        with self._lock:
            if last_cmd_servo_angles is None:
                self._last_cmd_servo_angles = None
            else:
                self._last_cmd_servo_angles = np.asarray(last_cmd_servo_angles, dtype=float).copy()

    def get_last_cmd_servo_angles(self) -> np.ndarray | None:
        with self._lock:
            if self._last_cmd_servo_angles is None:
                return None

            return np.asarray(self._last_cmd_servo_angles, dtype=float).copy()

    def set_last_cmd_time(self, last_cmd_time: float | None):
        with self._lock:
            self._last_cmd_time = None if last_cmd_time is None else float(last_cmd_time)

    def get_last_cmd_time(self) -> float | None:
        with self._lock:
            return self._last_cmd_time
        
    def set_last_cmd_id(self, last_cmd_id: int | None):
        with self._lock:
            self._last_cmd_id = None if last_cmd_id is None else int(last_cmd_id)

    def get_last_cmd_id(self) -> int | None:
        with self._lock:
            return self._last_cmd_id

    def set_last_cmd_data(self, last_cmd_id: int, last_cmd_servo_angles: np.ndarray | None, last_cmd_time: float | None):
        with self._lock:
            if last_cmd_servo_angles is None:
                self._last_cmd_servo_angles = None
            else:
                self._last_cmd_servo_angles = np.asarray(last_cmd_servo_angles, dtype=float).copy()

            self._last_cmd_id  = None if last_cmd_id is None else int(last_cmd_id)
            self._last_cmd_time = None if last_cmd_time is None else float(last_cmd_time)

    def get_last_cmd_data(self) -> tuple[int, np.ndarray | None, float | None]:
        with self._lock:
            if self._last_cmd_servo_angles is None:
                angles = None
            else:
                angles = np.asarray(self._last_cmd_servo_angles, dtype=float).copy()

            return self._last_cmd_id, angles, self._last_cmd_time
        
    def set_latest_endpoint_telemetry(self, telemetry: dict):
        with self._lock:
            self._latest_endpoint_telemetry = telemetry.copy()


    def get_latest_endpoint_telemetry(self):
        with self._lock:
            if self._latest_endpoint_telemetry is None:
                return None

            return self._latest_endpoint_telemetry.copy()

    # ----------------------------
    # primary -> endpoint
    # ----------------------------

    def set_active_plan_snapshot(self, active_plan: Plan | None):
        with self._lock:
            self._active_plan_snapshot = copy.deepcopy(active_plan)

    def get_active_plan_snapshot(self) -> Plan | None:
        with self._lock:
            return copy.deepcopy(self._active_plan_snapshot)

    def set_platform_mode_snapshot(self, platform_mode: PlatformMode):
        with self._lock:
            self._platform_mode_snapshot = platform_mode

    def get_platform_mode_snapshot(self) -> PlatformMode:
        with self._lock:
            return self._platform_mode_snapshot
    
    def set_triggering_halted(self, triggering_halted):
        with self._lock:
            self._triggering_halted = triggering_halted

    def get_triggering_halted(self) -> bool:
        with self._lock:
            return self._triggering_halted

    def set_platform_snapshot(self, active_plan: Plan | None, platform_mode: PlatformMode, triggering_halted: bool):
        with self._lock:
            self._active_plan_snapshot = copy.deepcopy(active_plan)
            self._platform_mode_snapshot = platform_mode
            self._triggering_halted = bool(triggering_halted)

    def get_platform_snapshot(self) -> tuple[Plan | None, PlatformMode | None, bool | None]:
        with self._lock:
            return copy.deepcopy(self._active_plan_snapshot), self._platform_mode_snapshot, self._triggering_halted
            


# could also be called comm_thread_main, once i add more functionality here
def cmd_thread_main(
    comm_buffer: CommBuffer, 
    stop_event: threading.Event,
    link: UdpLink | None = None,  #default None incase i restructure hardware eventually 
    cmd_frequency_hz=30.0
):
     
    cmd_period = 1.0/cmd_frequency_hz
    # spin_margin = 0.001 # 1ms, tune later according to packet tx jitter <--seemed to cause keyboard issues

    next_time = time.perf_counter()

    # probably None, None, None on startup
    last_cmd_id, last_cmd_servo_angles, last_cmd_time = comm_buffer.get_last_cmd_data() 

    try:

        ### FOR DEBUG ONLY ###
        # last_debug_print_time = None
        # debug_print_period = 0.1
        ######################

        while not stop_event.is_set():
            now = time.perf_counter()

            # Sleep most of the way, but wake up slightly before the needed time.
            # wait_time = next_time - now - spin_margin  <--spin_margin seemed to cause keyboard issues
            wait_time = next_time - now
            if wait_time > 0:
                stop_event.wait(wait_time)
            
            if stop_event.is_set():
                break

            # Spin for the final tiny interval to reduce timing overshoot.
            while not stop_event.is_set() and time.perf_counter() < next_time:
                pass

            if stop_event.is_set():
                break

            now = time.perf_counter()

            # ------------------------------------------------------------
            # UDP RX: receive telemetry/errors, but do NOT use received
            # msg_id values to generate our outgoing msg_id.
            # ------------------------------------------------------------
            if link is not None:
                for telemetry in link.recv_telemetry_available():
                    comm_buffer.set_latest_endpoint_telemetry(telemetry)

                    print(telemetry) # for debug only

                #Todo: consider storing errors in the comm_buffer and/or a log file
                for error_msg in link.recv_errors_available():
                    print(f"Endpoint error: {error_msg}")
            
            active_plan, platform_mode, triggering_halted = comm_buffer.get_platform_snapshot()

            cmd_servo_angles = compute_filtered_cmd_servo_angles(
                active_plan=active_plan,
                platform_mode=platform_mode,
                last_cmd_servo_angles=last_cmd_servo_angles,
                last_cmd_time=last_cmd_time,
                now=now
            )
            
            trigger = should_trigger(
                triggering_halted=triggering_halted,
                active_plan=active_plan,
                platform_mode=platform_mode,
                now=now
            )

            # ------------------------------------------------------------
            # UDP TX: this is where msg_id advances.
            # ------------------------------------------------------------

            cmd_id = next_msg_id(last_cmd_id)

            cmd_payload = {
                "platform_mode": None if platform_mode is None else platform_mode.name,
                "track_id": None if active_plan is None or active_plan.track_id is None else int(active_plan.track_id),
                "pan_deg": float(cmd_servo_angles[config.SERVO_IDX["pan"]]),
                "tilt_deg": float(cmd_servo_angles[config.SERVO_IDX["tilt"]]),
                "triggering_halted": bool(triggering_halted),
                "trigger": bool(trigger),
            }

            ### FOR DEBUG ONLY ###
            # if last_debug_print_time is None or (now - last_debug_print_time) >= debug_print_period:
            #     print(
            #         f"now={now:.3f}, "
            #         f"cmd_id={cmd_id}, "
            #         f"payload={cmd_payload}"
            #     )
            #     last_debug_print_time = now
            # print(
            #     f"now={now:.3f}, "
            #     f"cmd_id={cmd_id}, "
            #     f"payload={cmd_payload}"
            # )
            ########################

            if link is not None:
                link.send_cmd(
                    msg_id=cmd_id,
                    sender_time=now,
                    cmd_name=CMD_PLATFORM_CONTROL,
                    cmd_payload=cmd_payload
                )


            last_cmd_id, last_cmd_servo_angles, last_cmd_time = cmd_id, cmd_servo_angles, now
            
            comm_buffer.set_last_cmd_data(
                last_cmd_id=last_cmd_id,
                last_cmd_servo_angles=last_cmd_servo_angles, 
                last_cmd_time=last_cmd_time
            )

            next_time += cmd_period
            
            # If we fell behind badly, resync instead of sending catch-up bursts.
            if next_time < time.perf_counter() - cmd_period:
                next_time = time.perf_counter() + cmd_period

        
    finally:
        # Optional later:
        #
        # On shutdown, you may want to send one final safe command.
        # For example: no trigger, hold current angle or go neutral.
        #
        # if link is not None and last_cmd_servo_angles is not None:
        #     link.send_cmd(
        #         cmd_id=(0 if last_cmd_id is None else (last_cmd_id + 1) % (MAX_CMD_ID + 1)),
        #         cmd_servo_angles=last_cmd_servo_angles,
        #         trigger=False,
        #         platform_mode=PlatformMode.OFF,
        #         track_id=None,
        #         laptop_time=time.perf_counter(),
        #     )
        #
        # If this thread owns the UDP socket, close it here.
        # But if main.py owns the socket, close it in main.py instead.
        #
        # if link is not None:
        #     link.close()
        pass    
        
    
#todo: rpi side should know how to handle jitter nand missing packets



def should_trigger(triggering_halted: bool, active_plan: Plan, platform_mode: PlatformMode, now) -> bool:

    if triggering_halted is None or platform_mode is None or active_plan is None:
        return False
    
    if triggering_halted or platform_mode == PlatformMode.OFF:
        return False

    # ToDo: add more comprehensive triggering logic, based on timing, ACKs, and user enable/disable input
    trigger = platform_mode == PlatformMode.FOLLOWING_LEAD
    return trigger


def compute_filtered_cmd_servo_angles(active_plan: Plan, platform_mode: PlatformMode, last_cmd_servo_angles, last_cmd_time, now) -> np.ndarray:

    # ------------------------------------------------------------
    # 1. Decide the raw angle first.
    #
    # OFF / bad snapshot:
    #     default servo angles
    #
    # Normal platform modes:
    #     active_plan.raw_servo_angles
    # ------------------------------------------------------------

    q_raw = None

    if platform_mode is None or active_plan is None:
        q_raw = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()

    elif platform_mode == PlatformMode.OFF:
        q_raw = np.asarray(config.DEFAULT_SERVO_ANGLES, dtype=float).copy()

    else:
        if active_plan.raw_servo_angles is None: # <-- eventually remove this check
            raise ValueError("active_plan has None type raw_servo_angles")

        q_raw = np.asarray(active_plan.raw_servo_angles, dtype=float).copy()

    q_raw = np.clip(q_raw, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)

    q_cmd = None
    
    # ------------------------------------------------------------
    # 2. First cmd: no previous actuator cmd exists
    # ------------------------------------------------------------

    if last_cmd_servo_angles is None:
        q_cmd = q_raw.copy()
    
    # ------------------------------------------------------------
    # 3. Subsequent cmd: filter / deadband / speed-limit
    # ------------------------------------------------------------

    else:
        # Applies:
        # 1. clipping
        # 2. optional smoothing
        # 3. deadband
        # 4. max servo speed limiting
        # 5. final clipping

        if last_cmd_time is None: # <-- eventually remove this check
            raise ValueError("last_cmd_time is None type when last_cmd_servo_angles is not None type")

        dt_cmd = now - last_cmd_time
        
        if dt_cmd < 0.0: # <-- eventually remove this check
            raise ValueError("dt_cmd < 0.0")

        q_prev = np.asarray(last_cmd_servo_angles, dtype=float).copy()  
        # clippling, probably not needed since should be clipped already
        q_prev = np.clip(q_prev, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)

        q_goal = None
        # SEARCHING is already a smooth sine scan, so avoid over-smoothing/deadbanding it.
        # OFF should move toward default angles directly.
        # In both cases, skip smoothing/deadband, but still apply speed limiting below.
        if platform_mode in {PlatformMode.SEARCHING, PlatformMode.OFF}:
            q_goal = q_raw.copy()
        else:
            # Exponential smoothing:
            #
            # alpha near 1.0 -> follows goal quickly
            # alpha near 0.0 -> very smooth / slow response
            tau = config.CMD_SMOOTHING_TAU
            if tau <= 0.0:
                q_goal = q_raw.copy()
            else:
                alpha = dt_cmd/(tau + dt_cmd)
                q_goal = q_prev + alpha*(q_raw - q_prev)

            # Deadband should be based on the real error, not the smoothed step.
            # Otherwise smoothing could make every step tiny and accidentally freeze motion.
            deadband = np.asarray(config.SERVO_DEADBAND, dtype=float).copy()
            inside_deadband = np.abs(q_raw - q_prev) <= deadband
            q_goal = np.where(inside_deadband, q_prev, q_goal)

        # Rate limit: the command cannot move faster than the servo speed limit.
        max_step = np.asarray(config.MAX_SERVO_SPEEDS, dtype=float).copy()*dt_cmd

        step = q_goal - q_prev
        step = np.clip(step, -max_step, max_step)
        q_cmd = q_prev + step

    # final clip to deal with misc/rare numerical or bugged edge cases
    return np.clip(q_cmd, config.MIN_SERVO_ANGLES, config.MAX_SERVO_ANGLES)


