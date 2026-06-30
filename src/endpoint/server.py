import time
import threading

from src.comm.link import UdpLink
from src.comm.protocol import (
    CMD_PLATFORM_CONTROL,
    ERR_CODE_UNKNOWN_CMD_NAME,
    ERR_CODE_UNEXPECTED_ENDPOINT_ERROR,
    next_msg_id, # a function
    # telemetry names
    TELEMETRY_ENDPOINT_STATE,
    TELEMETRY_CONFIG,
    TELEMETRY_HEARTBEAT,   
)
from src.endpoint.controller import EndpointController, CmdResult



class EndpointServer:
    """
    Endpoint communication loop.

    Responsibilities:
        - receive endpoint commands from UdpLink
        - keep only the latest platform-control command per loop
        - pass decoded command info to EndpointController
        - send endpoint telemetry
        - send error messages for communication/command handling failures

    Not responsible for:
        - servo math
        - safe mode policy
        - platform state logic
        - raw UDP socket handling
        - protocol encoding/decoding
        - closing link/hardware resources
    """
    
    def __init__(
        self, 
        endpoint_controller: EndpointController, 
        link: UdpLink,
        refresh_frequency_hz: float = 120.0,
        telemetry_frequency_hz: float = 15.0,
    ):

        if refresh_frequency_hz <= 0.0:
            raise ValueError("refresh_frequency_hz must be positive")

        self.endpoint_controller = endpoint_controller
        self.link = link
       
        # Useful later if server.run() is moved to its own thread.
        self.stop_event = threading.Event()
       
        self.last_sent_telemetry_time: float | None = None
        self.last_sent_error_time: float | None = None
        self.last_sent_msg_time: float | None = None
        self.last_sent_msg_id: float | None = None

        self.last_recv_cmd_time: float | None = None
        self.last_recv_cmd_platform_control_time: float | None = None
        
        self.last_handled_cmd_time: float | None = None
        self.last_handled_cmd_platform_control_time: float | None = None

        self.refresh_frequency_hz = float(refresh_frequency_hz)
        self.telemetry_frequency_hz = float(telemetry_frequency_hz)
    

    def run(self):

        print("Endpoint server running. Waiting for commands...", flush=True)

        try:
            refresh_period = 1.0/self.refresh_frequency_hz
            telemetry_period = 1.0/self.telemetry_frequency_hz

            self.stop_event.clear()

            while not self.stop_event.is_set():     
                iter_start_time = time.perf_counter()
                self._handle_latest_cmd(now=iter_start_time)

                telemetry_due = (
                    self.last_sent_telemetry_time is None
                    or (iter_start_time - self.last_sent_telemetry_time) >= telemetry_period
                )

                # Todo: - might change what now time is passed to _send_telemetry depending on implementation...
                #       - put more state info in the endpoint state telemetry sent, especially foam mechanism state
                if telemetry_due:
                    self._send_telemetry(
                        now=iter_start_time,
                        telemetry_name=TELEMETRY_ENDPOINT_STATE,
                        telemetry_payload=self._make_endpoint_state_payload(now=iter_start_time),
                        reply_to_msg_id=None
                    )
                    # self._send_telemetry(
                    #     now=iter_start_time,
                    #     telemetry_name=TELEMETRY_HEARTBEAT,
                    #     telemetry_payload=None,
                    #     reply_to_msg_id=None
                    # )

                wait_time = max(0.0, refresh_period - (time.perf_counter() - iter_start_time))
                self.stop_event.wait(wait_time)

        finally:
            pass
    
    
    def _make_endpoint_state_payload(self, now: float) -> dict:
        state = self.endpoint_controller.get_state()

        return {
            # Controller / mechanism state
            "pan_deg": state.pan_deg,
            "tilt_deg": state.tilt_deg,
            "safe" : state.safe, # (endpoint side defn of safe)

            # Command receive health
            "last_recv_cmd_age_s": self._age_s(now, self.last_recv_cmd_time),
            "last_recv_cmd_platform_control_age_s": self._age_s(now, self.last_recv_cmd_platform_control_time),

            # Command handling health
            "last_handled_cmd_age_s": self._age_s(now, self.last_handled_cmd_time),
            "last_handled_cmd_platform_control_age_s": self._age_s(now, self.last_handled_cmd_platform_control_time),
        }


    def _age_s(self, now: float, t: float | None) -> float | None:
        if t is None:
            return None
        return now - t


    def stop(self) -> None:
        """
        Request the server loop to stop.
        Only useful if run() is executing in another thread.
        """
        self.stop_event.set()    


    def _handle_latest_cmd(self, now: float) -> None:
        
        # ignoring all cmds except the latest one recieved
        cmds = self.link.recv_cmds_available()
        if not cmds:
            return
        
        # Server-level comms state: at least one command was received.
        self.last_recv_cmd_time = now
        
        latest_platform_control_cmd = None
        #latest_<other_type_of_command>_cmd = None

        for cmd in cmds:
            cmd_name = cmd.get("cmd_name")
            if cmd_name == CMD_PLATFORM_CONTROL:
                latest_platform_control_cmd = cmd
                self.last_recv_cmd_platform_control_time = now
            
            # elif cmd_name == ... #Todo: implement other cases for other types of commands, if ever applicable
            
            else: # Error for unknown cmd name
                self._send_error(
                    now=now, 
                    error_text=f"Unknown cmd_name: {cmd_name}", 
                    error_code=ERR_CODE_UNKNOWN_CMD_NAME,
                    reply_to_msg_id=cmd.get("msg_id")
                )
        
        if latest_platform_control_cmd is None:
            return
        
        try:
            
            
            print(latest_platform_control_cmd) # FOR DEBUG ONLY
            
            cmd_result = self.endpoint_controller.handle_cmd(latest_platform_control_cmd, now)
            if cmd_result.is_error:
                self._send_error(
                    now=now,
                    error_text=cmd_result.error_text,
                    error_code=cmd_result.error_code,
                    reply_to_msg_id=latest_platform_control_cmd.get("msg_id")
                )
            else:
                self.last_handled_cmd_platform_control_time = now
                self.last_handled_cmd_time = now
        except Exception as e:
            self._send_error(
                now=now,
                error_text=f"Unexpected endpoint error: {e}",
                error_code=ERR_CODE_UNEXPECTED_ENDPOINT_ERROR,
                reply_to_msg_id=latest_platform_control_cmd.get("msg_id")
            )


    def _send_error(self, now: float, error_text: str, error_code: int, reply_to_msg_id: int | None) -> None:
        msg_id = next_msg_id(self.last_sent_msg_id)
        self.link.send_error(
            msg_id=msg_id,
            sender_time=now,
            error_text=error_text,
            error_code=error_code,
            reply_to_msg_id=reply_to_msg_id,
        )
        self.last_sent_msg_id = msg_id
        self.last_sent_msg_time = now
        self.last_sent_error_time = now


    def _send_telemetry(self, now: float, telemetry_name: str, telemetry_payload: dict | None, reply_to_msg_id: int | None) -> None:
        msg_id = next_msg_id(self.last_sent_msg_id)
        self.link.send_telemetry(
            msg_id=msg_id,
            sender_time=now,
            telemetry_name=telemetry_name,
            telemetry_payload=telemetry_payload,
            reply_to_msg_id=reply_to_msg_id
        )
        self.last_sent_msg_id = msg_id
        self.last_sent_telemetry_time = now
        self.last_sent_msg_time = now
    
    