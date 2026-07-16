import math

from src.endpoint.mechanisms.foam_mechanism import FoamMechanism, FoamMechanismError
from src.endpoint.mechanisms.orient_mechanism import OrientMechanism, OrientMechanismError
from src.comm.protocol import (
    CMD_PLATFORM_CONTROL,
    ERR_CODE_UNKNOWN_CMD_NAME,
    ERR_CODE_UNEXPECTED_ENDPOINT_ERROR, 
    ERR_CODE_BAD_CMD_PAYLOAD  
)

import src.endpoint.config as config

class CmdResult:
    def __init__(
        self,
        is_error: bool,
        error_code: int | None = None,
        error_text: str | None = None,
    ):
        self.is_error = is_error
        self.error_code = error_code
        self.error_text = error_text


class EndpointState:
    def __init__(
        self,
        safe: bool,
        pan_deg: float | None,
        tilt_deg: float | None,
    ):
        self.safe = safe
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        
        #Todo: add params for relevant foam mechanism state info...


class EndpointController:
    def  __init__(self, orient_mechanism: OrientMechanism, foam_mechanism: FoamMechanism):
        
        
        self.orient_mechanism = orient_mechanism
        self.foam_mechanism = foam_mechanism

        self.safe = False # an endpoint side state only, not for primary to know
            

    def go_safe(self) -> CmdResult:
        """
        Put the endpoint into safe mode.

        Safe mode means:
            - triggering is halted/disabled
            - platform is moved to default orientation
        """

        try:
            self.foam_mechanism.halt_trigger()

        except FoamMechanismError as e:
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_UNEXPECTED_ENDPOINT_ERROR,
                error_text=f"Failed to halt triggering while entering safe mode: {e}",
            )

        try:
            self.orient_mechanism.set_angles_deg(
                pan_deg=config.DEFAULT_PAN_ANGLE,
                tilt_deg=config.DEFAULT_TILT_ANGLE,
            )

        except OrientMechanismError as e:
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_UNEXPECTED_ENDPOINT_ERROR,
                error_text=f"Failed to move endpoint to safe pose: {e}",
            )

        self.safe = True
        return CmdResult(is_error=False)
    
    
    def handle_cmd(self, cmd: dict, now: float) -> CmdResult:
        cmd_name = cmd.get("cmd_name")

        if cmd_name == CMD_PLATFORM_CONTROL:
            return self._handle_cmd_platform_control(cmd, now)
        # elif cmd_name == <some other command name>:
        #     return self._<handler for that other command type>(cmd, now)

        # Server level should also do these checks, but just incase some names not known at controller level
        return CmdResult(
            is_error=True,
            error_code=ERR_CODE_UNKNOWN_CMD_NAME,
            error_text=f"Unknown cmd_name: {cmd_name}",
        )
        

    def _handle_cmd_platform_control(self, cmd: dict, now: float) -> CmdResult:
        # Receiving a platform-control command means the endpoint is no longer safe.
        self.safe = False

        payload = cmd.get("cmd_payload")

        if not isinstance(payload, dict):
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_BAD_CMD_PAYLOAD,
                error_text="Platform control payload must be a dict",
            )

        try:
            pan_deg = self._parse_float_field(payload, "pan_deg")
            tilt_deg = self._parse_float_field(payload, "tilt_deg")

            triggering_halted = self._parse_bool_field(
                payload,
                "triggering_halted",
                default=True,
            )

            trigger = self._parse_bool_field(
                payload,
                "trigger",
                default=False,
            )

        except ValueError as e:
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_BAD_CMD_PAYLOAD,
                error_text=f"Invalid platform control payload: {e}",
            )

        try:
            self.orient_mechanism.set_angles_deg(
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
            )

        except OrientMechanismError as e:
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_BAD_CMD_PAYLOAD,
                error_text=f"Rejected platform orientation command: {e}",
            )

        try:
            if triggering_halted:
                # Apply the halt only when transitioning into the halted state.
                if not self.foam_mechanism.trigger_is_halted():
                    self.foam_mechanism.halt_trigger()
            elif trigger:
                # trigger() automatically clears the temporary halt.
                self.foam_mechanism.trigger()

        except FoamMechanismError as e:
            return CmdResult(
                is_error=True,
                error_code=ERR_CODE_UNEXPECTED_ENDPOINT_ERROR,
                error_text=f"Platform trigger command failed: {e}",
            )

        return CmdResult(is_error=False)


    def get_state(self) -> EndpointState:
        pan_deg = self.orient_mechanism.get_last_pan_deg()
        tilt_deg = self.orient_mechanism.get_last_tilt_deg()

        if pan_deg is not None:
            pan_deg = float(pan_deg)

        if tilt_deg is not None:
            tilt_deg = float(tilt_deg)
        
        return EndpointState(
            safe=self.safe,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,

            #Todo: add params for relevant foam mechanism state info...
        )
    

    @staticmethod
    def _parse_float_field(payload: dict, field_name: str) -> float:
        if field_name not in payload:
            raise ValueError(f"Missing field: {field_name}")

        value = payload[field_name]

        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be numeric, not bool")

        if not isinstance(value, int | float):
            raise ValueError(
                f"{field_name} must be numeric, got {type(value).__name__}"
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")

        return value


    @staticmethod
    def _parse_bool_field(payload: dict, field_name: str, default: bool) -> bool:
        value = payload.get(field_name, default)

        if isinstance(value, bool):
            return value

        if isinstance(value, int) and value in (0, 1):
            return bool(value)

        raise ValueError(
            f"{field_name} must be bool or 0/1, got {value!r}"
        )