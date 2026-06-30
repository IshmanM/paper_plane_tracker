import json
from typing import Any


PROTOCOL_VERSION = 1

# Message types
MSG_TYPE_CMD = "cmd"
MSG_TYPE_TELEMETRY = "telemetry"
MSG_TYPE_ERROR = "error"

# Common node IDs
NODE_PRIMARY = "primary"
NODE_PLATFORM = "platform"
NODE_BROADCAST = "broadcast"

# Common command names
CMD_PLATFORM_CONTROL = "platform_control"
CMD_REQUEST_TELEMETRY = "request_telemetry"
CMD_REQUEST_CONFIG = "request_config"
CMD_SET_CONFIG = "set_config"

# Common telemetry names
TELEMETRY_ENDPOINT_STATE = "endpoint_state"
TELEMETRY_CONFIG = "config"
TELEMETRY_HEARTBEAT = "heartbeat"

# Error codes
ERR_CODE_UNKNOWN_CMD_NAME = 1
ERR_CODE_UNEXPECTED_ENDPOINT_ERROR = 2
ERR_CODE_BAD_CMD_PAYLOAD = 3

MAX_MSG_ID = 0xFFFFFFFF


def _as_uint32(value: int) -> int:
    return int(value) & MAX_MSG_ID


def _optional_uint32(value):
    if value is None:
        return None
    return _as_uint32(value)


def _json_dumps(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def _json_loads(data: bytes) -> dict:
    msg = json.loads(data.decode("utf-8"))

    if not isinstance(msg, dict):
        raise ValueError("Decoded packet is not a dict")

    return msg


def _validate_common_header(msg: dict) -> None:
    if msg.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Bad protocol version: {msg.get('version')}")

    required_fields = [
        "msg_type",
        "msg_id",
        "reply_to_msg_id",
        "sender_id",
        "receiver_id",
        "sender_time",
    ]

    for field in required_fields:
        if field not in msg:
            raise ValueError(f"Missing {field}")


def _make_base_msg(
    msg_type: str,
    msg_id: int,
    sender_id: str,
    receiver_id: str,
    sender_time: float,
    reply_to_msg_id=None,
) -> dict:
    return {
        "version": PROTOCOL_VERSION,

        # Type of this message: "cmd", "telemetry", "error", etc.
        "msg_type": str(msg_type),

        # ID of this specific message.
        "msg_id": _as_uint32(msg_id),

        # Optional ID of the message this is responding to.
        # None means this is not a direct reply.
        "reply_to_msg_id": _optional_uint32(reply_to_msg_id),

        # Logical node IDs, not hardware types.
        # Examples: "primary", "platform", "broadcast".
        "sender_id": str(sender_id),
        "receiver_id": str(receiver_id),

        # Local time on the sender.
        # Do not assume sender/receiver clocks are synced.
        "sender_time": float(sender_time),
    }


# ---------------------------------------------------------------------
# Encode functions
# ---------------------------------------------------------------------

def encode_cmd_msg(
    msg_id: int,
    sender_id: str,
    receiver_id: str,
    sender_time: float,
    cmd_name: str,
    cmd_payload: dict[str, Any] | None = None,
    reply_to_msg_id=None,
) -> bytes:
    """
    Generic command message.

    Examples:
        cmd_name = "platform_control"
        cmd_name = "request_telemetry"
        cmd_name = "request_config"
    """

    msg = _make_base_msg(
        msg_type=MSG_TYPE_CMD,
        msg_id=msg_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        sender_time=sender_time,
        reply_to_msg_id=reply_to_msg_id,
    )

    msg["cmd_name"] = str(cmd_name)
    msg["cmd_payload"] = {} if cmd_payload is None else dict(cmd_payload)

    return _json_dumps(msg)


def encode_telemetry_msg(
    msg_id: int,
    sender_id: str,
    receiver_id: str,
    sender_time: float,
    telemetry_name: str,
    telemetry_payload: dict[str, Any] | None = None,
    reply_to_msg_id=None,
    device_type: str | None = None,
) -> bytes:
    """
    Generic telemetry message.

    Can be periodic telemetry or a response to a request.
    """

    msg = _make_base_msg(
        msg_type=MSG_TYPE_TELEMETRY,
        msg_id=msg_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        sender_time=sender_time,
        reply_to_msg_id=reply_to_msg_id,
    )

    msg["telemetry_name"] = str(telemetry_name)
    msg["telemetry_payload"] = (
        {} if telemetry_payload is None else dict(telemetry_payload)
    )

    # Optional metadata.
    # Examples: "rpi4", "stm32", "laptop", "jetson".
    if device_type is not None:
        msg["device_type"] = str(device_type)

    return _json_dumps(msg)


def encode_error_msg(
    msg_id: int,
    sender_id: str,
    receiver_id: str,
    sender_time: float,
    error_text: str,
    error_code: int = 0,
    reply_to_msg_id=None,
) -> bytes:
    """
    Generic error message.

    Useful for malformed packets, unsupported commands, unsafe commands, etc.
    """

    msg = _make_base_msg(
        msg_type=MSG_TYPE_ERROR,
        msg_id=msg_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        sender_time=sender_time,
        reply_to_msg_id=reply_to_msg_id,
    )

    msg["error_code"] = int(error_code)
    msg["error_text"] = str(error_text)

    return _json_dumps(msg)


# ---------------------------------------------------------------------
# Generic decode
# ---------------------------------------------------------------------

def decode_any_msg(data: bytes) -> dict:
    """
    Decode any valid protocol message.

    This only decodes JSON and validates the common header.
    It does not assume whether the message is cmd, telemetry, or error.
    """

    msg = _json_loads(data)
    _validate_common_header(msg)

    msg_type = msg.get("msg_type")

    if msg_type not in (
        MSG_TYPE_CMD,
        MSG_TYPE_TELEMETRY,
        MSG_TYPE_ERROR,
    ):
        raise ValueError(f"Unknown msg_type: {msg_type}")

    return msg


# ---------------------------------------------------------------------
# Message-specific validators
# ---------------------------------------------------------------------

def validate_cmd_msg(msg: dict) -> None:
    if msg.get("msg_type") != MSG_TYPE_CMD:
        raise ValueError(f"Expected {MSG_TYPE_CMD}, got {msg.get('msg_type')}")

    if "cmd_name" not in msg:
        raise ValueError("Missing cmd_name")

    if "cmd_payload" not in msg:
        raise ValueError("Missing cmd_payload")

    if not isinstance(msg["cmd_payload"], dict):
        raise ValueError("cmd_payload must be a dict")


def validate_telemetry_msg(msg: dict) -> None:
    if msg.get("msg_type") != MSG_TYPE_TELEMETRY:
        raise ValueError(f"Expected {MSG_TYPE_TELEMETRY}, got {msg.get('msg_type')}")

    if "telemetry_name" not in msg:
        raise ValueError("Missing telemetry_name")

    if "telemetry_payload" not in msg:
        raise ValueError("Missing telemetry_payload")

    if not isinstance(msg["telemetry_payload"], dict):
        raise ValueError("telemetry_payload must be a dict")


def validate_error_msg(msg: dict) -> None:
    if msg.get("msg_type") != MSG_TYPE_ERROR:
        raise ValueError(f"Expected {MSG_TYPE_ERROR}, got {msg.get('msg_type')}")

    if "error_code" not in msg:
        raise ValueError("Missing error_code")

    if "error_text" not in msg:
        raise ValueError("Missing error_text")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def msg_is_for_node(msg: dict, local_node_id: str) -> bool:
    """
    True if the message is addressed to this node or broadcast.
    """

    receiver_id = msg.get("receiver_id")
    return receiver_id == local_node_id or receiver_id == NODE_BROADCAST


# Note: each endpoint increments its own message counter
def next_msg_id(last_msg_id) -> int:
    """
    None -> 0
    0xFFFFFFFF -> 0
    otherwise increment
    """

    if last_msg_id is None:
        return 0
    
    last_msg_id = int(last_msg_id)
    return last_msg_id + 1 if (last_msg_id is not None and last_msg_id < MAX_MSG_ID) else 0
