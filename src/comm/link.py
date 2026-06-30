import socket

from src.comm.protocol import (
    MSG_TYPE_CMD,
    MSG_TYPE_TELEMETRY,
    MSG_TYPE_ERROR,
    NODE_BROADCAST,
    decode_any_msg,
    encode_cmd_msg,
    encode_telemetry_msg,
    encode_error_msg,
    msg_is_for_node,
    validate_cmd_msg,
    validate_telemetry_msg,
    validate_error_msg,
)


# class TcpLink: ... (implement if needed, maybe make a higher abstract class)

class UdpLink:
    def __init__(
        self,
        local_ip: str,
        remote_ip: str,
        port: int,
        local_node_id: str,
        remote_node_id: str,
        max_packet_bytes: int = 4096,
        check_remote_ip: bool = True,
    ):
        self.local_addr = (local_ip, port)
        self.remote_addr = (remote_ip, port)

        self.local_node_id = local_node_id
        self.remote_node_id = remote_node_id

        self.max_packet_bytes = max_packet_bytes
        self.check_remote_ip = check_remote_ip

        # Create a non-blocking IPv4 UDP socket for sending/receiving datagrams.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.local_addr)
        self.sock.setblocking(False)

        # Stores valid received messages that have not been consumed yet.
        # This prevents recv_telemetry_available() from accidentally discarding
        # cmd/error messages that arrived in the same UDP drain.
        self._rx_queue = []

    def close(self):
        self.sock.close()

    # ------------------------------------------------------------------
    # Low-level receive/send
    # ------------------------------------------------------------------

    def send_raw(self, packet: bytes):
        self.sock.sendto(packet, self.remote_addr)

    def _drain_socket(self):
        """
        Drain all currently available UDP packets into self._rx_queue.

        Non-blocking:
            - does not wait for new packets
            - only processes packets already waiting in the OS socket buffer
        """

        while True:
            try:
                data, addr = self.sock.recvfrom(self.max_packet_bytes)
            except BlockingIOError:
                break

            if self.check_remote_ip and addr[0] != self.remote_addr[0]:
                print(f"Ignoring UDP packet from unexpected IP: {addr}")
                continue

            try:
                msg = decode_any_msg(data)
            except Exception as e:
                print(f"Bad UDP packet from {addr}: {e}")
                continue

            if not msg_is_for_node(msg, self.local_node_id):
                continue

            self._rx_queue.append(msg)

    def recv_available(self) -> list[dict]:
        """
        Return all currently available valid messages for this node.
        """

        self._drain_socket()

        msgs = self._rx_queue
        self._rx_queue = []

        return msgs

    def _recv_by_type_available(self, msg_type: str) -> list[dict]:
        """
        Return currently available messages of one type.

        Messages of other types stay in self._rx_queue for later.
        """

        self._drain_socket()

        selected = []
        remaining = []

        for msg in self._rx_queue:
            if msg.get("msg_type") == msg_type:
                selected.append(msg)
            else:
                remaining.append(msg)

        self._rx_queue = remaining

        return selected

    # ------------------------------------------------------------------
    # Generic send helpers
    # ------------------------------------------------------------------

    def send_cmd(
        self,
        msg_id: int,
        sender_time: float,
        cmd_name: str,
        cmd_payload: dict | None = None,
        receiver_id: str | None = None,
        reply_to_msg_id=None,
    ):
        packet = encode_cmd_msg(
            msg_id=msg_id,
            sender_id=self.local_node_id,
            receiver_id=self.remote_node_id if receiver_id is None else receiver_id,
            sender_time=sender_time,
            cmd_name=cmd_name,
            cmd_payload=cmd_payload,
            reply_to_msg_id=reply_to_msg_id,
        )

        self.send_raw(packet)

    def send_telemetry(
        self,
        msg_id: int,
        sender_time: float,
        telemetry_name: str,
        telemetry_payload: dict | None = None,
        receiver_id: str | None = None,
        reply_to_msg_id=None,
        device_type: str | None = None,
    ):
        packet = encode_telemetry_msg(
            msg_id=msg_id,
            sender_id=self.local_node_id,
            receiver_id=self.remote_node_id if receiver_id is None else receiver_id,
            sender_time=sender_time,
            telemetry_name=telemetry_name,
            telemetry_payload=telemetry_payload,
            reply_to_msg_id=reply_to_msg_id,
            device_type=device_type,
        )

        self.send_raw(packet)

    def send_error(
        self,
        msg_id: int,
        sender_time: float,
        error_text: str,
        error_code: int = 0,
        receiver_id: str | None = None,
        reply_to_msg_id=None,
    ):
        packet = encode_error_msg(
            msg_id=msg_id,
            sender_id=self.local_node_id,
            receiver_id=self.remote_node_id if receiver_id is None else receiver_id,
            sender_time=sender_time,
            error_text=error_text,
            error_code=error_code,
            reply_to_msg_id=reply_to_msg_id,
        )

        self.send_raw(packet)

    # ------------------------------------------------------------------
    # Generic receive helpers
    # ------------------------------------------------------------------

    def recv_cmds_available(self) -> list[dict]:
        msgs = self._recv_by_type_available(MSG_TYPE_CMD)

        valid_msgs = []
        for msg in msgs:
            try:
                validate_cmd_msg(msg)
            except Exception as e:
                print(f"Bad cmd msg: {e}")
                continue

            valid_msgs.append(msg)

        return valid_msgs

    def recv_telemetry_available(self) -> list[dict]:
        msgs = self._recv_by_type_available(MSG_TYPE_TELEMETRY)

        valid_msgs = []
        for msg in msgs:
            try:
                validate_telemetry_msg(msg)
            except Exception as e:
                print(f"Bad telemetry msg: {e}")
                continue

            valid_msgs.append(msg)

        return valid_msgs

    def recv_errors_available(self) -> list[dict]:
        msgs = self._recv_by_type_available(MSG_TYPE_ERROR)

        valid_msgs = []
        for msg in msgs:
            try:
                validate_error_msg(msg)
            except Exception as e:
                print(f"Bad error msg: {e}")
                continue

            valid_msgs.append(msg)

        return valid_msgs