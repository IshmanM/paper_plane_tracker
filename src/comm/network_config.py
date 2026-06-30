from src.comm.protocol import NODE_PRIMARY, NODE_PLATFORM

DEFAULT_UDP_PORT = 50000
DEFAULT_MAX_PACKET_BYTES = 4096

# Logical node IDs. These are not hardware-specific.
PRIMARY_NODE_ID = NODE_PRIMARY
ENDPOINT_NODE_ID = NODE_PLATFORM

# Safe fallback values for direct Ethernet setup.
# These are private LAN IPs, not public internet IPs.
DEFAULT_PRIMARY_IP = "192.168.50.1"
DEFAULT_ENDPOINT_IP = "192.168.50.2"

try:
    from src.comm.network_config_local import (
        PRIMARY_IP,
        ENDPOINT_IP,
        UDP_PORT,
    )
except ImportError:
    PRIMARY_IP = DEFAULT_PRIMARY_IP
    ENDPOINT_IP = DEFAULT_ENDPOINT_IP
    UDP_PORT = DEFAULT_UDP_PORT