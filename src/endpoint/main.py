import argparse
import board

import src.endpoint.config as config
from src.endpoint.drivers.servo_driver import ServoDriver
from src.endpoint.mechanisms.foam_mechanism import FoamMechanism
from src.endpoint.mechanisms.orient_mechanism import OrientMechanism
from src.endpoint.controller import EndpointController, CmdResult
from src.endpoint.server import EndpointServer
from src.endpoint.drivers.dc_motor_driver import DCMotorDriver

from src.comm.link import UdpLink
from src.comm.network_config import(
    PRIMARY_IP, 
    ENDPOINT_IP,
    UDP_PORT,
    PRIMARY_NODE_ID,
    ENDPOINT_NODE_ID,
    DEFAULT_MAX_PACKET_BYTES
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the endpoint server.")
    parser.add_argument(
        "--no-servo-calibration",
        action="store_true",
        help="ONLY for empirical servo characterization: bypass servo trim and polynomial/lookup calibration.",
    )
    args = parser.parse_args()
    use_servo_calibration = not args.no_servo_calibration

    i2c = board.I2C()

    servo_driver = ServoDriver(
        i2c=i2c,
        frequency_hz=config.PCA9685_FREQUENCY_HZ, 
        num_channels=config.PCA9685_NUM_CHANNELS,
        default_calibration=config.DEFAULT_SERVO_CALIBRATION,
        pca_reference_clock_frequency_hz=config.PCA9685_REFERENCE_CLOCK_FREQUENCY_HZ,
        use_calibration=use_servo_calibration,
    )

    print(f"Servo calibration: {'ENABLED' if use_servo_calibration else 'BYPASSED FOR CHARACTERIZATION'}")

    for channel, calibration in config.SERVO_CALIBRATIONS.items():
        servo_driver.set_channel_calibration(channel, calibration)

    # dc_motor_driver = DCMotorDriver(
    #     motor_1_gpio_pins=(board.D5, board.D6,),
    #     motor_2_gpio_pins=(board.D16,board.D20,),
    #     pwm_frequency_hz=20000
    # )
    dc_motor_driver = DCMotorDriver(
        motor_1_gpio_pins=(12, 18),
        motor_2_gpio_pins=(13, 19),
        pwm_frequency_hz=20_000,
        sleep_gpio=None
    )

    orient_mechanism = OrientMechanism(
        servo_driver=servo_driver,
        pan_channel=config.PAN_CHANNEL,
        tilt_channel=config.TILT_CHANNEL,
        default_pan_deg=config.DEFAULT_PAN_ANGLE,
        default_tilt_deg=config.DEFAULT_TILT_ANGLE
    )
    
    foam_mechanism = FoamMechanism(
        servo_driver=servo_driver,
        dc_motor_driver=dc_motor_driver,
        foam_channel=config.FOAM_CHANNEL,
        motor_1_speed=config.FOAM_MOTOR_SPEED_MAGNITUDE,
        motor_2_speed=config.FOAM_MOTOR_SPEED_MAGNITUDE
    )
    
    controller = EndpointController(orient_mechanism, foam_mechanism)

    link = UdpLink(
        local_ip=ENDPOINT_IP,
        remote_ip=PRIMARY_IP,
        port=UDP_PORT,
        local_node_id=ENDPOINT_NODE_ID,
        remote_node_id=PRIMARY_NODE_ID,
        max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
        check_remote_ip=True
    )

    server = EndpointServer(
        controller, 
        link, 
        refresh_frequency_hz=120,
        telemetry_frequency_hz=15
    )

    try:
        go_safe_result = controller.go_safe()
        if go_safe_result.is_error:
            raise RuntimeError(go_safe_result.error_text)
        
        server.run()
    
    finally:
    
        try:
            go_safe_result = controller.go_safe()
            if go_safe_result.is_error:
                print(f"Failed to enter safe mode during shutdown: {go_safe_result.error_text}")

        except Exception as e:
            print(f"Unexpected failure while entering safe mode during shutdown: {e}")

        try:
            # Stop the worker thread and ensure the trigger servo is reset.
            foam_mechanism.stop()
        except Exception as e:
            print(f"Failed to stop foam mechanism: {e}")
    
        try:
            link.close()
        except Exception as e:
            print(f"Failed to close UDP link: {e}")

        try:
            # Release the PWM GPIO resources after FoamMechanism has stopped.
            dc_motor_driver.close()
        except Exception as e:
            print(f"Failed to close DC motor driver: {e}")

        try:
            servo_driver.close(release=False)
        except Exception as e:
            print(f"Failed to close servo driver: {e}")