from src.endpoint.drivers.servo_driver import ServoDriver


class FoamMechanismError(Exception):
    """
    Raised when the foam mechanism rejects or fails to apply a command.
    """
    pass

class FoamMechanism:
    def __init__(
        self,
        servo_driver: ServoDriver,
        foam_channel: int # probably just 1 trigger servo
    ):
        #
        #
        #
        return
        raise NotImplementedError
    
    def trigger(self):
        #
        #
        #
        return
        raise NotImplementedError
    
    def halt_trigger(self):
        #
        #
        #
        return
        raise NotImplementedError