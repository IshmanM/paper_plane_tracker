import math
import numpy as np

from src.endpoint.drivers.servo_driver import ServoDriver


class OrientMechanismError(Exception):
    """
    Raised when the orientation mechanism rejects or fails to apply a command.
    """
    pass


class OrientMechanism:
    """
    Mechanism-level wrapper for platform orientation.

    This class gives physical meaning to two servo channels:
        - pan
        - tilt

    It uses ServoDriver underneath, but it does not own the PCA9685 directly.

    Responsibilities:
        - command pan/tilt angles together
        - command pan angle alone
        - command tilt angle alone
        - move to default orientation pose
        - remember last commanded pan/tilt angles

    Not responsible for:
        - UDP
        - msg_id tracking
        - command timeout
        - endpoint safe-mode decision
        - object tracking
        - intercept planning
        - platform state machine
        - high-level angle feasibility checks

    Note:
        The endpoint controller should decide WHEN to enter safe/default state.
        Orient only knows HOW to move the orientation mechanism to its default pose.
    """

    def __init__(
        self,
        servo_driver: ServoDriver,
        pan_channel: int,
        tilt_channel: int,
        default_pan_deg: float,
        default_tilt_deg: float,
    ):
        self.servo_driver = servo_driver

        self.pan_channel = int(pan_channel)
        self.tilt_channel = int(tilt_channel)

        self.default_pan_deg = float(default_pan_deg)
        self.default_tilt_deg = float(default_tilt_deg)

        self.last_pan_deg: float | None = None
        self.last_tilt_deg: float | None = None

        try:
            self._validate_angle(self.default_pan_deg, "default_pan_deg")
            self._validate_angle(self.default_tilt_deg, "default_tilt_deg")
            
            self._validate_angle_in_channel_calibration(
                channel=self.pan_channel,
                angle_deg=self.default_pan_deg,
                name="default_pan_deg",
            )

            self._validate_angle_in_channel_calibration(
                channel=self.tilt_channel,
                angle_deg=self.default_tilt_deg,
                name="default_tilt_deg",
            )

        except (TypeError, ValueError) as e:
            raise OrientMechanismError(f"Invalid default orientation: {e}") from e


    def set_angles_deg(
        self,
        pan_deg: float,
        tilt_deg: float,
    ) -> tuple[float, float]:
        """
        Command both orientation servos.

        Returns:
            (used_pan_deg, used_tilt_deg)

        Raises:
            OrientMechanismError if either angle is invalid or rejected by ServoDriver.
        """

        try:
            pan_deg = float(pan_deg)
            tilt_deg = float(tilt_deg)

            self._validate_angle(pan_deg, "pan_deg")
            self._validate_angle(tilt_deg, "tilt_deg")

            # Validate both before moving either servo.
            # This avoids pan moving successfully and then tilt failing.
            self._validate_angle_in_channel_calibration(
                channel=self.pan_channel,
                angle_deg=pan_deg,
                name="pan_deg",
            )

            self._validate_angle_in_channel_calibration(
                channel=self.tilt_channel,
                angle_deg=tilt_deg,
                name="tilt_deg",
            )

            used_pan_deg = self.servo_driver.set_angle_deg(
                channel=self.pan_channel,
                angle_deg=pan_deg,
            )

            used_tilt_deg = self.servo_driver.set_angle_deg(
                channel=self.tilt_channel,
                angle_deg=tilt_deg,
            )

        except (TypeError, ValueError) as e:
            raise OrientMechanismError(
                f"Failed to set orientation angles "
                f"pan_deg={pan_deg}, tilt_deg={tilt_deg}: {e}"
            ) from e

        self.last_pan_deg = used_pan_deg
        self.last_tilt_deg = used_tilt_deg

        return used_pan_deg, used_tilt_deg

    def set_angles_array_deg(
        self,
        angles_deg: np.ndarray,
        pan_idx: int = 0,
        tilt_idx: int = 1,
    ) -> tuple[float, float]:
        """
        Convenience method for commanding from an angle array.

        Example:
            angles_deg = [pan_deg, tilt_deg]

        Raises:
            OrientMechanismError if the array is invalid or the command fails.
        """

        try:
            angles_deg = np.asarray(angles_deg, dtype=float).reshape(-1)

            if angles_deg.shape[0] <= max(pan_idx, tilt_idx):
                raise ValueError(
                    f"angles_deg must contain pan_idx={pan_idx} and tilt_idx={tilt_idx}"
                )

            pan_deg = float(angles_deg[pan_idx])
            tilt_deg = float(angles_deg[tilt_idx])

        except (TypeError, ValueError, IndexError) as e:
            raise OrientMechanismError(
                f"Invalid orientation angle array: {e}"
            ) from e

        return self.set_angles_deg(
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
        )

    def set_pan_deg(self, pan_deg: float) -> float:
        """
        Command only the pan servo.

        Returns:
            used_pan_deg

        Raises:
            OrientMechanismError if pan_deg is invalid or rejected by ServoDriver.
        """

        try:
            pan_deg = float(pan_deg)

            self._validate_angle(pan_deg, "pan_deg")

            self._validate_angle_in_channel_calibration(
                channel=self.pan_channel,
                angle_deg=pan_deg,
                name="pan_deg",
            )

            used_pan_deg = self.servo_driver.set_angle_deg(
                channel=self.pan_channel,
                angle_deg=pan_deg,
            )

        except (TypeError, ValueError) as e:
            raise OrientMechanismError(
                f"Failed to set pan angle pan_deg={pan_deg}: {e}"
            ) from e

        self.last_pan_deg = used_pan_deg
        return used_pan_deg

    def set_tilt_deg(self, tilt_deg: float) -> float:
        """
        Command only the tilt servo.

        Returns:
            used_tilt_deg

        Raises:
            OrientMechanismError if tilt_deg is invalid or rejected by ServoDriver.
        """

        try:
            tilt_deg = float(tilt_deg)

            self._validate_angle(tilt_deg, "tilt_deg")

            self._validate_angle_in_channel_calibration(
                channel=self.tilt_channel,
                angle_deg=tilt_deg,
                name="tilt_deg",
            )

            used_tilt_deg = self.servo_driver.set_angle_deg(
                channel=self.tilt_channel,
                angle_deg=tilt_deg,
            )

        except (TypeError, ValueError) as e:
            raise OrientMechanismError(
                f"Failed to set tilt angle tilt_deg={tilt_deg}: {e}"
            ) from e

        self.last_tilt_deg = used_tilt_deg
        return used_tilt_deg

    def go_default(self) -> tuple[float, float]:
        """
        Move orientation mechanism to its default pose.

        This does not decide whether the endpoint should be in safe mode.
        It only performs the default orientation action.

        Raises:
            OrientMechanismError if the default orientation command fails.
        """

        try:
            return self.set_angles_deg(
                pan_deg=self.default_pan_deg,
                tilt_deg=self.default_tilt_deg,
            )

        except OrientMechanismError as e:
            raise OrientMechanismError(
                f"Failed to move orientation mechanism to default pose: {e}"
            ) from e

    def get_last_angles_deg(self) -> np.ndarray | None:
        """
        Return last commanded orientation angles as [pan_deg, tilt_deg].

        This is commanded state, not measured servo position.
        """

        if self.last_pan_deg is None or self.last_tilt_deg is None:
            return None

        return np.array(
            [self.last_pan_deg, self.last_tilt_deg],
            dtype=float,
        )
    
    def get_last_pan_deg(self) -> float | None:
        return self.last_pan_deg


    def get_last_tilt_deg(self) -> float | None:
        return self.last_tilt_deg
    

    def _validate_angle_in_channel_calibration(
        self,
        channel: int,
        angle_deg: float,
        name: str,
    ) -> None:
        calibration = self.servo_driver.get_calibration(channel)

        if not (
            calibration.min_angle_deg
            <= angle_deg
            <= calibration.max_angle_deg
        ):
            raise ValueError(
                f"{name}={angle_deg} is outside channel {channel} calibration range "
                f"[{calibration.min_angle_deg}, {calibration.max_angle_deg}]"
            )

    @staticmethod
    def _validate_angle(angle_deg: float, name: str) -> None:
        if not math.isfinite(float(angle_deg)):
            raise ValueError(f"{name} must be finite")