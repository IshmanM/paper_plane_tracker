from enum import Enum, auto
import numpy as np


class PlatformGeometrySpecId(Enum):
    PLATFORM_1 = auto()


class PlatformGeometrySpec:
    def __init__(
        self,
        foam_mechanism_origin_offset_m: np.ndarray,
        rotation_platform_from_foam_mechanism_at_forward: np.ndarray,
        laser_origin_offset_foam_mechanism_m: np.ndarray | None = None,
    ):
        # Platform origin -> dart exit when pan/tilt are forward, in platform FLU coordinates.
        self.foam_mechanism_origin_offset_m = np.asarray(foam_mechanism_origin_offset_m, dtype=float).copy()

        # Foam-mechanism FLU frame -> platform FLU frame when pan/tilt are forward.
        self.rotation_platform_from_foam_mechanism_at_forward = np.asarray(
            rotation_platform_from_foam_mechanism_at_forward, dtype=float
        ).copy()

        # Dart exit -> laser origin, expressed in the foam-mechanism FLU frame. (Laser used in calibration)
        self.laser_origin_offset_foam_mechanism_m = (
            np.zeros(3, dtype=float)
            if laser_origin_offset_foam_mechanism_m is None
            else np.asarray(laser_origin_offset_foam_mechanism_m, dtype=float).copy()
        )

        if self.foam_mechanism_origin_offset_m.shape != (3,) or not np.all(np.isfinite(self.foam_mechanism_origin_offset_m)):
            raise ValueError("foam_mechanism_origin_offset_m must be a finite length-3 vector")

        if self.rotation_platform_from_foam_mechanism_at_forward.shape != (3, 3) or not np.all(np.isfinite(self.rotation_platform_from_foam_mechanism_at_forward)):
            raise ValueError("rotation_platform_from_foam_mechanism_at_forward must be a finite 3x3 matrix")

        R = self.rotation_platform_from_foam_mechanism_at_forward
        if not np.allclose(R.T@R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("rotation_platform_from_foam_mechanism_at_forward must be a proper rotation matrix")

        if self.laser_origin_offset_foam_mechanism_m.shape != (3,) or not np.all(np.isfinite(self.laser_origin_offset_foam_mechanism_m)):
            raise ValueError("laser_origin_offset_foam_mechanism_m must be a finite length-3 vector")


# TODO: tune based on CAD/real measurements
PLATFORM_GEOMETRY_SPECS = {
    PlatformGeometrySpecId.PLATFORM_1: PlatformGeometrySpec(
        foam_mechanism_origin_offset_m=np.array([
            0.0,     # +ve forward
            -0.005,  # +ve left 
            0.0525,  # +ve up
        ]),
        rotation_platform_from_foam_mechanism_at_forward=np.eye(3),
        laser_origin_offset_foam_mechanism_m=np.array([
            0.0,     # +ve forward
           -0.052,   # +ve left 
           -0.005,   # +ve up   
        ]),
    ),
}