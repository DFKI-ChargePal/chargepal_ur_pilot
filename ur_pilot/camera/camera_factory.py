from __future__ import annotations
# local
from ur_pilot.camera.camera_base import CameraBase
from ur_pilot.camera.camera_build_in import CameraBuildIn
from ur_pilot.camera.camera_realsense import CameraRealSense


class CameraFactory:

    # @staticmethod
    # def available_cameras() -> list[str]:
    #     return [CameraBuildIn.type_id, CameraRealSense.type_id]

    @staticmethod
    def create(name: str) -> CameraBase:
        if name.startswith('realsense'):
            return CameraRealSense(name)
        elif name.startswith('build_in'):
            return CameraBuildIn(name)
        else:
            raise KeyError(f"Cannot map '{name}' to camera class. Make sure camera name starts with ")
