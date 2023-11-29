from __future__ import annotations
# global
import logging
import cv2 as cv
# local
from ur_pilot.camera.camera_base import CameraBase


LOGGER = logging.getLogger(__name__)


class CameraBuildIn(CameraBase):

    _type_id = "build_in"

    def __init__(self,
                 name: str,
                 size: tuple[int, int] = (1280, 720),
                 launch: bool = True) -> None:
        # Initialize super class
        super().__init__(name, size)
        # OpenCV camera capture
        self.cap: cv.VideoCapture | None = None
        if launch:
            self.start()

    def start(self) -> None:
        # Create OpenCV video capture and start video stream
        self.cap = cv.VideoCapture(0)
        self.alive = True
        self.thread.start()

    def update(self) -> None:
        assert self.cap
        while self.alive:
            self.alive, raw_frame = self.cap.read()
            if self.alive:
                self.color_frame = cv.resize(raw_frame, self.size, interpolation=cv.INTER_CUBIC)

    def destroy(self) -> None:
        self.alive = False
        if self.cap is not None:
            self.cap.release()
