from __future__ import annotations
# global
import logging
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
# local
from ur_pilot.camera.camera_base import CameraBase
# typing
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


class CameraRealSense(CameraBase):

    _type_id = "realsense"

    def __init__(self,
                 name: str,
                 size: tuple[int, int] = (1280, 720),
                 launch: bool = True) -> None:
        # Initialize super class
        super().__init__(name, size)
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        LOGGER.debug(f"\nHello RealSense camera {device_product_line}")
        # Search for rgb sensor
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            LOGGER.error("The demo requires Depth camera with Color sensor")
            exit(0)
        # Get depth sensor scale
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        LOGGER.debug(f"Depth Scale is: {self.depth_scale}")
        # Set up additional depth frame
        self.depth_frame = np.zeros((3,) + self.size, dtype=np.uint8).T
        if launch:
            self.start()

    def start(self) -> None:
        # Configure streams
        self.config.enable_stream(rs.stream.depth, self.size[0], self.size[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.size[0], self.size[1], rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(self.config)
        # Start thread
        self.alive = True
        self.thread.start()

    def update(self) -> None:

        align_to = rs.stream.color
        align = rs.align(align_to)

        while self.alive:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to the color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            else:
                # Convert images to numpy arrays
                self.color_frame = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                pixel_distance_in_meters = aligned_depth_frame.get_distance(2 * 35, 2 * 117)
                # print(pixel_distance_in_meters)
                depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)
                # print((self.depth_scale * depth_image[2*224, 2*207]))
                # self.depth_frame = depth_image
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                self.depth_frame = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_TURBO)

    def get_depth_frame(self) -> npt.NDArray[np.uint8]:
        return self.depth_frame

    def destroy(self) -> None:
        if self.alive:
            self.alive = False
            self.pipeline.stop()
