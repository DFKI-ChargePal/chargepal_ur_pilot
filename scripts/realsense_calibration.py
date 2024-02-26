""" Script to calibrate the robot camera. """
import logging
import camera_kit as ck


_CHECKER_SIZE = 16
_CHESSBOARD_SIZE = (11, 17)
_chessboard = ck.ChessboardDescription(_CHESSBOARD_SIZE, _CHECKER_SIZE)


def realsense_calibration() -> None:

    with ck.camera_manager(name='realsense_tcp_cam', logger_level=logging.INFO) as camera:
        # Record some images
        ck.CameraCalibration().record_images(camera)
        # Run calibration
        cc = ck.CameraCalibration().find_coeffs(camera, _chessboard, display=True)
        camera.save_coefficients(cc)


if __name__ == '__main__':
    realsense_calibration()
