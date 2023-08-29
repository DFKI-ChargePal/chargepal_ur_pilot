""" Script to calibrate the robot camera. """
import logging
import chargepal_aruco as ca


_CHECKER_SIZE = 16
_CHESSBOARD_DIM = (11, 17)


def realsense_calibration() -> None:

    # Create calibration instance
    ca.set_logging_level(logging.INFO)
    cam = ca.RealSenseCamera('tcp_cam_realsense')
    board = ca.Chessboard(_CHESSBOARD_DIM, _CHECKER_SIZE)

    calib = ca.Calibration(cam)

    # Record calibration images
    calib.camera_calibration_record_images()

    # Run camera calibration and save coefficients
    coeffs = calib.camera_calibration_find_coeffs(board=board, display=True)
    cam.save_coefficients(coeffs)

    cam.destroy()


if __name__ == '__main__':
    realsense_calibration()
