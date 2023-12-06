from __future__ import annotations

# global
import os
import tomli
import shutil
import tomli_w
import logging
import cv2 as cv
import numpy as np
from pathlib import Path
from tomlkit import document

# local
from camera_kit import CameraBase

# typing
from typing import Any
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


class HandEyeCalibration:

    @staticmethod
    def _create_paths(dir_path: Path) -> tuple[Path, Path, Path]:
        """ Extend the given path by the needed extensions for the hand eye calibration. If no path are given,
            use default one of the class.

        Args:
            dir_path: Path to the parent folder

        Returns:
            Tuple with path objects
        """
        dp_imgs = dir_path.joinpath('imgs')
        dp_cam2tgt = dir_path.joinpath('cam2tgt')
        dp_base2tcp = dir_path.joinpath('base2tcp')
        return dp_imgs, dp_cam2tgt, dp_base2tcp

    @staticmethod
    def clear_directories(camera: CameraBase, dir_path: str = "") -> None:
        """ Method to prepare hand eye calibration recording """

        # Get path
        target_dir = Path(dir_path) if dir_path else camera.cam_info_dir.joinpath('calibration_hand_eye')
        dir_paths = HandEyeCalibration._create_paths(target_dir)
        # Delete all directories with old files
        for dpath in dir_paths:
            if dpath.exists():
                shutil.rmtree(dpath)

    @staticmethod
    def record_sample(camera: CameraBase,
                      file_prefix: str,
                      T_base2tcp: npt.NDArray[np.float_],
                      T_cam2tgt: npt.NDArray[np.float_],
                      dir_path: str = ""
                      ) -> None:
        """ Method to record a hand eye calibration sample

        Args:
            camera:      The camera object
            file_prefix: Unique sample identifier
            T_base2tcp:  Transformation matrix of the TCP wrt. the robot arm base frame
            T_cam2tgt:   Transformation matrix of the target wrt. to the camera frame
            dir_path:    Optional a path to the directory where the data should be stored.

        """
        # Create directories if needed.
        target_dir = Path(dir_path) if dir_path else camera.cam_info_dir.joinpath('calibration_hand_eye')
        dir_paths = HandEyeCalibration._create_paths(target_dir)
        for dpath in dir_paths:
            dpath.mkdir(parents=True, exist_ok=True)
        dp_imgs, dp_cam2tgt, dp_base2tcp = dir_paths
        # Build file paths
        img_fp = dp_imgs.joinpath(f"{file_prefix}_h2eye.png")
        cam2tgt_fp = dp_cam2tgt.joinpath(f"{file_prefix}_h2eye.toml")
        base2tcp_fp = dp_base2tcp.joinpath(f"{file_prefix}_h2eye.toml")
        # Save image
        cv.imwrite(os.fspath(img_fp), camera.get_color_frame())

        def dump_np2toml(data_array: npt.NDArray[Any], name: str, fp: Path) -> None:
            toml_doc = document()
            toml_doc.add(name, data_array.tolist())
            with fp.open(mode='wb') as f:
                tomli_w.dump(toml_doc, f)

        # Save transformation matrices
        dump_np2toml(T_cam2tgt, 'T_cam2tgt', cam2tgt_fp)
        dump_np2toml(T_base2tcp, 'T_base2tcp', base2tcp_fp)

    @staticmethod
    def est_transformation(camera: CameraBase, dir_path: str = "") -> npt.NDArray[np.float_]:
        """ Method to estimate the transformation between robot hand and camera.

        Args:
            camera:   The camera object
            dir_path: Optional a path to the directory where the data are stored.

        Returns:
            Transformation matrix from camera to TCP
        """
        target_dir = Path(dir_path) if dir_path else camera.cam_info_dir.joinpath('calibration_hand_eye')
        dir_paths = HandEyeCalibration._create_paths(target_dir)
        for dp in dir_paths:
            if not dp.exists():
                raise NotADirectoryError(f"Folder with path {dp} not found.")
        # Unpack paths
        dp_imgs, dp_cam2tgt, dp_base2tcp = dir_paths
        # Create lists to store transformations
        R_cam2tgt: list[npt.NDArray[np.float_]] = []
        t_cam2tgt: list[npt.NDArray[np.float_]] = []
        R_base2tcp: list[npt.NDArray[np.float_]] = []
        t_base2tcp: list[npt.NDArray[np.float_]] = []

        # Read transformations
        for cam2tgt_fp in dp_cam2tgt.glob('*.toml'):
            # Build base-2-tcp file_path equivalent
            file_name = cam2tgt_fp.name
            base2tcp_fp = dp_base2tcp.joinpath(file_name)
            if base2tcp_fp.exists():
                # Load transformation matrices with the same prefix
                with base2tcp_fp.open(mode='rb') as fp:
                    toml_reading = tomli.load(fp)
                    T_base2tcp = np.asarray(toml_reading["T_base2tcp"])
                with cam2tgt_fp.open(mode='rb') as fp:
                    toml_reading = tomli.load(fp)
                    T_cam2tgt = np.asarray(toml_reading["T_cam2tgt"])
                # Convert to rotation matrix and translation vector
                R_cam2tgt.append(T_cam2tgt[:3, :3])
                t_cam2tgt.append(T_cam2tgt[:3, 3])
                R_base2tcp.append(T_base2tcp[:3, :3])
                t_base2tcp.append(T_base2tcp[:3, 3])
            else:
                LOGGER.warning(
                    f"There is no corresponding file for {cam2tgt_fp}. Item can not used for calibration.")

        # Run OpenCV calibration
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # !!! OpenCV convention and the convention in this program do not match. !!! #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        R_tcp2cam, t_tcp2cam = cv.calibrateHandEye(
            R_gripper2base=R_base2tcp, t_gripper2base=t_base2tcp,
            R_target2cam=R_cam2tgt, t_target2cam=t_cam2tgt)
        # Build transformation matrix
        T_tcp2cam = np.identity(4)
        T_tcp2cam[:3, :3] = R_tcp2cam
        T_tcp2cam[:3, 3] = np.squeeze(t_tcp2cam)
        return T_tcp2cam

    @staticmethod
    def save_transformation(camera: CameraBase, T_tcp2cam: npt.NDArray[np.float_], dir_path: str = "") -> None:
        """ Helper function to save the transformation matrix between TCP and camera

        Args:
            camera: Camera object to get data path
            T_tcp2cam: The transformation matrix as numpy array
            dir_path: Optional a path to the directory where the transformation is stored.

        Returns:
            None
        """
        dir_pathlib = Path(dir_path) if dir_path else camera.cam_info_dir.joinpath('calibration_hand_eye', 'tcp2cam')
        # Create target directory
        dir_pathlib.mkdir(parents=True, exist_ok=True)
        # Write to matrix to toml
        fp = dir_pathlib.joinpath('T_tcp2cam.toml')
        toml_doc = document()
        toml_doc.add('T_tcp2cam', T_tcp2cam.tolist())
        with fp.open(mode='wb') as f:
            tomli_w.dump(toml_doc, f)
        LOGGER.info(f"Save new calibration matrix in folder {str(fp.parent)}")

    @staticmethod
    def load_transformation(camera: CameraBase, dir_path: str = "") -> npt.NDArray[np.float_]:
        """ Helper function to load the calibration transformation matrix between TCP and camera.

        Args:
            camera: Camera object to get data path
            dir_path: Optional a path to the directory where the transformation is stored.

        Returns:
            None
        """
        # Create file path
        dir_pathlib = Path(dir_path) if dir_path else camera.cam_info_dir.joinpath('calibration_hand_eye', 'tcp2cam')
        file_path = dir_pathlib.joinpath('T_tcp2cam.toml')
        # Check if file exist
        if not file_path.exists():
            raise FileNotFoundError(f"File with path '{file_path}' not found!")
            # Read matrix
        with file_path.open(mode='rb') as f:
            toml_reading = tomli.load(f)
            T_tcp2cam = np.asarray(toml_reading["T_tcp2cam"])
        LOGGER.info(f"Load calibration matrix from folder {str(file_path.parent)}")
        return T_tcp2cam
