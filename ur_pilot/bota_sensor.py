"""Bota force torque sensor."""
from __future__ import annotations

# global
import time
import struct
import ctypes
import pysoem
import logging
import collections
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

#typing
from typing import Any
from numpy import typing as npt
from typing_extensions import Literal


LOGGER = logging.getLogger(__name__)


class BotaFtSensor:

    """Bota Force Torque Sensor class."""

    BOTA_VENDOR_ID = 0xB07A
    BOTA_PRODUCT_CODE = 0x00000001
    MEM_SIZE_BYTES = 77
    SINC_LENGTHS = (51, 64, 128, 205, 256, 512)

    def __init__(
        self,
        adapter: str,
        slave_pos: int = 0,
        filter_sinc_length: int = 512,
        filter_fir: bool = False,
        filter_fast: bool = False,
        filter_chop: bool = False,
    ) -> None:
        """
        Bota Force Torque Sensor.

        Parameters:
            adapter: string
                String identifier of the ethernet adapter interface the sensor is connected to
            slave_pos: int, default 0
                Index of the sensor instance as a SOEM slave in the slave list
            filter_sinc_length: int, default 512
                Length of the Sinc filter,
                value restricted to 51, 64, 128, 205, 256, 512
            filter_fir: bool, default False
                Flag for activating the FIR filter
            filter_fast: bool, default False
                Flag for activating the FAST option of the FIR filter,
                it is recommended to be disabled for real-time applications
            filter_chop: bool, default False
                Flag for activating the CHOP filter,
                it is recommended to be disabled for real-time applications
        """
        self.adapter = adapter
        if filter_sinc_length not in self.SINC_LENGTHS:
            raise ValueError(
                f"Value {filter_sinc_length} not valid for sinc_length. "
                f"Values are restricted to {', '.join(map(str, self.SINC_LENGTHS))}."
            )
        self.filter_sinc_length = filter_sinc_length
        self.filter_fir = filter_fir
        self.filter_fast = filter_fast
        self.filter_chop = filter_chop

        self._master = pysoem.Master()
        self.sampling_rate = 0
        self.time_step = 0.0
        SlaveSet = collections.namedtuple("SlaveSet", "slave_name product_code config_func")
        self._expected_slave_mapping = {
            slave_pos: SlaveSet("BFT-MEDS-ECAT-M8", self.BOTA_PRODUCT_CODE, self._sensor_setup)
        }
        self.update_process = multiprocessing.Process()
        self._init_master()
        self.slave = self._master.slaves[slave_pos]

    def start(self) -> None:
        LOGGER.debug("Starting value update process")
        self.write_buffer = shared_memory.SharedMemory(create=True, size=self.MEM_SIZE_BYTES)
        self.update_process = multiprocessing.Process(target=self._update_values, args=())
        self.read_buffer = shared_memory.SharedMemory(self.write_buffer.name)
        self.update_process.start()
        time.sleep(0.125)

    def stop(self) -> None:
        LOGGER.debug("Stopping value update process")
        self.update_process.terminate()

        LOGGER.debug("Releasing shared memory")
        self.write_buffer.close()
        self.read_buffer.close()
        self.write_buffer.unlink()

    def __enter__(self) -> BotaFtSensor:
        """
        Context manager __enter__ method.

        Use, e.g., like:
        >>> with BotaFtSensor(ether_if) as sensor:
        >>>     while True:
        >>>         print(f"Fx: {sensor.Fx:.5f} N", end="\r")

        Returns:
            self
        """
        self.start()
        return self

    def __exit__(self, ex_type: Any, ex_value: Any, ex_traceback: Any) -> Literal[False]:
        """
        Context manager __exit__ method.

        Use, e.g., like:
        >>> with BotaFtSensor(ether_if) as sensor:
        >>>     while True:
        >>>         print(f"Fx: {sensor.Fx:.5f} N", end="\r")
        """
        self.stop()
        return False

    def _sensor_setup(self, slave_pos: int) -> None:
        """
        Setup of sensor.

        Parameters:
            slave_pos: int
                Index of the sensor in the slaves list
        """
        LOGGER.debug("Setting up sensor")
        slave = self._master.slaves[slave_pos]

        # -- Set sensor configuration:
        # calibration matrix active
        slave.sdo_write(0x8010, 1, bytes(ctypes.c_uint8(1)))
        # temperature compensation
        slave.sdo_write(0x8010, 2, bytes(ctypes.c_uint8(0)))
        # IMU active
        slave.sdo_write(0x8010, 3, bytes(ctypes.c_uint8(1)))

        # -- Set force torque filter:
        slave.sdo_write(index=0x8006, subindex=2, data=struct.pack('b', int(not self.filter_fir)))  # Has to be inverted
        slave.sdo_write(index=0x8006, subindex=3, data=struct.pack('b', int(self.filter_fast)))
        slave.sdo_write(index=0x8006, subindex=4, data=struct.pack('b', int(self.filter_chop)))
        slave.sdo_write(index=0x8006, subindex=1, data=struct.pack('H', self.filter_sinc_length))
        
        LOGGER.info(f"Filter settings: sinc length {self.filter_sinc_length}, "
                    f"FIR {self.filter_fir}, FAST {self.filter_fast}, CHOP {self.filter_chop}")

        # Get sampling rate:
        self.sampling_rate = struct.unpack("h", slave.sdo_read(0x8011, 0))[0]
        LOGGER.info(f"Sampling rate {self.sampling_rate} Hz")
        if self.sampling_rate > 0:
            self.time_step = 1.0 / float(self.sampling_rate)

    def _init_master(self) -> None:
        """Initialization of the SOEM master."""
        self._master.open(self.adapter)

        # config_init returns the number of slaves found
        if self._master.config_init() > 0:
            LOGGER.debug(f"{len(self._master.slaves)} slave(s) found and configured")

            for i, slave in enumerate(self._master.slaves):
                assert slave.man == self.BOTA_VENDOR_ID
                assert slave.id == self._expected_slave_mapping[i].product_code
                slave.config_func = self._expected_slave_mapping[i].config_func

            # PREOP_STATE to SAFEOP_STATE request - each slave's config_func is called
            self._master.config_map()

            # wait 50 ms for all slaves to reach SAFE_OP state
            if self._master.state_check(pysoem.SAFEOP_STATE, 50000) != pysoem.SAFEOP_STATE:
                self._master.read_state()
                for slave in self._master.slaves:
                    if not slave.state == pysoem.SAFEOP_STATE:
                        LOGGER.error(f"{slave.name} did not reach SAFEOP state")
                        LOGGER.error(
                            f"al status code {hex(slave.al_status)} ",
                            f"({pysoem.al_status_code_to_string(slave.al_status)})",
                        )
                raise Exception("not all slaves reached SAFEOP state")

            self._master.state = pysoem.OP_STATE
            self._master.write_state()

            self._master.state_check(pysoem.OP_STATE, 50000)
            if self._master.state != pysoem.OP_STATE:
                self._master.read_state()
                for slave in self._master.slaves:
                    if not slave.state == pysoem.OP_STATE:
                        LOGGER.error(f"{slave.name} did not reach OP state")
                        LOGGER.error(
                            f"al status code {hex(slave.al_status)} ",
                            f"({pysoem.al_status_code_to_string(slave.al_status)})",
                        )
                raise Exception("Not all slaves reached OP state")
        else:
            raise RuntimeError(f"Can not connect to sensor slaves. Is adapter name '{self.adapter}' correct?")

    def _update_values(self) -> None:
        """Update process for the sensor values."""
        try:
            while True:
                # Run update loop:
                self._master.send_processdata()
                self._master.receive_processdata(2000)

                self.write_buffer.buf[:] = bytearray(self.slave.input)

                time.sleep(self.time_step)
        except Exception as e:
            LOGGER.error(f"Aborted sensor value update: {e}")
            raise
        finally:
            self._master.state = pysoem.INIT_STATE
            # request INIT state for all slaves
            self._master.write_state()
            self._master.close()

    def _read_value(self, type: str, position: int) -> Any:
        """
        Read current sensor value from the value buffer.

        Parameters:
            type: string
                Type identifier according to
                https://docs.python.org/3/library/struct.html#format-characters
            position: integer
                Position in the bytearray
        Returns:
            Unpacked value from value buffer
        """
        return struct.unpack_from(type, self.read_buffer.buf, position)[0]
    
    def set_ft_offset(self, offset: npt.NDArray[np.float64]) -> None:
        """ Set the internal force-torque offset

        Args:
            offset: 6 dimensional offset
        """
        if self.update_process.is_alive():
            self.slave.sdo_write(index=0x8000, subindex=1, data=struct.pack("f", float(offset[0])))
            self.slave.sdo_write(index=0x8000, subindex=2, data=struct.pack("f", float(offset[1])))
            self.slave.sdo_write(index=0x8000, subindex=3, data=struct.pack("f", float(offset[2])))
            self.slave.sdo_write(index=0x8000, subindex=4, data=struct.pack("f", float(offset[3])))
            self.slave.sdo_write(index=0x8000, subindex=5, data=struct.pack("f", float(offset[4])))
            self.slave.sdo_write(index=0x8000, subindex=6, data=struct.pack("f", float(offset[5])))
        else:
            LOGGER.error("Error while setting sensor offset. Is the sensor connected properly?")

    def clear_ft_offset(self) -> None:
        """ Method to set force torque offset to zero
        """
        self.set_ft_offset(np.zeros(6))

    def zero_ft_readings(self) -> npt.NDArray[np.float64]:
        """ Zeroing of the sensor readings.
            This method sets an offset of the mean values over one second of sensor
            measurements. Therefore, the update process of the sensor has to be started
            already by entering the context manager of the sensor.

        Returns:
            The calculated offset
        """
        LOGGER.info(f"Zeroing sensor with mean values over a sample size of {self.sampling_rate}.")
        buffer = np.empty(shape=(6, self.sampling_rate), dtype=np.float32)
        for i in range(self.sampling_rate):
            buffer[:, i] = np.array([self.Fx, self.Fy, self.Fz, self.Tx, self.Ty, self.Tz])
            time.sleep(self.time_step)
        offset: npt.NDArray[np.float64] = buffer.mean(axis=1)
        self.set_ft_offset(-offset)
        return -offset

    @property
    def status(self) -> int:
        """Status of sensor readings."""
        status: int = self._read_value("B", 0)
        return status

    @property
    def warningsErrorsFatals(self) -> int:
        """Count of warnings, errors or fatals."""
        wef: int = self._read_value("I", 1)
        return wef

    @property
    def Fx(self) -> float:
        """Force in x direction."""
        fx: float = self._read_value("f", 5)
        return fx

    @property
    def Fy(self) -> float:
        """Force in y direction."""
        fy: float = self._read_value("f", 9)
        return fy

    @property
    def Fz(self) -> float:
        """Force in z direction."""
        fz: float = self._read_value("f", 13)
        return fz

    @property
    def Tx(self) -> float:
        """Torque about x axis."""
        tx: float = self._read_value("f", 17)
        return tx

    @property
    def Ty(self) -> float:
        """Torque about y axis."""
        ty: float = self._read_value("f", 21)
        return ty

    @property
    def Tz(self) -> float:
        """Torque about z axis."""
        tz: float = self._read_value("f", 25)
        return tz
    
    @property
    def FT(self) -> npt.NDArray[np.float64]:
        """ Force torque readings as array
        """
        return np.array([self.Fx, self.Fy, self.Fz, self.Tx, self.Ty, self.Tz])

    @property
    def FTSaturated(self) -> int:
        """Saturation status of force or torque measurements."""
        ft_sat: int =  self._read_value("H", 29)
        return ft_sat

    @property
    def Ax(self) -> float:
        """Acceleration in x direction."""
        ax: float = self._read_value("f", 31)
        return ax

    @property
    def Ay(self) -> float:
        """Acceleration in y direction."""
        ay: float = self._read_value("f", 35)
        return ay

    @property
    def Az(self) -> float:
        """Acceleration in z direction."""
        az: float = self._read_value("f", 39)
        return az

    @property
    def accelerationSaturated(self) -> int:
        """Saturation status of acceleration measurements."""
        acc_sat: int = self._read_value("B", 43)
        return acc_sat

    @property
    def Rx(self) -> float:
        """Angular rate about x axis."""
        rx: float = self._read_value("f", 44)
        return rx

    @property
    def Ry(self) -> float:
        """Angular rate about y axis."""
        ry: float = self._read_value("f", 48)
        return ry

    @property
    def Rz(self) -> float:
        """Angular rate about z axis."""
        rz: float = self._read_value("f", 52)
        return rz

    @property
    def angularRateSaturated(self) -> int:
        """Saturation status of angular rate measurements."""
        ang_sat: int = self._read_value("B", 56)
        return ang_sat

    @property
    def IMU(self) -> npt.NDArray[np.float64]:
        """ IMU readings as array
        """
        return np.array([self.Ax, self.Ay, self.Az, self.Rx, self.Ry, self.Rz])

    @property
    def temperature(self) -> float:
        """Temperature."""
        temp: float = self._read_value("f", 57)
        return temp

    @property
    def orientationX(self) -> float:
        """
        Not available right now.

        The sensor does not return any other value than 0.0 when orientation
        estimation is turned on.
        """
        ox: float = self._read_value("f", 61)
        return ox

    @property
    def orientationY(self) -> float:
        """
        Not available right now.

        The sensor does not return any other value than 0.0 when orientation
        estimation is turned on.
        """
        oy: float = self._read_value("f", 65)
        return oy

    @property
    def orientationZ(self) -> float:
        """
        Not available right now.

        The sensor does not return any other value than 0.0 when orientation
        estimation is turned on.
        """
        oz: float = self._read_value("f", 69)
        return oz

    @property
    def orientationW(self) -> float:
        """
        Not available right now.

        The sensor does not return any other value than 0.0 when orientation
        estimation is turned on.
        """
        ow: float = self._read_value("f", 73)
        return ow
