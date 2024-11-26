import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Self

from gpiozero import DigitalOutputDevice

from app.models import SensorData

__all__ = ["MLX90393"]


# Communication Interface
class CommunicationInterface(ABC):
    @abstractmethod
    async def transfer(self, data: bytes, read_length: int = 0) -> bytes:
        pass


# SPI Communication
class SPICommunication(CommunicationInterface):
    def __init__(self, *, chip_select_pin, clock_pin, mosi_pin, miso_pin, **kwargs):
        from gpiozero.pins.pigpio import PiGPIOHardwareSPI

        self._spi = PiGPIOHardwareSPI(
            select_pin=chip_select_pin,
            clock_pin=clock_pin,
            mosi_pin=mosi_pin,
            miso_pin=miso_pin,
            **kwargs,
        )
        self._spi._set_clock_mode(0x11)  # CPOL=1, CPHA=1
        self._spi._set_rate(1000000)  # Default 1 MHz
        self._spi._set_bits_per_word(8)
        self._cs = DigitalOutputDevice(chip_select_pin)
        self.lock = asyncio.Lock()

    async def transfer(self, data: bytes, read_length: int = 0) -> bytes:
        async with self.lock:
            self._cs.on()
            response = await asyncio.to_thread(
                self._spi.transfer, data + bytes([0] * read_length)
            )
            self._cs.off()
        return bytes(response)


# I2C Communication
class I2CCommunication(CommunicationInterface):
    def __init__(self, *, i2c_address, bus=1):
        from smbus2 import SMBus

        self.i2c_address = i2c_address
        self._bus = SMBus(bus)
        self.lock = asyncio.Lock()

    async def transfer(self, data: bytes, read_length: int = 0) -> bytes:
        async with self.lock:
            from smbus2 import i2c_msg

            if data:
                write_msg = i2c_msg.write(self.i2c_address, data)
                if read_length > 0:
                    read_msg = i2c_msg.read(self.i2c_address, read_length)
                    await asyncio.to_thread(self._bus.i2c_rdwr, write_msg, read_msg)
                    response = bytes(read_msg)
                else:
                    await asyncio.to_thread(self._bus.i2c_rdwr, write_msg)
                    response = b""
            else:
                if read_length > 0:
                    read_msg = i2c_msg.read(self.i2c_address, read_length)
                    await asyncio.to_thread(self._bus.i2c_rdwr, read_msg)
                    response = bytes(read_msg)
                else:
                    response = b""
        return response


class MLX90393:
    def __init__(self, comm_interface: CommunicationInterface):
        self.comm_interface = comm_interface

    async def start_burst_mode(self, axes=0x07):
        """
        Start burst mode for the specified axes (X, Y, Z).
        Args:
            axes (int): Axes selection bits (0x07 for XYZ).
        Returns:
            bool: True if burst mode was successfully activated, False otherwise.
        """
        command = 0x10  # SB command for starting burst mode
        data = bytes([command, axes])
        response = await self.comm_interface.transfer(data, read_length=1)
        return bool(response[0] & 0x80)  # Check if burst mode was activated

    async def read_burst_data(self):
        """
        Read data from the sensor in burst mode.
        Returns:
            SensorData: Contains x, y, z as signed integers.
        """
        # Read 7 bytes: 1 status byte + 6 data bytes
        response = await self.comm_interface.transfer(bytes(), read_length=7)
        if len(response) < 7:
            raise IOError("Failed to read data from sensor.")
        # Parse response
        x = (response[1] << 8) | response[2]
        y = (response[3] << 8) | response[4]
        z = (response[5] << 8) | response[6]

        # Convert to signed 16-bit integers
        x = x if x < 32768 else x - 65536
        y = y if y < 32768 else y - 65536
        z = z if z < 32768 else z - 65536

        return SensorData(x=x, y=y, z=z)

    async def continuous_read(self, interval_ms=2):
        """
        Continuously read burst data at the specified interval.
        Args:
            interval_ms (int): Interval between reads in milliseconds.
        Yields:
            SensorData: Magnetic field values.
        """
        # Start burst mode for XYZ axes
        if await self.start_burst_mode(axes=0x07):
            print("Burst mode successfully activated!")
        else:
            print("Failed to activate burst mode. Check configuration.")

        while True:
            data = await self.read_burst_data()
            yield data
            await asyncio.sleep(interval_ms / 1000.0)

    @classmethod
    async def continuous_read_all(cls, devices: Dict[str, Self], interval_ms=2):
        """
        Continuously read burst data from all devices at the specified interval.
        Args:
            devices (dict[str, MLX90393]): Dictionary of devices with their names as keys.
            interval_ms (int): Interval between reads in milliseconds.
        Yields:
            tuple: (name, SensorData) for each device.
        """
        queue = asyncio.Queue()

        async def read_device(name, device):
            while True:
                data = await device.read_burst_data()
                await queue.put((name, data))
                await asyncio.sleep(interval_ms / 1000.0)

        # Start coroutines for each device
        coroutines = [read_device(name, device) for name, device in devices.items()]
        asyncio.create_task(asyncio.gather(*coroutines))

        # Yield data as it becomes available
        while True:
            name, data = await queue.get()
            yield name, data
