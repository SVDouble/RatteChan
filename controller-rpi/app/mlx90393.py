import asyncio
from multiprocessing import get_logger
from typing import Self

from gpiozero import SPIDevice, DigitalOutputDevice

from app.models import SensorData

__all__ = ["MLX90393"]

logger = get_logger("mlx90393")


class MLX90393(SPIDevice):
    lock = asyncio.Lock()

    def __init__(self, *, chip_select_pin, clock_pin, mosi_pin, miso_pin, **kwargs):
        from gpiozero.pins.pigpio import PiGPIOHardwareSPI

        self._spi: PiGPIOHardwareSPI
        super().__init__(
            chip_select_pin=chip_select_pin,
            clock_pin=clock_pin,
            mosi_pin=mosi_pin,
            miso_pin=miso_pin,
            **kwargs,
        )
        self._spi._set_clock_mode(0x11)  # CPOL=1, CPHA=1
        self._spi._set_rate(1000000)  # Default 1 MHz
        self._spi._set_bits_per_word(8)
        self._cs = DigitalOutputDevice(chip_select_pin)

    async def _transfer(self, data):
        async with self.lock:
            self._cs.on()
            response = await asyncio.to_thread(self._spi.transfer, data)
            self._cs.off()
        return response

    async def start_burst_mode(self, axes=0x07):
        """
        Start burst mode for the specified axes (X, Y, Z).
        Args:
            axes (int): Axes selection bits (0x07 for XYZ).
        Returns:
            bool: True if burst mode was successfully activated, False otherwise.
        """
        command = 0x10  # SB command for starting burst mode
        data = [axes]
        response = await self._transfer([command] + data)
        return bool(
            response[0] & 0x80
        )  # Check if burst mode was activated (BURST_MODE bit)

    async def read_burst_data(self):
        """
        Read data from the sensor in burst mode.
        Returns:
            tuple: (x, y, z) as signed integers.
        """
        buffer = await self._transfer([0x00] * 7)
        x = (buffer[1] << 8) | buffer[2]
        y = (buffer[3] << 8) | buffer[4]
        z = (buffer[5] << 8) | buffer[6]

        # Convert to signed values
        if x > 32767:
            x -= 65536
        if y > 32767:
            y -= 65536
        if z > 32767:
            z -= 65536

        return SensorData(x=x, y=y, z=z)

    async def continuous_read(self, interval_ms=2):
        """
        Continuously read burst data at the specified interval.
        Args:
            interval_ms (int): Interval between reads in milliseconds.
        Yields:
            tuple: (x, y, z) magnetic field values.
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
    async def continuous_read_all(cls, devices: dict[str, Self], interval_ms=2):
        """
        Continuously read burst data from all devices at the specified interval.
        Args:
            devices (dict[str, MLX90393]): Dictionary of MLX90393 devices with their names as keys.
            interval_ms (int): Interval between reads in milliseconds.
        Yields:
            tuple: (name, SensorData) magnetic field values for each device.
        """
        queue = asyncio.Queue()

        async def read_device(name, device):
            """
            Coroutine to read data from a single device and add it to the queue.
            Args:
                name (str): Name of the device.
                device (MLX90393): The device instance.
            """
            while True:
                data = await device.read_burst_data()
                await queue.put((name, data))
                await asyncio.sleep(interval_ms / 1000.0)

        # Start a coroutine for each device
        coroutines = [read_device(name, device) for name, device in devices.items()]
        asyncio.create_task(asyncio.gather(*coroutines))

        # Yield data as it becomes available in the queue
        while True:
            name, data = await queue.get()
            yield name, data
