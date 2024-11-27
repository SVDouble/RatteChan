import asyncio
from typing import Protocol

from gpiozero import SPIDevice

from app.models import SensorData

__all__ = ["Backend", "SPI", "I2C", "MLX90393"]


# Communication Interface
class Backend(Protocol):
    async def transfer(self, data: bytes, response_length: int = 0) -> bytes: ...


# SPI Communication
class SPI(SPIDevice):
    def __init__(self, *, chip_select_pin, clock_pin, mosi_pin, miso_pin, **kwargs):
        from gpiozero.pins.lgpio import LGPIOHardwareSPI

        super().__init__(
            clock_pin=clock_pin,
            mosi_pin=mosi_pin,
            miso_pin=miso_pin,
            select_pin=chip_select_pin,
            **kwargs,
        )
        # Set SPI mode and speed
        self._spi: LGPIOHardwareSPI
        self._spi._set_clock_mode(0b11)  # CPOL=1, CPHA=1
        self._spi._set_rate(1000000)  # Default 1 MHz
        self._spi._set_bits_per_word(8)  # Default 8 bits
        self.lock = asyncio.Lock()

    async def transfer(self, data: bytes, response_length: int = 0) -> bytes:
        async with self.lock:
            total_data = list(data) + [0] * response_length
            response = await asyncio.to_thread(self._spi.transfer, total_data)
            return bytes(response)


# I2C Communication
class I2C:
    def __init__(self, *, i2c_address, bus=1):
        from smbus2 import SMBus

        self.i2c_address = i2c_address
        self._bus = SMBus(bus)
        self.lock = asyncio.Lock()

    async def transfer(self, data: bytes, response_length: int = 0) -> bytes:
        async with self.lock:
            from smbus2 import i2c_msg

            if data:
                write_msg = i2c_msg.write(self.i2c_address, data)
                if response_length > 0:
                    read_msg = i2c_msg.read(self.i2c_address, response_length)
                    await asyncio.to_thread(self._bus.i2c_rdwr, write_msg, read_msg)
                    response = bytes(read_msg)
                else:
                    await asyncio.to_thread(self._bus.i2c_rdwr, write_msg)
                    response = b""
            else:
                if response_length > 0:
                    read_msg = i2c_msg.read(self.i2c_address, response_length)
                    await asyncio.to_thread(self._bus.i2c_rdwr, read_msg)
                    response = bytes(read_msg)
                else:
                    response = b""
            return response


# MLX90393 Sensor Class with All Commands
class MLX90393:
    def __init__(self, backend: Backend):
        self.backend = backend

    # Count set bits in zyxt to determine data length
    def count_set_bits(self, n: int) -> int:
        return bin(n).count("1")

    # EX Command
    async def ex(self) -> bytes:
        command = bytes([0x80])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # SB Command
    async def sb(self, zyxt: int) -> bytes:
        command = bytes([0x10 | (zyxt & 0x0F)])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # SWOC Command
    async def swoc(self, zyxt: int) -> bytes:
        command = bytes([0x20 | (zyxt & 0x0F)])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # SM Command
    async def sm(self, zyxt: int) -> bytes:
        command = bytes([0x30 | (zyxt & 0x0F)])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # RM Command
    async def rm(self, zyxt: int) -> bytes:
        command = bytes([0x40 | (zyxt & 0x0F)])
        data_length = 1 + 2 * self.count_set_bits(zyxt & 0x0F)
        response = await self.backend.transfer(command, response_length=data_length)
        return response

    # RR Command
    async def rr(self, address: int) -> bytes:
        command = bytes([0x50, (address << 2) & 0xFC])
        response = await self.backend.transfer(command, response_length=3)
        return response

    # WR Command
    async def wr(self, address: int, data: int) -> bytes:
        command = bytes(
            [
                0x60,
                (data >> 8) & 0xFF,
                data & 0xFF,
                (address << 2) & 0xFC,
            ]
        )
        response = await self.backend.transfer(command, response_length=1)
        return response

    # RT Command
    async def rt(self) -> bytes:
        command = bytes([0xF0])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # NOP Command
    async def nop(self) -> bytes:
        command = bytes([0x00])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # HR Command
    async def hr(self) -> bytes:
        command = bytes([0xD0])
        response = await self.backend.transfer(command, response_length=1)
        return response

    # HS Command
    async def hs(self) -> bytes:
        command = bytes([0xE0])
        response = await self.backend.transfer(command, response_length=1)
        return response

    async def start_burst_mode(self, zyxt=0b1110) -> bool:
        """
        Start burst mode for the specified axes (X, Y, Z).
        Args:
            zyxt (int): Axes selection bits (0x07 for XYZ).
        Returns:
            bool: True if burst mode was successfully activated, False otherwise.
        """

        response = await self.sb(zyxt)
        return bool(response[0] & 0x80)

    async def read_burst_data(self, zyxt=0b1110) -> SensorData:
        """
        Read data from the sensor in burst mode.
        Returns:
            tuple: (x, y, z) as signed integers.
        """
        # TODO: double check
        buffer = await self.rm(zyxt)
        keys = ["txyz"[i] for i in range(4) if bool(zyxt & (1 << i))]
        # bit 0 is reserved for status, so we start from 1
        values = [buffer[i] << 8 | buffer[i + 1] for i in range(1, len(buffer), 2)]
        signed_values = [(v - 65536) if v > 32767 else v for v in values]
        return SensorData(**dict(zip(keys, signed_values)))
