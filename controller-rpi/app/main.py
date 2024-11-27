import asyncio

from app.mlx90393 import MLX90393, I2C
from app.repository import Repository
from app.utils import get_settings, get_logger

logger = get_logger(__name__)


async def log_message_rate(counter):
    """
    Logs the message rate (messages per second) at regular intervals.
    Args:
        counter: An asyncio-compatible object to count messages (e.g., asyncio.Lock + variable).
    """

    while True:
        await asyncio.sleep(1)  # Log every second
        async with counter["lock"]:
            rate = counter["count"]
            counter["count"] = 0  # Reset count for the next interval
        logger.info(f"Messages per second: {rate}")


async def read(devices: dict[str, MLX90393], interval_ms: int):
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


async def main():
    settings = get_settings()
    repository = Repository(settings)

    # Initialize sensors
    sensors = {
        addr: MLX90393(I2C(i2c_address=addr)) for addr in settings.sensor_addresses
    }

    # Shared counter for message logging
    counter = {"count": 0, "lock": asyncio.Lock()}

    # Start logging task
    asyncio.create_task(log_message_rate(counter))

    # Main data processing loop
    async for name, data in read(
        sensors, interval_ms=1000 // settings.publish_frequency_hz
    ):
        await repository.publish_sensor_data(data)
        async with counter["lock"]:
            counter["count"] += 1
