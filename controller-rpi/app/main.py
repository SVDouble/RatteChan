import asyncio

from app.mlx90393 import MLX90393
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


async def main():
    settings = get_settings()
    repository = Repository(settings)

    # Initialize sensors
    sensors = {
        addr: MLX90393(
            chip_select_pin=addr,
            clock_pin=settings.clock_pin,
            mosi_pin=settings.mosi_pin,
            miso_pin=settings.miso_pin,
        )
        for addr in settings.sensor_addresses
    }

    # Shared counter for message logging
    counter = {"count": 0, "lock": asyncio.Lock()}

    # Start logging task
    asyncio.create_task(log_message_rate(counter))

    # Main data processing loop
    async for name, data in MLX90393.continuous_read_all(sensors):
        await repository.publish_sensor_data(data)
        async with counter["lock"]:
            counter["count"] += 1
