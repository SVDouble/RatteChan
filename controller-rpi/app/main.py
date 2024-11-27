import asyncio
import time

from app.metrics import Metrics
from app.mlx90393 import MLX90393, I2C
from app.repository import Repository
from app.utils import get_settings, get_logger

logger = get_logger(__name__)


async def log_metrics(metrics: Metrics):
    """
    Logs the current metrics at regular intervals.
    Args:
        metrics: Metrics object to track arbitrary counters.
    """
    while True:
        await asyncio.sleep(1)  # Log every second
        snapshot = await metrics.reset()
        log_message = ", ".join(f"{name}: {value}" for name, value in snapshot.items())
        logger.info(f"Metrics: {log_message}")


async def read(devices: dict[str, MLX90393], interval_ms: int, metrics: Metrics):
    queue = asyncio.Queue()

    async def read_device(name, device):
        """
        Task to periodically read data from a device.
        """
        try:
            await asyncio.wait_for(device.start_burst_mode(), timeout=1)
        except asyncio.TimeoutError:
            raise RuntimeError("Failed to start burst mode")

        while True:
            start_time = time.time()
            try:
                data = await device.read_burst_data()
                await queue.put((name, data))
                await metrics.increment("messages")
                await metrics.increment(f"{name}_messages")
            except Exception as e:
                await metrics.increment("errors")
                await metrics.increment(f"{name}_errors")
                logger.error(f"Error reading data from {name}: {e}")

            elapsed_time = (time.time() - start_time) * 1000.0
            await asyncio.sleep(max(0.0, interval_ms - elapsed_time) / 1000.0)

    # Start a coroutine for each device
    for name, device in devices.items():
        asyncio.create_task(read_device(name, device))

    # Yield data as it becomes available in the queue
    while True:
        name, data = await queue.get()
        yield name, data


async def main():
    settings = get_settings()
    repository = Repository(settings)
    sensors = {addr: MLX90393(I2C(address=addr)) for addr in settings.sensors}

    # Create a metrics object for tracking counts
    metrics = Metrics()

    # Start logging metrics in the background
    asyncio.create_task(log_metrics(metrics))

    # Main data processing loop
    async with repository:
        async for name, data in read(
                sensors, interval_ms=1000 // settings.publish_frequency_hz, metrics=metrics
        ):
            await repository.publish_sensor_data(data)
