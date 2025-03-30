import asyncio
import time

from ratte.sensors.metrics import Metrics
from ratte.sensors.mlx90393 import MLX90393, I2C
from ratte.sensors.repository import Repository
from ratte.sensors.utils import get_settings, get_logger

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
        log_message = ", ".join(f"{name}: {value}" for name, value in sorted(snapshot.items()))
        logger.info(f"Metrics: {log_message}")


async def read_device(
    name: str,
    device: MLX90393,
    *,
    queue: asyncio.Queue,
    interval_ms: int,
    metrics: Metrics,
):
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
            await metrics.increment("M[*]")
            await metrics.increment(f"M[id={name}]")
        except Exception as e:
            await metrics.increment("E[*]")
            await metrics.increment(f"E[id={name}]")
            logger.debug(f"Error reading data from {name}: {e}")

        elapsed_time = (time.time() - start_time) * 1000.0
        await asyncio.sleep(max(0.0, interval_ms - elapsed_time) / 1000.0)


async def main():
    settings = get_settings()
    repository = Repository(settings)
    sensors = {sensor.name: MLX90393(I2C(bus=sensor.bus, address=sensor.address)) for sensor in settings.sensors}

    # Create a metrics object for tracking counts
    metrics = Metrics()

    # Start logging metrics in the background
    asyncio.create_task(log_metrics(metrics))

    # Start a coroutine for each device
    queue = asyncio.Queue()
    interval_ms = 1000 // settings.publish_frequency_hz
    for name, device in sensors.items():
        asyncio.create_task(
            read_device(
                name,
                device,
                queue=queue,
                interval_ms=interval_ms,
                metrics=metrics,
            )
        )

    # Main data processing loop
    async with repository:
        while True:
            name, data = await queue.get()
            await repository.publish_sensor_data(name, data)
