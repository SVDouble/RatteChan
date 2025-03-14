import asyncio
import json
import os
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "generator")
MQTT_PORT = int(os.getenv("MQTT_PORT", 9001))

# Read frequency from environment variable, default to 100 Hz if not set
FREQUENCY_HZ = float(os.getenv("FREQUENCY_HZ", 100))

# Create an MQTT client with WebSocket transport
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")


async def publish_data():
    interval = (
        1.0 / FREQUENCY_HZ
    )  # Calculate the interval in seconds based on the frequency
    while True:
        # Generate timestamp with microsecond precision
        timestamp = (
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            + f".{datetime.now().microsecond:06d}"
        )

        # Calculate the sine values based on time
        time_elapsed = (asyncio.get_event_loop().time()) % (
            2 * np.pi
        )  # Loop time over 2π for smooth cycling
        angles = np.array(
            [time_elapsed, time_elapsed + np.pi / 2, time_elapsed + np.pi]
        )  # Use np.pi for phase shifts

        def publish_data(data: np.ndarray, topic: str):
            # Create a JSON message
            message = {
                "timestamp": timestamp,
                "x": float(data[0]),
                "y": float(data[1]),
                "z": float(data[2]),
            }

            # Publish the message
            client.publish(topic, json.dumps(message))

        publish_data(np.sin(angles), f"{MQTT_TOPIC}/clean")
        publish_data(
            np.sin(angles) + np.random.uniform(0.0, 0.1, size=3), f"{MQTT_TOPIC}/noisy"
        )

        # Sleep for the calculated interval to achieve the configured frequency
        await asyncio.sleep(interval)


async def main():
    # Connect to the MQTT broker via WebSocket
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Start the loop to process callbacks
    client.loop_start()

    try:
        await publish_data()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
