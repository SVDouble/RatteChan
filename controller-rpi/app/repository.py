import paho.mqtt.client as mqtt

from app.models import SensorData
from app.settings import Settings


class Repository:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="tcp")

    def publish_sensor_data(self, data: SensorData):
        self.mqtt.publish(self._settings.mqtt_topic, data.model_dump_json(), qos=0)
