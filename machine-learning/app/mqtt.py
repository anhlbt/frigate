import logging
import threading
from typing import Any, Callable, Optional,List
from pydantic import BaseModel, Field, validator
import os
import paho.mqtt.client as mqtt
import yaml
from collections import Counter
import json
import time
import requests
import numpy as np
import cv2
from datetime import datetime, timedelta
from fastapi import UploadFile, File, HTTPException

logger = logging.getLogger(__name__)
YAML_EXT = (".yaml", ".yml")
# Dictionary to track attempts and timestamps by topic and event_id
event_tracker = {}
# Constants
MAX_ATTEMPTS = 10
T_SLEEP = 1
EXPIRATION_TIME = timedelta(minutes=10)  # Time after which entries should be removed

# FRIGATE_HOST = "http://localhost:5000"
# MQTT_TOPIC = "frigate-devcontainer/events"

def load_config_with_no_duplicates(raw_config) -> dict:
    class PreserveDuplicatesLoader(yaml.loader.SafeLoader):
        pass

    def map_constructor(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                raise ValueError(f"Config input {key} is defined multiple times for the same field, this is not allowed.")
            else:
                data[key] = val
        return data

    PreserveDuplicatesLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, map_constructor)
    return yaml.load(raw_config, PreserveDuplicatesLoader)

class MqttConfig(BaseModel):
    enabled: bool = Field(title="Enable MQTT Communication.", default=True)
    host: str = Field(default="", title="MQTT Host")
    port: int = Field(default=1883, title="MQTT Port")
    topic_prefix: str = Field(default="frigate", title="MQTT Topic Prefix")
    client_id: str = Field(default="frigate", title="MQTT Client ID")
    stats_interval: int = Field(default=60, title="MQTT Camera Stats Interval")
    user: Optional[str] = Field(title="MQTT Username")
    password: Optional[str] = Field(title="MQTT Password")
    tls_ca_certs: Optional[str] = Field(title="MQTT TLS CA Certificates")
    tls_client_cert: Optional[str] = Field(title="MQTT TLS Client Certificate")
    tls_client_key: Optional[str] = Field(title="MQTT TLS Client Key")
    tls_insecure: Optional[bool] = Field(title="MQTT TLS Insecure")
    frigate_host: str = Field(default="http://frigate:5000", title="MQTT Host")
    mqtt_topic: List[str] = Field(default=["+/+/person"], title="MQTT Topics subscribed by face recognizer")
    face_conf_cosin: float = Field(default=0.5, title="face confidence score with metric cosine")


    @validator("password", pre=True, always=True)
    def validate_password(cls, v, values):
        if (v is None) != (values["user"] is None):
            raise ValueError("Password must be provided with username.")
        return v

class FrigateConfig(BaseModel):
    mqtt: MqttConfig = Field(title="MQTT Configuration.")
    
    @classmethod
    def parse_file(cls, config_file):
        with open(config_file) as f:
            raw_config = f.read()

        if config_file.endswith(YAML_EXT):
            config = load_config_with_no_duplicates(raw_config)
        elif config_file.endswith(".json"):
            config = json.loads(raw_config)

        return cls.parse_obj(config)

class MqttClient():
    """Frigate wrapper for mqtt client."""
    def __init__(self) -> None:
        self.connected: bool = False
        self.init_config()
        self.mqtt_config = self.config.mqtt
        self._dispatcher: Optional[Callable] = None
        
    def init_config(self) -> None:
        config_file = os.environ.get("CONFIG_FILE", "/usr/src/app/config_mqtt.yml")
        config_file_yaml = config_file.replace(".yml", ".yaml")
        if os.path.isfile(config_file_yaml):
            config_file = config_file_yaml
        self.config = FrigateConfig.parse_file(config_file)

    def subscribe(self, receiver: Callable) -> None:
        """
        Wrapper for allowing dispatcher to subscribe.
        """
        self._dispatcher = receiver
        
    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        """Wrapper for publishing when client is in valid state."""
        if not self.connected:
            logger.error(f"Unable to publish to {topic}: client is not connected")
            return
        self.client.publish(f"{self.mqtt_config.topic_prefix}/{topic}", payload, retain=retain)

    def stop(self) -> None:
        self.client.disconnect()

    def _set_initial_topics(self) -> None:
        """Set initial state topics."""
        self.publish("f_available", "online", retain=True)

    ##  Define the callback for when a message is receive
    def _on_message(self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        try:
            # logger.info(f"Received MQTT message on topic: {message.topic}")
            topic = message.topic.replace(f"{self.mqtt_config.topic_prefix}/", "", 1)
            # payload = json.loads(message.payload.decode("utf-8"))  # not image
            if 'events' in topic:
                self.recognize_events(client, userdata, message)
            if 'person' in topic:
                self.recognize_snapshot(client, userdata, message)
            if self._dispatcher:
                # self._dispatcher(topic, payload, retain=True)
                self._dispatcher(client, userdata, message)
        except json.JSONDecodeError:
            print("Received a non-JSON message:", message.payload)

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, rc: int) -> None:
        """Mqtt connection callback."""
        threading.current_thread().name = "mqtt"
        if rc != 0:
            if rc == 3:
                logger.error("Unable to connect to MQTT server: MQTT Server unavailable")
            elif rc == 4:
                logger.error("Unable to connect to MQTT server: MQTT Bad username or password")
            elif rc == 5:
                logger.error("Unable to connect to MQTT server: MQTT Not authorized")
            else:
                logger.error(f"Unable to connect to MQTT server: Connection refused. Error code: {rc}")
        else:
            self.connected = True
            logger.info("MQTT connected")
            # client.subscribe(f"{self.mqtt_config.topic_prefix}/#")
            # client.subscribe(self.mqtt_config.mqtt_topic)
            for topic in self.mqtt_config.mqtt_topic:
                client.subscribe(topic)
            self._set_initial_topics()

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Mqtt disconnection callback."""
        self.connected = False
        logger.error("MQTT disconnected")

    def _start(self) -> None:
        """Start mqtt client."""
        self.client = mqtt.Client() # client_id=self.mqtt_config.client_id
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.will_set(self.mqtt_config.topic_prefix + "/f_available", payload="offline", qos=1, retain=True)

        if self.mqtt_config.tls_ca_certs is not None:
            if self.mqtt_config.tls_client_cert is not None and self.mqtt_config.tls_client_key is not None:
                self.client.tls_set(self.mqtt_config.tls_ca_certs, self.mqtt_config.tls_client_cert, self.mqtt_config.tls_client_key)
            else:
                self.client.tls_set(self.mqtt_config.tls_ca_certs)
        if self.mqtt_config.tls_insecure:
            self.client.tls_insecure_set(self.mqtt_config.tls_insecure)
        if self.mqtt_config.user is not None:
            self.client.username_pw_set(self.mqtt_config.user, password=self.mqtt_config.password)

        try:
            self.client.connect_async(self.mqtt_config.host, self.mqtt_config.port, 60)
            self.client.loop_start()
        #     self.client.connect(self.mqtt_config.host, self.mqtt_config.port, 60)
        #     self.client.loop_forever()
        except Exception as e:
            logger.error(f"Unable to connect to MQTT server: {e}")



    # # Function to publish a message to an MQTT topic
    # def publish_to_mqtt_topic(self, client, userdata, message):
    #     # event_id = message["after"]["id"]
    #     # logger.info(f"Processing frame from MQTT. Topic: {topic}, Payload id : {event_id}")
    #     # requests.post(f"{self.mqtt_config.frigate_host}/api/events/{event_id}/sub_label", json={"subLabel": "face_recognition", "subLabelScore": round(0.232424, 2)})
    #     # self.client.publish(topic, json.dumps(message), retain=retain)

    #     # Decode JPEG byte array to a NumPy array  ## only with topic snapshot
    #     nparr = np.frombuffer(message.payload, np.uint8)
    #     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     cv2.imwrite(f'saved_image_{time.time()}.jpg', frame)
    #     ret, frame_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    #     logger.info(f"Processing frame from MQTT. save image .................: ")

    def convert_bbox_to_xywh(self, box, image_width, image_height):
        x1, y1, x2, y2 = box
 
        w = x2 - x1
        h = y2 - y1
        
        x = x1 / image_width
        y = y1 / image_height
        w /= image_width
        h /= image_height
        
        return x, y, w, h

    def recognize_events(self, client, userdata, message):
        try:
            image_byte = None
            topic = message.topic
            payload = json.loads(message.payload.decode("utf-8"))
            event_id = payload['before']['id']

            # Initialize topic in event_tracker if not already set
            if topic not in event_tracker:
                event_tracker[topic] = {}

            # Check for expired entries and remove them
            current_time = datetime.now()
            if event_id in event_tracker[topic]:
                attempts, timestamp, face_score = event_tracker[topic][event_id]
                if current_time - timestamp > EXPIRATION_TIME:
                    del event_tracker[topic][event_id]
                    logger.warn(f"Removed expired event_id {event_id} from attempt tracker for topic {topic}")

            # Initialize attempts and timestamp for this event_id if not already set
            if event_id not in event_tracker[topic]:
                event_tracker[topic][event_id] = (0, current_time, 1.0)  # num get snapshot, datetime, face score
            else:
                # Increment attempts
                event_tracker[topic][event_id] = (event_tracker[topic][event_id][0] + 1, current_time, event_tracker[topic][event_id][2])
                if event_tracker[topic][event_id][2] < self.mqtt_config.face_conf_cosin and event_tracker[topic][event_id][0] > 0.5 * MAX_ATTEMPTS:  # face recognized exactly
                    # time.sleep(T_SLEEP)
                    logger.warn(f"by pass, face score: {event_tracker[topic][event_id]} -  {event_tracker[topic][event_id][2]}")
                    # del event_tracker[topic][event_id]
                    return

            if payload['type'] != 'end' and event_tracker[topic][event_id][0] < MAX_ATTEMPTS:
                snapshot_url = f"{self.mqtt_config.frigate_host}/api/events/{event_id}/snapshot.jpg?crop=1&bbox=0"
                try:
                    response = requests.get(snapshot_url)
                    if response.status_code == 200:
                        image_byte = {'image': ('snapshot.jpg', response.content, 'image/jpeg')}
                except Exception as ce:
                    logger.error(f"Connection error: {ce}")
            if image_byte:
                try:
                    response = requests.post(
                        "http://0.0.0.0:3003/recognize",
                        files=image_byte
                    )
                    if response.status_code == 200:
                        prediction = response.json()
                        camera = payload['before']['camera']
                        # client.publish(f"recognize_events_{camera}", json.dumps(prediction), retain=True)
                        # requests.post(f"{self.mqtt_config.frigate_host}/api/events/{event_id}/sub_label",
                        #               json={"subLabel": prediction['name'], "subLabelScore": round(prediction['distance'], 3)})
                        

                        box= payload['before']['box']
                        f_width= payload['before']['width']
                        f_height= payload['before']['height']
                        score = prediction['distance']
                        requests.post(f"{self.mqtt_config.frigate_host}/api/events/{camera}/customer/create",
                                      json={"label": "customer", "sub_label": prediction['name'], \
                                            "draw": {
                                                    # optional annotations that will be drawn on the snapshot
                                                    "boxes": [
                                                    {
                                                        "box": self.convert_bbox_to_xywh(box, f_width, f_height), #[x, y, width, height], # box consists of x, y, width, height which are on a scale between 0 - 1
                                                        "color": [255, 0, 0], # color of the box, default is red
                                                        "score": round(score, 2) # optional score associated with the box
                                                    }
                                                    ]
                                                }
                                      }
                                    )                        
                        event_tracker[topic][event_id] = (event_tracker[topic][event_id][0], current_time, round(prediction['distance'], 3))
                    else:
                        logger.error(f"Failed to get prediction: {response.status_code} - {response.text}")
                except requests.RequestException as re:
                    logger.error(f"Error during recognition request: {re}")
            else:
                logger.error(f"Delete, failed to retrieve a valid image for event {event_id} after {event_tracker[topic][event_id][0]} attempts")
                del event_tracker[topic][event_id]
        except Exception as e:
            logger.error(f"Error processing frame from MQTT: {e}")



    def recognize_snapshot(self,client, userdata, message):
        try:
            logger.warn(f"processing frame from MQTT: {message.topic}")
            # if message.topic.split("/")[-2] == 'person':    
            response = requests.post(
                "http://0.0.0.0:3003/recognize",
                files={"image": message.payload}  # frame_bytes.tobytes
            )
            prediction = response.json()
            logger.info(f"Prediction result: {prediction}")
            camera = message.topic.split("/")[-3]
            # requests.post(f"{self.mqtt_config.frigate_host}/api/events/{event_id}/sub_label", json={"subLabel": "face_recognition", "subLabelScore": round(0.232424, 2)})
            client.publish(f"face_{camera}", json.dumps(prediction), retain=True)
        except Exception as e:
            logger.error(f"Error processing frame from MQTT: {e}")

       

# def save_frame(frame_bytes, frame_id):
#     # Decode the frame bytes to an image
#     output_dir = "."
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     filename = os.path.join(output_dir, f"frame_{frame_id}.jpg")
#     cv2.imwrite(filename, frame)
#     print(f"Saved frame {frame_id} to {filename}")


# ## process frame from zmq
# async def receive_frames():
#     frame_id = 0
#     while True:
#         try:
#             [topic, frame_bytes] = socket.recv_multipart()
            
#             # log.info("Received frame from the detector.")
#             # save_frame(frame_bytes, f"{topic.decode('utf-8')}_{frame_id}")
#             frame_id += 1

#             # Call /predict endpoint
#             response = requests.post(
#                 "http://localhost:3003/predict",
#                 data={"modelName": "buffalo_l", "modelType": "facial-recognition"},
#                 files={"image": frame_bytes}
#             )
#             prediction = response.json()
#             log.info(f"Prediction result: {prediction}")

#         except zmq.ZMQError as e:
#             log.error(f"ZeroMQ error occurred: {e}")
#         except Exception as e:
#             log.error(f"Error occurred during frame processing: {e}")
#         finally:
#             # Perform any necessary cleanup
#             pass


#### option 2
def process_frame_from_mqtt(client, userdata, message): # topic: str, payload: str
    try:
        # topic = message.topic.replace(f"{self.mqtt_config.topic_prefix}/", "", 1)
        # payload = json.loads(message.payload.decode("utf-8")        
        
        # Decode JPEG byte array to a NumPy array
        nparr = np.frombuffer(message.payload, np.uint8)
        # ## save image 
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imwrite(f'tmp/saved_image_fastapi_{time.time()}.jpg', frame)
        ret, frame_bytes = cv2.imencode('.jpg', frame) #[int(cv2.IMWRITE_JPEG_QUALITY), 70]
        if not ret:
            raise ValueError("Could not encode image to JPEG format")
        ## Call /predict endpoint
        response = requests.post(
            "http://localhost:3003/predict",
            data={"modelName": "buffalo_l", "modelType": "facial-recognition"},
            files={"image": frame_bytes.tobytes()}  # frame_bytes.tobytes | frame
        )
        prediction = response.json()
        logger.info(f"Prediction result: {prediction}")
        # requests.post(f"{self.mqtt_config.frigate_host}/api/events/{event_id}/sub_label", json={"subLabel": "face_recognition", "subLabelScore": round(0.232424, 2)})
        client.publish("face_recognition", json.dumps(prediction), retain=False)

    except Exception as e:
        logger.error(f"Error processing frame from MQTT: {e}")


        
# when run local need to check: variables in config_mqtt.yml: host(localhost, mqtt) and check CONFIG_FILE
# CONFIG_FILE=/workspaces/frigate/machine-learning/config_mqtt.yml python mqtt.py


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     mqtt_client = MqttClient()
#     mqtt_client._start()  # Connect to MQTT broker

#     # # Wait for the client to connect before publishing
#     # while not mqtt_client.connected:
#     #     time.sleep(0.1)
#     # while True:    
#     mqtt_client.subscribe(mqtt_client.publish_to_mqtt_topic)

        
        
        
        