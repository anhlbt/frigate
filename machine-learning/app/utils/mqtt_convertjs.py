import os
import uuid
import requests
import paho.mqtt.client as mqtt
import json
from pathlib import Path

# Assuming fs and jwt utility functions are defined somewhere
from fs_util import writer, delete
from auth_util import jwt
from constants import AUTH, SERVER, MQTT, FRIGATE, CAMERAS, STORAGE, UI, config

PREVIOUS_MQTT_LENGTHS = []
JUST_SUBSCRIBED = False
CLIENT = None
PERSON_RESET_TIMEOUT = {}
STATUS = None

def log_status(status, log_func):
    global STATUS
    STATUS = status
    log_func(f"MQTT: {status}")

def camera_topics():
    return [
        CAMERAS[key]['SNAPSHOT']['TOPIC']
        for key in CAMERAS
        if CAMERAS[key].get('SNAPSHOT') and CAMERAS[key]['SNAPSHOT'].get('TOPIC')
    ] if CAMERAS else []

def process_message(topic, message):
    async def init():
        if (('/snapshot' in topic or topic in camera_topics()) and not JUST_SUBSCRIBED):
            await snapshot()
        if '/events' in topic:
            await frigate()

    async def snapshot():
        found_camera = next(
            (key for key in CAMERAS if CAMERAS[key]['SNAPSHOT']['TOPIC'] == topic), None)
        camera = (found_camera or topic.split('/')[1]).lower()
        filename = f"{uuid.uuid4()}.jpg"
        buffer = message

        attempts = config['frigate'](camera) if found_camera else {'ATTEMPTS': {'MQTT': True}}

        if not attempts['ATTEMPTS']['MQTT'] or len(buffer) in PREVIOUS_MQTT_LENGTHS:
            return
        PREVIOUS_MQTT_LENGTHS.insert(0, len(buffer))

        writer(f"{STORAGE['TMP']['PATH']}/{filename}", buffer)
        requests.get(
            f"http://0.0.0.0:{SERVER['PORT']}{UI['PATH']}/api/recognize",
            headers={'Authorization': jwt.sign({'route': 'recognize'})} if AUTH else None,
            params={
                'url': f"http://0.0.0.0:{SERVER['PORT']}{UI['PATH']}/api/{STORAGE['TMP']['PATH']}/{filename}",
                'type': 'mqtt',
                'camera': camera
            }
        )
        delete(f"{STORAGE['TMP']['PATH']}/{filename}")
        PREVIOUS_MQTT_LENGTHS = PREVIOUS_MQTT_LENGTHS[:10]

    async def frigate():
        payload = json.loads(message.decode('utf-8'))
        print(f"Incoming event from frigate: {message.decode('utf-8')}")
        if payload['type'] == 'end':
            return

        requests.post(
            f"http://0.0.0.0:{SERVER['PORT']}{UI['PATH']}/api/recognize",
            headers={'Authorization': jwt.sign({'route': 'recognize'})} if AUTH else None,
            json={**payload, 'topic': topic}
        )

    return init, snapshot, frigate

def connect():
    global CLIENT
    if not MQTT or not MQTT['HOST']:
        return

    try:
        CLIENT = mqtt.Client(client_id=MQTT.get('CLIENT_ID', f"double-take-{uuid.uuid4().hex[:8]}"))
        CLIENT.username_pw_set(MQTT.get('USERNAME', MQTT.get('USER')), MQTT.get('PASSWORD', MQTT.get('PASS')))
        CLIENT.tls_set(
            ca_certs=MQTT['TLS']['CA'] if MQTT['TLS']['CA'] else None,
            certfile=MQTT['TLS']['CERT'] if MQTT['TLS']['CERT'] else None,
            keyfile=MQTT['TLS']['KEY'] if MQTT['TLS']['KEY'] else None
        )
        CLIENT.tls_insecure_set(MQTT['TLS'].get('REJECT_UNAUTHORIZED', True))
        CLIENT.connect(MQTT['HOST'], port=MQTT.get('PORT', 8883 if MQTT.get('PROTOCOL') == 'mqtts' else 1883), keepalive=60)
        CLIENT.reconnect_delay_set(min_delay=1, max_delay=10)

        CLIENT.on_connect = lambda client, userdata, flags, rc: handle_connect(rc)
        CLIENT.on_message = lambda client, userdata, msg: handle_message(msg.topic, msg.payload)
        CLIENT.on_disconnect = lambda client, userdata, rc: log_status('disconnected', print)
        CLIENT.on_log = lambda client, userdata, level, buf: print(f"MQTT Log: {buf}")

        CLIENT.loop_start()
    except Exception as e:
        log_status(str(e), print)

def handle_connect(rc):
    if rc == 0:
        log_status('connected', print)
        publish({'topic': 'double-take/errors'})
        available('online')
        subscribe()
    else:
        log_status(f"Connection failed with code {rc}", print)

def handle_message(topic, message):
    init, snapshot, frigate = process_message(topic, message)
    init()

def available(state):
    if CLIENT:
        publish({'topic': 'double-take/available', 'retain': True, 'message': state})

def subscribe():
    topics = camera_topics()

    if FRIGATE and FRIGATE.get('URL') and MQTT['TOPICS'].get('FRIGATE'):
        frigate_topics = MQTT['TOPICS']['FRIGATE'] if isinstance(MQTT['TOPICS']['FRIGATE'], list) else [MQTT['TOPICS']['FRIGATE']]
        topics.extend(frigate_topics)
        for topic in frigate_topics:
            prefix = topic.split('/')[0]
            topics.extend(
                [f"{prefix}/{camera}/person/snapshot" for camera in FRIGATE.get('CAMERAS', [])] or [f"{prefix}/+/person/snapshot"]
            )

    if topics:
        CLIENT.subscribe(topics)
        print(f"MQTT: subscribed to {', '.join(topics)}")
        global JUST_SUBSCRIBED
        JUST_SUBSCRIBED = True
        # reset JUST_SUBSCRIBED after 5 seconds
        CLIENT.loop.call_later(5, lambda: setattr(JUST_SUBSCRIBED, False))

def recognize(data):
    try:
        if not MQTT or not MQTT['HOST']:
            return

        base_data = data.copy()
        id, duration, timestamp, attempts, zones, matches, misses, unknowns, counts, token = (
            base_data['id'], base_data['duration'], base_data['timestamp'], base_data['attempts'],
            base_data['zones'], base_data['matches'], base_data['misses'], base_data['unknowns'],
            base_data['counts'], base_data['token']
        )
        camera = base_data['camera'].lower()

        payload = {
            'base': {
                'id': id, 'duration': duration, 'timestamp': timestamp, 'attempts': attempts,
                'camera': camera, 'zones': zones, 'token': token
            }
        }
        payload['unknown'] = {**payload['base'], 'unknown': unknowns[0], 'unknowns': unknowns}
        payload['match'] = {**payload['base']}
        payload['camera'] = {
            **payload['base'], 'matches': matches, 'misses': misses,
            'unknowns': unknowns, 'personCount': counts['person'], 'counts': counts
        }
        payload['cameraReset'] = {
            **payload['camera'], 'personCount': 0, 'counts': {'person': 0, 'match': 0, 'miss': 0, 'unknown': 0}
        }

        messages = []

        messages.append({
            'topic': f"{MQTT['TOPICS']['CAMERAS']}/{camera}/person",
            'retain': True, 'message': str(counts['person'])
        })

        if unknowns:
            messages.append({
                'topic': f"{MQTT['TOPICS']['MATCHES']}/unknown",
                'retain': False, 'message': json.dumps(payload['unknown'])
            })
            if MQTT['TOPICS'].get('HOMEASSISTANT'):
                messages.extend([
                    {
                        'topic': f"{MQTT['TOPICS']['HOMEASSISTANT']}/sensor/double-take/unknown/config",
                        'retain': True,
                        'message': json.dumps({
                            'name': 'double_take_unknown', 'icon': 'mdi:account', 'value_template': '{{ value_json.camera }}',
                            'state_topic': f"{MQTT['TOPICS']['HOMEASSISTANT']}/sensor/double-take/unknown/state",
                            'json_attributes_topic': f"{MQTT['TOPICS']['HOMEASSISTANT']}/sensor/double-take/unknown/state",
                            'availability_topic': 'double-take/available', 'unique_id': 'double_take_unknown', 'expire_after': 600
                        })
                    },
                    {
                        'topic': f"{MQTT['TOPICS']['HOMEASSISTANT']}/sensor/double-take/unknown/state",
                        'retain': True, 'message': json.dumps(payload['unknown'])
                    }
                ])

        for match in matches:
            topic = match['name'].
