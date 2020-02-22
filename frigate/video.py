import os
import time
import datetime
import cv2
import queue
import threading
import ctypes
import multiprocessing as mp
import subprocess as sp
import numpy as np
import hashlib
import pyarrow.plasma as plasma
import SharedArray as sa
import copy
import itertools
import json
from collections import defaultdict
from frigate.util import draw_box_with_label, area, calculate_region, clipped, intersection_over_union, intersection, EventsPerSecond
from frigate.objects import ObjectTracker
from frigate.edgetpu import RemoteObjectDetector
from frigate.motion import MotionDetector

# TODO: add back opencv fallback
def get_frame_shape(source):
    ffprobe_cmd = " ".join([
        'ffprobe',
        '-v',
        'panic',
        '-show_error',
        '-show_streams',
        '-of',
        'json',
        '"'+source+'"'
    ])
    print(ffprobe_cmd)
    p = sp.Popen(ffprobe_cmd, stdout=sp.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    info = json.loads(output)
    print(info)

    video_info = [s for s in info['streams'] if s['codec_type'] == 'video'][0]

    if video_info['height'] != 0 and video_info['width'] != 0:
        return (video_info['height'], video_info['width'], 3)
    
    # fallback to using opencv if ffprobe didnt succeed
    video = cv2.VideoCapture(source)
    ret, frame = video.read()
    frame_shape = frame.shape
    video.release()
    return frame_shape

def get_ffmpeg_input(ffmpeg_input):
    frigate_vars = {k: v for k, v in os.environ.items() if k.startswith('FRIGATE_')}
    return ffmpeg_input.format(**frigate_vars)

def filtered(obj, objects_to_track, object_filters, mask):
    object_name = obj[0]

    if not object_name in objects_to_track:
        return True
    
    if object_name in object_filters:
        obj_settings = object_filters[object_name]

        # if the min area is larger than the
        # detected object, don't add it to detected objects
        if obj_settings.get('min_area',-1) > obj[3]:
            return True
        
        # if the detected object is larger than the
        # max area, don't add it to detected objects
        if obj_settings.get('max_area', 24000000) < obj[3]:
            return True

        # if the score is lower than the threshold, skip
        if obj_settings.get('threshold', 0) > obj[1]:
            return True
    
        # compute the coordinates of the object and make sure
        # the location isnt outside the bounds of the image (can happen from rounding)
        y_location = min(int(obj[2][3]), len(mask)-1)
        x_location = min(int((obj[2][2]-obj[2][0])/2.0)+obj[2][0], len(mask[0])-1)

        # if the object is in a masked location, don't add it to detected objects
        if mask[y_location][x_location] == [0]:
            return True
        
        return False

def create_tensor_input(frame, region):
    cropped_frame = frame[region[1]:region[3], region[0]:region[2]]

    # Resize to 300x300 if needed
    if cropped_frame.shape != (300, 300, 3):
        cropped_frame = cv2.resize(cropped_frame, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    
    # Expand dimensions since the model expects images to have shape: [1, 300, 300, 3]
    return np.expand_dims(cropped_frame, axis=0)

def track_camera(name, config, ffmpeg_global_config, global_objects_config, detect_lock, detect_ready, frame_ready, detected_objects_queue, fps, skipped_fps, detection_fps):
    print(f"Starting process for {name}: {os.getpid()}")

    # Merge the ffmpeg config with the global config
    ffmpeg = config.get('ffmpeg', {})
    ffmpeg_input = get_ffmpeg_input(ffmpeg['input'])
    ffmpeg_global_args = ffmpeg.get('global_args', ffmpeg_global_config['global_args'])
    ffmpeg_hwaccel_args = ffmpeg.get('hwaccel_args', ffmpeg_global_config['hwaccel_args'])
    ffmpeg_input_args = ffmpeg.get('input_args', ffmpeg_global_config['input_args'])
    ffmpeg_output_args = ffmpeg.get('output_args', ffmpeg_global_config['output_args'])

    # Merge the tracked object config with the global config
    camera_objects_config = config.get('objects', {})    
    # combine tracked objects lists
    objects_to_track = set().union(global_objects_config.get('track', ['person', 'car', 'truck']), camera_objects_config.get('track', []))
    # merge object filters
    global_object_filters = global_objects_config.get('filters', {})
    camera_object_filters = camera_objects_config.get('filters', {})
    objects_with_config = set().union(global_object_filters.keys(), camera_object_filters.keys())
    object_filters = {}
    for obj in objects_with_config:
        object_filters[obj] = {**global_object_filters.get(obj, {}), **camera_object_filters.get(obj, {})}

    expected_fps = config['fps']
    take_frame = config.get('take_frame', 1)

    frame_shape = get_frame_shape(ffmpeg_input)
    frame_size = frame_shape[0] * frame_shape[1] * frame_shape[2]

    try:
        sa.delete(name)
    except:
        pass

    frame = sa.create(name, shape=frame_shape, dtype=np.uint8)

    # load in the mask for object detection
    if 'mask' in config:
        mask = cv2.imread("/config/{}".format(config['mask']), cv2.IMREAD_GRAYSCALE)
    else:
        mask = None

    if mask is None:
        mask = np.zeros((frame_shape[0], frame_shape[1], 1), np.uint8)
        mask[:] = 255

    motion_detector = MotionDetector(frame_shape, mask, resize_factor=6)
    object_detector = RemoteObjectDetector('/labelmap.txt', detect_lock, detect_ready, frame_ready)

    object_tracker = ObjectTracker(10)

    ffmpeg_cmd = (['ffmpeg'] +
            ffmpeg_global_args +
            ffmpeg_hwaccel_args +
            ffmpeg_input_args +
            ['-i', ffmpeg_input] +
            ffmpeg_output_args +
            ['pipe:'])

    print(" ".join(ffmpeg_cmd))
    
    ffmpeg_process = sp.Popen(ffmpeg_cmd, stdout = sp.PIPE, bufsize=frame_size)
    
    plasma_client = plasma.connect("/tmp/plasma")
    frame_num = 0
    fps_tracker = EventsPerSecond()
    skipped_fps_tracker = EventsPerSecond()
    fps_tracker.start()
    skipped_fps_tracker.start()
    object_detector.fps.start()
    while True:
        frame_bytes = ffmpeg_process.stdout.read(frame_size)

        if not frame_bytes:
            break

        # limit frame rate
        frame_num += 1
        if (frame_num % take_frame) != 0:
            continue

        fps_tracker.update()
        fps.value = fps_tracker.eps()
        detection_fps.value = object_detector.fps.eps()

        frame_time = datetime.datetime.now().timestamp()
        
        # Store frame in numpy array
        frame[:] = (np
                    .frombuffer(frame_bytes, np.uint8)
                    .reshape(frame_shape))
        
        # look for motion
        motion_boxes = motion_detector.detect(frame)

        # skip object detection if we are below the min_fps
        # TODO: its about more than just the FPS. also look at avg wait or min wait
        if frame_num > 100 and fps.value < expected_fps-1:
            skipped_fps_tracker.update()
            skipped_fps.value = skipped_fps_tracker.eps()
            continue
        
        skipped_fps.value = skipped_fps_tracker.eps()

        tracked_objects = object_tracker.tracked_objects.values()

        # merge areas of motion that intersect with a known tracked object into a single area to look at
        areas_of_interest = []
        used_motion_boxes = []
        for obj in tracked_objects:
            x_min, y_min, x_max, y_max = obj['box']
            for m_index, motion_box in enumerate(motion_boxes):
                if area(intersection(obj['box'], motion_box))/area(motion_box) > .5:
                    used_motion_boxes.append(m_index)
                    x_min = min(obj['box'][0], motion_box[0])
                    y_min = min(obj['box'][1], motion_box[1])
                    x_max = max(obj['box'][2], motion_box[2])
                    y_max = max(obj['box'][3], motion_box[3])
            areas_of_interest.append((x_min, y_min, x_max, y_max))
        unused_motion_boxes = set(range(0, len(motion_boxes))).difference(used_motion_boxes)
        
        # compute motion regions
        motion_regions = [calculate_region(frame_shape, motion_boxes[i][0], motion_boxes[i][1], motion_boxes[i][2], motion_boxes[i][3], 1.2)
            for i in unused_motion_boxes]
        
        # compute tracked object regions
        object_regions = [calculate_region(frame_shape, a[0], a[1], a[2], a[3], 1.2)
            for a in areas_of_interest]
        
        # merge regions with high IOU
        merged_regions = motion_regions+object_regions
        while True:
            max_iou = 0.0
            max_indices = None
            region_indices = range(len(merged_regions))
            for a, b in itertools.combinations(region_indices, 2):
                iou = intersection_over_union(merged_regions[a], merged_regions[b])
                if iou > max_iou:
                    max_iou = iou
                    max_indices = (a, b)
            if max_iou > 0.1:
                a = merged_regions[max_indices[0]]
                b = merged_regions[max_indices[1]]
                merged_regions.append(calculate_region(frame_shape,
                    min(a[0], b[0]),
                    min(a[1], b[1]),
                    max(a[2], b[2]),
                    max(a[3], b[3]),
                    1
                ))
                del merged_regions[max(max_indices[0], max_indices[1])]
                del merged_regions[min(max_indices[0], max_indices[1])]
            else:
                break

        # resize regions and detect
        detections = []
        for region in merged_regions:

            tensor_input = create_tensor_input(frame, region)

            region_detections = object_detector.detect(tensor_input)

            for d in region_detections:
                box = d[2]
                size = region[2]-region[0]
                x_min = int((box[1] * size) + region[0])
                y_min = int((box[0] * size) + region[1])
                x_max = int((box[3] * size) + region[0])
                y_max = int((box[2] * size) + region[1])
                det = (d[0],
                    d[1],
                    (x_min, y_min, x_max, y_max),
                    (x_max-x_min)*(y_max-y_min),
                    region)
                if filtered(det, objects_to_track, object_filters, mask):
                    continue
                detections.append(det)

        #########
        # merge objects, check for clipped objects and look again up to N times
        #########
        refining = True
        refine_count = 0
        while refining and refine_count < 4:
            refining = False

            # group by name
            detected_object_groups = defaultdict(lambda: [])
            for detection in detections:
                detected_object_groups[detection[0]].append(detection)

            selected_objects = []
            for group in detected_object_groups.values():

                # apply non-maxima suppression to suppress weak, overlapping bounding boxes
                boxes = [(o[2][0], o[2][1], o[2][2]-o[2][0], o[2][3]-o[2][1])
                    for o in group]
                confidences = [o[1] for o in group]
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for index in idxs:
                    obj = group[index[0]]
                    if clipped(obj, frame_shape): #obj['clipped']:
                        box = obj[2]
                        # calculate a new region that will hopefully get the entire object
                        region = calculate_region(frame_shape, 
                            box[0], box[1],
                            box[2], box[3])
                        
                        tensor_input = create_tensor_input(frame, region)
                        # run detection on new region
                        refined_detections = object_detector.detect(tensor_input)
                        for d in refined_detections:
                            box = d[2]
                            size = region[2]-region[0]
                            x_min = int((box[1] * size) + region[0])
                            y_min = int((box[0] * size) + region[1])
                            x_max = int((box[3] * size) + region[0])
                            y_max = int((box[2] * size) + region[1])
                            det = (d[0],
                                d[1],
                                (x_min, y_min, x_max, y_max),
                                (x_max-x_min)*(y_max-y_min),
                                region)
                            if filtered(det, objects_to_track, object_filters, mask):
                                continue
                            selected_objects.append(det)

                        refining = True
                    else:
                        selected_objects.append(obj)
                
            # set the detections list to only include top, complete objects
            # and new detections
            detections = selected_objects

            if refining:
                refine_count += 1
        
        # now that we have refined our detections, we need to track objects
        object_tracker.match_and_update(frame_time, detections)

        # put the frame in the plasma store
        object_id = hashlib.sha1(str.encode(f"{name}{frame_time}")).digest()
        plasma_client.put(frame, plasma.ObjectID(object_id))
        # add to the queue
        detected_objects_queue.put((name, frame_time, object_tracker.tracked_objects))

        # if (frames >= 700 and frames <= 1635) or (frames >= 2500):
        # if (frames >= 300 and frames <= 600):
        # if (frames >= 0):
            # row1 = cv2.hconcat([gray, cv2.convertScaleAbs(avg_frame)])
            # row2 = cv2.hconcat([frameDelta, thresh])
            # cv2.imwrite(f"/lab/debug/output/{frames}.jpg", cv2.vconcat([row1, row2]))
            # # cv2.imwrite(f"/lab/debug/output/resized-frame-{frames}.jpg", resized_frame)
            # for region in motion_regions:
            #     cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (255,128,0), 2)
            # for region in object_regions:
            #     cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (0,128,255), 2)
            # for region in merged_regions:
            #     cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (0,255,0), 2)
            # for box in motion_boxes:
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
            # for detection in detections:
            #     box = detection[2]
            #     draw_box_with_label(frame, box[0], box[1], box[2], box[3], detection[0], f"{detection[1]*100}%")
            # for obj in object_tracker.tracked_objects.values():
            #     box = obj['box']
            #     draw_box_with_label(frame, box[0], box[1], box[2], box[3], obj['label'], obj['id'], thickness=1, color=(0,0,255), position='bl')
            # cv2.putText(frame, str(total_detections), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
            # cv2.putText(frame, str(frame_detections), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
            # cv2.imwrite(f"/lab/debug/output/frame-{frames}.jpg", frame)
            # break

    # start a thread to publish object scores
    # mqtt_publisher = MqttObjectPublisher(self.mqtt_client, self.mqtt_topic_prefix, self)
    # mqtt_publisher.start()

    # create a watchdog thread for capture process
    # self.watchdog = CameraWatchdog(self)


