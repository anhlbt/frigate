"""Model Utils"""

import logging
import os

import cv2
import numpy as np
import onnxruntime as ort

from frigate.const import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


### Post Processing


def post_process_dfine(
    tensor_output: np.ndarray, width: int, height: int
) -> np.ndarray:
    class_ids = tensor_output[0][tensor_output[2] > 0.4]
    boxes = tensor_output[1][tensor_output[2] > 0.4]
    scores = tensor_output[2][tensor_output[2] > 0.4]

    input_shape = np.array([height, width, height, width])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.4)
    detections = np.zeros((20, 6), np.float32)

    for i, (bbox, confidence, class_id) in enumerate(
        zip(boxes[indices], scores[indices], class_ids[indices])
    ):
        if i == 20:
            break

        detections[i] = [
            class_id,
            confidence,
            bbox[1],
            bbox[0],
            bbox[3],
            bbox[2],
        ]

    return detections


def post_process_rfdetr(tensor_output: list[np.ndarray, np.ndarray]) -> np.ndarray:
    boxes = tensor_output[0]
    raw_scores = tensor_output[1]

    # apply soft max to scores
    exp = np.exp(raw_scores - np.max(raw_scores, axis=-1, keepdims=True))
    all_scores = exp / np.sum(exp, axis=-1, keepdims=True)

    # get highest scoring class from every detection
    scores = np.max(all_scores[0, :, 1:], axis=-1)
    labels = np.argmax(all_scores[0, :, 1:], axis=-1)

    idxs = scores > 0.4
    filtered_boxes = boxes[0, idxs]
    filtered_scores = scores[idxs]
    filtered_labels = labels[idxs]

    # convert boxes from [x_center, y_center, width, height]
    x_center, y_center, w, h = (
        filtered_boxes[:, 0],
        filtered_boxes[:, 1],
        filtered_boxes[:, 2],
        filtered_boxes[:, 3],
    )
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    filtered_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

    # apply nms
    indices = cv2.dnn.NMSBoxes(
        filtered_boxes, filtered_scores, score_threshold=0.4, nms_threshold=0.4
    )
    detections = np.zeros((20, 6), np.float32)

    for i, (bbox, confidence, class_id) in enumerate(
        zip(filtered_boxes[indices], filtered_scores[indices], filtered_labels[indices])
    ):
        if i == 20:
            break

        detections[i] = [
            class_id,
            confidence,
            bbox[1],
            bbox[0],
            bbox[3],
            bbox[2],
        ]

    return detections


def post_process_yolov9(predictions: np.ndarray, width, height) -> np.ndarray:
    predictions = np.squeeze(predictions).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > 0.4, :]
    scores = scores[scores > 0.4]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Rescale box
    boxes = predictions[:, :4]

    input_shape = np.array([width, height, width, height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.4)
    detections = np.zeros((20, 6), np.float32)
    for i, (bbox, confidence, class_id) in enumerate(
        zip(boxes[indices], scores[indices], class_ids[indices])
    ):
        if i == 20:
            break

        detections[i] = [
            class_id,
            confidence,
            bbox[1] - bbox[3] / 2,
            bbox[0] - bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
            bbox[0] + bbox[2] / 2,
        ]

    return detections


### ONNX Utilities


def get_ort_providers(
    force_cpu: bool = False, device: str = "AUTO", requires_fp16: bool = False
) -> tuple[list[str], list[dict[str, any]]]:
    if force_cpu:
        return (
            ["CPUExecutionProvider"],
            [
                {
                    "enable_cpu_mem_arena": False,
                }
            ],
        )

    providers = []
    options = []

    for provider in ort.get_available_providers():
        if provider == "CUDAExecutionProvider":
            device_id = 0 if not device.isdigit() else int(device)
            providers.append(provider)
            options.append(
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "device_id": device_id,
                }
            )
        elif provider == "TensorrtExecutionProvider":
            # TensorrtExecutionProvider uses too much memory without options to control it
            # so it is not enabled by default
            if device == "Tensorrt":
                os.makedirs(
                    os.path.join(MODEL_CACHE_DIR, "tensorrt/ort/trt-engines"),
                    exist_ok=True,
                )
                device_id = 0 if not device.isdigit() else int(device)
                providers.append(provider)
                options.append(
                    {
                        "device_id": device_id,
                        "trt_fp16_enable": requires_fp16
                        and os.environ.get("USE_FP_16", "True") != "False",
                        "trt_timing_cache_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_timing_cache_path": os.path.join(
                            MODEL_CACHE_DIR, "tensorrt/ort"
                        ),
                        "trt_engine_cache_path": os.path.join(
                            MODEL_CACHE_DIR, "tensorrt/ort/trt-engines"
                        ),
                    }
                )
            else:
                continue
        elif provider == "OpenVINOExecutionProvider":
            os.makedirs(os.path.join(MODEL_CACHE_DIR, "openvino/ort"), exist_ok=True)
            providers.append(provider)
            options.append(
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cache_dir": os.path.join(MODEL_CACHE_DIR, "openvino/ort"),
                    "device_type": device,
                }
            )
        elif provider == "CPUExecutionProvider":
            providers.append(provider)
            options.append(
                {
                    "enable_cpu_mem_arena": False,
                }
            )
        else:
            providers.append(provider)
            options.append({})

    return (providers, options)
