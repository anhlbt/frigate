import ctypes
import logging
from pathlib import Path
import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda

    TRT_VERSION = int(trt.__version__[0 : trt.__version__.find(".")])

    TRT_SUPPORT = True
except ModuleNotFoundError:
    TRT_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal
from typing import List, Tuple, Union
from numpy import ndarray

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum

logger = logging.getLogger(__name__)

DETECTOR_KEY = "tensorrt"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


class TensorRTDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: int = Field(default=0, title="GPU Device Index")


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem, nbytes, size, dtype):
        self.host = host_mem
        err, self.host_dev = cuda.cuMemHostGetDevicePointer(self.host, 0)
        self.device = device_mem
        self.nbytes = nbytes
        self.size = size
        self.dtype = dtype

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        cuda.cuMemFreeHost(self.host)
        cuda.cuMemFree(self.device)


class TensorRtDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def _load_engine(self, model_path):
        if self.model_type not in (ModelTypeEnum.yolov5, ModelTypeEnum.yolov8):
            try:
                trt.init_libnvinfer_plugins(self.trt_logger, "")

                ctypes.cdll.LoadLibrary("/usr/local/lib/libyolo_layer.so")
                # ctypes.cdll.LoadLibrary("/models/libmyplugins.so")
            except OSError as e:
                logger.error(
                    "ERROR: failed to load libraries. %s",
                    e,
                )

            with open(model_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:    
            trt.init_libnvinfer_plugins(self.trt_logger, "")
            weight = Path(model_path) if isinstance(model_path, str) else model_path
            with trt.Runtime(self.trt_logger) as runtime:
                model = runtime.deserialize_cuda_engine(weight.read_bytes())
                return model            

    def _binding_is_input(self, binding):
        if TRT_VERSION < 10:
            return self.engine.binding_is_input(binding)
        else:
            return binding == "input"

    def _get_binding_dims(self, binding):
        if TRT_VERSION < 10:
            return self.engine.get_binding_shape(binding)
        else:
            return self.engine.get_tensor_shape(binding)

    def _get_binding_dtype(self, binding):
        if TRT_VERSION < 10:
            return self.engine.get_binding_dtype(binding)
        else:
            return self.engine.get_tensor_dtype(binding)

    def _execute(self):
        if TRT_VERSION < 10:
            return self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream
            )
        else:
            return self.context.execute_v2(self.bindings)

    def _get_input_shape(self):
        """Get input shape of the TensorRT YOLO engine."""
        binding = self.engine[0]
        assert self._binding_is_input(binding)
        binding_dims = self._get_binding_dims(binding)
        if len(binding_dims) == 4:
            return (
                tuple(binding_dims[2:]),
                trt.nptype(self._get_binding_dtype(binding)),
            )
        elif len(binding_dims) == 3:
            return (
                tuple(binding_dims[1:]),
                trt.nptype(self._get_binding_dtype(binding)),
            )
        else:
            raise ValueError(
                "bad dims of binding %s: %s" % (binding, str(binding_dims))
            )

    def _allocate_buffers(self):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        output_idx = 0
        for binding in self.engine:
            binding_dims = self._get_binding_dims(binding)
            if len(binding_dims) == 4:
                # explicit batch case (TensorRT 7+)
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                # implicit batch case (TensorRT 6 or older)
                size = trt.volume(binding_dims) * self.engine.max_batch_size
            else:
                # raise ValueError(
                #     "bad dims of binding %s: %s" % (binding, str(binding_dims))
                # )
                size = trt.volume(binding_dims)
            nbytes = size * self._get_binding_dtype(binding).itemsize
            dtype = trt.nptype(self._get_binding_dtype(binding))
            # Allocate host and device buffers
            err, host_mem = cuda.cuMemHostAlloc(
                nbytes, Flags=cuda.CU_MEMHOSTALLOC_DEVICEMAP
            )
            assert err is cuda.CUresult.CUDA_SUCCESS, f"cuMemAllocHost returned {err}"
            logger.debug(
                f"Allocated Tensor Binding {binding} Memory {nbytes} Bytes ({size} * {self._get_binding_dtype(binding)})"
            )
            err, device_mem = cuda.cuMemAlloc(nbytes)
            assert err is cuda.CUresult.CUDA_SUCCESS, f"cuMemAlloc returned {err}"
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._binding_is_input(binding):
                logger.debug(f"Input has Shape {binding_dims}")
                inputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size, dtype))
            else:
                # each grid has 3 anchors, each anchor generates a detection
                # output of 7 float32 values
                # assert size % 7 == 0, f"output size was {size}"  ##  by pass with yolov8
                logger.debug(f"Output has Shape {binding_dims}")
                outputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size, dtype))
                output_idx += 1
        assert len(inputs) == 1, f"inputs len was {len(inputs)}"
        # assert len(outputs) == 1, f"output len was {len(outputs)}"   ##  by pass with yolov8
        return inputs, outputs, bindings

    def _do_inference(self):
        """do_inference (for TensorRT 7.0+)
        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Push CUDA Context
        cuda.cuCtxPushCurrent(self.cu_ctx)

        # Transfer input data to the GPU.
        [
            cuda.cuMemcpyHtoDAsync(inp.device, inp.host, inp.nbytes, self.stream)
            for inp in self.inputs
        ]

        # Run inference.
        if not self._execute():
            logger.warning("Execute returned false")

        # Transfer predictions back from the GPU.
        [
            cuda.cuMemcpyDtoHAsync(out.host, out.device, out.nbytes, self.stream)
            for out in self.outputs
        ]

        # Synchronize the stream
        cuda.cuStreamSynchronize(self.stream)

        # Pop CUDA Context
        cuda.cuCtxPopCurrent()

        # Return only the host outputs.
        # return [
        #     np.array(
        #         (ctypes.c_float * out.size).from_address(out.host), dtype=np.float32
        #     )
        #     for out in self.outputs
        # ]
        ctypes_outputs = [ctypes.c_float if out.dtype == np.float32 else ctypes.c_int32 for out in self.outputs]
        return [
            np.array(
                (ctypes * out.size).from_address(out.host), dtype=out.dtype
            )
            for (ctypes, out) in zip(ctypes_outputs, self.outputs)
        ]
    def __init__(self, detector_config: TensorRTDetectorConfig):
        assert (
            TRT_SUPPORT
        ), f"TensorRT libraries not found, {DETECTOR_KEY} detector not present"

        (cuda_err,) = cuda.cuInit(0)
        assert (
            cuda_err == cuda.CUresult.CUDA_SUCCESS
        ), f"Failed to initialize cuda {cuda_err}"
        err, dev_count = cuda.cuDeviceGetCount()
        logger.debug(f"Num Available Devices: {dev_count}")
        assert (
            detector_config.device < dev_count
        ), f"Invalid TensorRT Device Config. Device {detector_config.device} Invalid."
        err, self.cu_ctx = cuda.cuCtxCreate(
            cuda.CUctx_flags.CU_CTX_MAP_HOST, detector_config.device
        )

        
        self.model_type = detector_config.model.model_type
        self.conf_th = 0.4  ##TODO: model config parameter
        self.nms_threshold = 0.4
        err, self.stream = cuda.cuStreamCreate(0)
        self.trt_logger = TrtLogger()
        self.engine = self._load_engine(detector_config.model.path)
        self.input_shape = self._get_input_shape()
        self.height = detector_config.model.height
        self.width = detector_config.model.width
        
        self.bindings: List[int] = [0] * self.engine.num_bindings        

        try:
            self.context = self.engine.create_execution_context()
            (
                self.inputs,
                self.outputs,
                self.bindings,
            ) = self._allocate_buffers()
        except Exception as e:
            logger.error(e)
            raise RuntimeError("fail to allocate CUDA resources") from e

        logger.debug("TensorRT loaded. Input shape is %s", self.input_shape)
        logger.debug("TensorRT version is %s", TRT_VERSION)

    def __del__(self):
        """Free CUDA memories."""
        if self.outputs is not None:
            del self.outputs
        if self.inputs is not None:
            del self.inputs
        if self.stream is not None:
            cuda.cuStreamDestroy(self.stream)
            del self.stream
        del self.engine
        del self.context
        del self.trt_logger
        cuda.cuCtxDestroy(self.cu_ctx)

    def _postprocess_yolo(self, trt_outputs, conf_th):
        """Postprocess TensorRT outputs.
        # Args
            trt_outputs: a list of 2 or 3 tensors, where each tensor
                        contains a multiple of 7 float32 numbers in
                        the order of [x, y, w, h, box_confidence, class_id, class_prob]
            conf_th: confidence threshold
        # Returns
            boxes, scores, classes
        """
        # filter low-conf detections and concatenate results of all yolo layers
        detection_list = []
        for o in trt_outputs:
            detections = o.reshape((-1, 7))
            detections = detections[detections[:, 4] * detections[:, 6] >= conf_th]
            detection_list.append(detections)
        detection_list = np.concatenate(detection_list, axis=0) # detection_list = raw_detections

        if len(detection_list) == 0:
            return np.zeros((20, 6), np.float32)

        # detection_list: Nx7 numpy arrays of
        #             [[x, y, w, h, box_confidence, class_id, class_prob],

        # Calculate score as box_confidence x class_prob
        detection_list[:, 4] = detection_list[:, 4] * detection_list[:, 6]
        # Reorder elements by the score, best on top, remove class_prob
        ordered = detection_list[detection_list[:, 4].argsort()[::-1]][:, 0:6]
        # transform width to right with clamp to 0..1
        ordered[:, 2] = np.clip(ordered[:, 2] + ordered[:, 0], 0, 1)
        # transform height to bottom with clamp to 0..1
        ordered[:, 3] = np.clip(ordered[:, 3] + ordered[:, 1], 0, 1)
        # put result into the correct order and limit to top 20
        detections = ordered[:, [5, 4, 1, 0, 3, 2]][:20]

        # pad to 20x6 shape
        append_cnt = 20 - len(detections)
        if append_cnt > 0:
            detections = np.append(
                detections, np.zeros((append_cnt, 6), np.float32), axis=0
            )

        return detections
    
    ## anhlbt
    def _postprocess_yolov5_8(self, trt_outputs):
        """
        Processes yolov8 output.

        Args:
        results: array with shape: (84, n) where n depends on yolov8 model size (for 320x320 model n=2100)
        yolov8: (89, 1000)
        Returns:
        detections: array with shape (20, 6) with 20 rows of (class, confidence, y_min, x_min, y_max, x_max)
        """

        # results = np.transpose(np.array(results[0]).reshape((84, -1)))  # array shape (2100, 84)
        results = np.transpose(np.array(trt_outputs[0][1:]).reshape((89, -1)))   
        scores = np.max(
            results[:, 4:], axis=1
        )  # array shape (2100,); max confidence of each row

        # remove lines with score scores < 0.4
        filtered_arg = np.argwhere(scores > self.conf_th)
        results = results[filtered_arg[:, 0]]
        scores = scores[filtered_arg[:, 0]]

        num_detections = len(scores)

        if num_detections == 0:
            return np.zeros((20, 6), np.float32)

        if num_detections > 20:
            top_arg = np.argpartition(scores, -20)[-20:]
            results = results[top_arg]
            scores = scores[top_arg]
            num_detections = 20

        classes = np.argmax(results[:, 4:], axis=1)

        boxes = np.transpose(
            np.vstack(
                (
                    (results[:, 1] - 0.5 * results[:, 3]) / self.height,
                    (results[:, 0] - 0.5 * results[:, 2]) / self.width,
                    (results[:, 1] + 0.5 * results[:, 3]) / self.height,
                    (results[:, 0] + 0.5 * results[:, 2]) / self.width,                                        
                )
            )
        )

        detections = np.zeros((20, 6), np.float32)
        detections[:num_detections, 0] = classes
        detections[:num_detections, 1] = scores
        detections[:num_detections, 2:] = boxes
        return detections


    # trt_outputs: num_dets, bboxes, scores, labels
    def det_postprocess(self, data: Tuple[ndarray, ndarray, ndarray, ndarray]):
        '''
        Postprocesses the detection data and returns an array with shape (20, 6) with 20 rows of (class, confidence, y_min, x_min, y_max, x_max).
        Args:
            data: A tuple containing the detection data (num_dets, bboxes, scores, labels).
        Returns:
            detections: An array with shape (20, 6) containing the postprocessed detection results.
        '''
        max_obj = 20
        assert len(data) == 4, f"outputs len was {len(data)}"
        num_dets, bboxes, scores, labels = (i for i in data)                   
        nums = min(num_dets[0], max_obj)
        if nums == 0:
            # return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
            return np.hstack((
                    np.array([[0., 0., 0., 0., 0.]], dtype=np.float32), 
                    np.array([[0]], dtype=np.int32)
                ))
        # check score negative
        scores[scores < 0] = 1 + scores[scores < 0]
        bboxes = bboxes.reshape(100, -1)[:nums, :]
        scores = scores[:nums]
        labels = labels[:nums]        
        boxes = np.transpose(
            np.vstack(
                (
                    (bboxes[:, 1]) / self.width,  # x_min
                    (bboxes[:, 0]) / self.height, # y_min
                    (bboxes[:, 3]) / self.width,  # x_max
                    (bboxes[:, 2]) / self.height, # y_max                   
                )
            )
        )

        detections = np.zeros((max_obj, 6), dtype=np.float32) # 20
        detections[:nums, 0] = labels
        detections[:nums, 1] = scores
        detections[:nums, 2:] = boxes
                        
        return detections


    def detect_raw(self, tensor_input):
        # Input tensor has the shape of the [height, width, 3]
        # Output tensor of float32 of shape [20, 6] where:
        # O - class id
        # 1 - score
        # 2..5 - a value between 0 and 1 of the box: [top, left, bottom, right]
        # normalize
        if self.input_shape[-1] != trt.int8:
            tensor_input = tensor_input.astype(self.input_shape[-1])
            tensor_input /= 255.0

        self.inputs[0].host = np.ascontiguousarray(
            tensor_input.astype(self.input_shape[-1])
        )
        trt_outputs = self._do_inference()
                  
        if self.model_type in (ModelTypeEnum.yolov5, ModelTypeEnum.yolov8):
            # return self._postprocess_yolov5_8(trt_outputs)
            return self.det_postprocess(trt_outputs) # yolov8
   
        return self._postprocess_yolo(trt_outputs, self.conf_th)     
