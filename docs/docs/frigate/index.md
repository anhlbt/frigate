---
id: index
title: Introduction
slug: /
---

A complete and local NVR designed for Home Assistant with AI object detection. Uses OpenCV and Tensorflow to perform realtime object detection locally for IP cameras.

Use of a [Recommended Detector](/frigate/hardware#detectors) is optional, but strongly recommended. CPU detection should only be used for testing purposes.

- Tight integration with Home Assistant via a [custom component](https://github.com/blakeblackshear/frigate-hass-integration)
- Designed to minimize resource use and maximize performance by only looking for objects when and where it is necessary
- Leverages multiprocessing heavily with an emphasis on realtime over processing every frame
- Uses a very low overhead motion detection to determine where to run object detection
- Object detection with TensorFlow runs in separate processes for maximum FPS
- Communicates over MQTT for easy integration into other systems
- Recording with retention based on detected objects
- Re-streaming via RTSP to reduce the number of connections to your camera
- A dynamic combined camera view of all tracked cameras.

## Screenshots

![Live View](/img/live-view.png)

![Review Items](/img/review-items.png)

![Media Browser](/img/media_browser-min.png)

![Notification](/img/notification-min.png)
