from ultralytics import YOLOWorld, YOLO
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# Enable CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class YoloWorld:
    def __init__(self, model_path):
        self.yolo_world = YOLO(model_path)
        self.iou_threshold = 0.1

    def find_object(self, color_frame, conf, iou, max_det, visualize=True):
        results = self.yolo_world.predict(color_frame, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
        
        # Debug: Print the raw results
        # print("Raw results:", results)

        # Convert results to list
        bboxes = results.boxes.xyxy.cpu().tolist()
        cls = results.boxes.cls.cpu().tolist()
        confidences = results.boxes.conf.cpu().tolist()
        
        # Remove rows with NaN values
        valid_bboxes = []
        valid_confidences = []
        for bbox, conf in zip(bboxes, confidences):
            if not any(np.isnan(x) for x in bbox) and all(x is not None for x in bbox):
                valid_bboxes.append(bbox)
                valid_confidences.append(conf)
            else:
                print("Invalid bbox detected and removed:", bbox)

        # Check if we have valid bboxes left
        if not valid_bboxes:
            return None, 0

        # Use the first valid bbox
        bbox = valid_bboxes[0]
        confidence = valid_confidences[0]

        # Convert to integers
        bbox = [int(x) for x in bbox]
        object_name = results.names[cls[0]]
        print("object_name: ", object_name, cls[0])
        if visualize:
            # Annotate the frame with the bounding box
            label = f'{object_name} {confidence:.2f}'
            self.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), bbox, label)

        return bbox, confidence

    def set_object_to_find(self, object_to_find: list):
        self.yolo_world.set_classes(object_to_find)

    @staticmethod
    def draw_text(
            img,
            text,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            pos=(0, 0),
            font_scale=1,
            font_thickness=2,
            text_color=(0, 255, 0),
            text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO-World live')
    parser.add_argument("--src", default=None, type=str)
    parser.add_argument("--obj-name", default=None, type=str)
    args = parser.parse_args()
    return args


def main():
    obj_name = ["happy face","human baby", "computer monitor", "smiling person", "happy man", "man is smiling", "person with glasses", "woman"]
    src = 0
    yolo_world = YoloWorld('yolov8x-worldv2.pt')


    # with torch.no_grad():
    #     # Initialize a YOLO model with pretrained weights
    #     yolo_world = YOLO('yolov8l-world.pt')  # You can also choose yolov8m/l-world.pt based on your needs
    #     # Define custom classes specific to your application
    #     custom_classes = ["happy face","human baby", "computer monitor"]
    #     yolo_world.set_classes(custom_classes)
    #     # Save the model with the custom classes defined (modified code)
    #     yolo_world.save("custom_yolov8l.pt")  # This saves extra metadata required for CoreML conversion
    #     # Load the saved model with custom classes
    #     yolo_world = YOLO("custom_yolov8l.pt")
    #     # Export the model to CoreML format with non-maximum suppression enabled
    #     yolo_world.export(format="coreml", nms=True)


    if obj_name == None:
        obj_name = ""
    yolo_world.set_object_to_find(obj_name)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    frame_count = 0
    start_time = time.time()

    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots()

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        # Predict the image
        bbox, confidence = yolo_world.find_object(img, conf=0.5, iou=0.3, max_det=10, visualize=True)
        if bbox is not None:
            print("Bounding Box:", bbox, "Confidence:", confidence)
            

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_value = frame_count / elapsed_time

        # Overlay FPS information onto the frame
        cv2.putText(img, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert image from BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.clear()
        ax.imshow(img_rgb)
        ax.axis('off')
        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    plt.ioff()  # Turn off interactive mode
    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
