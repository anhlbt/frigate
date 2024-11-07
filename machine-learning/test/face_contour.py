from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image

det_size = 640
ctx_id = -1
app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

class HeadPose(object):
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ], dtype="float32")

    def get_face_landmark(self, image):
        face = app.get(image)
        if len(face) > 0:
            return face[0].landmark_2d_106
        return None

    def get_image_points_from_landmark(self, face_landmark):
        image_points = np.array([
            face_landmark[86],  # Nose tip
            face_landmark[0],  # Chin
            face_landmark[93],  # Left eye left corner
            face_landmark[35],  # Right eye right corner
            face_landmark[61],  # Left Mouth corner
            face_landmark[52]  # Right mouth corner
        ], dtype="float32")
        return image_points

    def calculate_pose_vector(self, image):
        face_2d_landmark = self.get_face_landmark(image)
        if face_2d_landmark is None:
            return None, None
        ldk_min = np.min(face_2d_landmark, axis=0)
        ldk_max = np.max(face_2d_landmark, axis=0)
        ldk_w_h = ldk_max - ldk_min  # w, h
        face_2d = face_2d_landmark - ldk_min
        image_points = self.get_image_points_from_landmark(face_2d)
        camera_matrix = np.array(
            [[ldk_w_h[0], 0, ldk_w_h[0] / 2],
             [0, ldk_w_h[0], ldk_w_h[1] / 2],
             [0, 0, 1]], dtype="float32")
        dist_coeffs = np.zeros((4, 1), dtype="float32")  # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                    dist_coeffs)
        return rotation_vector.T, translation_vector.T

    def get_yaw_pitch_roll(self, rotation_vector):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)

def main():
    head_pose = HeadPose()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rotation_vector, translation_vector = head_pose.calculate_pose_vector(frame)
        if rotation_vector is not None:
            pitch, yaw, roll = head_pose.get_yaw_pitch_roll(rotation_vector)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if yaw > 10:
                direction_yaw = "Right"
            elif yaw < -10:
                direction_yaw = "Left"
            else:
                direction_yaw = "Center"

            if pitch > 10:
                direction_pitch = "Down"
            elif pitch < -10:
                direction_pitch = "Up"
            else:
                direction_pitch = "Center"

            direction = f"{direction_yaw}, {direction_pitch}"

            cv2.putText(frame, f"Direction: {direction}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

