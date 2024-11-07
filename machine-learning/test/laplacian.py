import cv2
import os

# Hàm tính độ rõ nét của ảnh
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Trích xuất khung hình từ video
video_path = '/workspaces/frigate/storage/recordings/2024-06-15/08/esp-cam/29.41.mp4'
output_folder = './frames'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()

# Chọn khung hình tốt nhất
best_focus = 0
best_frame = None

for i in range(frame_count):
    frame_path = os.path.join(output_folder, f'frame_{i:04d}.png')
    image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    focus_measure = variance_of_laplacian(image)
    if focus_measure > best_focus:
        best_focus = focus_measure
        best_frame = cv2.imread(frame_path)

# Lưu khung hình tốt nhất
if best_frame is not None:
    cv2.imwrite('best_frame.png', best_frame)
