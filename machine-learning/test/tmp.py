import cv2
from insightface.app import FaceAnalysis
from skimage.filters import laplace
from deepface import DeepFace
import numpy as np
from scipy.spatial import distance as dist
from skimage.metrics import structural_similarity as ssim
from insightface.utils.face_align import norm_crop
import os
import numpy as np
import time
from os.path import dirname, realpath, join, basename, splitext, relpath
import sys
C_DIR = dirname(realpath(__file__))
ML_DIR = dirname(C_DIR)

tmp = "/workspaces/frigate/machine-learning/app/models"
sys.path.insert(0, tmp)

from emotion import FaceEmotionEstimator

# object detection: person

SIM_THRESHOLD = 0.28
EYE_THRESHOLD = 0.19 # default: 0.15
MOUTH_THRESHOLD = 0.15

FACE_RL_THRESHOLD = 2.5 # default: 2.5
FACE_UP_THRESHOLD = 0.25
FACE_DOWN_THRESHOLD = 0.55

INPUT_DIR_MODEL_ESTIMATION      = join(ML_DIR, "models/estimation/")


# Initialize the InsightFace model
app = FaceAnalysis(name="buffalo_l")  # antelopev2
app.prepare(ctx_id=0, det_thresh=0.5)

# Function to assess sharpness
def assess_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Function to check if an image is blurred
def is_blurred(image, threshold=100.0):
    sharpness = assess_sharpness(image)
    return sharpness < threshold, sharpness

# Function to assess lighting
def assess_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Function to assess expression
def assess_expression(image):
    result = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
    if isinstance(result, list):
        result = result[0]
    return result['dominant_emotion'], result['emotion']['happy']

# Function to calculate face angles (yaw and pitch)
def calculate_face_angles(lmk):
    result = ""
    face12, face28 = lmk[12], lmk[28]
    nose86 = lmk[86]
    A = dist.euclidean(face12, nose86)
    B = dist.euclidean(face28, nose86)
    face_RL_rate = A / B
    if face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE RIGHT {face_RL_rate}"
    elif 1 / face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE LEFT {face_RL_rate}"
    chin0 = lmk[0]
    forehead72 = lmk[72]
    A = dist.euclidean(forehead72, nose86)
    B = dist.euclidean(chin0, nose86)
    face_UD_rate = A / B
    if face_UD_rate <= FACE_UP_THRESHOLD:
        result += f" FACE UP {face_UD_rate}"
    elif face_UD_rate >= FACE_DOWN_THRESHOLD:
        result += f" FACE DOWN {face_UD_rate}"
    return result

# Corrected function to check if mouth is open
def is_mouth_open(landmarks, threshold=MOUTH_THRESHOLD):
    mouth66, mouth54 = landmarks[66], landmarks[54]
    mouth70, mouth57 = landmarks[70], landmarks[57]
    mouth65, mouth69 = landmarks[65], landmarks[69]
    A = dist.euclidean(mouth66, mouth54)
    B = dist.euclidean(mouth70, mouth57)
    C = dist.euclidean(mouth65, mouth69)
    mouth_rate = (A + B) / (2.0 * C)
    return mouth_rate >= threshold, mouth_rate

# Corrected function to check if eyes are closed
def are_eyes_closed(landmarks, threshold=EYE_THRESHOLD):
    eye41, eye42, eye36, eye37 = landmarks[41], landmarks[42], landmarks[36], landmarks[37]
    eye35, eye39 = landmarks[35], landmarks[39]
    A_left = dist.euclidean(eye41, eye36)
    B_left = dist.euclidean(eye42, eye37)
    C_left = dist.euclidean(eye35, eye39)
    lefteye_rate = (A_left + B_left) / (2.0 * C_left)
    eye95, eye90, eye96, eye91 = landmarks[95], landmarks[90], landmarks[96], landmarks[91]
    eye89, eye93 = landmarks[89], landmarks[93]
    A_right = dist.euclidean(eye95, eye90)
    B_right = dist.euclidean(eye96, eye91)
    C_right = dist.euclidean(eye89, eye93)
    righteye_rate = (A_right + B_right) / (2.0 * C_right)
    eye_rate = (lefteye_rate + righteye_rate) / 2.0
    return eye_rate <= threshold, eye_rate

def face_analysis(lmk):
    eye41, eye42, eye36, eye37 = lmk[41], lmk[42], lmk[36], lmk[37]
    eye35, eye39 = lmk[35], lmk[39]
    A = dist.euclidean(eye41, eye36)
    B = dist.euclidean(eye42, eye37)
    C = dist.euclidean(eye35, eye39)
    lefteye_rate = (A + B) / (2.0 * C)
    eye95, eye90, eye96, eye91 = lmk[95], lmk[90], lmk[96], lmk[91]
    eye89, eye93 = lmk[89], lmk[93]
    A = dist.euclidean(eye95, eye90)
    B = dist.euclidean(eye96, eye91)
    C = dist.euclidean(eye89, eye93)
    righteye_rate = (A + B) / (2.0 * C)
    eye_rate = (lefteye_rate + righteye_rate) / 2.0
    result = ""
    if eye_rate <= EYE_THRESHOLD:
        result += f"EYE_CLOSE {eye_rate}"
    else:
        result += f"EYE_OPEN {eye_rate}"
    mouth66, mouth54 = lmk[66], lmk[54]
    mouth70, mouth57 = lmk[70], lmk[57]
    mouth65, mouth69 = lmk[65], lmk[69]
    A = dist.euclidean(mouth66, mouth54)
    B = dist.euclidean(mouth70, mouth57)
    C = dist.euclidean(mouth65, mouth69)
    mouth_rate = (A + B) / (2.0 * C)
    if mouth_rate >= MOUTH_THRESHOLD:
        result += f" MOUTH_OPEN {mouth_rate}"
    else:
        result += f" MOUTH_CLOSE {mouth_rate}"
    face12, face28 = lmk[12], lmk[28]
    nose86 = lmk[86]
    A = dist.euclidean(face12, nose86)
    B = dist.euclidean(face28, nose86)
    face_RL_rate = A / B
    if face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE_RIGHT {face_RL_rate}"
    elif 1 / face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE_LEFT {face_RL_rate}"
    chin0 = lmk[0]
    forehead72 = lmk[72]
    A = dist.euclidean(forehead72, nose86)
    B = dist.euclidean(chin0, nose86)
    face_UD_rate = A / B
    if face_UD_rate <= FACE_UP_THRESHOLD:
        result += f" FACE_UP {face_UD_rate}"
    elif face_UD_rate >= FACE_DOWN_THRESHOLD:
        result += f" FACE_DOWN {face_UD_rate}"
    return result

def calculate_min_margin(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    
    left_margin = x_min
    right_margin = image_width - x_max
    top_margin = y_min
    bottom_margin = image_height - y_max
    
    min_margin = min(left_margin, right_margin, top_margin, bottom_margin)
    
    return min_margin

def compare_frames(frame1, frame2, threshold=0.9):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold

def extract_distinct_frames(video_path, output_folder="./outputs", num_frames=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    differences = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i - 1][1], frames[i][1])
        differences.append(np.mean(diff))
    selected_frames = []
    selected_indices = np.argsort(differences)[-num_frames:]
    for idx in selected_indices:
        selected_frames.append(frames[idx])
    for idx, (frame_idx, frame) in enumerate(selected_frames):
        output_path = os.path.join(output_folder, f'{os.path.basename(video_path)}_frame_{frame_idx}.jpg')
        cv2.imwrite(output_path, frame)
    return [frame for _, frame in selected_frames]



# Function to get the 5 main landmarks from the 106 landmarks
def get_image_points_from_landmark(face_landmark):
    image_points = np.array([
        # face_landmark[0],  # Chin
        face_landmark[93],  # Left eye left corner
        face_landmark[35],  # Right eye right corne
        face_landmark[86],  # Nose tip
        face_landmark[61],  # Left Mouth corner
        face_landmark[52]  # Right mouth corner
    ], dtype="float32")
    return image_points

# Updated evaluate_face function
def evaluate_face(face, img):
    main_landmarks = get_image_points_from_landmark(face.landmark_2d_106)
    face_img = norm_crop(img, landmark=main_landmarks)
    
    blurred, sharpness = is_blurred(face_img)
    if blurred:
        return 0

    lighting = assess_lighting(face_img)
    expression, happiness_score = assess_expression(face_img)
    print("expression: ", expression)
    
    result = calculate_face_angles(face.landmark_2d_106)
    print("face_angles: ", result)

    mouth_open, m_score = is_mouth_open(face.landmark_2d_106)
    print("mouth_open: ", mouth_open, m_score)
    
    eyes_closed, e_score = are_eyes_closed(face.landmark_2d_106)
    print("eyes_closed: ", eyes_closed, e_score)
    if mouth_open or eyes_closed:
        return 0, ()
    # Assess sharpness
    sharpness_score = sharpness / 100.0  # Normalize sharpness score between 0 and 1


    quality_score = 0.25 * sharpness_score + 0.3 * lighting
    return quality_score, (sharpness_score, lighting)

def drawLms(img, lms, color=(255, 0, 0),x="1"):
    img1 = img.copy()
    for i,lm in enumerate(lms):
        cv2.circle(img1, tuple(lm), 1, color, 2)
        cv2.putText(img,'%d'%(i), (lm[0]-2, lm[1]-3),cv2.FONT_HERSHEY_DUPLEX,0.3,(0,0,255),1)

# Function to draw face mask and key points
def draw_face_mask_and_keypoints(frame, face):
    mask = np.zeros_like(frame)
    landmarks = face.landmark_2d_106.astype(int)

    # # Correct face contour
    face_contour = [1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,
                        43,48,49,51,50,
                        102,103,104,105,101,
                        72,73,74,86,78,79,80,85,84,
                        35,41,42,39,37,36,
                        89,95,96,93,91,90,
                        52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54
                        ]
    for i in range(len(face_contour) - 1):
        cv2.line(mask, tuple(landmarks[face_contour[i]]), tuple(landmarks[face_contour[i + 1]]), (0, 255, 0), 2)

    # Draw key points with their order numbers
    for i, point in enumerate(landmarks):
        cv2.circle(mask, tuple(point), 2, (0, 0, 255), 2)
        cv2.putText(mask, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(mask,'%d'%(i), (point[0]-2, point[1]-3),cv2.FONT_HERSHEY_DUPLEX,0.3,(0,0,255),1)

    combined = cv2.addWeighted(frame, 0.8, mask, 0.2, 0)
    return combined

# Function to draw only the 5 main key points
def draw_main_keypoints(frame, face):
    mask = np.zeros_like(frame)
    main_landmarks = get_image_points_from_landmark(face.landmark_2d_106).astype(int)
    for i, point in enumerate(main_landmarks):
        cv2.circle(mask, tuple(point), 2, (0, 0, 255), -1)
        cv2.putText(mask, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    combined = cv2.addWeighted(frame, 0.8, mask, 0.2, 0)
    return combined

# Function to extract frames and filter based on rules
def get_best_frame(video_path, output_folder="./outputs", num_frames=5, offset = 30):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(join(output_folder, "invalid"), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_bin = max(num_frames, 10)
    frame_indices = np.linspace(offset, total_frames - 1 - offset, num_bin, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))  # Save frame index along with frame
    cap.release()
    selected_frames = []
    for idx, (frame_idx, frame) in enumerate(frames):
        faces = app.get(frame)
        valid_frame = True
        for face in faces:
            result = face_analysis(face.landmark_2d_106)
            if any(keyword in result for keyword in ['FACE_RIGHT', 'FACE_LEFT', 'FACE_UP', 'FACE_DOWN', 'MOUTH_OPEN', 'EYE_CLOSE']):
                valid_frame = False
                print("* " * 5, result, " *" * 5)  
                ignore_img = join(output_folder, "invalid", f'{basename(video_path)}_bypass_frame_{frame_idx}.jpg')
                cv2.imwrite(ignore_img, frame)
                break
        if valid_frame:
            selected_frames.append((frame_idx, frame))        
        # Limit to num_frames frames
        if len(selected_frames) >= num_frames:
            break
    for idx, (frame_idx, frame) in enumerate(selected_frames):           
        save_frame(frame, video_path, input_dir, output_folder, frame_idx)   
    return selected_frames


def save_frame(frame, video_path, root_dir, output_base_path, frame_id):
    # Calculate relative path
    relative_path = relpath(video_path, "/workspaces/frigate/storage/recordings") # root_dir
    cam_id = video_path.split('/')[-2]
    store_id = cam_id.split('-')[0]
    date_str = relative_path.split('/')[0]
    # relative_path = splitext(relative_path)[0]  # Remove the file extension
    # Construct the output directory path
    output_dir = join(output_base_path,date_str, store_id) # relative_path
    os.makedirs(output_dir, exist_ok=True)
    # Construct the output file path
    output_file = join(output_dir, f"{relative_path.replace('/', '_').lstrip('_')}_{frame_id}.jpg")
    # Save the frame
    cv2.imwrite(output_file, frame)
    print(f"Saved frame to: {output_file}")

def process_videos(input_dir, output_dir):
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(join(root, file))

    for video_file in video_files:
        selected_frames = get_best_frame(video_file)
        # Process selected frames
        for frame_count, frame in selected_frames:
            faces = app.get(frame)
            for face in faces:
                score, all_score = evaluate_face(face, frame)
                annotated_frame = draw_face_mask_and_keypoints(frame, face)
                save_frame(annotated_frame, video_file, input_dir, output_dir, frame_count)
                print("# " * 10, frame_count, " #" * 10, score, all_score)

                # Optionally, show the annotated frame
                # cv2.imshow("Best Frame with Face Mask and Key Points", annotated_frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                
if __name__ == "__main__":
    input_dir = "/workspaces/frigate/storage/recordings/2024-06-15/07/front-cam"  # Update this to your input directory
    output_dir="albums"
    process_videos(input_dir, output_dir)
