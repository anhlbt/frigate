# import os
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.cluster import KMeans

# # Function to extract frames from videos
# def extract_frames(video_path, frame_skip=30):
#     frames = []
#     frame_info = []
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % frame_skip == 0:
#             frames.append(frame)
#             frame_info.append(frame_count)
#         frame_count += 1
#     cap.release()
#     return frames, frame_info

# # Function to detect faces and get embeddings
# def get_face_embeddings(frames, face_analysis_app):
#     face_embeddings = []
#     for frame in frames:
#         faces = face_analysis_app.get(frame)
#         if len(faces) > 0:
#             face_embeddings.append(faces[0].embedding)
#         else:
#             face_embeddings.append(None)
#     return face_embeddings

# # Function to cluster embeddings
# def cluster_embeddings(face_embeddings, n_clusters=5):
#     valid_embeddings = [embedding for embedding in face_embeddings if embedding is not None]
#     if len(valid_embeddings) < n_clusters:
#         n_clusters = len(valid_embeddings)
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(valid_embeddings)
#     return kmeans.labels_, kmeans.cluster_centers_

# # Function to select representative frames from clusters
# def select_representative_frames(face_embeddings, frames, frame_info, labels, n_clusters):
#     selected_frames = []
#     selected_frame_info = []
#     for i in range(n_clusters):
#         cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
#         if cluster_indices:
#             selected_idx = cluster_indices[0]  # Choose the first frame from each cluster
#             selected_frames.append(frames[selected_idx])
#             selected_frame_info.append(frame_info[selected_idx])
#     return selected_frames, selected_frame_info

# # Initialize InsightFace
# face_analysis_app = FaceAnalysis()
# face_analysis_app.prepare(ctx_id=0, det_thresh=0.4)

# # Paths
# video_path = '/workspaces/frigate/storage/recordings/2024-06-06/09/esp-cam/42.36.mp4'
# output_folder = './outputs'
# os.makedirs(output_folder, exist_ok=True)

# # # Process each video
# # video_paths = [os.path.join(video_folder, vid) for vid in os.listdir(video_folder) if vid.endswith(('.mp4', '.avi'))]
# # for video_path in video_paths:

# frames, frame_info = extract_frames(video_path)
# face_embeddings = get_face_embeddings(frames, face_analysis_app)

# if len([embedding for embedding in face_embeddings if embedding is not None]) >= 5:
#     labels, cluster_centers = cluster_embeddings(face_embeddings, n_clusters=5)
#     selected_frames, selected_frame_info = select_representative_frames(face_embeddings, frames, frame_info, labels, n_clusters=5)
# else:
#     selected_frames = [frames[i] for i, emb in enumerate(face_embeddings) if emb is not None]
#     selected_frame_info = [frame_info[i] for i, emb in enumerate(face_embeddings) if emb is not None]

# # Save selected frames
# for idx, (frame, frame_number) in enumerate(zip(selected_frames, selected_frame_info)):
#     output_path = os.path.join(output_folder, f'{os.path.basename(video_path)}_frame_{frame_number}.jpg')
#     cv2.imwrite(output_path, frame)


import os
import cv2
import numpy as np

def process_video(video_path, output_folder, num_frames=20, num_representatives=5):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))  # Save frame index along with frame
    cap.release()

    # Calculate frame differences
    differences = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i - 1][1], frames[i][1])  # Calculate difference between consecutive frames
        differences.append(np.mean(diff))

    # Select representative frames
    selected_frames = []
    selected_indices = np.argsort(differences)[-num_representatives:]
    for idx in selected_indices:
        selected_frames.append(frames[idx])

    # Save representative frames
    for idx, (frame_idx, frame) in enumerate(selected_frames):
        output_path = os.path.join(output_folder, f'{os.path.basename(video_path)}_frame_{frame_idx}.jpg')
        cv2.imwrite(output_path, frame)

    return selected_frames

# Paths
video_path = '/workspaces/frigate/storage/recordings/2024-06-15/08/front-cam/33.11.mp4'
output_folder = './outputs'

# Process the video
representative_frames = process_video(video_path, output_folder, num_frames=20, num_representatives=10)
