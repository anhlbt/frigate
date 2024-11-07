import os
import cv2

def extract_and_save_frames(root_path, output_base_path="albums"):
    # Walk through the directory
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(subdir, file)
                save_frames_from_video(video_path, root_path, output_base_path)

def save_frames_from_video(video_path, root_path, output_base_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Extract first and last frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        save_frame(first_frame, video_path, root_path, output_base_path, frame_id="first")

    # Get last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if ret:
        save_frame(last_frame, video_path, root_path, output_base_path, frame_id="last")

    cap.release()

def save_frame(frame, video_path, root_path, output_base_path, frame_id):
    # Calculate relative path
    relative_path = os.path.relpath(video_path, root_path)
    relative_path = os.path.splitext(relative_path)[0]  # Remove the file extension

    # Construct the output directory path
    output_dir = os.path.join(output_base_path, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    output_file = os.path.join(output_dir, f"{frame_id}.jpg")

    # Save the frame
    cv2.imwrite(output_file, frame)
    print(f"Saved frame to: {output_file}")

# Example usage
root_path = "/workspaces/frigate/storage/recordings/"
extract_and_save_frames(root_path)
