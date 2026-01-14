## A quick python code to create a avi video from frames in a folder

import cv2
import os

def create_video_from_frames(frames_dir, output_video_path, fps=30):
    # Get list of frame files and sort them
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    frame_files.sort()

    if not frame_files:
        print("No frames found in the specified directory.")
        return

    # Read the first frame to get the dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
create_video_from_frames(
    frames_dir="/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/left",
    output_video_path="/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/left_video.avi",
    fps=4
)