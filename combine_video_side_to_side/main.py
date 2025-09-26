import cv2
import numpy as np

# --- Inputs ---
video1_path = "/home/tomass/tomass/magistrs/video_annotating/annotated_output/pidgeon_annotations4/vid_4_gt_slowed_half1.avi"
video2_path = "/home/tomass/tomass/magistrs/video_annotating/annotated_output/pidgeon_annotations4/vid_4_gt_slowed_half2.avi"
output_path = "/home/tomass/tomass/magistrs/video_annotating/annotated_output/pidgeon_annotations4/vid_4_gt_split.avi"

# --- Open videos ---
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

fps = int(cap1.get(cv2.CAP_PROP_FPS))  # assume same FPS
h1, w1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h2, w2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))

# Resize second video to match height of first
target_height = min(h1, h2)
scale1 = target_height / h1
scale2 = target_height / h2
new_w1, new_h1 = int(w1 * scale1), target_height
new_w2, new_h2 = int(w2 * scale2), target_height

out_w = new_w1 + new_w2
out_h = target_height

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize to same height
    frame1 = cv2.resize(frame1, (new_w1, new_h1))
    frame2 = cv2.resize(frame2, (new_w2, new_h2))

    # Concatenate horizontally
    combined = np.hstack((frame1, frame2))

    out.write(combined)

cap1.release()
cap2.release()
out.release()
print(f"Saved side-by-side video as {output_path}")
