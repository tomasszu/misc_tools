import cv2
import pandas as pd
import os

# --- Inputs ---
video_path = "/home/tomass/tomass/magistrs/video_annotating/vid_4.MOV"
gt_path = "/home/tomass/tomass/magistrs/video_annotating/annotated_output/pidgeon_annotations4.csv"
output_path = "/home/tomass/tomass/magistrs/video_annotating/vid_4_gt.avi"

# --- Detect ground-truth format (csv or txt) ---
ext = os.path.splitext(gt_path)[1].lower()
cols = ["frame_id", "track_id", "x1", "y1", "x2", "y2"]

if ext == ".csv":
    # Your CSV format: frame_id,track_id,x1,y1,x2,y2
    gt = pd.read_csv(gt_path)
    gt["frame_id"] = gt["frame_id"].astype(int)   # remove decimals
    gt["track_id"] = gt["track_id"].astype(int)
elif ext == ".txt":
    # MOTChallenge-like: frame,id,left,top,width,height,conf,?,? ...
    cols = ["frame", "id", "left", "top", "width", "height", "conf", "x", "y", "z"]
    gt = pd.read_csv(gt_path, header=None, names=cols)
else:
    raise ValueError("Unsupported GT file format. Use .csv or .txt")

# --- Open video ---
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Codec: mp4v is widely supported (change to XVID if needed)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0  # your CSV starts at frame 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if ext == ".csv":
        # Get boxes for this frame
        boxes = gt[gt["frame_id"] == frame_idx]
        for _, row in boxes.iterrows():
            x1, y1, x2, y2, obj_id = int(row.x1), int(row.y1), int(row.x2), int(row.y2), int(row.track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 9)

            # Draw ID label with background
            text = str(obj_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 7.0
            thickness = 14
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

            text_w, text_h = text_size
            bg_x1, bg_y1 = x1, y1 - text_h - 4
            bg_x2, bg_y2 = x1 + text_w + 4, y1

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 5),
                        font, font_scale, (0, 0, 0), thickness)

    else:  # .txt (MOT style)
        boxes = gt[gt["frame"] == frame_idx + 1]  # MOT starts at frame=1
        for _, row in boxes.iterrows():
            x, y, bw, bh, obj_id = int(row.left), int(row.top), int(row.width), int(row.height), int(row.id)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 3)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Saved annotated video to {output_path}")
