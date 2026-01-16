"""
STEP 1:

Object detection and tracking on video.

It saves the original bboxes after detection to use for annotations instead of the distorted ones that are returned after passing through bytetrack.

Additionally this code employes a ROI mask to focus detection on relevant areas of the frame.
"""

# COCO class ID for vehicles: car, motorcycle, bus, truck
classes = [2, 5, 7]

import supervision as sv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# print out ultralytics version
import ultralytics
print("Ultralytics version:", ultralytics.__version__)

from ultralytics import YOLO
model = YOLO(f'yolov8x.pt')
tracker = sv.ByteTrack()
classes = [2, 5, 7]
conf_thres = 0.3

def apply_roi_mask(frame, mask):
    if mask.shape[:2] != frame.shape[:2]:
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    else:
        mask_resized = mask

    return cv2.bitwise_and(frame, frame, mask=mask_resized)

mask = cv2.imread(
    "/home/tomass/tomass/Cam_record/ROIs/fisheye_cam_right_perspective_strict_roi.png",
    cv2.IMREAD_GRAYSCALE
)
# Make binary mask (0 / 255)
mask = (mask > 0).astype(np.uint8) * 255

folder_name = "right"
frames = sorted(Path(f"/home/tomass/tomass/Cam_record/12.01.26/perspective_views_fisheye_record/{folder_name}").glob("*.jpg"))
# Define output directory
output_dir = f"/home/tomass/tomass/Cam_record/12.01.26/perspective_views_fisheye_record/{folder_name}"

rows = []

for frame_id, frame_path in enumerate(frames):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    print(f"Processing frame {frame_id}: {frame_path}", end='\r')

    masked_frame = apply_roi_mask(frame, mask)

    # ---- YOLO DETECTION (NO TRACKING) ----
    result = model(masked_frame, conf=conf_thres, classes=classes, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    if len(detections) == 0:
        tracker.update_with_detections(detections)
        continue

    # Save original detector boxes
    original_xyxy = detections.xyxy.copy()

    # ---- BYTE TRACKING ----
    tracked = tracker.update_with_detections(detections)

    if len(tracked) == 0:
        continue


    iou = sv.box_iou_batch(original_xyxy, tracked.xyxy)
    matches = iou.argmax(axis=0)

    matched_boxes = original_xyxy[matches]

    for i in range(len(tracked)):
        x1, y1, x2, y2 = matched_boxes[i]

        rows.append([
            Path(folder_name,frame_path.name),
            frame_id,
            int(tracked.tracker_id[i]),
            x1, y1, x2, y2,
            float(tracked.confidence[i]),
            int(tracked.class_id[i])
        ])

df = pd.DataFrame(
    rows,
    columns=["filename", "frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"]
)

# Save as CSV
csv_path = f"{output_dir}/annotations_test.csv"
df.to_csv(csv_path, index=False)

print(f"Annotations saved to {csv_path}")