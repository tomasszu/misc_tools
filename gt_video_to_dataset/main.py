import cv2
import os
import csv

# Config
video_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/fisheye_vdo.avi"
gt_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt_fisheye.txt"
output_dir = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/fisheye_dataset"
csv_output = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/fisheye_dataset.csv"

os.makedirs(output_dir, exist_ok=True)

# Read GT file
# Format: frame,id,x,y,w,h,rest...
gt_data = {}
with open(gt_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        vid = int(parts[1])
        x, y, w, h = map(int, [float(p) for p in parts[2:6]])
        if frame not in gt_data:
            gt_data[frame] = []
        gt_data[frame].append((vid, x, y, w, h))

# Open video
cap = cv2.VideoCapture(video_path)
frame_id = 0
csv_rows = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    if frame_id not in gt_data:
        continue

    for idx, (vid, x, y, w, h) in enumerate(gt_data[frame_id]):
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # filename format: frame_vehicleindex.jpg
        filename = f"{frame_id:06d}_{vid}_{idx}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, crop)

        # relative path for CSV
        csv_rows.append([os.path.join(output_dir, filename), vid])

cap.release()

# Write CSV
with open(csv_output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "id"])
    writer.writerows(csv_rows)

print(f"Dataset saved in {output_dir}, CSV written to {csv_output}")
