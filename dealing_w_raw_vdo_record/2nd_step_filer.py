import cv2
import os
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 5, 7}

def filter_vehicle_frames_from_folder(
    frame_dir,
    out_dir,
    roi_mask_path,
    yolo_weights="yolov8n.pt",
    min_area=150,
    min_aspect=0.3,
    max_aspect=3.5,
    yolo_conf=0.3
):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    assert len(files) > 1, "Need at least 2 frames"

    roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    assert roi_mask is not None, "ROI mask missing"

    detector = YOLO(yolo_weights)

    prev = None

    for fname in files:
        path = os.path.join(frame_dir, fname)
        frame = cv2.imread(path)

        frame_masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
        gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

        if prev is None:
            prev = gray
            continue

        diff = cv2.absdiff(gray, prev)
        _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
        diff = cv2.medianBlur(diff, 5)

        num, _, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)

        vehicle_found = False

        for i in range(1, num):
            x, y, w, h, area = stats[i]

            if area < min_area:
                continue

            aspect = w / float(h)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            pad = 30
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)

            roi = frame[y0:y1, x0:x1]

            results = detector(
                roi,
                conf=yolo_conf,
                classes=list(VEHICLE_CLASSES),
                verbose=False
            )

            if len(results[0].boxes) > 0:
                vehicle_found = True
                break

        if vehicle_found:
            cv2.imwrite(os.path.join(out_dir, fname), frame)

        prev = gray

# Example usage:
filter_vehicle_frames_from_folder(
    frame_dir="/home/tomass/tomass/Cam_record/04.09.25_3/fisheye_record_1756991341.9092808_frames",
    out_dir="/home/tomass/tomass/Cam_record/04.09.25_3/fisheye_record_1756991341.9092808_frames_refined",
    roi_mask_path="/home/tomass/tomass/Cam_record/ROIs/fisheye_cam_movement_roi.png")
