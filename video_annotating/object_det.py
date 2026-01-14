"""
STEP 1:

Object detection and tracking on video.
"""

# COCO class ID for vehicles: car, motorcycle, bus, truck
classes = [2, 5, 7]

from ultralytics import YOLO
import pandas as pd

from pathlib import Path

model = YOLO(f'yolov8x.pt')

folder_name = "right"
# Define output directory
output_dir = f"/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/{folder_name}"

# Run inference with tracking enabled
results = model.track(
    source=f"/home/tomass/tomass/Cam_record/04.09.25_2/perspective_views_fisheye_record_1756987992.9123657_frames_refined/{folder_name}/*.jpg",  # Input frames path
    save=True, 
    save_txt=False,  # We handle saving manually
    conf=0.3,
    classes=classes,  
    project=output_dir,
    name=f"{folder_name}",
    tracker="bytetrack.yaml"  # Use ByteTrack for tracking
)


csv_data = []
# Iterate over the results to extract frame data and tracking information
frame_id = 0  # Initialize frame ID

# Iterate through the results
for result in results:
    for box in result.boxes.data:
        x1, y1, x2, y2, track_id, conf, cls = box.tolist()

        filename = Path(folder_name,Path(result.path).name)
        
        # Save frame_id, track_id, bounding box, confidence, and class_id to CSV
        csv_data.append([
            filename,
            frame_id,
            int(track_id),
            x1, y1, x2, y2,
            conf,
            int(cls)
        ])
    
    # Increment frame_id for each frame processed
    frame_id += 1

# Convert to DataFrame
df = pd.DataFrame(csv_data, columns=["filename", "frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"])

# Save as CSV
csv_path = f"{output_dir}/annotations.csv"
df.to_csv(csv_path, index=False)

print(f"Annotations saved to {csv_path}")