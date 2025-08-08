import cv2

import argparse

from detector import VehicleDetector
from visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path1', type=str, default='/home/tomass/tomass/docker/dockerized_reid_pipeline/detection/videos/vdo4.avi', help='Path to the first video file. (Re-Identification FROM)')
    parser.add_argument('--roi_path1', type=str, default="/home/tomass/tomass/docker/dockerized_reid_pipeline/detection/videos/vdo4_roi.png", help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--detection_model_path', type=str, default='yolov8x.pt', choices=['yolov8x.pt', 'yolov8l.pt', 'yolov5su.pt'] , help='Path to the YOLO model file.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Device to run the model on (e.g., "cuda" or "cpu").')

    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')

    # Args concerning the establishment of crop zones for video 1 and video 2
    parser.add_argument('--crop_zone_rows_vid1', type=int, default=7, help='Number of rows in the crop zone grid for the first video.')
    parser.add_argument('--crop_zone_cols_vid1', type=int, default=6, help='Number of columns in the crop zone grid for the first video.')
    parser.add_argument('--crop_zone_area_bottom_left_vid1', type=tuple, default=(0, 1000), help='Bottom-left corner of the crop zone area as a tuple (x, y) for the first video.')
    parser.add_argument('--crop_zone_area_top_right_vid1', type=tuple, default=(1750, 320), help='Top-right corner of the crop zone area as a tuple (x, y) for the first video.')

    return parser.parse_args()

def run_demo(video_path1, roi_path1, detection_model, device, crop_zone_rows_1, crop_zone_cols_1, crop_zone_area_bottom_left_1, crop_zone_area_top_right_1, play_mode):

    print("Starting vehicle detection demo...")

    # Initialize the vehicle detectors for both videos
    detector = VehicleDetector(video_path=video_path1, roi_path=roi_path1, model_path=detection_model, device=device)

    roi_mask = detector._load_roi(roi_path1, video_path1)

    # The visualizers will annotate the frames with the detections and matched IDs
    visualizer = Visualizer(detector.class_names, rows=crop_zone_rows_1, cols=crop_zone_cols_1, area_bottom_left= crop_zone_area_bottom_left_1, area_top_right=crop_zone_area_top_right_1, roi_mask=roi_mask)


    while True:
        ret1, frame = detector.read_frame()

        if not ret1:
            print("End of video stream.")
            break

        # Process the frames from both videos
        detections, frame = detector.process_frame(frame)
        
        vis_frame = visualizer.annotate(frame, detections)

        frame = cv2.resize(vis_frame, (1280, 720))


        cv2.imshow("Vehicle Re-ID Demo", frame)
        if cv2.waitKey(play_mode) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    args = parse_args()

    run_demo(video_path1=args.video_path1, roi_path1=args.roi_path1, detection_model=args.detection_model_path, device=args.device, crop_zone_rows_1 = args.crop_zone_rows_vid1, crop_zone_cols_1 = args.crop_zone_cols_vid1, crop_zone_area_bottom_left_1 = args.crop_zone_area_bottom_left_vid1, crop_zone_area_top_right_1 = args.crop_zone_area_top_right_vid1, play_mode=args.play_mode)

