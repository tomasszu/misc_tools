import cv2
import os
import csv
from collections import deque

def extract_motion_sequences(
    video_path,
    out_dir,
    min_fg_ratio_start=0.01,    # threshold to detect new motion
    min_fg_ratio_continue=0.005,# threshold to continue motion sequence
    frame_skip=8,
    backtrack_frames=10,
    stop_grace_frames=8          # allow N frames below threshold before stopping
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Cannot open video"

    # Sniegam es nonesu threshold no 64 uz 16, jo video ir ar mazaku izskatu izmainamibu
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

    frame_idx = 0
    saved = 0
    buffer = deque(maxlen=backtrack_frames)
    motion_active = False
    frames_below_threshold = 0

    timestamp_file = os.path.join(out_dir, "frame_timestamps.csv")
    with open(timestamp_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_ms", "filename"])

        while True:
            ret, orig_frame = cap.read()

            if not ret:
                break

            frame_idx += 1
            # SKIP initial frames if needed -----------------------------
            if frame_idx < 35911:
                print(f"Skipping frame {frame_idx}", end='\r')
                continue  # skip initial frames if needed
            # -----------------------------------------------------------
            buffer.append((frame_idx, orig_frame.copy(), cap.get(cv2.CAP_PROP_POS_MSEC)))
            print(f"Processing frame {frame_idx}", end='\r')

            # overlaying a ROI mask to focus on the important parts of frame
            roi_mask = cv2.imread('/home/tomass/tomass/Cam_record/ROIs/fisheye_cam_movement_roi.png', cv2.IMREAD_GRAYSCALE)
            if roi_mask is not None:
                frame = cv2.bitwise_and(orig_frame, orig_frame, mask=roi_mask)
                # cv2.imshow('Masked Frame', frame)
                # cv2.waitKey(1)

            # motion detection every frame_skip frames
            if frame_idx % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = bg.apply(gray)
                fg_mask = cv2.medianBlur(fg_mask, 5)
                fg_ratio = (fg_mask > 0).sum() / fg_mask.size

                if motion_active:
                    if fg_ratio >= min_fg_ratio_continue:
                        frames_below_threshold = 0  # reset grace counter
                    else:
                        frames_below_threshold += frame_skip
                        if frames_below_threshold > stop_grace_frames:
                            # motion ended
                            motion_active = False
                            buffer.clear()
                else:
                    if fg_ratio >= min_fg_ratio_start:
                        motion_active = True
                        frames_below_threshold = 0
                        # save buffered frames
                        for idx, buf_frame, ts in buffer:
                            filename = f"frame_{idx:08d}.jpg"
                            cv2.imwrite(os.path.join(out_dir, filename), buf_frame)
                            writer.writerow([idx, ts, filename])
                            saved += 1
                        buffer.clear()
            
            # save current frame if motion is active
            if motion_active:
                filename = f"frame_{frame_idx:08d}.jpg"
                cv2.imwrite(os.path.join(out_dir, filename), orig_frame)
                writer.writerow([frame_idx, cap.get(cv2.CAP_PROP_POS_MSEC), filename])
                saved += 1

    cap.release()
    print(f"Saved {saved} frames with timestamps in {timestamp_file}")


# Example usage:

extract_motion_sequences(
    video_path="/home/tomass/tomass/Cam_record/12.01.26/fisheye_record_1768215168.9960444.avi",
    out_dir="/home/tomass/tomass/Cam_record/12.01.26/fisheye_record_1768215168.9960444.avi_frames"
)