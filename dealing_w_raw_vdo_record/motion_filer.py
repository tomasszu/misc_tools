import cv2
import os
import csv
from collections import deque

def extract_motion_sequences(
    video_path,
    out_dir,
    min_fg_ratio_start=0.05,    # threshold to detect new motion
    min_fg_ratio_continue=0.01,# threshold to continue motion sequence
    frame_skip=5,
    backtrack_frames=10,
    stop_grace_frames=5          # allow N frames below threshold before stopping
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Cannot open video"

    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

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
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            buffer.append((frame_idx, frame.copy(), cap.get(cv2.CAP_PROP_POS_MSEC)))

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
                cv2.imwrite(os.path.join(out_dir, filename), frame)
                writer.writerow([frame_idx, cap.get(cv2.CAP_PROP_POS_MSEC), filename])
                saved += 1

    cap.release()
    print(f"Saved {saved} frames with timestamps in {timestamp_file}")


# Example usage:

extract_motion_sequences(
    video_path="/home/tomass/tomass/Cam_record/04.09.25_2/fisheye_record_1756987992.9123657.avi",
    out_dir="fisheye_record_1756987992.9123657_frames"
)