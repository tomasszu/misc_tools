import cv2

video_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c001/vdo.avi"
frame_number = 100   # <-- pick the frame index you want
output_path = "still_from_vid.jpg"

cap = cv2.VideoCapture(video_path)

# Jump to frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()
if ret:
    cv2.imwrite(output_path, frame)
    print(f"Saved frame {frame_number} to {output_path}")
else:
    print(f"Could not read frame {frame_number}")

cap.release()
