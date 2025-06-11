import cv2

input_path = 'cam1_cuts.mp4'
output_path = 'cam1_cuts_4fps.mp4'
target_fps = 4

# Open the input video
cap = cv2.VideoCapture(input_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

# Calculate the frame skipping interval
frame_interval = int(original_fps / target_fps)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        out.write(frame)

    frame_idx += 1

cap.release()
out.release()

print(f"Video saved as {output_path} with {target_fps} FPS.")