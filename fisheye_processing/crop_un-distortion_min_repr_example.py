import numpy as np
import cv2

# Step 1: Dummy fisheye image
H, W = 720, 720
fisheye = np.zeros((H, W, 3), dtype=np.uint8)
cv2.circle(fisheye, (W//2, H//2), 360, (255, 255, 255), -1)
for r in range(360):
    cv2.circle(fisheye, (W//2, H//2), r, (r, r//2, 255 - r), 1)

# Step 2: Camera intrinsics & distortion coefficients
K = np.array([[360, 0, W/2],
              [0, 360, H/2],
              [0,   0,   1]], dtype=np.float64)
D = np.array([-0.1, 0.01, 0.0, 0.0], dtype=np.float64)

# Step 3: Undistortion maps for the **full** image
map1_full, map2_full = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, (W, H), cv2.CV_32FC1
)

# Step 4: Crop region
crop_x, crop_y, crop_w, crop_h = 100, 100, 200, 200
crop_img = fisheye[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

# Step 5: Extract remap maps for the crop region (ABSOLUTE coordinates!)
map1_crop_abs = map1_full[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
map2_crop_abs = map2_full[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

# Step 6: Shift remap maps to crop-relative input image (crop_img)
# Since we're remapping crop_img (smaller image), subtract crop offsets
map1_crop_rel = map1_crop_abs - crop_x
map2_crop_rel = map2_crop_abs - crop_y

# Step 7: Now remap the crop_img using the adjusted crop maps
undistorted_crop = cv2.remap(
    crop_img,
    map1_crop_rel,
    map2_crop_rel,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT
)

# Step 8: Visualise
cv2.imshow("Original Full Fisheye", fisheye)
cv2.imshow("Cropped Fisheye", crop_img)
cv2.imshow("Undistorted Crop (local maps)", undistorted_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
