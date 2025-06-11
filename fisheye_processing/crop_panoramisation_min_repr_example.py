import numpy as np
import cv2

# Step 1: Dummy fisheye image
H, W = 720, 720
fisheye = np.zeros((H, W, 3), dtype=np.uint8)
cv2.circle(fisheye, (W//2, H//2), 360, (255, 255, 255), -1)
for r in range(360):
    cv2.circle(fisheye, (W//2, H//2), r, (r, r//2, 255 - r), 1)

# Step 2: Generate full pano remap
pano_H, pano_W = 480, 1028
theta = np.linspace(0, 2 * np.pi, pano_W)
radius = np.linspace(0, H//2, pano_H)
theta_grid, radius_grid = np.meshgrid(theta, radius)

map_x_full = (W/2 + radius_grid * np.cos(theta_grid)).astype(np.float32)
map_y_full = (H/2 + radius_grid * np.sin(theta_grid)).astype(np.float32)

# Step 3: Define crop in fisheye image
crop_x, crop_y, crop_w, crop_h = 50, 50, 200, 200
crop_img = fisheye[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

# Step 4: Mask which pano pixels fall within the crop
# Make a mask of valid coordinates
in_crop_mask = (
    (map_x_full >= crop_x) & (map_x_full < crop_x + crop_w) &
    (map_y_full >= crop_y) & (map_y_full < crop_y + crop_h)
)

# Create adjusted remap just for the crop
map_x_crop = np.zeros_like(map_x_full, dtype=np.float32)
map_y_crop = np.zeros_like(map_y_full, dtype=np.float32)

map_x_crop[in_crop_mask] = map_x_full[in_crop_mask] - crop_x
map_y_crop[in_crop_mask] = map_y_full[in_crop_mask] - crop_y

# For pixels outside the crop, map to -1 (invalid)
map_x_crop[~in_crop_mask] = -1
map_y_crop[~in_crop_mask] = -1

# Step 5: Remap using adjusted map and cropped image
warped_crop = cv2.remap(
    crop_img,
    map_x_crop,
    map_y_crop,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

# Provide for the cropped warped image to contain only the cropped area, not the full pano dimensions:

# Find bounding box of valid pano area
ys, xs = np.where(in_crop_mask)
min_y, max_y = ys.min(), ys.max()
min_x, max_x = xs.min(), xs.max()

# Crop the warped image to valid area
warped_crop_cropped = warped_crop[min_y:max_y+1, min_x:max_x+1]

# Step 6: Display
cv2.imshow("Fisheye Full", fisheye)
cv2.imshow("Panorama Full", cv2.remap(fisheye, map_x_full, map_y_full, cv2.INTER_LINEAR))
cv2.imshow("Cropped Fisheye", crop_img)
cv2.imshow("Warped Crop to Pano", warped_crop_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
