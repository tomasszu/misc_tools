import os
import cv2
import numpy as np
from glob import glob
from random import randint
from tqdm import tqdm

# Configuration
input_dir = "/home/tomass/tomass/data/EDI_Cam_testData/summer_vid/vid1/cam1"  # folder with clean vehicle crops
output_dir = "fisheye_simulation/reid_images_fisheye_simulated"
os.makedirs(output_dir, exist_ok=True)


fisheye_img_size = (2048, 2048)
# Simulated fisheye camera intrinsics
fx = fy = fisheye_img_size[0] // 2
cx, cy = fisheye_img_size[0] // 2, fisheye_img_size[1] // 2  # center of image
K = np.array([[fx,  0, cx],
              [0,  fy, cy],
              [0,   0,  1]], dtype=np.float64)
# Realistic fisheye distortion coefficients for ~180° FOV fisheye lens
D = np.array([-0.3, 0.1, 0.0, 0.0], dtype=np.float64)

def apply_fisheye_distortion_to_crop(crop_img):
    h, w = crop_img.shape[:2]

    # Random top-left position in fisheye canvas where crop will be placed
    max_x = fisheye_img_size[0] - w
    max_y = fisheye_img_size[1] - h

    # Avoid placing the crop too close to borders (padding ~400 px)
    margin = 500
    safe_max_x = fisheye_img_size[0] - w - margin
    safe_max_y = fisheye_img_size[1] - h - margin

    x_offset = np.random.randint(margin, safe_max_x + 1)
    y_offset = np.random.randint(margin, safe_max_y + 1)

    # Embed crop into black fisheye canvas
    canvas = np.zeros((fisheye_img_size[1], fisheye_img_size[0], 3), dtype=np.uint8)
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = crop_img

    # Create new camera matrix (optional, but safe)
    new_K = K.copy()

    # Undistort + distort maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, fisheye_img_size, cv2.CV_16SC2
    )
    distorted_canvas = cv2.remap(canvas, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Crop the same region after distortion
    distorted_crop = distorted_canvas[y_offset:y_offset + h, x_offset:x_offset + w]

    # cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + w, y_offset + h), (0, 255, 0), 2)
    # cv2.imshow("canvas", canvas)
    # cv2.imshow("distorted", distorted_canvas)
    # cv2.waitKey(0)

    return distorted_crop

# Process images
image_paths = glob(os.path.join(input_dir, "*"))
for img_path in tqdm(image_paths, desc="Simulating fisheye distortion"):
    img = cv2.imread(img_path)
    if img is None:
        continue

    distorted = apply_fisheye_distortion_to_crop(img)



    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, filename), distorted)
    print(f"Processed {filename} with fisheye distortion.")