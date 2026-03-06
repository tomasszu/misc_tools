# square-root gamma brightening
# + increase local contrast and edge strength

import cv2
import numpy as np
from pathlib import Path

input_dir = "image_adjust/test/img_946687013500409520.png"
output_dir = "image_adjust/test/img_946687013500409520_mod.png"

img = cv2.imread(input_dir)

# --- 1. square root gamma ---
img_norm = img.astype(np.float32) / 255.0
img_gamma = np.sqrt(img_norm)

# --- 2. contrast stretch ---
p2, p98 = np.percentile(img_gamma, (5, 100))
img_contrast = np.clip((img_gamma - p2) / (p98 - p2), 0, 1)

# --- 3. convert back to uint8 ---
img_uint8 = (img_contrast * 255).astype(np.uint8)

# # --- 4. unsharp mask (edge enhancement) ---
# blur = cv2.GaussianBlur(img_uint8, (0,0), sigmaX=2.0)
# sharp = cv2.addWeighted(img_uint8, 2, blur, -1, 0)

cv2.imwrite(output_dir, img_uint8)