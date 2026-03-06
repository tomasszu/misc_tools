# square-root gamma brightening
# + increase local contrast and edge strength

import cv2
import numpy as np
from pathlib import Path

input_dir = "image_adjust/test/img_946687013500409520.png"
output_dir = "image_adjust/test/img_946687013500409520_mod5.png"

img = cv2.imread(input_dir)

# --- 1. square root gamma ---
img_norm = img.astype(np.float32) / 255.0
img_gamma = np.sqrt(img_norm)

# --- 2. contrast stretch ---
p2, p98 = np.percentile(img_gamma, (5, 100))
img_contrast = np.clip((img_gamma - p2) / (p98 - p2), 0, 1)

# --- 3. convert back to uint8 ---
img_uint8 = (img_contrast * 255).astype(np.uint8)

# --- 4. unsharp mask (edge enhancement) ---
ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2YCrCb)
y = ycrcb[:,:,0]

blur = cv2.GaussianBlur(y, (0,0), 1.0)
y_sharp = cv2.addWeighted(y, 1.5, blur, -0.5, 0)

ycrcb[:,:,0] = y_sharp
sharp = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

sharp = cv2.detailEnhance(sharp, sigma_s=10, sigma_r=0.15)

cv2.imwrite(output_dir, sharp)