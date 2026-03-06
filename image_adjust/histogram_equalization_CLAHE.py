import cv2
import numpy as np
from pathlib import Path

input_dir = "image_adjust/test/img_946687013500409520.png"
output_dir = "image_adjust/test/img_946687013500409520_mod_clahe.png"

img = cv2.imread(input_dir)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])

img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


cv2.imwrite(output_dir, img_clahe)