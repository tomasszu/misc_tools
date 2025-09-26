import numpy as np
from PIL import Image
import os
import glob

# Path where your npy files are stored
input_dir = "/home/tomass/Downloads/rosbag2_2025_09_09-19_49_04_converted_bag/"
output_dir = "/home/tomass/Downloads/rosbag2_2025_09_09-19_49_04_converted_bag/png/"

os.makedirs(output_dir, exist_ok=True)

# Loop through all npy files
for file in glob.glob(os.path.join(input_dir, "*.npy")):
    # Load numpy array
    arr = np.load(file)

    # If image is float, normalize to [0,255]
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-8)).astype(np.uint8)

    # Convert to image
    img = Image.fromarray(arr)

    # Save as png
    base = os.path.splitext(os.path.basename(file))[0]
    out_path = os.path.join(output_dir, base + ".png")
    img.save(out_path)

    print(f"Saved {out_path}")
