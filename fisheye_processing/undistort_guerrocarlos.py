import cv2
import numpy as np
import math

PI = math.pi

def fisheye_to_rectilinear(image_path, output_path, fov_deg=120):
    """
    Convert a fisheye image to a rectilinear projection (vectorized).
    
    Based on guerrerocarlos' StackOverflow algorithm (CC-BY-SA 4.0).
    Vectorized by ChatGPT (2025-11-05).
    
    Parameters:
        image_path (str): path to input fisheye image
        output_path (str): path to save output image
        fov_deg (float): horizontal field of view in degrees (default 180)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read input image: {image_path}")

    h, w = img.shape[:2]

    # === 1. Prepare coordinate grids ===
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # Normalize pixel coordinates to [-0.5, 0.5]
    theta = math.pi * (xv / w - 0.5)
    phi = math.pi * (yv / h - 0.5)

    # === 2. Convert to 3D spherical coordinates ===
    psph_x = np.cos(phi) * np.sin(theta)
    psph_y = np.cos(phi) * np.cos(theta)
    psph_z = np.sin(phi) * np.cos(theta)

    # === 3. Back-project to fisheye ===
    theta2 = np.arctan2(psph_z, psph_x)
    phi2 = np.arctan2(np.sqrt(psph_x ** 2 + psph_z ** 2), psph_y)

    fov = math.radians(fov_deg)
    r = w * phi2 / fov
    r2 = h * phi2 / fov

    # Fisheye coordinates
    map_x = (0.5 * w + r * np.cos(theta2)).astype(np.float32)
    map_y = (0.5 * h + r2 * np.sin(theta2)).astype(np.float32)

    # === 4. Remap ===
    rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite(output_path, rectified)
    print(f"Saved rectilinear image to {output_path}")


# Example usage:
fisheye_to_rectilinear("/home/tomass/Downloads/conceptf_images/images_00/right_images/img_1760063675559624435.png", "fisheye_rosbag_right1.png")

