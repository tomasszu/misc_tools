"""
This script undistorts fisheye images or converts them to panoramic images.
It uses OpenCV for image processing and assumes a specific camera model with known intrinsics.
It can operate in two modes: 'undistort' and 'pano'.

Usage:  
    python main.py --mode <undistort|pano>

Input data:
    - Fisheye images located in the specified folder.
Output:
    - Undistorted images displayed in a window.
    - Panoramic images displayed in a window if 'pano' mode is selected.

Default input folder:
    - /home/tomass/tomass/data/VIP_CUP_2020_fisheye_dataset/fisheye_video_1

"""
# main.py

import numpy as np
import cv2
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['undistort', 'pano'], default='undistort')
    return parser.parse_args()


def get_camera_intrinsics(image_width, image_height):
    """
    Returns the camera intrinsics matrix K for a given image size.
    
    Parameters:
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    
    Returns:
    - K: Camera intrinsics matrix as a numpy array.
    """
    fx = fy = image_width / 2  # Assuming square pixels and centered principal point
    cx = cy = image_width / 2   # Center of the image
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K

def get_distortion_coefficients():
    D = np.array([[-0.05], [0.01], [0.0], [0.0]])  # example fisheye distortion coeffs
    return D

def undistort_image(image, K, D):
    """
    Undistorts an image using the camera intrinsics and distortion coefficients.
    
    Parameters:
    - image: Input image to be undistorted.
    - K: Camera intrinsics matrix.
    - D: Distortion coefficients.
    
    Returns:
    - undistorted_image: The undistorted image.
    """
    h, w = image.shape[:2]

    # Undistort using fisheye model
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  

    return undistorted_image

def fisheye_to_panorama(img, fov_deg=360, output_width=2048, output_height=512):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    max_radius = min(cx, cy)

    # Convert FOV to radians
    fov_rad = np.deg2rad(fov_deg)

    # Prepare destination grid
    theta = np.linspace(0, fov_rad, output_width)
    r = np.linspace(1, 0, output_height) * max_radius

    theta, r = np.meshgrid(theta, r)

    # Polar to Cartesian
    map_x = (r * np.cos(theta) + cx).astype(np.float32)
    map_y = (r * np.sin(theta) + cy).astype(np.float32)

    # Remap to get panoramic image
    pano = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return pano

def main(args):
    # Open folder and read images

    image_folder = Path("/home/tomass/tomass/data/VIP_CUP_2020_fisheye_dataset/fisheye_video_1")

    img_paths = sorted(image_folder.glob("*.[jp][pn]g"))  # matches .jpg, .jpeg, .png

    # Loop through all image files (common formats)
    for image_path in img_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        if args.mode == 'undistort':
                
            image_height, image_width = image.shape[:2]
            # Get camera intrinsics and distortion coefficients
            K = get_camera_intrinsics(image_width, image_height)
            D = get_distortion_coefficients()
            # Undistort the image
            undistorted_image = undistort_image(image, K, D)

        elif args.mode == 'pano':
            # Convert fisheye image to panorama
            undistorted_image = fisheye_to_panorama(image)
        
        # Show the undistorted image
        cv2.imshow("Undistorted Image", undistorted_image)
        cv2.waitKey(0)  # Wait for a key press to close the window

    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)






    



    




