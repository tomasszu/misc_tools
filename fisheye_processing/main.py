"""
This script undistorts fisheye images or converts them to panoramic images.
It uses OpenCV for image processing and assumes a specific camera model with known intrinsics.
It can operate in two modes: 'undistort' and 'pano'.

Usage:  
    python main.py --mode <undistort|pano>

Input data:
    - A fisheye image.
Output:
    - Undistorted image displayed in a window.
    - Panoramic image displayed in a window if 'pano' mode is selected.

Default input folder:   /home/tomass/Pictures/IPcam1_screens/vlcsnap-2025-08-27-15h29m55s778.png
    - 

"""
# main.py

import numpy as np
import cv2
from pathlib import Path
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['undistort', 'pano', 'multiview'], default='undistort')
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
    fx = fy = image_width /2 # Assuming square pixels and centered principal point
    cx = cy = image_width / 2   # Center of the image
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K

def get_distortion_coefficients(args):
    if args.mode == 'undistort':
        D = np.array([[-0.05], [0.01], [0.0], [0.0]])  # example fisheye distortion coeffs
    elif args.mode == 'multiview':
        #Šitais dod kkadu normalu attelu kad ir kamera itka pagriezta uz divam tam pretejam puseem
        D = np.array([[-0.25], [0.07], [0.0], [0.0]])  # example fisheye distortion coeffs
    #D = np.array([[-0.33], [0.075], [0.0], [0.0]])  # example fisheye distortion coeffs
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

def get_rotation_matrix(yaw_deg=0, pitch_deg=0, roll_deg=0):
    """
    Returns a rotation matrix for given yaw/pitch/roll (in degrees).
    Rotation order: roll -> pitch -> yaw.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Rotation matrices
    R_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])

    # Combined
    R = R_yaw @ R_pitch @ R_roll
    return R.astype(np.float32)

def undistort_with_rotation(image, K, D, yaw=0, pitch=0, roll=0):
    """
    Undistorts fisheye image with a rotated rectification (simulated pinhole view).
    """
    h, w = image.shape[:2]
    R = get_rotation_matrix(yaw, pitch, roll)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K, (w, h), cv2.CV_16SC2
    )

    rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rectified

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

    
    # load a sample fisheye image
    image_path = Path("/home/tomass/Pictures/IPcam1_screens/vlcsnap-2025-08-27-15h29m55s778.png")    

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")

    if args.mode == 'undistort':
            
        image_height, image_width = image.shape[:2]
        # Get camera intrinsics and distortion coefficients
        K = get_camera_intrinsics(image_width, image_height)
        D = get_distortion_coefficients(args)
        # Undistort the image
        undistorted_image = undistort_image(image, K, D)

        #resize for display
        undistorted_image = cv2.resize(undistorted_image, (960, 960))

        cv2.imshow("Undistorted Image", undistorted_image)

    elif args.mode == 'pano':
        # Convert fisheye image to panorama
        undistorted_image = fisheye_to_panorama(image)

        cv2.imshow("Pano Image", undistorted_image)

    if args.mode == 'multiview':
        image_height, image_width = image.shape[:2]
        K = get_camera_intrinsics(image_width, image_height)
        D = get_distortion_coefficients(args)

        #Piemeeram nemam 3 skatus - forward, left, right

        views = {
            "forward": undistort_with_rotation(image, K, D, yaw=0, pitch=0),
            "-45": undistort_with_rotation(image, K, D, yaw=270, pitch=-45),
            "45": undistort_with_rotation(image, K, D, yaw=90, pitch=45)
        }

        for name, view in views.items():
            cv2.imshow(name, cv2.resize(view, (640, 640)))

    
    cv2.waitKey(0)  # Wait for a key press to close the window

    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)






    



    




