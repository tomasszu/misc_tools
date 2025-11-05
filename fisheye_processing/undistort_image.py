import cv2
import numpy as np

def fisheye_to_panorama(img, fov_deg=360, output_width=2048, output_height=512, cx=None, cy=None):
    h, w = img.shape[:2]
    if cx is None:
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

def undistort_image_fisheye(image, K, D):
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
    
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistorted_image


# -----------------------------<<<<< Mode >>>>>------------------------------------------
mode = 'undistort_fisheye'  # options: 'undistort_fisheye','undistort_plumbob', 'panorama'

# ---------------------------------------------------------------------------------------

# Load your fisheye image
img = cv2.imread("/home/tomass/Downloads/images/images_00/left_images/img_1760063941558817131.png")
h, w = img.shape[:2]

# fisheye intrinsics
fx = 1191.758338836872
fy = 1191.629011218609
cx = 2004.954445058495
cy = 1508.553099629891
K_fish = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]], dtype=np.float32)
D_fish = np.array([-0.002763474413349704, -0.0001978800573972798, 0.00153185092420312, -0.0003027647426547098], dtype=np.float32)

if mode == 'undistort_fisheye':
            
    # Undistort the image
    undistorted_image = undistort_image_fisheye(img, K_fish, D_fish)

    #resize for display
    undistorted_image = cv2.resize(undistorted_image, (int(4048/4), int(3040/4)))

    cv2.imshow("Undistorted Image", undistorted_image)

elif mode == 'panorama':
    # Create panoramic image
    pano_image = fisheye_to_panorama(img, fov_deg=360, output_width=2048, output_height=512, cx=cx, cy=cy)

    cv2.imshow("Panoramic Image", pano_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
