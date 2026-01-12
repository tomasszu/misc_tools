import cv2
import numpy as np

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
img = cv2.imread("/home/tomass/Downloads/conceptf_images/images_00/right_images/img_1760063675559624435.png")
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


cv2.waitKey(0)
cv2.destroyAllWindows()
