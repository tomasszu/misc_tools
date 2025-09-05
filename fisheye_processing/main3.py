import cv2
import numpy as np

# Load your fisheye image
img = cv2.imread("/home/tomass/Pictures/IPcam1_screens/vlcsnap-2025-08-27-15h29m55s778.png")
h, w = img.shape[:2]

# Rough fisheye intrinsics
fx = fy = w/2
cx, cy = w/2, h/2
K_fish = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]], dtype=np.float32)
D_fish = np.array([-0.25, 0.07, 0.0, 0.0], dtype=np.float32)

# Big “perspective” output same size as fisheye
out_h, out_w = h, w

# Use identity rotation for testing (no yaw/pitch/roll)
R = np.eye(3)

# Try shifting the principal point to see effect
# E.g., move center slightly up/down/left/right
shift_x = 0  # pixels
shift_y = 900
f = 0.5 * out_w / np.tan(np.deg2rad(90)/2)  # FOV 90 deg
K_persp = np.array([[f, 0, out_w/2 + shift_x],
                    [0, f, out_h/2 + shift_y],
                    [0, 0, 1]], dtype=np.float32)

# Generate pixel grid
i, j = np.meshgrid(np.arange(out_w), np.arange(out_h))
homog = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)
rays = (np.linalg.inv(K_persp) @ homog.reshape(-1,3).T).T
rays /= np.linalg.norm(rays, axis=1, keepdims=True)

# Apply rotation (identity here)
rays = (R @ rays.T).T

# Project into fisheye
rays = rays.reshape(-1,1,3)
pts, _ = cv2.fisheye.projectPoints(rays, rvec=np.zeros(3), tvec=np.zeros(3),
                                   K=K_fish, D=D_fish)

map_x = pts[:,0,0].reshape(out_h, out_w).astype(np.float32)
map_y = pts[:,0,1].reshape(out_h, out_w).astype(np.float32)

# Remap
out_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

out_img = cv2.resize(out_img, (640, 640))
cv2.imshow("shifted perspective", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
