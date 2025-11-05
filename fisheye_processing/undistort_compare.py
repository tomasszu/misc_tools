import cv2
import numpy as np

# --- Paste your intrinsics here ---
K = np.array([[1191.758338836872, 0.0, 2004.954445058495],
              [0.0, 1191.629011218609, 1508.553099629891],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# D = np.array([-0.002763474413349704,
#               -0.0001978800573972798,
#                0.00153185092420312,
#               -0.0003027647426547098], dtype=np.float64)  # ambiguous: either plumb_bob (k1,k2,p1,p2) or fisheye (k1..k4)

D = np.array([0.25, -0.05, 0.01, -0.0124], dtype=np.float64)  # ambiguous: either plumb_bob (k1,k2,p1,p2) or fisheye (k1..k4)

# load image (should be exactly 4048x3040)
img = cv2.imread("/home/tomass/Downloads/images/images_00/left_images/img_1760063941558817131.png", cv2.IMREAD_COLOR)
h, w = img.shape[:2]
print("image shape:", (w, h))

# --- 1) Standard plumb_bob undistort ---
newK_std, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w,h))
map1_std, map2_std = cv2.initUndistortRectifyMap(K, D, None, newK_std, (w, h), cv2.CV_16SC2)
und_std = cv2.remap(img, map1_std, map2_std, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# --- 2) Fisheye undistort (treat D as fisheye params) ---
# cv2.fisheye expects D with shape (4,1) or (1,4)
D_fish = D.reshape(4,1).copy()  # interpret the 4 numbers as fisheye params
# build new K for fisheye (you can use same K or adjust)
newK_fish = K.copy()
# optional: balance/stretch param; here use identity rectification
map1_fish, map2_fish = cv2.fisheye.initUndistortRectifyMap(K, D_fish, np.eye(3), newK_fish, (w, h), cv2.CV_16SC2)
und_fish = cv2.remap(img, map1_fish, map2_fish, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# --- 3) Create a side-by-side visualization and save ---
vis = np.hstack([img, und_std, und_fish])
cv2.imwrite("compare_orig_std_fish.jpg", vis)
print("Saved compare_orig_std_fish.jpg (orig | plumb_bob | fisheye)")

# --- 4) Quantify pixel shifts at a grid of points ---
def sample_points_grid(w, h, nx=20, ny=15):
    xs = np.linspace(0, w-1, nx, dtype=np.float32)
    ys = np.linspace(0, h-1, ny, dtype=np.float32)
    pts = np.array([[x,y] for y in ys for x in xs], dtype=np.float32)
    return pts.reshape(-1,1,2)

pts = sample_points_grid(w,h, nx=30, ny=22)  # dense grid
# undistortPoints returns normalized coordinates unless P provided; pass P=K to get back pixel coords
und_pts_std = cv2.undistortPoints(pts, K, D, P=newK_std)  # shape Nx1x2
und_pts_fish = cv2.fisheye.undistortPoints(pts, K, D_fish, P=newK_fish)

orig = pts.reshape(-1,2)
std_p = und_pts_std.reshape(-1,2)
fish_p = und_pts_fish.reshape(-1,2)

shift_std = np.linalg.norm(std_p - orig, axis=1)
shift_fish = np.linalg.norm(fish_p - orig, axis=1)

print("std model: mean shift {:.3f} px, max shift {:.3f} px".format(shift_std.mean(), shift_std.max()))
print("fish model: mean shift {:.3f} px, max shift {:.3f} px".format(shift_fish.mean(), shift_fish.max()))

# Save shift maps as images for inspection (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.title("std shift (px)"); plt.hist(shift_std, bins=50)
plt.subplot(1,2,2); plt.title("fish shift (px)"); plt.hist(shift_fish, bins=50)
plt.tight_layout()
plt.savefig("shift_histograms.png")
print("Saved shift_histograms.png")
