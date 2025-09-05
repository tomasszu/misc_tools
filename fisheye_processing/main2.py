import cv2
import numpy as np

def rotation_matrix(yaw, pitch, roll):
    """Create 3D rotation matrix from yaw, pitch, roll in degrees."""
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def fisheye_to_perspective(fisheye_img, K_fish, D_fish, fov=90, out_size=(640, 640),
                           yaw=0, pitch=0, roll=0):
    """
    Warp fisheye image into a perspective view with virtual camera rotation.
    """
    h_out, w_out = out_size

    # Virtual perspective intrinsics
    f = 0.5 * w_out / np.tan(np.deg2rad(fov) / 2)
    K_persp = np.array([[f, 0, w_out/2],
                        [0, f, h_out/2],
                        [0, 0, 1]], dtype=np.float32)

    # Pixel grid in perspective cam
    i, j = np.meshgrid(np.arange(w_out), np.arange(h_out))
    homog = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)
    rays = (np.linalg.inv(K_persp) @ homog.reshape(-1,3).T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # Apply rotation (yaw, pitch, roll)
    R = rotation_matrix(yaw, pitch, roll)
    rays = (R @ rays.T).T

    # Project into fisheye
    rays = rays.reshape(-1,1,3)
    pts, _ = cv2.fisheye.projectPoints(rays, rvec=np.zeros(3), tvec=np.zeros(3), K=K_fish, D=D_fish)

    # Build mapping
    map_x = pts[:,0,0].reshape(h_out, w_out).astype(np.float32)
    map_y = pts[:,0,1].reshape(h_out, w_out).astype(np.float32)

    persp_img = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return persp_img


if __name__ == "__main__":
    img = cv2.imread("/home/tomass/Pictures/IPcam1_screens/vlcsnap-2025-08-27-15h29m55s778.png")
    h, w = img.shape[:2]

    # Rough intrinsics (replace with real calibration if available)
    fx = fy = w/2
    cx, cy = w/2, h/2
    K_fish = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float32)
    D_fish = np.array([-0.25, 0.07, 0.0, 0.0], dtype=np.float32)

    # Example: rotate camera to look at horizon (yaw=90, pitch=45)
    persp = fisheye_to_perspective(img, K_fish, D_fish,
                                   fov=80, out_size=(640, 640),
                                   yaw=0, pitch=45, roll=30)
    persp2 = fisheye_to_perspective(img, K_fish, D_fish,
                                   fov=80, out_size=(640, 640),
                                   yaw=0, pitch=45, roll=-30)
    persp3 = fisheye_to_perspective(img, K_fish, D_fish,
                                   fov=80, out_size=(640, 640),
                                   yaw=0, pitch=15, roll=0)
    
    img = cv2.resize(img, (640, 640))
    cv2.imshow("fisheye", img)

    cv2.imshow("perspective right", persp)
    cv2.imshow("perspective left", persp2)
    cv2.imshow("perspective centre", persp3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
