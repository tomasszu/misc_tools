""" 
Fisheye Perspective Sectioning
---------------------------------------------
Given a folder with fisheye images, it generates perspective views by reprojecting the fisheye images into three different orientations (left, center, right) and saves them into separate folders.

"""



import cv2
import numpy as np
import os
from tqdm import tqdm


# ----------------- GEOMETRY ----------------- #

def rotation_matrix(yaw, pitch, roll):
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


def fisheye_to_perspective(
    fisheye_img,
    K_fish,
    D_fish,
    fov=80,
    out_size=(640, 640),
    yaw=0,
    pitch=0,
    roll=0
):
    h_out, w_out = out_size

    f = 0.5 * w_out / np.tan(np.deg2rad(fov) / 2)
    K_p = np.array([[f, 0, w_out / 2],
                    [0, f, h_out / 2],
                    [0, 0, 1]], dtype=np.float32)

    i, j = np.meshgrid(np.arange(w_out), np.arange(h_out))
    homog = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)

    rays = (np.linalg.inv(K_p) @ homog.reshape(-1, 3).T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    R = rotation_matrix(yaw, pitch, roll)
    rays = (R @ rays.T).T

    rays = rays.reshape(-1, 1, 3)
    pts, _ = cv2.fisheye.projectPoints(
        rays,
        rvec=np.zeros(3),
        tvec=np.zeros(3),
        K=K_fish,
        D=D_fish
    )

    map_x = pts[:, 0, 0].reshape(h_out, w_out).astype(np.float32)
    map_y = pts[:, 0, 1].reshape(h_out, w_out).astype(np.float32)

    return cv2.remap(
        fisheye_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )


# ----------------- PIPELINE ----------------- #

def process_fisheye_folder(
    input_dir,
    output_dir,
    K_fish,
    D_fish,
    out_size=(640, 640),
    fov=80
):
    views = {
        "left":   dict(yaw=0, pitch=45, roll=-30),
        "center": dict(yaw=0, pitch=15, roll=0),
        "right":  dict(yaw=0, pitch=45, roll=30),
    }

    for v in views:
        os.makedirs(os.path.join(output_dir, v), exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jpg"))

    for fname in tqdm(files, desc="Fisheye reprojection"):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue

        print(f"Processing {fname}", end='\r')

        for view, params in views.items():
            persp = fisheye_to_perspective(
                img,
                K_fish,
                D_fish,
                fov=fov,
                out_size=out_size,
                **params
            )

            cv2.imwrite(
                os.path.join(output_dir, view, fname),
                persp
            )


# ----------------- ENTRY POINT ----------------- #

if __name__ == "__main__":

    input_frames = "/home/tomass/tomass/Cam_record/04.09.25_3/fisheye_record_1756991341.9092808_frames_refined"
    output_root = "/home/tomass/tomass/Cam_record/04.09.25_3/perspective_views_fisheye_record_1756991341.9092808_frames_refined"

    # --- Intrinsics (replace with real calibration if you have it) ---
    sample = cv2.imread(os.path.join(input_frames, os.listdir(input_frames)[0]))
    h, w = sample.shape[:2]

    fx = fy = w / 2
    cx, cy = w / 2, h / 2

    K_fish = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float32)

    D_fish = np.array([-0.25, 0.07, 0.0, 0.0], dtype=np.float32)

    process_fisheye_folder(
        input_dir=input_frames,
        output_dir=output_root,
        K_fish=K_fish,
        D_fish=D_fish,
        out_size=(640, 640),
        fov=80
    )
    print("Processing completed.")