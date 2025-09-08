from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw

# -------------------------
# Camera distortion config
# -------------------------
@dataclass
class DistortionParams:
    k1: float = -0.6
    k2: float = -0.1
    k3: float = -0.05
    p1: float = 0.0
    p2: float = 0.0
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None


# -------------------------
# Intrinsics + distortion
# -------------------------
def _get_intrinsics(w: int, h: int, params: DistortionParams):
    fx = params.fx if params.fx is not None else 0.5 * w
    fy = params.fy if params.fy is not None else 0.5 * h
    cx = params.cx if params.cx is not None else 0.5 * w
    cy = params.cy if params.cy is not None else 0.5 * h
    return fx, fy, cx, cy

def _radial_tangential_distort(x: np.ndarray, y: np.ndarray, params: DistortionParams):
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    k1, k2, k3, p1, p2 = params.k1, params.k2, params.k3, params.p1, params.p2
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x_tangential = 2*p1*x*y + p2*(r2 + 2*x*x)
    y_tangential = p1*(r2 + 2*y*y) + 2*p2*x*y
    x_dist = x * radial + x_tangential
    y_dist = y * radial + y_tangential
    return x_dist, y_dist


# -------------------------
# Precompute distortion maps
# -------------------------
def precompute_maps(w, h, params: DistortionParams):
    fx, fy, cx, cy = _get_intrinsics(w, h, params)
    xs, ys = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(xs, ys)

    x = (xv - cx) / fx
    y = (yv - cy) / fy

    # inverse distortion iteration
    x_u, y_u = x.copy(), y.copy()
    for _ in range(3):
        x_d, y_d = _radial_tangential_distort(x_u, y_u, params)
        ex, ey = x_d - x, y_d - y
        x_u -= 0.5 * ex
        y_u -= 0.5 * ey

    mapx = (x_u * fx + cx).astype(np.float32)
    mapy = (y_u * fy + cy).astype(np.float32)
    return mapx, mapy


def distort_with_maps(img: np.ndarray, mapx, mapy) -> np.ndarray:
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# -------------------------
# Bbox distortion
# -------------------------
def map_bboxes_through_distortion(
    bboxes_xyxy: np.ndarray,
    img_size: Tuple[int, int],
    mapx, mapy
) -> np.ndarray:
    """Distort each bbox independently using precomputed maps."""
    w, h = img_size
    out = np.zeros_like(bboxes_xyxy, dtype=np.float32)

    for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy.astype(int)):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # distort this mask
        mask_distorted = cv2.remap(mask, mapx, mapy,
                                   interpolation=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT)

        ys, xs = np.nonzero(mask_distorted)
        if len(xs) > 0 and len(ys) > 0:
            out[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
        else:
            out[i] = [x1, y1, x2, y2]  # fallback if distortion empties bbox

    return out



# -------------------------
# Drawing utils
# -------------------------
def draw_bboxes(img: np.ndarray, bboxes: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    for x1, y1, x2, y2 in bboxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


# -------------------------
# Process one frame
# -------------------------
def ground_truth_for_frame(frame_idx, last_read_line, frame_nr, curr_line, frame, lines, mapx, mapy):

    distorted_img = distort_with_maps(frame, mapx, mapy)
    bboxes_frame = []

    if (last_read_line != 0 and frame_idx == frame_nr):
        x, y, w, h = map(float, curr_line[2:6])
        bboxes_frame.append([x, y, x+w, y+h])

    if (last_read_line == 0 or frame_idx == frame_nr):
        while frame_idx == frame_nr:
            line = lines[last_read_line]
            curr_line = line.split(",", maxsplit=6)
            frame_idx = int(curr_line[0])
            x, y, w, h = map(float, curr_line[2:6])
            if frame_idx == frame_nr:
                last_read_line += 1
                bboxes_frame.append([x, y, x+w, y+h])
            else:
                last_read_line += 1
                break

    if len(bboxes_frame) > 0:
        bboxes_frame = np.array(bboxes_frame, dtype=np.float32)
        distorted_bboxes = map_bboxes_through_distortion(
            bboxes_frame, distorted_img.shape[1::-1], mapx, mapy
        )
        distorted_img = draw_bboxes(distorted_img, bboxes_frame, color=(0,255,0))      # green = original
        distorted_img = draw_bboxes(distorted_img, distorted_bboxes, color=(0,0,255))  # red = distorted

    return frame_idx, last_read_line, curr_line, distorted_img


# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    vid_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi"
    gt_path  = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt"

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Mild fisheye (subtle barrel) — looks natural, small curvature
    # params = DistortionParams(
    #     k1 = -0.2,
    #     k2 = -0.03,
    #     k3 = -0.005,
    #     p1 = 0.0, p2 = 0.0,
    #     fx = 1100.0, fy = 1100.0,    # focal length in pixels (square pixels)
    #     cx = 960.0, cy = 540.0
    # )
    # Medium fisheye (noticeable curvature, good for simulating consumer action cams)
    # params = DistortionParams(
    #     k1 = -0.5,
    #     k2 = -0.08,
    #     k3 = -0.02,
    #     p1 = 0.0, p2 = 0.0,
    #     fx = 850.0, fy = 850.0,
    #     cx = 960.0, cy = 540.0
    # )
    # Strong fisheye (extreme barrel, strong edge compression)
    params = DistortionParams(
        k1 = -0.5,
        k2 = -0.08,
        k3 = -0.02,
        p1 = 0.0, p2 = 0.0,
        fx = 700.0, fy = 700.0,
        cx = 960.0, cy = 540.0
    )

    mapx, mapy = precompute_maps(w, h, params)

    with open(gt_path, "r") as f:
        lines = f.readlines()

    curr_line = None
    last_read_line = 0
    frame_idx = 1

    for frame_nr in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame_nr += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx, last_read_line, curr_line, labeled_frame = ground_truth_for_frame(
            frame_idx, last_read_line, frame_nr, curr_line, frame, lines, mapx, mapy
        )

        labeled_frame = cv2.resize(labeled_frame, (1280, 720))
        cv2.imshow('Distorted Frame with BBoxes', labeled_frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
