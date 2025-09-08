from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2

# Try to import OpenCV for fast remapping. If unavailable, we'll fallback.
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


@dataclass
class DistortionParams:
    # Radial polynomial distortion (OpenCV-style, barrel if k1 < 0)
    k1: float = -0.6
    k2: float = -0.1
    k3: float = -0.05
    p1: float = 0.0  # tangential
    p2: float = 0.0  # tangential
    # Focal lengths and principal point in pixel units
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

def _get_intrinsics(w: int, h: int, params: DistortionParams):
    fx = params.fx if params.fx is not None else 0.5 * w
    fy = params.fy if params.fy is not None else 0.5 * h
    cx = params.cx if params.cx is not None else 0.5 * w
    cy = params.cy if params.cy is not None else 0.5 * h
    return fx, fy, cx, cy

def _radial_tangential_distort(x: np.ndarray, y: np.ndarray, params: DistortionParams):
    # Apply OpenCV-style distortion in normalized coordinates.
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

def distort_image(img: Image.Image, params: DistortionParams) -> Image.Image:
    """Add barrel/fisheye-like distortion using polynomial model. Returns a new PIL Image."""
    w, h = img.size
    fx, fy, cx, cy = _get_intrinsics(w, h, params)

    # Build a pixel grid for the *output* and map back to input.
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)

    # Normalize to camera coordinates
    x = (xv - cx) / fx
    y = (yv - cy) / fy

    # To distort an image, we need to find for each OUTPUT pixel, where to sample in the INPUT.
    # We'll invert the distortion numerically. For small distortions, one Newton step works ok.
    # Start with x_u = x, y_u = y (undistorted guess). Iterate to find (x_u, y_u) s.t. distort(x_u, y_u) = (x, y).
    x_u = x.copy()
    y_u = y.copy()
    for _ in range(3):  # 2-3 iterations are usually enough for this mild model
        x_d, y_d = _radial_tangential_distort(x_u, y_u, params)
        # Compute error and do a simple gradient-free update
        ex = x_d - x
        ey = y_d - y
        # Small step back; scale factor chosen empirically for stability
        x_u -= 0.5 * ex
        y_u -= 0.5 * ey

    # Map back to pixel coordinates in the input image
    u = (x_u * fx + cx).astype(np.float32)
    v = (y_u * fy + cy).astype(np.float32)

    if HAS_CV2:
        src = np.array(img)
        mapx = u
        mapy = v
        remapped = cv2.remap(src, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return Image.fromarray(remapped)
    else:
        # Slow fallback using PIL's sampling
        arr = np.array(img)
        # Clip sampling coords
        u_clipped = np.clip(u, 0, w-1)
        v_clipped = np.clip(v, 0, h-1)
        # Bilinear sampling (manual)
        x0 = np.floor(u_clipped).astype(int)
        x1 = np.clip(x0 + 1, 0, w-1)
        y0 = np.floor(v_clipped).astype(int)
        y1 = np.clip(y0 + 1, 0, h-1)
        wa = (x1 - u_clipped) * (y1 - v_clipped)
        wb = (u_clipped - x0) * (y1 - v_clipped)
        wc = (x1 - u_clipped) * (v_clipped - y0)
        wd = (u_clipped - x0) * (v_clipped - y0)

        out = (arr[y0, x0] * wa[..., None] +
               arr[y0, x1] * wb[..., None] +
               arr[y1, x0] * wc[..., None] +
               arr[y1, x1] * wd[..., None]).astype(np.uint8)
        return Image.fromarray(out)


def map_bboxes_through_distortion(
    bboxes_xyxy: np.ndarray,
    img_size: Tuple[int,int],
    params: DistortionParams
) -> np.ndarray:
    """
    Distort bounding boxes by rasterizing them into a binary mask,
    applying the same distortion to the mask as the image,
    and extracting the bounding rectangle afterwards.
    This guarantees consistency with the image warp.
    """
    w, h = img_size
    out = np.zeros_like(bboxes_xyxy, dtype=np.float32)

    for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
        # --- create binary mask for this bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

        # --- distort the mask with the same function used on the image
        mask_pil = Image.fromarray(mask)
        mask_distorted = distort_image(mask_pil, params)
        mask_distorted = np.array(mask_distorted)

        # --- find nonzero pixels (distorted bbox region)
        ys, xs = np.nonzero(mask_distorted > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
        else:
            # if mask vanished (e.g., distorted out of FOV), keep original
            x_min, y_min, x_max, y_max = x1, y1, x2, y2

        out[i] = [x_min, y_min, x_max, y_max]

    return out

def draw_bboxes(img: Image.Image, bboxes: np.ndarray, color=(255,0,0)) -> Image.Image:
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for x1,y1,x2,y2 in bboxes:
        dr.rectangle((x1,y1,x2,y2), outline=color, width=3)
    return im

def ground_truth_for_frame(frame_idx, last_read_line, frame_nr, curr_line, frame, lines, params):

    labeled_frame = frame.copy()
    labeled_frame = Image.fromarray(cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB))
    distorted_img = distort_image(labeled_frame, params)
    if(last_read_line != 0 and frame_idx == frame_nr):
        x, y, w, h = map(float, curr_line[2:6])
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        bboxes = np.array([[x1,y1,x2,y2]], dtype=np.float32)
        distorted_img = draw_bboxes(distorted_img, bboxes, color=(0,255,0))

        distorted_bboxes = map_bboxes_through_distortion(bboxes, distorted_img.size, params)
        distorted_img = draw_bboxes(distorted_img, distorted_bboxes, color=(255,0,0))

    if(last_read_line == 0 or frame_idx == frame_nr):
        while frame_idx == frame_nr:
            line = lines[last_read_line]
            curr_line = line.split(",", maxsplit=6)
            frame_idx = int(curr_line[0])
            x, y, w, h = map(float, curr_line[2:6])
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            bboxes = np.array([[x1,y1,x2,y2]], dtype=np.float32)
            if frame_idx == frame_nr:
                last_read_line = last_read_line+1
                distorted_img = draw_bboxes(distorted_img, bboxes, color=(0,255,0))

                distorted_bboxes = map_bboxes_through_distortion(bboxes, distorted_img.size, params)
                distorted_img = draw_bboxes(distorted_img, distorted_bboxes, color=(255,0,0))
            else:
                last_read_line = last_read_line+1
                break
    

    return frame_idx, last_read_line, curr_line, distorted_img
    

# Example usage with video and bboxes form ground truth txt file

if __name__ == "__main__":
    vid_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi"
    gt_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt"

    
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fx, fy, cx, cy = _get_intrinsics(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), DistortionParams())
    params = DistortionParams(fx=fx, fy=fy, cx=cx, cy=cy)

    gt_file = open(gt_path, 'r')
    if gt_file is None:
        print("Error: Could not open ground truth file.")
        exit()

    lines = gt_file.readlines()
    gt_file.close()

    curr_line = None
    last_read_line = 0
    frame_idx = 1

    for frame_nr in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame_nr += 1

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_idx, last_read_line, curr_line, labeled_frame = ground_truth_for_frame(frame_idx, last_read_line, frame_nr, curr_line, frame, lines, params)
        labeled_frame = cv2.cvtColor(np.array(labeled_frame), cv2.COLOR_RGB2BGR)
        labeled_frame = cv2.resize(labeled_frame, (1280, 720))
        cv2.imshow('Distorted Frame with BBoxes', labeled_frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
