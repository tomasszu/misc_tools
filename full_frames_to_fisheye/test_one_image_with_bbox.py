from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

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
    k2: float = 0.1
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
    fy = params.fy if params.fy is not None else 0.5 * w  # square pixels assumption
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

def _apply_to_points(points_xy: np.ndarray, img_size: Tuple[int,int], params: DistortionParams) -> np.ndarray:
    """Apply forward distortion to 2D points (N,2) given image size, returns distorted points in pixel coords."""
    w, h = img_size
    fx, fy, cx, cy = _get_intrinsics(w, h, params)
    x = (points_xy[:,0] - cx) / fx
    y = (points_xy[:,1] - cy) / fy
    xd, yd = _radial_tangential_distort(x, y, params)
    u = xd * fx + cx
    v = yd * fy + cy
    return np.stack([u, v], axis=1)

def map_bboxes_through_distortion(
    bboxes_xyxy: np.ndarray,  # (N,4) in pixel coords [x1,y1,x2,y2]
    img_size: Tuple[int,int],
    params: DistortionParams
) -> np.ndarray:
    """Approximate bbox mapping by distorting the 4 corners and taking min/max."""
    N = bboxes_xyxy.shape[0]
    w, h = img_size
    out = np.zeros_like(bboxes_xyxy, dtype=np.float32)
    for i in range(N):
        x1,y1,x2,y2 = bboxes_xyxy[i]
        corners = np.array([
            [x1,y1], [x2,y1], [x1,y2], [x2,y2]
        ], dtype=np.float32)
        dc = _apply_to_points(corners, img_size, params)
        xmins = np.clip(dc[:,0].min(), 0, w-1)
        xmaxs = np.clip(dc[:,0].max(), 0, w-1)
        ymins = np.clip(dc[:,1].min(), 0, h-1)
        ymaxs = np.clip(dc[:,1].max(), 0, h-1)
        out[i] = [xmins, ymins, xmaxs, ymaxs]
    return out

def draw_bboxes(img: Image.Image, bboxes: np.ndarray, color=(255,0,0)) -> Image.Image:
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for x1,y1,x2,y2 in bboxes:
        dr.rectangle((x1,y1,x2,y2), outline=color, width=3)
    return im
    

# Example usage with image and bboxes form ground truth txt file

if __name__ == "__main__":
    img_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/still_from_vid.jpg"  # Replace with your image path
    gt_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt_for_still.txt"
    img = Image.open(img_path).convert("RGB")
    fx, fy, cx, cy = _get_intrinsics(img.size[0], img.size[1], DistortionParams())
    params = DistortionParams(fx=fx, fy=fy, cx=cx, cy=cy)
    distorted_img = distort_image(img, params)

    # Load bboxes from gt file
    bboxes = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            x, y, w, h = map(float, parts[2:6])
            x1, y1, x2, y2 = x, y, x+w, y+h
            bboxes.append([x1,y1,x2,y2])
    bboxes = np.array(bboxes, dtype=np.float32)

    # Draw original bboxes
    img_with_bboxes = draw_bboxes(img, bboxes, color=(0,255,0))
    img_with_bboxes.show()

    #wait till window is closed
    input("Press Enter to continue...")

    # Map bboxes through distortion
    distorted_bboxes = map_bboxes_through_distortion(bboxes, img.size, params)
    print(bboxes)
    print(distorted_bboxes)
    distorted_img_with_bboxes = draw_bboxes(distorted_img, bboxes, color=(0,255,0))
    distorted_img_with_bboxes = draw_bboxes(distorted_img_with_bboxes, distorted_bboxes, color=(255,0,0))
    distorted_img_with_bboxes.show()
