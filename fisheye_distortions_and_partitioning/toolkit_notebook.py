# This notebook builds a small, self-contained toolkit to simulate fisheye/barrel distortion,
# partition images into tiles (e.g., 6 x 640x640), and keep track of bbox coordinate mappings.
# It includes a quick demo on a synthetic image so you can see outputs immediately.
#
# Usage (outside this demo):
# - Put your images in a folder, set INPUT_DIR, set OUTPUT_DIR, and run process_dataset().
# - If you have YOLO-format bboxes (txt per image), add a loader to feed them into `map_bboxes_through_distortion`.
#
# NOTE: Tries OpenCV for speed; falls back to a pure-Pillow/Numpy remapper if OpenCV isn't available.

import os
import io
import json
import math
import zipfile
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

# -----------------------------
# Geometry helpers
# -----------------------------

@dataclass
class DistortionParams:
    # Radial polynomial distortion (OpenCV-style, barrel if k1 < 0)
    k1: float = -0.35
    k2: float = 0.10
    k3: float = -0.02
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

# -----------------------------
# Partitioning & Letterbox (YOLO-style)
# -----------------------------

def letterbox(img: Image.Image, new_size=(640,640), stride=32) -> Tuple[Image.Image, Tuple[float,float], Tuple[int,int]]:
    """Resize + pad to new_size keeping aspect ratio. Returns (image, scale, pad)."""
    w, h = img.size
    r = min(new_size[0]/w, new_size[1]/h)
    # resize
    new_unpad = (int(round(w * r)), int(round(h * r)))
    img_resized = img.resize(new_unpad, Image.BILINEAR)
    # compute padding
    dw = new_size[0] - new_unpad[0]
    dh = new_size[1] - new_unpad[1]
    left = dw // 2
    top = dh // 2
    right = dw - left
    bottom = dh - top
    img_padded = ImageOps.expand(img_resized, border=(left, top, right, bottom), fill=0)
    return img_padded, (r, r), (left, top)

def partition_image(img: Image.Image, grid: Tuple[int,int]=(3,2), tile_out=(640,640)) -> Tuple[List[Image.Image], List[Tuple[int,int,int,int]]]:
    """Split image into grid (cols, rows), crop tiles, letterbox each to tile_out.
       Returns tiles and their bbox regions in the original image.
    """
    w, h = img.size
    cols, rows = grid
    tile_w = w // cols
    tile_h = h // rows
    tiles = []
    regions = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * tile_w
            y1 = r * tile_h
            x2 = w if c == cols-1 else (c+1) * tile_w
            y2 = h if r == rows-1 else (r+1) * tile_h
            crop = img.crop((x1,y1,x2,y2))
            lb, scale, pad = letterbox(crop, new_size=tile_out)
            tiles.append(lb)
            regions.append((x1,y1,x2,y2))
    return tiles, regions

def remap_bbox_to_tile(
    bbox_xyxy: np.ndarray, region_xyxy: Tuple[int,int,int,int], tile_out=(640,640)
) -> Optional[np.ndarray]:
    """Map a bbox in global coords into letterboxed tile coords. Returns None if no overlap."""
    x1,y1,x2,y2 = bbox_xyxy
    rx1,ry1,rx2,ry2 = region_xyxy
    # intersect
    ix1 = max(x1, rx1); iy1 = max(y1, ry1)
    ix2 = min(x2, rx2); iy2 = min(y2, ry2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None  # no overlap

    # map into crop-local
    cx1, cy1 = ix1 - rx1, iy1 - ry1
    cx2, cy2 = ix2 - rx1, iy2 - ry1

    # compute letterbox mapping for this region
    rw = rx2 - rx1
    rh = ry2 - ry1
    r = min(tile_out[0]/rw, tile_out[1]/rh)
    new_w = int(round(rw * r))
    new_h = int(round(rh * r))
    dw = tile_out[0] - new_w
    dh = tile_out[1] - new_h
    left = dw // 2
    top = dh // 2

    # scale + pad
    tx1 = cx1 * r + left
    ty1 = cy1 * r + top
    tx2 = cx2 * r + left
    ty2 = cy2 * r + top
    return np.array([tx1, ty1, tx2, ty2], dtype=np.float32)

# -----------------------------
# Demo: generate a synthetic image + bboxes, distort, partition, remap
# -----------------------------

def draw_bboxes(img: Image.Image, bboxes: np.ndarray, color=(255,0,0)) -> Image.Image:
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for x1,y1,x2,y2 in bboxes:
        dr.rectangle((x1,y1,x2,y2), outline=color, width=3)
    return im

def synthetic_demo(save_dir="/mnt/data/fisheye_partition_demo"):
    os.makedirs(save_dir, exist_ok=True)
    # Create a synthetic checkerboard with rectangles standing in for "cars"
    W, H = 1920, 1080
    base = Image.new("RGB", (W,H), (40,40,40))
    dr = ImageDraw.Draw(base)
    # checker
    step = 60
    for y in range(0, H, step):
        for x in range(0, W, step):
            if ((x//step) + (y//step)) % 2 == 0:
                dr.rectangle([x,y,x+step-1,y+step-1], fill=(60,60,60))
    # add a few "vehicles" as bright rectangles
    gt_bboxes = np.array([
        [300, 300, 500, 450],
        [900, 200, 1100, 380],
        [1400, 700, 1700, 980],
    ], dtype=np.float32)
    base = draw_bboxes(base, gt_bboxes, color=(0,255,0))
    base.save(os.path.join(save_dir, "00_base_with_gt.png"))

    # Distort
    params = DistortionParams(k1=-0.35, k2=0.1, k3=-0.02, p1=0.0, p2=0.0)
    distorted = distort_image(base, params)
    # Map bboxes (approx via corner mapping)
    mapped = map_bboxes_through_distortion(gt_bboxes, img_size=base.size, params=params)
    distorted_with_boxes = draw_bboxes(distorted, mapped, color=(255,0,0))
    distorted_with_boxes.save(os.path.join(save_dir, "01_distorted_with_mapped_boxes.png"))

    # Partition distorted image into 3x2 grid and letterbox to 640
    tiles, regions = partition_image(distorted, grid=(3,2), tile_out=(640,640))

    # For each tile, remap bboxes that overlap that tile region
    manifest = {"tiles": []}
    for idx, (tile_img, region) in enumerate(zip(tiles, regions)):
        tile_bboxes = []
        for bb in mapped:
            tb = remap_bbox_to_tile(bb, region, tile_out=(640,640))
            if tb is not None:
                tile_bboxes.append(tb.tolist())
        # draw per-tile
        drawn = draw_bboxes(tile_img, np.array(tile_bboxes, dtype=np.float32), color=(255,255,255)) if tile_bboxes else tile_img
        out_path = os.path.join(save_dir, f"tile_{idx:02d}.png")
        drawn.save(out_path)
        manifest["tiles"].append({
            "index": idx,
            "region_xyxy": list(region),
            "tile_image": out_path,
            "mapped_bboxes_xyxy": tile_bboxes,
        })

    # Save manifest
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Also create a ZIP for easy download
    zip_path = os.path.join(save_dir, "demo_outputs.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(save_dir):
            for fn in files:
                if fn.endswith(".png") or fn.endswith(".json"):
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, save_dir))
    return save_dir, zip_path

save_dir, zip_path = synthetic_demo()
save_dir, zip_path
