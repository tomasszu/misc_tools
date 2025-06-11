
import argparse
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
import supervision as sv



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['undistort', 'pano'], default='undistort')
    return parser.parse_args()

def get_camera_intrinsics(image_width, image_height):
    """
    Returns the camera intrinsics matrix K for a given image size.
    
    Parameters:
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    
    Returns:
    - K: Camera intrinsics matrix as a numpy array.
    """
    fx = fy = image_width / 2  # Assuming square pixels and centered principal point
    cx = cy = image_width / 2   # Center of the image
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K

def get_distortion_coefficients():
    D = np.array([[-0.05], [0.01], [0.0], [0.0]])  # example fisheye distortion coeffs
    return D


def crop_from_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2], [(x1, y1), (x2 - x1, y2 - y1)]  # Return crop and top-left corner as (x, y) and size (width, height)

def extract_local_remap(map_x, map_y, top_left, crop_size):
    full_map_x = map_x
    full_map_y = map_y

    crop_x, crop_y = top_left
    crop_h, crop_w = crop_size

    # Extract relevant portion of the mapping and shift coordinates
    local_map_x = full_map_x[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] - crop_x
    local_map_y = full_map_y[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] - crop_y


    return local_map_x, local_map_y

def undistort_crop(crop, local_map_x, local_map_y):

    return cv2.remap(crop, local_map_x, local_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def process_detections(image, detections, mapping, K=None, D=None):
    """
    Processes the detections and applies undistortion to each detected object.
    
    Parameters:
    - image: Input image with detections.
    - detections: Detections object containing bounding boxes.
    - mapping: Mapping arrays for remapping the fisheye image to panorama.
    
    Returns:
    - List of undistorted and list of original crops from the detected objects.
    """
    
    corrected_crops = []
    original_crops = []
    map_x_full, map_y_full = mapping

    for xyxy in detections.xyxy:


        crop, coords = crop_from_bbox(image, xyxy)

        crop_h, crop_w = coords[1][1], coords[1][0]  # height and width of the crop
        crop_x, crop_y = coords[0]  # (x, y) coordinates of the top-left corner of the crop

        if crop_h < 50 or crop_w < 50:  # Skip invalid crops
            continue


        if args.mode == 'undistort':

            in_crop_mask = (
                (map_x_full >= crop_x) & (map_x_full < crop_x + crop_w) &
                (map_y_full >= crop_y) & (map_y_full < crop_y + crop_h)
            )

            # Create adjusted remap just for the crop
            map_x_crop = np.zeros_like(map_x_full, dtype=np.float32)
            map_y_crop = np.zeros_like(map_y_full, dtype=np.float32)

            map_x_crop[in_crop_mask] = map_x_full[in_crop_mask] - crop_x
            map_y_crop[in_crop_mask] = map_y_full[in_crop_mask] - crop_y

                        # For pixels outside the crop, map to -1 (invalid)
            map_x_crop[~in_crop_mask] = -1
            map_y_crop[~in_crop_mask] = -1

            # Step 5: Remap using adjusted map and cropped image
            warped_crop = cv2.remap(
                crop,
                map_x_crop,
                map_y_crop,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Find bounding box of valid pano area
            ys, xs = np.where(in_crop_mask)
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()

            # Crop the warped image to valid area
            warped_crop_cropped = warped_crop[min_y:max_y+1, min_x:max_x+1]

        elif args.mode == 'pano':

            in_crop_mask = (
                (map_x_full >= crop_x) & (map_x_full < crop_x + crop_w) &
                (map_y_full >= crop_y) & (map_y_full < crop_y + crop_h)
            )

            # Create adjusted remap just for the crop
            map_x_crop = np.zeros_like(map_x_full, dtype=np.float32)
            map_y_crop = np.zeros_like(map_y_full, dtype=np.float32)

            map_x_crop[in_crop_mask] = map_x_full[in_crop_mask] - crop_x
            map_y_crop[in_crop_mask] = map_y_full[in_crop_mask] - crop_y

            # For pixels outside the crop, map to -1 (invalid)
            map_x_crop[~in_crop_mask] = -1
            map_y_crop[~in_crop_mask] = -1

            # Step 5: Remap using adjusted map and cropped image
            warped_crop = cv2.remap(
                crop,
                map_x_crop,
                map_y_crop,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Provide for the cropped warped image to contain only the cropped area, not the full fisheye dimensions:

            # Find bounding box of valid pano area
            ys, xs = np.where(in_crop_mask)
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()

            # Crop the warped image to valid area
            warped_crop_cropped = warped_crop[min_y:max_y+1, min_x:max_x+1]

        # Append the original crop to the list
        if crop is not None and warped_crop_cropped is not None:
            original_crops.append(crop)
            corrected_crops.append(warped_crop_cropped)
            print(f"Processed bbox {xyxy}: Original crop shape {crop.shape}, Corrected crop shape {warped_crop_cropped.shape}")
        else:
            print(f"Warning: Crop or corrected crop is None for bbox {xyxy}. Skipping this detection.")
            continue
            

    # Return both original and undistorted crops
    return corrected_crops, original_crops
    

def detections_process(model, image, class_ids=None):
    """
    Processes the image with the YOLO model to detect objects.
    
    Parameters:
    - model: YOLO model instance.
    - image: Input image for detection.
    - class_ids: List of class IDs to filter detections.
    """

    confidence_threshold = 0.6

    results = model(image, conf=confidence_threshold, classes=class_ids)[0]

    detections = sv.Detections.from_ultralytics(results)

    return detections

def frame_annotations(image, detections):

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame

def get_undistortion_mapping(image, K, D):
    """
    Generates undistortion mapping for a fisheye image.
    
    Parameters:
    - image: Input fisheye image.
    - K: Camera intrinsics matrix.
    - D: Distortion coefficients.
    
    Returns:
    - map1, map2: Mapping arrays for remapping the fisheye image.
    """
    h, w = image.shape[:2]

    # Generate undistortion maps
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_32FC1
    )

    return map_x, map_y
    

def get_pano_mapping(img, fov_deg=360, output_width=2048, output_height=512):
    """
    Generates a mapping for converting fisheye images to panoramic images.
    
    Parameters:
    - img: Input fisheye image.
    - fov_deg: Field of view in degrees for the panorama.
    - output_width: Width of the output panoramic image.
    - output_height: Height of the output panoramic image.
    
    Returns:
    - map_x, map_y: Mapping arrays for remapping the fisheye image to panorama.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
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

    return map_x, map_y

def fisheye_to_panorama(img, mapping):


    # Polar to Cartesian
    map_x = mapping[0]
    map_y = mapping[1]

    # Remap to get panoramic image
    pano = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return pano

def fisheye_to_undistorted(image, mapping):
    """
    Applies undistortion mapping to a fisheye image.
    
    Parameters:
    - image: Input fisheye image.
    - mapping: Mapping arrays for undistortion.
    
    Returns:
    - undistorted_image: The undistorted image.
    """
    map_x, map_y = mapping

    # Apply the undistortion mapping
    undistorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image

def resize_to_height(image, target_height):
    """Resize image to a specific height while keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

def ensure_3_channels(image):
    """Convert grayscale or 4-channel image to 3-channel BGR."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def show_crop_pairs(warped_crops, original_crops, window_name="Crops", max_height=256):
    display_rows = []

    for orig, warp in zip(original_crops, warped_crops):
        orig_resized = ensure_3_channels(resize_to_height(orig, max_height))
        warp_resized = ensure_3_channels(resize_to_height(warp, max_height))

        # Pad to same width if needed
        max_width = max(orig_resized.shape[1], warp_resized.shape[1])
        orig_resized = cv2.copyMakeBorder(orig_resized, 0, 0, 0, max_width - orig_resized.shape[1], cv2.BORDER_CONSTANT)
        warp_resized = cv2.copyMakeBorder(warp_resized, 0, 0, 0, max_width - warp_resized.shape[1], cv2.BORDER_CONSTANT)

        combined = np.vstack([orig_resized, warp_resized])
        display_rows.append(combined)

    # Stack all vertically
    if display_rows:
        final_display = np.hstack(display_rows)
        cv2.imshow(window_name, final_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pad_to_width(img, target_width):
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    pad_right = target_width - w
    return cv2.copyMakeBorder(img, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def resize_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

def ensure_3_channels(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def main(args):

    # Load YOLOv8 model
    model = YOLO("yolov8x.pt")  # Load a pre-trained YOLOv8 model
    model.to(device='cuda')

    # class_ids of interest - car, motorcycle, bus and truck
    CLASS_ID = [2, 3, 5, 7]

    # Open folder and read images

    image_folder = Path("/home/tomass/tomass/data/VIP_CUP_2020_fisheye_dataset/fisheye_video_1")

    img_paths = sorted(image_folder.glob("*.[jp][pn]g"))  # matches .jpg, .jpeg, .png

    if args.mode == 'pano':
        # Establish pano mapping

        image = iter(img_paths).__next__()
        image = cv2.imread(str(image))
        if image is None:
            print(f"Failed to load: {image}")
            return

        mapping = get_pano_mapping(image)

    elif args.mode == 'undistort':
        # Get camera intrinsics and distortion coefficients
        image = iter(img_paths).__next__()
        image = cv2.imread(str(image))
        if image is None:
            print(f"Failed to load: {image}")
            return

        image_height, image_width = image.shape[:2]
        K = get_camera_intrinsics(image_width, image_height)
        D = get_distortion_coefficients()

        # Generate undistortion maps
        mapping = get_undistortion_mapping(image, K, D)

    
    # Loop through all image files (common formats)
    for image_path in img_paths:
        orig_image = cv2.imread(str(image_path))
        if orig_image is None:
            print(f"Failed to load: {image_path}")
            continue

        detections = detections_process(model, orig_image, class_ids=CLASS_ID)
        annotated_original = frame_annotations(orig_image, detections)

        if args.mode == 'undistort':
            
            corrected_crops, orig_crops = process_detections(orig_image, detections, mapping, K, D)
            
            undist_image = fisheye_to_undistorted(annotated_original, mapping)
            cv2.imshow("Undistorted Image", undist_image)
            cv2.imshow("Original Image", annotated_original)
            show_crop_pairs(corrected_crops, orig_crops, window_name="Crops")

            # Apply undistortion mapping to the original image
        elif args.mode == 'pano':

            corrected_crops, orig_crops = process_detections(orig_image, detections, mapping)

            # Convert fisheye image to panorama
            pano_image = fisheye_to_panorama(annotated_original, mapping)

            cv2.imshow("Panorama Image", pano_image)
            cv2.imshow("Original Image", annotated_original)

            show_crop_pairs(corrected_crops, orig_crops, window_name="Crops")

    #         if cv2.waitKey(0) & 0xFF == ord('q'):
    #             break
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
