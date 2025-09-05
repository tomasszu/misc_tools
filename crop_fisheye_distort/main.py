import cv2
import numpy as np
from PIL import Image

def local_fisheye_patch(img, strength=0.0005, direction='radial'):
    """
    Simulate being in a fisheye patch (not the whole image).
    strength: distortion strength (higher = more bend).
    direction: 'radial' (stretch from center), 'tangential' (bend sideways).
    """
    h, w = img.shape[:2]
    # meshgrid
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    if direction == 'radial':
        r = np.sqrt(xv**2 + yv**2)
        map_x = xv * (1 + strength * r**2)
        map_y = yv * (1 + strength * r**2)
    else:  # tangential bend along x
        map_x = xv + strength * (yv**2)
        map_y = yv

    # normalize back to pixel coords
    map_x = ((map_x + 1) * 0.5 * (w-1)).astype(np.float32)
    map_y = ((map_y + 1) * 0.5 * (h-1)).astype(np.float32)

    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    return distorted

# Example
img = np.array(Image.open("/home/tomass/tomass/data/VeRi/image_test/0002_c003_00084300_1.jpg"))
distorted = local_fisheye_patch(img, strength=0.2, direction='tangential')
Image.fromarray(distorted).save("vehicle_crop_localfisheye.jpg")
