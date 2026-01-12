import cv2
import numpy as np
import math

PI = math.pi

def get_input_point(x, y, src_width, src_height):
    """
    Compute the corresponding fisheye-space point for a rectilinear target pixel.
    Directly from guerrerocarlos's C++ implementation.
    """
    FOV = PI / 180 * 180
    FOV2 = PI / 180 * 180
    width = float(src_width)
    height = float(src_height)

    # Polar angles (range -pi/2 to pi/2)
    theta = PI * (x / width - 0.5)
    phi = PI * (y / height - 0.5)

    # Vector in 3D space
    psph_x = math.cos(phi) * math.sin(theta)
    psph_y = math.cos(phi) * math.cos(theta)
    psph_z = math.sin(phi) * math.cos(theta)

    # Calculate fisheye angle and radius
    theta = math.atan2(psph_z, psph_x)
    phi = math.atan2(math.sqrt(psph_x ** 2 + psph_z ** 2), psph_y)

    r = width * phi / FOV
    r2 = height * phi / FOV2

    # Pixel in fisheye space
    pfish_x = 0.5 * width + r * math.cos(theta)
    pfish_y = 0.5 * height + r2 * math.sin(theta)

    return int(pfish_x), int(pfish_y)


def fisheye_to_rectilinear(input_path, output_path):
    """
    Converts a fisheye image to a rectilinear projection using the
    algorithm by guerrerocarlos (StackOverflow CC-BY-SA 4.0).
    """
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    h, w = original.shape[:2]
    out_img = np.zeros_like(original)

    for j in range(h):
        for i in range(w):
            x_in, y_in = get_input_point(i, j, w, h)

            if 0 <= x_in < w and 0 <= y_in < h:
                out_img[j, i] = original[y_in, x_in]

    cv2.imwrite(output_path, out_img)
    print(f"Saved rectilinear image to {output_path}")


# Example usage:
fisheye_to_rectilinear("/home/tomass/Downloads/images/images_00/left_images/img_1760063941558817131.png", "output.png")

