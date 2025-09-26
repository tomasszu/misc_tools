import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# coarse mask 4x4
coarse_mask = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
], dtype=float)

# upscale factor
scale = 8 // coarse_mask.shape[0]  # upscale to 8x8

# nearest neighbor upsample first (repeat each coarse cell)
upsampled = np.repeat(np.repeat(coarse_mask, scale, axis=0), scale, axis=1)

# apply gaussian filter to smooth out subdivisions
gaussian_upsampled = gaussian_filter(upsampled, sigma=1)

# normalize back to [0,1]
gaussian_upsampled = (gaussian_upsampled - gaussian_upsampled.min()) / (gaussian_upsampled.max() - gaussian_upsampled.min())

# plot
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(coarse_mask, cmap='gray')
axs[0].set_title("Coarse 4x4 mask")
axs[1].imshow(upsampled, cmap='gray')
axs[1].set_title("Nearest-neighbor 8x8")
axs[2].imshow(gaussian_upsampled, cmap='gray')
axs[2].set_title("Gaussian upsampled 8x8")

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.show()
