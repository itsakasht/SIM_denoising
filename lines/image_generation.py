import sys
import os
import numpy as np
import skimage as ski

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import modules.optics as op

# Image size
width, height = 1024, 1024

# Create a black background
image = np.ones((height, width), dtype=np.uint8) * 3

# Pixel sixe = 25nm
# Spacings in pixels between lines
spacings = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# spacings = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40])

print('Spacings (nm): ', 25*spacings)
# Starting x-coordinate
x = 20

spacing_index = 0
for space in spacings:
    image[height//8:7*height//8, x] = 255  # Draw a vertical line
    x += space+1
    image[height//8:7*height//8, x] = 255  # Draw a vertical line
    x += 40

image = (image / 255).astype(np.float32)

plt.figure()
plt.set_cmap('gray')
plt.imshow(image)
plt.axis('off')

# op.display_fourier(image)

plt.show()

image_path = os.path.join(current_dir, "lines_gt.tif")
ski.io.imsave(image_path, image)

widefield = op.otf_incoherent(image, NA=1.2, wavelength=480, pixelsize=25)
ski.io.imsave(os.path.join(current_dir, 'lines_wf.tif'), widefield)