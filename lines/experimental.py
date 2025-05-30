import sys
import os
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
import bm3d
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import modules.optics as op

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12
# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

# Load widefield image
wf_path = os.path.join(current_dir, 'lines_wf.tif')
widefield = ski.io.imread(wf_path)

SNR = [2, 5, 10, 20, 50, 100]
noise_types = ['Gaussian', 'Poisson', 'SaltPepper', 'Mix']

img = widefield

# Load 2 different types of noisy images
noisy = np.zeros((2, len(SNR), len(noise_types), img.shape[0], img.shape[1]))
for j in range(2):
    for i in range(len(SNR)):
        image = ski.io.imread(os.path.join(current_dir, f"results/noisy_images/noisy_wf_{j}_snr{SNR[i]}.tif"))
        noisy[j, i] = np.moveaxis(image, -1, 0)

base_psnr_color = "#1f77a4"
base_ssim_color = "#e51212"

psnr_color = ["#7b17eb", "#004aad", "#009fbc", "#00bf97"]
ssim_color = ["#e51212", "#a55318", "#ff8c2f", "#ffd323"]

x = np.arange(0, 10, 0.1)
y1 = 1.5*np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)+0.3
y4 = 0.8*np.cos(3*x)-0.2
y5 = 0.5*np.sin(4*x)+0.1
y6 = 0.2*np.cos(5*x)-0.3
y7 = 0.1*np.sin(6*x)+0.4
y8 = 0.3*np.cos(7*x)-0.5
cl = psnr_color
plt.figure()
plt.plot(x, y1, label='y1', color=cl[0])
plt.plot(x, y2, label='y2', color=cl[1])
plt.plot(x, y3, label='y3', color=cl[2])
plt.plot(x, y4, label='y4', color=cl[3])
plt.plot(x, y5, label='y5', color=cl[0])
plt.plot(x, y6, label='y6', color=cl[1])
# plt.plot(x, y7, label='y7', color=ssim_color[2])
# plt.plot(x, y8, label='y8', color=ssim_color[3])

plt.show()
exit()

### PSF comparison
noisy_img = noisy[0, 0, 2]
size = 17
gpsf = op.gaussian_psf(size)
ipsf = op.inchoerent_psf(psfsize=size, wavelength=480, NA=1.2, pixelsize=25)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(ipsf, cmap='gray')
plt.title('Incoherent PSF')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gpsf, cmap='gray')
plt.title('Gaussian PSF')
plt.axis('off')
plt.tight_layout()

# Wiener filter
wiener0 = ski.restoration.richardson_lucy(noisy_img, ipsf, num_iter=10)
wiener1 = ski.restoration.richardson_lucy(noisy_img, gpsf, num_iter=10)


plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(wiener0, cmap='gray')
plt.title('Wiener filter (Incoherent PSF)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(wiener1, cmap='gray')
plt.title('Wiener filter (Gaussian PSF)')
plt.axis('off')
plt.tight_layout()
plt.show()
