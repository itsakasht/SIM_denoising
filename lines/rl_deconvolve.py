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
import time
start_time = time.perf_counter()

os.makedirs(os.path.join(current_dir, "results/plots/optimum_params"), exist_ok=True)

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12
# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

# Load widefield image
wf_path = os.path.join(current_dir, 'lines_wf.tif')
widefield = ski.io.imread(wf_path)

psf = op.incoherent_psf(psfsize=21, wavelength=480, NA=1.2, pixelsize=25)
plt.figure(figsize=(10, 10))
plt.imshow(widefield, cmap='gray')
plt.title("Widefield Image")
plt.axis('off')
plt.tight_layout()

for i in range(3):
    bma = bm3d.bm3d(widefield, sigma_psd=0.2*(i+1), stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    bmb = bm3d.bm3d(widefield, sigma_psd=0.2*(i+1), stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    ssim_a = ski.metrics.structural_similarity(widefield, bma, data_range=1)
    ssim_b = ski.metrics.structural_similarity(widefield, bmb, data_range=1)

    print(f"SSIM for iteration {i+1}: {ssim_a:.4f}, {ssim_b:.4f}")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(bma, cmap='gray')
    # plt.title(f"RL Iterations: {i+1}")
    # plt.axis('off')
    # plt.tight_layout()

# plt.show()