import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski
import bm3d
import tifffile as tiff

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
os.makedirs(os.path.join(current_dir, "results/illumination/unfiltered"), exist_ok=True)

import modules.optics as op
from modules.noise import mix_noise

# Set plotting styles
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

# Load widefield image
wf_path = os.path.join(current_dir, 'lines_wf.tif')
widefield = ski.io.imread(wf_path)

test_path = os.path.join(current_dir, 'Test.tif')
test = ski.io.imread(test_path)

SNR = [2, 5, 10, 20, 50, 100]

angles = np.array([[0, 120, 240], # 0 deg deviation
                   [0, 119, 241], # 1 deg deviation
                   [0, 121, 241], # 1 deg deviation
                   [0, 119, 239], # 1 deg deviation
                   [0, 121, 239], # 1 deg deviation
                   [0, 118, 242], # 2 deg deviation
                   [0, 122, 242], # 2 deg deviation
                   [0, 118, 238], # 2 deg deviation
                   [0, 122, 238], # 2 deg deviation
                   [0, 110, 250], # 5 deg deviation
                   [0, 130, 250], # 5 deg deviation
                   [0, 110, 230], # 5 deg deviation
                   [0, 130, 230], # 5 deg deviation
                   [0, 100, 260], # 10 deg deviation
                   [0, 140, 260], # 10 deg deviation
                   [0, 100, 220], # 10 deg deviation
                   [0, 140, 220]])# 10 deg deviation

phases = np.array([[0, 120, 240], # 0 deg deviation
                   [0, 119, 241], # 1 deg deviation
                   [0, 121, 241], # 1 deg deviation
                   [0, 119, 239], # 1 deg deviation
                   [0, 121, 239], # 1 deg deviation
                   [0, 118, 242], # 2 deg deviation
                   [0, 122, 242], # 2 deg deviation
                   [0, 118, 238], # 2 deg deviation
                   [0, 122, 238], # 2 deg deviation
                   [0, 110, 250], # 5 deg deviation
                   [0, 130, 250], # 5 deg deviation
                   [0, 110, 230], # 5 deg deviation
                   [0, 130, 230], # 5 deg deviation
                   [0, 100, 260], # 10 deg deviation
                   [0, 140, 260], # 10 deg deviation
                   [0, 100, 220], # 10 deg deviation
                   [0, 140, 220]])# 10 deg deviation


image = gt
n = 9
illuminated = np.zeros((n, image.shape[0], image.shape[1]))
raw = np.zeros((n, image.shape[0]//2, image.shape[1]//2))

for i in range(3):
    for j in range(3):
        illuminated[3*i + j] = image * op.grating(len(image), angle=30+angles[0, i],  phase=phases[0, j], NA=1.2, wavelength=500, pixelsize=25)
        raw_double = op.otf_incoherent(illuminated[3*i + j], NA=1.2, wavelength=480, pixelsize=25)
        raw[3*i + j] = op.fourier_downsample(raw_double)

tiff.imwrite(os.path.join(current_dir, "results/illumination/illuminated.tif"), raw.astype(np.float32), imagej=True)

raw = ski.io.imread(os.path.join(current_dir, "results/illumination/illuminated.tif"))

exit()

mix_df = []
for i in range(len(SNR)):
    mix_a = np.zeros((9, raw.shape[1], raw.shape[2]))
    mix_b = np.zeros((9, raw.shape[1], raw.shape[2]))
    msnr = np.zeros((9))
    for j in range(9):
        mix_a[j], msnra = mix_noise(raw[j], snr=SNR[i], seed=0)
        mix_b[j], msnrb = mix_noise(raw[j], snr=SNR[i], seed=1)
        msnr[j] = np.mean([msnra, msnrb])
    data = {'SNR': SNR[i],
            'MSNR_A': msnra,
            'MSNR_B': msnrb}
    mix_df.append(data)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_a_snr{SNR[i]}.tif"), mix_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_b_snr{SNR[i]}.tif"), mix_b.astype(np.float32), imagej=True)

# Save results to CSV
mix_df = pd.DataFrame(mix_df)
mix_df.to_csv(os.path.join(current_dir, "results/illumination/unfiltered/mix_metrics.csv"), index=False)
