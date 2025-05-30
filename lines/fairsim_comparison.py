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

import modules.optics as op
from modules.noise import mix_noise

# Set plotting styles
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Colors
base_psnr_color = "#1f77a4"
base_ssim_color = "#e51212"
psnr_color = ["#7b17eb", "#004aad", "#009fbc", "#00bf97"]
ssim_color = ["#e51212", "#a55318", "#ff8c2f", "#ffd323"]

# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

filters = ['Wiener', 'TV', 'RL', 'BM3D']
result = []

SNR = [2, 5, 10, 20, 50, 100]

noisefree_sim = ski.io.imread(os.path.join(current_dir, 'results/illumination/noisefree_sim.tif'))[:, :, 0]
noisefree_sim[noisefree_sim < 0] = 0
noisefree_sim = noisefree_sim / (np.max(noisefree_sim)+1e-10)

metric = gt

nf_psnr = ski.metrics.peak_signal_noise_ratio(metric, noisefree_sim, data_range=1)
nf_ssim = ski.metrics.structural_similarity(metric, noisefree_sim, data_range=1)


for snr in SNR:
    noisy_img_a = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_a_snr{snr}_sim.tif"))
    noisy_img_a[noisy_img_a < 0] = 0
    noisy_img_a = noisy_img_a / (np.max(noisy_img_a)+1e-10)
    noisy_img_b = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_b_snr{snr}_sim.tif"))
    noisy_img_b[noisy_img_b < 0] = 0
    noisy_img_b = noisy_img_b / (np.max(noisy_img_b)+1e-10)

    noisy_psnr = np.mean([ski.metrics.peak_signal_noise_ratio(metric, noisy_img_a, data_range=1),
                          ski.metrics.peak_signal_noise_ratio(metric, noisy_img_b, data_range=1)])
    noisy_ssim = np.mean([ski.metrics.structural_similarity(metric, noisy_img_a, data_range=1),
                          ski.metrics.structural_similarity(metric, noisy_img_b, data_range=1)])

    # Wiener
    wiener = ski.io.imread(os.path.join(current_dir, f"results/illumination/filtered/wiener_snr{snr}_sim.tif"))
    wiener[wiener < 0] = 0
    wiener = wiener / (np.max(wiener)+1e-10)
    w_psnr = ski.metrics.peak_signal_noise_ratio(metric, wiener, data_range=1)
    w_ssim = ski.metrics.structural_similarity(metric, wiener, data_range=1)

    # TV
    tv = ski.io.imread(os.path.join(current_dir, f"results/illumination/filtered/tv_snr{snr}_sim.tif"))
    tv[tv < 0] = 0
    tv = tv / (np.max(tv)+1e-10)
    t_psnr = ski.metrics.peak_signal_noise_ratio(metric, tv, data_range=1)
    t_ssim = ski.metrics.structural_similarity(metric, tv, data_range=1)

    # RL
    rl = ski.io.imread(os.path.join(current_dir, f"results/illumination/filtered/rl_snr{snr}_sim.tif"))
    rl[rl < 0] = 0
    rl = rl / (np.max(rl)+1e-10)
    r_psnr = ski.metrics.peak_signal_noise_ratio(metric, rl, data_range=1)
    r_ssim = ski.metrics.structural_similarity(metric, rl, data_range=1)

    # BM3D
    bm = ski.io.imread(os.path.join(current_dir, f"results/illumination/filtered/bm3d_snr{snr}_sim.tif"))
    bm[bm < 0] = 0
    bm = bm / (np.max(bm)+1e-10)
    b_psnr = ski.metrics.peak_signal_noise_ratio(metric, bm, data_range=1)
    b_ssim = ski.metrics.structural_similarity(metric, bm, data_range=1)
    
    result.append({"SNR": snr,
                   "Noisy_PSNR": noisy_psnr,
                   "Noisy_SSIM": noisy_ssim,
                   
                   "Wiener_PSNR": w_psnr,
                   "Wiener_SSIM": w_ssim,

                   "TV_PSNR": t_psnr,
                   "TV_SSIM": t_ssim,
                   
                   "RL_PSNR": r_psnr,
                   "RL_SSIM": r_ssim,

                   "BM3D_PSNR": b_psnr,
                   "BM3D_SSIM": b_ssim,
        })
    
    psnr_values = [w_psnr, t_psnr, r_psnr, b_psnr]
    ssim_values = [w_ssim, t_ssim, r_ssim, b_ssim]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.bar(filters, psnr_values, color=psnr_color[:4])
    ax1.axhline(y=noisy_psnr, color=base_psnr_color, linestyle='--', label='Noisy PSNR', alpha=0.8)
    ax1.axhline(y=nf_psnr, color='grey', linestyle='--', label='Noisefree PSNR', alpha=0.8)
    ax1.set_ylim(0, 50)
    ax1.set_title(f"PSNR Comparison (SNR={snr})")
    ax1.set_ylabel("PSNR")
    ax1.set_xlabel("Filter")
    ax1.legend()

    for i, v in enumerate(psnr_values):
        ax1.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_psnr_color)

    ax2.bar(filters, ssim_values, color=ssim_color[:4])
    ax2.axhline(y=noisy_ssim, color=base_ssim_color, linestyle='--', label='Noisy SSIM', alpha=0.8)
    ax2.axhline(y=nf_ssim, color='grey', linestyle='--', label='Noisefree SSIM', alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"SSIM Comparison (SNR={snr})")
    ax2.set_ylabel("SSIM")
    ax2.set_xlabel("Filter")
    ax2.legend()

    for i, v in enumerate(ssim_values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_ssim_color)

    fig.suptitle(f"Prefiltered SIM reconstruction for SNR={snr}", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(current_dir, f"results/illumination/filtered/gt_prefiltered_sim_snr{snr}.png"))
    plt.close(fig)

df = pd.DataFrame(result)
df.to_csv(os.path.join(current_dir, "results/illumination/filtered/gt_prefiltered_sim_performance.csv"), index=False)
