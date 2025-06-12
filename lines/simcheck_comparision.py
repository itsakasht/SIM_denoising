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

cool_color = ["#631cfe", "#0da6fe", "#11cbd1", "#25ffba"]
warm_color = ["#e51212", "#cb6820", "#ffb222", "#ffed23"]

# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

filters = ['Wiener', 'TV', 'RL', 'BM3D']
result = []

SNR = [2, 5, 10, 20, 50]

noisefree_sim = ski.io.imread(os.path.join(current_dir, 'results/illumination/noisefree_sim.tif'))
noisefree_sim[noisefree_sim < 0] = 0
noisefree_sim = noisefree_sim / (np.max(noisefree_sim)+1e-10)

metric = noisefree_sim

nf_psnr = ski.metrics.peak_signal_noise_ratio(metric, noisefree_sim, data_range=1)
nf_ssim = ski.metrics.structural_similarity(metric, noisefree_sim, data_range=1)


for snr in SNR:
    
    fairsim_a = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_a_snr{snr}_sim.tif"))
    fairsim_a[fairsim_a < 0] = 0
    fairsim_a = fairsim_a / (np.max(fairsim_a)+1e-10)
    fairsim_b = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_b_snr{snr}_sim.tif"))
    fairsim_b[fairsim_b < 0] = 0
    fairsim_b = fairsim_b / (np.max(fairsim_b)+1e-10)

    fairsim_psnr = np.mean([ski.metrics.peak_signal_noise_ratio(metric, fairsim_a, data_range=1),
                          ski.metrics.peak_signal_noise_ratio(metric, fairsim_b, data_range=1)])
    fairsim_ssim = np.mean([ski.metrics.structural_similarity(metric, fairsim_a, data_range=1),
                          ski.metrics.structural_similarity(metric, fairsim_b, data_range=1)])
    
    simcheck_a = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_a_snr{snr}_simcheck.tif"))
    simcheck_a[simcheck_a < 0] = 0
    simcheck_a = simcheck_a / (np.max(simcheck_a)+1e-10)
    simcheck_b = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_b_snr{snr}_simcheck.tif"))
    simcheck_b[simcheck_b < 0] = 0
    simcheck_b = simcheck_b / (np.max(simcheck_b)+1e-10)

    simcheck_psnr = np.mean([ski.metrics.peak_signal_noise_ratio(metric, simcheck_a, data_range=1),
                          ski.metrics.peak_signal_noise_ratio(metric, simcheck_b, data_range=1)])
    simcheck_ssim = np.mean([ski.metrics.structural_similarity(metric, simcheck_a, data_range=1),
                          ski.metrics.structural_similarity(metric, simcheck_b, data_range=1)])

    result.append({"SNR": snr,

                   "FairSIM_PSNR": fairsim_psnr,
                   "FairSIM_SSIM": fairsim_ssim,
                   
                   "SimCheck_PSNR": simcheck_psnr,
                   "SimCheck_SSIM": simcheck_ssim
        })

df = pd.DataFrame(result)
df.to_csv(os.path.join(current_dir, "results/illumination/unfiltered/simcheck_performance.csv"), index=False)


fairsim_psnr_values = df['FairSIM_PSNR'].tolist()
fairsim_ssim_values = df['FairSIM_SSIM'].tolist()
simcheck_psnr_values = df['SimCheck_PSNR'].tolist()
simcheck_ssim_values = df['SimCheck_SSIM'].tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(SNR, fairsim_psnr_values, marker='o', color=cool_color[0], label='FairSIM PSNR')
ax1.plot(SNR, simcheck_psnr_values, marker='o', color=cool_color[3], label='SimCheck PSNR')
ax1.set_ylim(0, 60)
ax1.set_title(f"PSNR Comparison")
ax1.set_ylabel("PSNR")
ax1.set_xlabel("SNR")
ax1.legend()

# for i, v in enumerate(psnr_values):
#     ax1.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_psnr_color)

ax2.plot(SNR, fairsim_ssim_values, marker='o', color=warm_color[0], label='FairSIM SSIM')
ax2.plot(SNR, simcheck_ssim_values, marker='o', color=warm_color[3], label='SimCheck SSIM')
ax2.set_ylim(0, 1.2)
ax2.set_title(f"SSIM Comparison")
ax2.set_ylabel("SSIM")
ax2.set_xlabel("SNR")
ax2.legend()

# for i, v in enumerate(ssim_values):
#     ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_ssim_color)

fig.suptitle(f"Unfiltered SIMcheck reconstruction comparision", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(current_dir, f"results/illumination/unfiltered/nf_simcheck.png"))

