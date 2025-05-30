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
os.makedirs(os.path.join(current_dir, "results/illumination/filtered"), exist_ok=True)

import modules.optics as op

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

# Load DataFrame
df = pd.read_csv(os.path.join(current_dir, 'results/plots/optimum_params/optimum_filter_parameters.csv'))

# Filter DataFrame for mix noise type
mix_noise_df = df[df['NoiseType'] == 'Mix']

SNR = [2, 5, 10, 20, 50, 100]


# Apply filtering using parameters from the DataFrame
for _, row in mix_noise_df.iterrows():
    snr = row["SNR"]

    # Load noisy image
    raw_a = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_a_snr{snr}.tif"))
    raw_b = ski.io.imread(os.path.join(current_dir, f"results/illumination/unfiltered/mix_raw_b_snr{snr}.tif"))

    # Wiener filter
    balance = row["Wiener_Balance"]
    psf_size = row["Wiener_PSF"]
    best_wpsf = row["Wiener_PSF"]
    if best_wpsf.startswith("Gaussian"):
        psf_size = int(best_wpsf.split()[-1])
        psf = op.gaussian_psf(psf_size)
    else:
        wavelength = int(best_wpsf.split()[-1].replace('nm', ''))
        psf = op.incoherent_psf(psfsize=21, wavelength=wavelength, NA=1.2, pixelsize=25)
    
    wiener_a = np.zeros((9, raw_a.shape[1], raw_a.shape[2]))
    wiener_b = np.zeros((9, raw_b.shape[1], raw_b.shape[2]))
    for i in range(9):
        wiener_a[i] = ski.restoration.wiener(raw_a[i], psf=psf, balance=balance)
        wiener_b[i] = ski.restoration.wiener(raw_b[i], psf=psf, balance=balance)
    
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/wiener_a_snr{snr}.tif"), wiener_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/wiener_b_snr{snr}.tif"), wiener_b.astype(np.float32), imagej=True)

    # TV filter
    tv_weight = row["TV_Weight"]

    tv_a = np.zeros((9, raw_a.shape[1], raw_a.shape[2]))
    tv_b = np.zeros((9, raw_b.shape[1], raw_b.shape[2]))
    for i in range(9):
        tv_a[i] = ski.restoration.denoise_tv_chambolle(raw_a[i], weight=tv_weight)
        tv_b[i] = ski.restoration.denoise_tv_chambolle(raw_b[i], weight=tv_weight)
    
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/tv_a_snr{snr}.tif"), tv_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/tv_b_snr{snr}.tif"), tv_b.astype(np.float32), imagej=True)

    # RL filter
    rl_iter = row["RL_Iterations"]
    rl_psf_size = row["RL_PSF"]
    best_psf = row["RL_PSF"]
    if best_psf.startswith("Gaussian"):
        psf_size = int(best_psf.split()[-1])
        psf = op.gaussian_psf(psf_size)
    else:
        wavelength = int(best_psf.split()[-1].replace('nm', ''))
        psf = op.incoherent_psf(psfsize=21, wavelength=wavelength, NA=1.2, pixelsize=25)

    rl_a = np.zeros((9, raw_a.shape[1], raw_a.shape[2]))
    rl_b = np.zeros((9, raw_b.shape[1], raw_b.shape[2]))

    for i in range(9):
        rl_a[i] = ski.restoration.richardson_lucy(raw_a[i], psf=psf, num_iter=rl_iter)
        rl_b[i] = ski.restoration.richardson_lucy(raw_b[i], psf=psf, num_iter=rl_iter)
    
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/rl_a_snr{snr}.tif"), rl_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/rl_b_snr{snr}.tif"), rl_b.astype(np.float32), imagej=True)

    # BM3D filter
    bm3d_sigma = row["BM3D_Sigma"]
    bm3d_stage = row["BM3D_Stage"]
    stage = bm3d.BM3DStages.ALL_STAGES if bm3d_stage == "ALL_STAGES" else bm3d.BM3DStages.HARD_THRESHOLDING

    bm_a = np.zeros((9, raw_a.shape[1], raw_a.shape[2]))
    bm_b = np.zeros((9, raw_b.shape[1], raw_b.shape[2]))

    for i in range(9):
        bm_a[i] = bm3d.bm3d(raw_a[i], sigma_psd=bm3d_sigma, stage_arg=stage)
        bm_b[i] = bm3d.bm3d(raw_b[i], sigma_psd=bm3d_sigma, stage_arg=stage)

    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/bm3d_a_snr{snr}.tif"), bm_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/illumination/filtered/bm3d_b_snr{snr}.tif"), bm_b.astype(np.float32), imagej=True)

    print(f"Filtered SNR {snr}")

mix_noise_df = mix_noise_df.drop(columns=['NoiseType', 'Noisy_PSNR', 'Noisy_SSIM', 'Wiener_PSNR', 'Wiener_SSIM', 'TV_PSNR', 'TV_SSIM', 'RL_PSNR', 'RL_SSIM', 'BM3D_PSNR', 'BM3D_SSIM'])
mix_noise_df.to_csv(os.path.join(current_dir, 'results/illumination/filtered/mix_noise_filter_parameters.csv'), index=False)
