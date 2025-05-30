import sys
import os
import numpy as np
import pandas as pd
import skimage as ski
import tifffile as tiff
import matplotlib.pyplot as plt
import bm3d
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import modules.optics as op
from modules.noise import gauss_noise, poisson_noise, salt_pepper_noise, mix_noise


# Load ground truth image
gt_path = os.path.join(current_dir, 'lines_gt.tif')
gt = ski.io.imread(gt_path)

# Plot widefield image
widefield = op.otf_incoherent(gt, NA=1.2, wavelength=480, pixelsize=25)
# op.display_fourier(widefield)
plt.rcParams['image.cmap'] = 'gray'

sc = ski.img_as_float(ski.io.imread(os.path.join(project_root, 'source_images/synthetic_circles.png'))[:, :, 0])
sc = op.normalize(sc)

img = widefield

ski.io.imsave(os.path.join(current_dir, 'lines_wf.tif'), widefield)
# Add different amounts of noise to the same image
measured_snr = []
SNR = [2, 5, 10, 20, 50, 100]
seeds = [0, 1]
pixel_size = 25 # in nm

os.makedirs(os.path.join(current_dir, "results/plots/noise_levels"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "results/noisy_images"), exist_ok=True)


gauss = np.zeros((len(SNR), img.shape[0], img.shape[1]))
poisson = np.zeros((len(SNR), img.shape[0], img.shape[1]))
snp = np.zeros((len(SNR), img.shape[0], img.shape[1]))
mix = np.zeros((len(SNR), img.shape[0], img.shape[1]))

mix_df = []

for seed in seeds:
    for i in range(len(SNR)):
        gauss[i], g_snr = gauss_noise(img, snr=SNR[i], seed=seed)
        poisson[i], p_snr = poisson_noise(img, scale_factor=2.4*SNR[i], seed=seed)
        snp[i], s_snr = salt_pepper_noise(img, snr=SNR[i], seed=seed)
        mix[i], m_snr = mix_noise(img, snr=SNR[i], seed=seed)

        data = {'SNR': SNR[i],
                'MSNR': m_snr,
                'PSNR': ski.metrics.peak_signal_noise_ratio(img, mix[i], data_range=1),
                'SSIM': ski.metrics.structural_similarity(img, mix[i], data_range=1)}
        mix_df.append(data)

        # continue

        noisy = np.stack([gauss[i], poisson[i], snp[i], mix[i]], axis=0, dtype=np.float32)
        tiff.imwrite(os.path.join(current_dir, f"results/noisy_images/noisy_wf_{seed}_snr{SNR[i]}.tif"), noisy, imagej=True)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f"Noisy images with SNR={SNR[i]}", fontsize=15)
        ax[0][0].imshow(gauss[i])
        ax[0][0].set_title("Gaussian noise")
        ax[0][0].axis('off')
        ax[0][1].imshow(poisson[i])
        ax[0][1].set_title("Poisson noise")
        ax[0][1].axis('off')
        ax[1][0].imshow(snp[i])
        ax[1][0].set_title("Salt and Pepper noise")
        ax[1][0].axis('off')
        ax[1][1].imshow(mix[i])
        ax[1][1].set_title("Mix noise")
        ax[1][1].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f"results/plots/noise_levels/noisy_{seed}_snr{SNR[i]}.png"), dpi=300)
        # fig.clear()

        # Collect metrics for each noise type
        noise_metrics = {
            'Gauss': {'MSNR': g_snr,
                    'PSNR': ski.metrics.peak_signal_noise_ratio(img, gauss[i], data_range=1),
                    'SSIM': ski.metrics.structural_similarity(img, gauss[i], data_range=1)},
            
            'Poisson': {'MSNR': p_snr,
                        'PSNR': ski.metrics.peak_signal_noise_ratio(img, poisson[i], data_range=1),
                        'SSIM': ski.metrics.structural_similarity(img, poisson[i], data_range=1)},
            
            'SaltPepper': {'MSNR': s_snr,
                        'PSNR': ski.metrics.peak_signal_noise_ratio(img, snp[i], data_range=1),
                        'SSIM': ski.metrics.structural_similarity(img, snp[i], data_range=1)},
            
            'Mix': {'MSNR': m_snr,
                    'PSNR': ski.metrics.peak_signal_noise_ratio(img, mix[i], data_range=1),
                    'SSIM': ski.metrics.structural_similarity(img, mix[i], data_range=1)}
        }

        # Add metrics as subrows under the current SNR value
        for noise_type, metrics in noise_metrics.items():
            row = {
                'RandomSeed': seed,
                'SNR': SNR[i],
                'NoiseType': noise_type,
                'MSNR': metrics['MSNR'],
                'PSNR': metrics['PSNR'],
                'SSIM': metrics['SSIM']
            }
            measured_snr.append(row)

    # tiff.imwrite(os.path.join(current_dir, "results/noisy_images/mix_noise.tif"), mix.astype(np.float32), imagej=True)

    # mix_df = pd.DataFrame(mix_df)
    # print(mix_df)
    # mix_df.to_csv(os.path.join(current_dir, 'results/noisy_images/mix_noise_metrics.csv'), index=False)
    # exit()

    # Create DataFrame with MultiIndex for SNR and NoiseType
df = pd.DataFrame(measured_snr)

# Set SNR and NoiseType as MultiIndex
df.set_index(['RandomSeed', 'SNR', 'NoiseType'], inplace=True)

# Save to CSV
df.to_csv(os.path.join(current_dir, 'results/noisy_images/noise_metrics.csv'))
# plt.show()