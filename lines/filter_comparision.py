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
from modules.frc import create_frc_map, plot_frc_map, generate_results_table

# Set plotting styles
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Colors
base_cool_color = "#1f77a4"
base_warm_color = "#e51212"

cool_color = ["#631cfe", "#0da6fe", "#11cbd1", "#25ffba"]
warm_color = ["#e51212", "#cb6820", "#ffb222", "#ffed23"]

# Load DataFrame
df = pd.read_csv(os.path.join(current_dir, 'results/plots/optimum_params_fixed_psf/optimum_filter_parameters.csv'))

# Ground truth
wf_path = os.path.join(current_dir, 'lines_wf.tif')
gt = ski.io.imread(wf_path)


# Make folders if not exist
os.makedirs(os.path.join(current_dir, "results/stacks"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "results/plots"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "results/plots/optimum_params_fixed_psf/frc"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "results/plots/best_filters"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "results/plots/optimum_params_fixed_psf/refilter_performance"), exist_ok=True)




# Storage for table
final_table = []
frc_final_table = []

filters = ['Wiener', 'TV', 'RL', 'BM3D']

all_filtered_stack_a = []
all_filtered_stack_b = []

for idx, row in df.iterrows():
    snr = row["SNR"]
    noise_type = row["NoiseType"]

    # Load noisy images
    noisy_a = ski.io.imread(os.path.join(current_dir, f"results/noisy_images/noisy_wf_0_snr{snr}.tif"))
    noisy_a = np.moveaxis(noisy_a, -1, 0)

    noisy_b = ski.io.imread(os.path.join(current_dir, f"results/noisy_images/noisy_wf_1_snr{snr}.tif"))
    noisy_b = np.moveaxis(noisy_b, -1, 0)
    
    # Pick correct noisy image
    noise_idx = ['Gaussian', 'Poisson', 'SaltPepper', 'Mix'].index(noise_type)
    noisy_img_a = noisy_a[noise_idx]
    noisy_img_b = noisy_b[noise_idx]

    # Filtered images
    filtered_images_a = []
    filtered_images_b = []

    # -- Wiener

    best_psf = row["Wiener_PSF"]
    if best_psf.startswith("Gaussian"):
        psf_size = int(best_psf.split()[-1])
        psf = op.gaussian_psf(psf_size)
    else:
        wavelength = int(best_psf.split()[-1].replace('nm', ''))
        psf = op.incoherent_psf(psfsize=21, wavelength=wavelength, NA=1.2, pixelsize=25)

    wiener_a = ski.restoration.wiener(noisy_img_a, psf, balance=row["Wiener_Balance"])
    wiener_b = ski.restoration.wiener(noisy_img_b, psf, balance=row["Wiener_Balance"])
    filtered_images_a.append(wiener_a)
    filtered_images_b.append(wiener_b)

    # -- TV
    tv_a = ski.restoration.denoise_tv_chambolle(noisy_img_a, weight=row["TV_Weight"])
    tv_b = ski.restoration.denoise_tv_chambolle(noisy_img_b, weight=row["TV_Weight"])
    filtered_images_a.append(tv_a)
    filtered_images_b.append(tv_b)

    # -- RL
    best_psf = row["RL_PSF"]
    if best_psf.startswith("Gaussian"):
        psf_size = int(best_psf.split()[-1])
        psf = op.gaussian_psf(psf_size)
    else:
        wavelength = int(best_psf.split()[-1].replace('nm', ''))
        psf = op.incoherent_psf(psfsize=21, wavelength=wavelength, NA=1.2, pixelsize=25)

    rl_a = ski.restoration.richardson_lucy(noisy_img_a, psf, num_iter=int(row["RL_Iterations"]), filter_epsilon=1e-10)
    rl_b = ski.restoration.richardson_lucy(noisy_img_b, psf, num_iter=int(row["RL_Iterations"]), filter_epsilon=1e-10)
    filtered_images_a.append(rl_a)
    filtered_images_b.append(rl_b)


    # -- BM3D
    stage = bm3d.BM3DStages.HARD_THRESHOLDING # if row["BM3D_Stage"] == "HARD_THRESHOLDING" else bm3d.BM3DStages.ALL_STAGES
    bm_a = bm3d.bm3d(noisy_img_a, sigma_psd=row["BM3D_Sigma"], stage_arg=stage)
    bm_b = bm3d.bm3d(noisy_img_b, sigma_psd=row["BM3D_Sigma"], stage_arg=stage)
    filtered_images_a.append(bm_a)
    filtered_images_b.append(bm_b)

    # Stack and save
    filtered_stack_a = np.stack(filtered_images_a, axis=0)
    filtered_stack_b = np.stack(filtered_images_b, axis=0)
    all_filtered_stack_a.append(filtered_stack_a)
    all_filtered_stack_b.append(filtered_stack_b)

    tiff.imwrite(os.path.join(current_dir, f"results/stacks/filtered_stack_a_snr{snr}_{noise_type}.tif"), filtered_stack_a.astype(np.float32), imagej=True)
    tiff.imwrite(os.path.join(current_dir, f"results/stacks/filtered_stack_b_snr{snr}_{noise_type}.tif"), filtered_stack_b.astype(np.float32), imagej=True)

    block_size = 64         # Block size in pixels
    pixel_size_nm = 25    # Pixel size in nanometers

    bins = 30
    metric = gt
    frc = np.zeros((len(filters)+1, 2, bins))
    # Plot FRC
    for i in range(len(filtered_images_a)):
        frc_map, frc_values = create_frc_map(filtered_images_a[i], filtered_images_b[i], block_size, pixel_size_nm, threshold=1/7)
        # Plot FRC map
        plt.figure(figsize=(8,6))
        plt.imshow(frc_map, cmap='coolwarm' , origin='lower')
        plt.colorbar(label='Resolution (nm)')
        plt.title('FRC Resolution Map')
        plt.savefig(os.path.join(current_dir, f"results/plots/optimum_params_fixed_psf/frc/frc_map_snr{snr}_{noise_type}_{filters[i]}.png"))
        plt.close()

        results_table = generate_results_table(frc_values)
        # print(results_table)
        # results_table.to_csv(os.path.join(current_dir, f"results/plots/frc/frc_results_snr{snr}_{noise_type}_{filters[i]}.csv"), index=False)

        results_table["SNR"] = snr
        results_table["NoiseType"] = noise_type
        results_table["Filter"] = filters[i]
        frc_final_table.append(results_table)

        frc[i+1] = op.frc(filtered_images_a[i], filtered_images_b[i], bins, pixelsize=25)
    
    frc[0] = op.frc(noisy_img_a, noisy_img_b, bins, pixelsize=25)
    # Plot the FRC curve
    plt.figure(figsize=(6, 4))

    plt.plot(frc[0, 1], frc[0, 0], label='Noisy', alpha=0.5)
    for i, filt in enumerate(filters):
        plt.plot(frc[i+1, 1], frc[i+1, 0], label=filt, alpha=0.5)


    plt.axhline(y=1/7, color='grey', linestyle='--', label="1/7 Threshold")
    plt.xlabel("Spatial Frequency (cycles per um)")
    plt.xlim(0, 8)
    plt.ylabel("FRC")
    plt.title(f"Fourier Ring Correlation (SNR={snr}, {noise_type})")
    plt.legend()
    plt.grid()
    
    plt.savefig(os.path.join(current_dir, f"results/plots/optimum_params_fixed_psf/frc/FRC_curve_snr{snr}_{noise_type}.png"))
    plt.close()

    # Compute PSNR and SSIM for each
    psnr_values = [np.mean([ski.metrics.peak_signal_noise_ratio(gt, filtered_images_a[i], data_range=1),
                            ski.metrics.peak_signal_noise_ratio(gt, filtered_images_b[i], data_range=1)]) for i in range(len(filtered_images_a))]
    ssim_values = [np.mean([ski.metrics.structural_similarity(gt, filtered_images_a[i], data_range=1),
                            ski.metrics.structural_similarity(gt, filtered_images_b[i], data_range=1)]) for i in range(len(filtered_images_a))]
    # Noisy PSNR and SSIM
    noisy_psnr = np.mean([ski.metrics.peak_signal_noise_ratio(gt, noisy_img_a, data_range=1),
                            ski.metrics.peak_signal_noise_ratio(gt, noisy_img_b, data_range=1)])
    noisy_ssim = np.mean([ski.metrics.structural_similarity(gt, noisy_img_a, data_range=1),
                            ski.metrics.structural_similarity(gt, noisy_img_b, data_range=1)])

    # Save to final table
    for i, filt in enumerate(filters):
        final_table.append({
            "SNR": snr,
            "NoiseType": noise_type,
            "Filter": filt,
            "PSNR": psnr_values[i],
            "SSIM": ssim_values[i]
        })

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.bar(filters, psnr_values, color=cool_color[:4])
    ax1.axhline(y=noisy_psnr, color='black', linestyle='--', label='Noisy PSNR', alpha=0.8)
    ax1.set_ylim(0, 60)
    ax1.set_title(f"PSNR Comparison (SNR={snr}, {noise_type})")
    ax1.set_ylabel("PSNR")
    ax1.set_xlabel("Filter")
    ax1.legend()

    for i, v in enumerate(psnr_values):
        ax1.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_cool_color)

    ax2.bar(filters, ssim_values, color=warm_color[:4])
    ax2.axhline(y=noisy_ssim, color='black', linestyle='--', label='Noisy SSIM', alpha=0.8)
    ax2.set_ylim(0, 1.2)
    ax2.set_title(f"SSIM Comparison (SNR={snr}, {noise_type})")
    ax2.set_ylabel("SSIM")
    ax2.set_xlabel("Filter")
    ax2.legend()

    for i, v in enumerate(ssim_values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10, color=base_warm_color)

    fig.suptitle(f"Filtered Performance for SNR={snr}, Noise={noise_type}", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(current_dir, f"results/plots/optimum_params_fixed_psf/refilter_performance/performance_snr{snr}_{noise_type}.png"))
    plt.close(fig)

    print(f"Filters compared (SNR={snr}, Noise={noise_type})")

# Save final table
final_df = pd.DataFrame(final_table)
final_df.to_csv(os.path.join(current_dir, 'results/plots/optimum_params_fixed_psf/refilter_performance/final_performance_table.csv'), index=False)

frc_final_df = pd.concat(frc_final_table, ignore_index=True)
frc_final_df.to_csv(os.path.join(current_dir, 'results/plots/optimum_params_fixed_psf/frc/final_frc_results_table.csv'), index=False)

# Save all filtered stacks
# Sequence: SNR -> Noise -> Filter

all_filtered_stack_a = np.concatenate(all_filtered_stack_a, axis=0)
all_filtered_stack_b = np.concatenate(all_filtered_stack_b, axis=0)
tiff.imwrite(os.path.join(current_dir, f"results/stacks/all_filtered_stack_a.tif"), all_filtered_stack_a.astype(np.float32), imagej=True)
tiff.imwrite(os.path.join(current_dir, f"results/stacks/all_filtered_stack_b.tif"), all_filtered_stack_b.astype(np.float32), imagej=True)