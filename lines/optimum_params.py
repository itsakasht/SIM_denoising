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

img = widefield

# Add different amounts of noise to the same image
SNR = [2, 5, 10, 20, 50, 100]
noise_types = ['Gaussian', 'Poisson', 'SaltPepper', 'Mix']

# Load 2 different types of noisy images
noisy = np.zeros((2, len(SNR), len(noise_types), img.shape[0], img.shape[1]))
for j in range(2):
    for i in range(len(SNR)):
        image = ski.io.imread(os.path.join(current_dir, f"results/noisy_images/noisy_wf_{j}_snr{SNR[i]}.tif"))
        noisy[j, i] = np.moveaxis(image, -1, 0)

# Filter parameters

results = []  # New list to collect all best results

psf_w = [op.gaussian_psf(3),
         op.gaussian_psf(7),
         op.gaussian_psf(9),
         op.gaussian_psf(11),
         op.incoherent_psf(psfsize=21, wavelength=480, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=360, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=240, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=120, NA=1.2, pixelsize=25),
         ]  # PSF for Wiener filter

psf_r = [op.gaussian_psf(13),
         op.gaussian_psf(17),
         op.gaussian_psf(19),
         op.gaussian_psf(21),
         op.incoherent_psf(psfsize=21, wavelength=480, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=360, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=240, NA=1.2, pixelsize=25),
         op.incoherent_psf(psfsize=21, wavelength=120, NA=1.2, pixelsize=25),
         ]  # PSF for RL filter


balance = np.array([0.1, 1, 2, 5, 10, 20]) # Wiener filtering
weight = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]) # TV filtering
iterations = np.array([1, 2, 3, 4, 5, 10, 20]) # RL filtering
sigma_psd = np.array([0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 1, 2, 5]) # BM3D filtering

base_cool_color = "#1f77a4"
base_warm_color = "#e51212"

cool_color = ["#7b17eb", "#004aad", "#009fbc", "#00bf97"]
warm_color = ["#e51212", "#a55318", "#ff8c2f", "#ffd323"]


for i in range(len(SNR)):
    for j in range(len(noise_types)):
        noisy_psnr = np.mean([ski.metrics.peak_signal_noise_ratio(img, noisy[0, i, j], data_range=1),
                              ski.metrics.peak_signal_noise_ratio(img, noisy[1, i, j], data_range=1)])
        noisy_ssim = np.mean([ski.metrics.structural_similarity(img, noisy[0, i, j], data_range=1),
                              ski.metrics.structural_similarity(img, noisy[1, i, j], data_range=1)])

        # Wiener filtering
        w_psnr = np.zeros(len(psf_w)*len(balance))
        w_ssim = np.zeros(len(psf_w)*len(balance))

        for l in range(len(psf_w)):
            for k in range(len(balance)):
                wiener0 = ski.restoration.wiener(noisy[0, i, j], psf_w[l], balance=balance[k])
                wiener1 = ski.restoration.wiener(noisy[1, i, j], psf_w[l], balance=balance[k])

                w_psnr[len(balance)*l+k] = np.mean([ski.metrics.peak_signal_noise_ratio(img, wiener0, data_range=1),
                                                    ski.metrics.peak_signal_noise_ratio(img, wiener1, data_range=1)])
                w_ssim[len(balance)*l+k] = np.mean([ski.metrics.structural_similarity(img, wiener0, data_range=1),
                                                    ski.metrics.structural_similarity(img, wiener1, data_range=1)])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(balance, w_psnr[0:len(balance)], label=f'Gaussian PSF 3', color=cool_color[0])
        ax1.plot(balance, w_psnr[len(balance):2*len(balance)], label=f'Gaussian PSF 7', color=cool_color[1])
        ax1.plot(balance, w_psnr[2*len(balance):3*len(balance)], label=f'Gaussian PSF 9', color=cool_color[2])
        ax1.plot(balance, w_psnr[3*len(balance):4*len(balance)], label=f'Gaussian PSF 11', color=cool_color[3])
        ax1.plot(balance, w_psnr[4*len(balance):5*len(balance)], label=f'Incoherent PSF 480nm', color=warm_color[0])
        ax1.plot(balance, w_psnr[5*len(balance):6*len(balance)], label=f'Incoherent PSF 360nm', color=warm_color[1])
        ax1.plot(balance, w_psnr[6*len(balance):7*len(balance)], label=f'Incoherent PSF 240nm', color=warm_color[2])
        ax1.plot(balance, w_psnr[7*len(balance):], label=f'Incoherent PSF 120nm', color=warm_color[3])

        ax1.axhline(y=noisy_psnr, color='black', linestyle='--', label='Noisy PSNR', alpha=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Balance parameter')
        ax1.set_ylabel('PSNR')
        ax1.set_ylim(0, 50)
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y')

        
        ax2.plot(balance, w_ssim[0:len(balance)], label=f'Gaussian PSF 3', color=cool_color[0])
        ax2.plot(balance, w_ssim[len(balance):2*len(balance)], label=f'Gaussian PSF 7', color=cool_color[1])
        ax2.plot(balance, w_ssim[2*len(balance):3*len(balance)], label=f'Gaussian PSF 9', color=cool_color[2])
        ax2.plot(balance, w_ssim[3*len(balance):4*len(balance)], label=f'Gaussian PSF 11', color=cool_color[3])
        ax2.plot(balance, w_ssim[4*len(balance):5*len(balance)], label=f'Incoherent PSF 480nm', color=warm_color[0])
        ax2.plot(balance, w_ssim[5*len(balance):6*len(balance)], label=f'Incoherent PSF 360nm', color=warm_color[1])
        ax2.plot(balance, w_ssim[6*len(balance):7*len(balance)], label=f'Incoherent PSF 240nm', color=warm_color[2])
        ax2.plot(balance, w_ssim[7*len(balance):], label=f'Incoherent PSF 120nm', color=warm_color[3])

        ax2.axhline(y=noisy_ssim, color='black', linestyle='--', label='Noisy SSIM', alpha=0.8)
        ax2.set_xscale('log')
        ax2.set_xlabel('Balance parameter')
        ax2.set_ylabel('SSIM')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.tick_params(axis='y')

        fig.suptitle(f"Wiener Filter (SNR={SNR[i]}, {noise_types[j]})", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f"results/plots/optimum_params/wiener_snr{SNR[i]}_{noise_types[j]}_balance.png"))
        plt.close(fig)

        # Total variation filtering
        t_psnr = np.zeros(len(weight))
        t_ssim = np.zeros(len(weight))
        
        for k in range(len(weight)):
            tv0 = ski.restoration.denoise_tv_chambolle(noisy[0, i, j], weight=weight[k])
            tv1 = ski.restoration.denoise_tv_chambolle(noisy[1, i, j], weight=weight[k])

            t_psnr[k] = np.mean([ski.metrics.peak_signal_noise_ratio(img, tv0, data_range=1),
                                 ski.metrics.peak_signal_noise_ratio(img, tv1, data_range=1)])
            t_ssim[k] = np.mean([ski.metrics.structural_similarity(img, tv0, data_range=1),
                                 ski.metrics.structural_similarity(img, tv1, data_range=1)])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(weight, t_psnr, label='PSNR', color=cool_color[1])
        ax1.axhline(y=noisy_psnr, color='black', linestyle='--', label='Noisy PSNR', alpha=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Weight parameter')
        ax1.set_ylabel('PSNR')
        ax1.set_ylim(0, 50)
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y')

        ax2.plot(weight, t_ssim, label='SSIM', color=warm_color[0])
        ax2.axhline(y=noisy_ssim, color='black', linestyle='--', label='Noisy SSIM', alpha=0.8)
        ax2.set_xscale('log')
        ax2.set_xlabel('Weight parameter')
        ax2.set_ylabel('SSIM')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='y')

        fig.suptitle(f"Total Variation Filter (SNR={SNR[i]}, {noise_types[j]})", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f"results/plots/optimum_params/tv_snr{SNR[i]}_{noise_types[j]}_weight.png"))
        plt.close(fig)
        

        # Richardson Lucy filtering
        r_psnr = np.zeros(len(psf_r)*len(iterations))
        r_ssim = np.zeros(len(psf_r)*len(iterations))
        
        for l in range(len(psf_r)):
            for k in range(len(iterations)):
                rl0 = ski.restoration.richardson_lucy(noisy[0, i, j], psf_r[l], num_iter=iterations[k], filter_epsilon=1e-10)
                rl1 = ski.restoration.richardson_lucy(noisy[1, i, j], psf_r[l], num_iter=iterations[k], filter_epsilon=1e-10)
                
                r_psnr[len(iterations)*l+k] = np.mean([ski.metrics.peak_signal_noise_ratio(img, rl0, data_range=1),
                                                       ski.metrics.peak_signal_noise_ratio(img, rl1, data_range=1)])
                r_ssim[len(iterations)*l+k] = np.mean([ski.metrics.structural_similarity(img, rl0, data_range=1),
                                                       ski.metrics.structural_similarity(img, rl1, data_range=1)])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(iterations, r_psnr[0:len(iterations)], label=f'Gaussian 13', color=cool_color[0])
        ax1.plot(iterations, r_psnr[len(iterations):2*len(iterations)], label=f'Gaussian 17', color=cool_color[1])
        ax1.plot(iterations, r_psnr[2*len(iterations):3*len(iterations)], label=f'Gaussian 19', color=cool_color[2])
        ax1.plot(iterations, r_psnr[3*len(iterations):4*len(iterations)], label=f'Gaussian 21', color=cool_color[3])
        ax1.plot(iterations, r_psnr[4*len(iterations):5*len(iterations)], label=f'Incoherent PSF 480nm', color=warm_color[0])
        ax1.plot(iterations, r_psnr[5*len(iterations):6*len(iterations)], label=f'Incoherent PSF 360nm', color=warm_color[1])
        ax1.plot(iterations, r_psnr[6*len(iterations):7*len(iterations)], label=f'Incoherent PSF 240nm', color=warm_color[2])
        ax1.plot(iterations, r_psnr[7*len(iterations):], label=f'Incoherent PSF 120nm', color=warm_color[3])

        ax1.axhline(y=noisy_psnr, color='black', linestyle='--', label='Noisy PSNR', alpha=0.8)
        ax1.set_xlabel('Number of iterations')
        ax1.set_xlim(left=0)
        ax1.set_ylabel('PSNR')
        ax1.set_ylim(0, 50)
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y')

        ax2.plot(iterations, r_ssim[0:len(iterations)], label=f'Gaussian 13', color=cool_color[0])
        ax2.plot(iterations, r_ssim[len(iterations):2*len(iterations)], label=f'Gaussian 17', color=cool_color[1])
        ax2.plot(iterations, r_ssim[2*len(iterations):3*len(iterations)], label=f'Gaussian 19', color=cool_color[2])
        ax2.plot(iterations, r_ssim[3*len(iterations):4*len(iterations)], label=f'Gaussian 21', color=cool_color[3])
        ax2.plot(iterations, r_ssim[4*len(iterations):5*len(iterations)], label=f'Incoherent PSF 480nm', color=warm_color[0])
        ax2.plot(iterations, r_ssim[5*len(iterations):6*len(iterations)], label=f'Incoherent PSF 360nm', color=warm_color[1])
        ax2.plot(iterations, r_ssim[6*len(iterations):7*len(iterations)], label=f'Incoherent PSF 240nm', color=warm_color[2])
        ax2.plot(iterations, r_ssim[7*len(iterations):], label=f'Incoherent PSF 120nm', color=warm_color[3])
        
        ax2.axhline(y=noisy_ssim, color='black', linestyle='--', label='Noisy SSIM', alpha=0.8)
        ax2.set_xlabel('Number of iterations')
        ax2.set_xlim(left=0)
        ax2.set_ylabel('SSIM')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='y')

        fig.suptitle(f"Richardson Lucy Filter (SNR={SNR[i]}, {noise_types[j]})", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f"results/plots/optimum_params/rl_snr{SNR[i]}_{noise_types[j]}_iterations.png"))
        plt.close(fig)
        


        # BM3d filtering
        b_psnr = np.zeros(2*len(sigma_psd))
        b_ssim = np.zeros(2*len(sigma_psd))
        sigma_est = np.mean((ski.restoration.estimate_sigma(noisy[0, i, j], channel_axis=None, average_sigmas=True),
                             ski.restoration.estimate_sigma(noisy[1, i, j], channel_axis=None, average_sigmas=True)))
        
        for l in range(2):
            for k in range(len(sigma_psd)):
                if l == 0:
                    bm0 = bm3d.bm3d(noisy[0, i, j], sigma_psd[k], stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
                    bm1 = bm3d.bm3d(noisy[1, i, j], sigma_psd[k], stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
                else:
                    bm0 = bm3d.bm3d(noisy[0, i, j], sigma_psd[k], stage_arg=bm3d.BM3DStages.ALL_STAGES)
                    bm1 = bm3d.bm3d(noisy[1, i, j], sigma_psd[k], stage_arg=bm3d.BM3DStages.ALL_STAGES)
                b_psnr[2*l+k] = np.mean([ski.metrics.peak_signal_noise_ratio(img, bm0, data_range=1),
                                         ski.metrics.peak_signal_noise_ratio(img, bm1, data_range=1)])
                b_ssim[2*l+k] = np.mean([ski.metrics.structural_similarity(img, bm0, data_range=1),
                                         ski.metrics.structural_similarity(img, bm1, data_range=1)])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(sigma_psd, b_psnr[0:len(sigma_psd)], label='PSNR HARD_THRESHOLDING', color=cool_color[0])
        ax1.plot(sigma_psd, b_psnr[len(sigma_psd):], label='PSNR ALL_STAGES', color=cool_color[1])
        ax1.axhline(y=noisy_psnr, color='black', linestyle='--', label='Noisy PSNR', alpha=0.8)
        ax1.axvline(x=sigma_est, color='gray', linestyle='--', label='Estimated sigma', alpha=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Sigma estimate')
        ax1.set_xlim(left=sigma_psd[0], right=sigma_psd[-1])
        ax1.set_ylabel('PSNR')
        ax1.set_ylim(0, 50)
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y')

        ax2.plot(sigma_psd, b_ssim[0:len(sigma_psd)], label='SSIM HARD_THRESHOLDING', color=warm_color[0])
        ax2.plot(sigma_psd, b_ssim[len(sigma_psd):], label='SSIM ALL_STAGES', color=warm_color[2])
        ax2.axhline(y=noisy_ssim, color='black', linestyle='--', label='Noisy SSIM', alpha=0.8)
        ax2.axvline(x=sigma_est, color='gray', linestyle='--', label='Estimated sigma', alpha=0.8)
        ax2.set_xscale('log')
        ax2.set_xlabel('Sigma estimate')
        ax2.set_ylabel('SSIM')
        ax2.set_xlim(left=sigma_psd[0], right=sigma_psd[-1])
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='y')

        fig.suptitle(f"BM3D Filter (SNR={SNR[i]}, {noise_types[j]})", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f"results/plots/optimum_params/bm3d_snr{SNR[i]}_{noise_types[j]}_sigma.png"))
        plt.close(fig)

        # Collect best parameters

        # For Wiener filter
        w_best_idx = np.argmax(w_ssim)
        w_best_balance = balance[w_best_idx % len(balance)]
        number = w_best_idx // len(balance)
        if number == 0:
            w_best_psf = f'Gaussian 13'
        elif number == 1:
            w_best_psf = f'Gaussian 17'
        elif number == 2:
            w_best_psf = f'Gaussian 19'
        elif number == 3:
            w_best_psf = f'Gaussian 21'
        elif number == 4:
            w_best_psf = f'Incoherent 480nm'
        elif number == 5:
            w_best_psf = f'Incoherent 360nm'
        elif number == 6:
            w_best_psf = f'Incoherent 240nm'
        elif number == 7:
            w_best_psf = f'Incoherent 120nm'
        w_best_psnr = w_psnr[w_best_idx]
        w_best_ssim = w_ssim[w_best_idx]

        # For TV filter
        t_best_idx = np.argmax(t_ssim)
        t_best_weight = weight[t_best_idx]
        t_best_psnr = t_psnr[t_best_idx]
        t_best_ssim = t_ssim[t_best_idx]

        # For RL filter
        r_best_idx = np.argmax(r_ssim)
        r_best_iter = iterations[r_best_idx % len(iterations)]
        number = r_best_idx // len(iterations)
        if number == 0:
            r_best_psf = f'Gaussian 3'
        elif number == 1:
            r_best_psf = f'Gaussian 7'
        elif number == 2:
            r_best_psf = f'Gaussian 9'
        elif number == 3:
            r_best_psf = f'Gaussian 11'
        elif number == 4:
            r_best_psf = f'Incoherent 480nm'
        elif number == 5:
            r_best_psf = f'Incoherent 360nm'
        elif number == 6:
            r_best_psf = f'Incoherent 240nm'
        elif number == 7:
            r_best_psf = f'Incoherent 120nm'

        r_best_psnr = r_psnr[r_best_idx]
        r_best_ssim = r_ssim[r_best_idx]

        # For BM3D filter
        b_best_idx = np.argmax(b_ssim[0:len(sigma_psd)]) # Use only HARD_THRESHOLDING
        b_best_sigma = sigma_psd[b_best_idx % len(sigma_psd)]
        b_best_stage = "HARD_THRESHOLDING" # if b_best_idx < len(sigma_psd) else "ALL_STAGES"
        b_best_psnr = b_psnr[b_best_idx]
        b_best_ssim = b_ssim[b_best_idx]

        # Save all
        results.append({
            "SNR": SNR[i],
            "NoiseType": noise_types[j],
            "Noisy_PSNR": noisy_psnr,
            "Noisy_SSIM": noisy_ssim,
            
            "Wiener_PSNR": w_best_psnr,
            "Wiener_SSIM": w_best_ssim,
            "Wiener_Balance": w_best_balance,
            "Wiener_PSF": w_best_psf,

            "TV_PSNR": t_best_psnr,
            "TV_SSIM": t_best_ssim,
            "TV_Weight": t_best_weight,

            "RL_PSNR": r_best_psnr,
            "RL_SSIM": r_best_ssim,
            "RL_Iterations": r_best_iter,
            "RL_PSF": r_best_psf,

            "BM3D_PSNR": b_best_psnr,
            "BM3D_SSIM": b_best_ssim,
            "BM3D_Sigma": b_best_sigma,
            "BM3D_Estimated_Sigma": sigma_est,
            "BM3D_Stage": b_best_stage,
        })

        print(f"Found optimum parameters ({noise_types[j]}, SNR={SNR[i]})")

df = pd.DataFrame(results)
df.to_csv(os.path.join(current_dir, "results/plots/optimum_params/optimum_filter_parameters.csv"), index=False)

end_time = time.perf_counter()
print(f"Execution time: {(end_time - start_time)/60:.2f} minutes")