import numpy as np

def ensure_batch(images):
    return images[None, ...] if images.ndim == 2 else images

def gauss_noise(images, snr=10, seed=2025):
    rng = np.random.default_rng(seed)  # For reproducibility
    noisy_images = []
    msnr = []
    
    ensure_batch(images)  # Add batch dim
    
    for image in images:
        signal_power = np.mean(image ** 2)
        noise_power = signal_power / snr
        print(f'{np.sqrt(noise_power):.5f}')
        noise = rng.normal(0, np.sqrt(noise_power), image.shape)
        raw_noisy = image + noise
        noisy_image = np.clip(raw_noisy, 0, 1).astype(np.float32)
        noisy_images.append(noisy_image)

        msnr.append(signal_power / np.mean((raw_noisy - image)**2)) # SNR before clipping
        # print('Measured gaussian SNR: ', msnr[-1])
    
    if len(noisy_images) == 1:
        return noisy_images[0], msnr[0]
    else:
        return np.stack(noisy_images), np.array(msnr)


def poisson_noise(images, scale_factor=100 , seed=2025):
    rng = np.random.default_rng(seed)  # For reproducibility
    noisy_images = []
    msnr = []
    
    ensure_batch(images)  # Add batch dim

    for image in images:
        scaled = image * scale_factor
        raw_noisy = rng.poisson(scaled) / scale_factor
        noisy_image = np.clip(raw_noisy, 0, 1).astype(np.float32)
        noisy_images.append(noisy_image)
        
        signal_power = np.mean(image ** 2)
        msnr.append(signal_power / np.mean((raw_noisy - image)**2)) # SNR before clipping
        # print('Measured gaussian SNR: ', msnr[-1])
    
    if len(noisy_images) == 1:
        return noisy_images[0], msnr[0]
    else:
        return np.stack(noisy_images), np.array(msnr)

def salt_pepper_noise(images, snr=10, seed=2025):
    rng = np.random.default_rng(seed)  # For reproducibility
    noisy_images = []
    msnr = []
    
    ensure_batch(images)  # Add batch dim
    
    for image in images:
        signal_power = np.mean(image ** 2)        
        mu = np.mean(image)
        noise_per_pixel = signal_power - mu + 0.5
        density = min(1.0, signal_power / (snr * noise_per_pixel + 1e-8)) # lower density = higher SNR

        noisy_image = image.copy()
        num_pixels = image.size
        num_salt = int(density * num_pixels / 2)
        num_pepper = int(density * num_pixels / 2)
        
        # Generate random pixel indices
        salt_coords = tuple(rng.integers(0, dim, num_salt) for dim in image.shape)
        pepper_coords = tuple(rng.integers(0, dim, num_pepper) for dim in image.shape)

        noisy_image[salt_coords] = 1  # Salt noise (white)
        noisy_image[pepper_coords] = 0  # Pepper noise (black)

        noisy_images.append(noisy_image.astype(np.float32))
        msnr.append(signal_power / np.mean((noisy_image - image)**2))
        # print('Measured gaussian SNR: ', msnr[-1])
    
    if len(noisy_images) == 1:
        return noisy_images[0], msnr[0]
    else:
        return np.stack(noisy_images), np.array(msnr)

def mix_noise(images, snr=10, seed=2025):
    noisy_images = []
    msnr = []
    
    ensure_batch(images)  # Add batch dim

    gauss_snr = 3*snr
    poisson_scale = 6*snr
    sp_snr = 3*snr

    for image in images:
        p, _ = poisson_noise(image, scale_factor=poisson_scale, seed=seed)
        g, _ = gauss_noise(p, gauss_snr, seed=seed)
        s, _ = salt_pepper_noise(g, sp_snr, seed=seed)
        noisy_images.append(s.astype(np.float32))

        signal_power = np.mean(image ** 2)  
        msnr.append(signal_power / np.mean((s - image)**2))
        # print('Measured gaussian SNR: ', msnr[-1])

    if len(noisy_images) == 1:
        return noisy_images[0], msnr[0]
    else:
        return np.stack(noisy_images), np.array(msnr)