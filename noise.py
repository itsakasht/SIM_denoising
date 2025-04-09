import numpy as np

def gauss_noise(images, sigma=0.05):
    """Adds Gaussian noise to an image or a batch of images."""
    gauss_noise = np.random.normal(0, sigma, images.shape)
    return np.clip(images + gauss_noise, 0, 1)

def poisson_noise(images, noise_percent=0.05):
    """Adds Poisson noise to an image or a batch of images."""
    poisson_noise = np.random.poisson(images * noise_percent)
    return np.clip(images + poisson_noise, 0, 1)

def salt_pepper_noise(images, noise_percent=0.05):
    """Adds salt & pepper noise to an image or a batch of images."""
    output = np.copy(images)
    total_pixels = np.prod(images.shape[-2:])  # Works for 2D (grayscale) and 3D (RGB) images
    
    num_salt = int(total_pixels * noise_percent / 2)
    num_pepper = int(total_pixels * noise_percent / 2)
    
    # Generate random pixel indices
    salt_coords = tuple(np.random.randint(0, dim, num_salt) for dim in images.shape)
    pepper_coords = tuple(np.random.randint(0, dim, num_pepper) for dim in images.shape)

    output[salt_coords] = 1  # Salt noise (white)
    output[pepper_coords] = 0  # Pepper noise (black)
    
    return output

def mix_noise(images, noise_percent=0.05):
    """Adds mix noise to an image or a batch of images."""
    gauss_noise = np.random.normal(0, noise_percent/3, images.shape)

    poisson_noise = np.random.poisson(images * noise_percent/3)

    output = np.copy(images)
    total_pixels = np.prod(images.shape[-2:])  # Works for 2D (grayscale) and 3D (RGB) images
    num_salt = int(total_pixels * noise_percent / 6)
    num_pepper = int(total_pixels * noise_percent / 6)
    
    # Generate random pixel indices
    salt_coords = tuple(np.random.randint(0, dim, num_salt) for dim in images.shape)
    pepper_coords = tuple(np.random.randint(0, dim, num_pepper) for dim in images.shape)

    output[salt_coords] = 1  # Salt noise (white)
    output[pepper_coords] = 0  # Pepper noise (black)
    return np.clip(images + gauss_noise + poisson_noise, 0, 1)