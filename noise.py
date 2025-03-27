import numpy as np

def gauss_noise(images, sigma=0.05):
    """Adds Gaussian noise to an image or a batch of images."""
    gauss = np.random.normal(0, sigma, images.shape)
    return np.clip(images + gauss, 0, 1)

def poisson_noise(images, noise_percent=0.05):
    """Adds Poisson noise to an image or a batch of images."""
    noise = np.random.poisson(images * noise_percent)
    return np.clip(images + noise, 0, 1)

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

def random_noise(images, noise_percent=0.05):
    """Adds uniform random noise to an image or a batch of images."""
    gauss = noise_percent*np.random.ranf(images.shape)
    return np.clip(images + gauss, 0, 1)