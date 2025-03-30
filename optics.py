import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski

def grating(size, angle, phase, NA=1.2, wavelength=500, pixelsize=100):
    # wavelength in pixels, angle in degree, phase in degree
    wavenumber = pixelsize * NA / wavelength
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y, sparse=True)
    return 0.5*(1 + np.sin(2*np.pi*wavenumber*(x*np.cos(np.pi/180*angle) + y*np.sin(np.pi/180*angle)) + np.pi*phase/180))

def normalize(arr):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    return arr

def gaussian_psf(size):
    # Calculate sigma based on the size (use size / 6 as a reasonable choice for sigma)
    sigma = size / 6
    
    # Create coordinate grid centered around 0
    x = np.arange(-(size // 2), size // 2 + 1)
    y = np.arange(-(size // 2), size // 2 + 1)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Gaussian PSF
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf /= np.sum(psf)  # Normalize so that the sum is 1

    return psf

def square_lowpass(input_image, factor):
    ft = np.fft.fft2(input_image)
    ft = np.fft.fftshift(ft)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.log(np.abs(ft)))
    m, n = ft.shape
    ft = ft[(factor-1)*m//(2*factor):(factor+1)*m//(2*factor),
            (factor-1)*n//(2*factor):(factor+1)*n//(2*factor)]
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.log(np.abs(ft)))
    # plt.show()
    ft = np.fft.ifftshift(ft)    
    reconstruct = np.fft.ifft2(ft)  # Inverse FFT
    reconstruct = np.real(reconstruct)
    print(reconstruct.shape, reconstruct.min(), reconstruct.max())
    reconstruct = normalize(reconstruct)
    return reconstruct

def circle_lowpass(images, r):
    """
    Applies a low-pass circular filter in the Fourier domain, crops the result, 
    and returns the inverse FFT of the cropped image.

    Parameters:
        images (numpy array): Single image (H, W) or batch of images (N, H, W).
        r (int): Radius of the low-pass filter.
    
    Returns:
        numpy array: The filtered and cropped images.
    """
    # Ensure input is at least 3D (N, H, W) for batch processing
    single_image = False
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Convert (H, W) to (1, H, W)
        single_image = True

    N, H, W = images.shape
    H_new, W_new = H // 2, W // 2  # Crop dimensions

    # Prepare output array
    output_images = np.zeros((N, H_new, W_new))

    # Generate circular low-pass filter mask
    y, x = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    mask = (x - center[1])**2 + (y - center[0])**2 <= r**2

    for i in range(N):
        img = images[i]

        # Compute FFT
        fft_img = np.fft.fftshift(np.fft.fft2(img))  # Shift zero frequency to center
        
        # Apply circular low-pass filter
        fft_filtered = fft_img * mask

        # Crop to half the size
        fft_cropped = fft_filtered[H_new//2:3*H_new//2, W_new//2:3*W_new//2]

        # Compute inverse FFT
        filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_cropped))
        filtered_image = np.real(filtered_image)

        output_images[i] = normalize(filtered_image)

    return output_images[0] if single_image else output_images

def otf_incoherent(images, NA=1.2, wavelength=500, pixelsize=100):
    """
    Computes the Optical Transfer Function (OTF) for an incoherent imaging system
    and applies it as a low-pass filter to the given images in the Fourier domain.

    Parameters:
        images (numpy array): Single image (H, W) or batch of images (N, H, W).
        NA (float): Numerical Aperture of the system.
        wavelength (float): Wavelength of the light in nanometers.
    
    Returns:
        numpy array: The filtered images.
    """
    # Ensure input is at least 3D (N, H, W) for batch processing
    single_image = False
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Convert (H, W) to (1, H, W)
        single_image = True

    N, H, W = images.shape
    fc = NA / wavelength  # Cutoff frequency
    fx = np.fft.fftshift(np.fft.fftfreq(W, pixelsize))  # Normalized frequency coordinates
    fy = np.fft.fftshift(np.fft.fftfreq(H, pixelsize))
    # print(fc)
    # print(fx)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)  # Spatial frequency magnitude

    # Compute OTF based on the diffraction-limited incoherent system
    OTF = np.zeros_like(f)
    mask = f <= fc
    OTF[mask] = (2 / np.pi) * (np.arccos(f[mask] / fc) - (f[mask] / fc) * np.sqrt(1 - (f[mask] / fc) ** 2))

    output_images = np.zeros((N, H//2, W//2))
    for i in range(N):
        img = images[i]
        fft_img = np.fft.fftshift(np.fft.fft2(img))  # Compute FFT and shift zero frequency to center
        fft_filtered = fft_img * OTF  # Apply OTF
        filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_filtered[H//4:3*H//4, W//4:3*W//4]))  # Compute inverse FFT
        output_images[i] = normalize(np.real(filtered_image))  # Take real part
    
    return output_images[0] if single_image else output_images
    
def display_fourier(image):
    """Display the Fourier transform of an image."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Original Image')
    
    ft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log(np.abs(ft) + 1)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')
    plt.show()

def frc(image1, image2, num_bins=50):
    """
    Computes the Fourier Ring Correlation (FRC) between two images.

    Parameters:
        image1 (numpy array): First input image (H, W).
        image2 (numpy array): Second input image (H, W).
        num_bins (int): Number of radial frequency bins.

    Returns:
        tuple: (FRC values, frequency bins)
    """
    assert image1.shape == image2.shape, "Input images must have the same dimensions"
    
    H, W = image1.shape
    
    # Compute FFT of both images
    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))
    
    # Compute spatial frequency grid
    fx = np.fft.fftshift(np.fft.fftfreq(W) * W)
    fy = np.fft.fftshift(np.fft.fftfreq(H) * H)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)  # Radial frequency values
    
    # Define bins for frequency rings
    max_freq = np.max(f)
    bins = np.linspace(0, max_freq, num_bins + 1)
    frc_values = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (f >= bins[i]) & (f < bins[i + 1])
        
        numerator = np.sum(fft1[mask] * np.conj(fft2[mask]))
        denominator = np.sqrt(np.sum(np.abs(fft1[mask])**2) * np.sum(np.abs(fft2[mask])**2))
        
        if denominator != 0:
            frc_values[i] = np.abs(numerator / denominator)
    
    return frc_values, bins[:-1]