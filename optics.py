import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski

def grating(size, wavelength, angle, phase):
    # wavelength in pixels, angle in degree, phase in degree
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y, sparse=True)
    return 0.5*(1 + np.sin(2*np.pi*(x*np.cos(np.pi/180*angle) + y*np.sin(np.pi/180*angle))/wavelength + np.pi*phase/180))

def normalize(arr):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    return arr

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
        # plt.imshow(np.log(np.abs(fft_cropped) + 2))
        # plt.show()

        # Compute inverse FFT
        filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_cropped))
        filtered_image = np.real(filtered_image)

        output_images[i] = normalize(filtered_image)

    return output_images[0] if single_image else output_images