import numpy as np
import matplotlib.pyplot as plt

def grating(size, angle, phase, wavelength=600, pixelsize=100):
    # wavelength in pixels, angle in degree, phase in degree
    wavenumber = pixelsize / wavelength
    x = np.arange(size)
    y = np.arange(size)
    x, y = np.meshgrid(x, y, sparse=True)
    return 0.5 + 0.45*np.sin(2*np.pi*wavenumber*(x*np.cos(np.pi/180*angle) + y*np.sin(np.pi/180*angle)) + np.pi*phase/180)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def fourier_downsample(images):
    """
    Downsamples an image by a factor of 2 using Fourier domain cropping.
    
    Parameters:
        img (numpy.ndarray): Input image to be downsampled.
    Returns:
        np.ndarray: Downsampled image as a NumPy array.
    """
    # Ensure input is at least 3D (N, H, W) for batch processing
    single_image = False
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Convert (H, W) to (1, H, W)
        single_image = True
    
    output_images = np.zeros((images.shape[0], images.shape[1]//2, images.shape[2]//2), dtype=images.dtype)
    for i in range(images.shape[0]):
        img = images[i]

        # Get original shape
        h, w = img.shape

        # Perform Fourier transform and shift the zero frequency to the center
        f_shift = np.fft.fftshift(np.fft.fft2(img))

        # Crop the center part of the frequency domain
        start_h = (h - h//2) // 2
        end_h = start_h + h//2
        start_w = (w - w//2) // 2
        end_w = start_w + w//2
        cropped = f_shift[start_h:end_h, start_w:end_w]

        # Inverse shift and inverse FFT to return to spatial domain
        img_back = np.fft.ifft2(np.fft.ifftshift(cropped))
        img_downsampled = np.abs(img_back)



        # Normalize to 0-255 and convert to uint8
        img_downsampled = normalize(img_downsampled)
        output_images[i] = img_downsampled.astype(img.dtype)

    return output_images[0] if single_image else output_images

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

def incoherent_psf(psfsize=65, NA=1.2, wavelength=500, pixelsize=100):
    """
    Computes the Incoherent Point Spread Function (PSF) for a given size, NA, wavelength, and pixel size.

    Parameters:
        size (int): Size of the PSF (size x size).
        NA (float): Numerical Aperture of the system.
        wavelength (float): Wavelength of the light in nanometers.
        pixelsize (float): Pixel size in nanometers.

    Returns:
        numpy array: The computed PSF.
    """
    size = 8*psfsize + 1
    k0 = 2 * np.pi / wavelength
    cutoff = NA * k0  # in radians/nanometer

    # Generate spatial frequency grids in radians/nanometer
    kx = np.fft.fftshift(np.fft.fftfreq(size, d=pixelsize)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(size, d=pixelsize)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2

    # Coherent Transfer Function (binary circular mask)
    CTF = np.zeros_like(K_squared)
    CTF[K_squared <= (cutoff**2)] = 1
    # print(CTF.max(), CTF.min())

    # Incoherent Transfer Function (OTF)
    cpsf = np.fft.ifft2(np.fft.ifftshift(CTF))
    ipsf =np.fft.fftshift(np.abs(cpsf)**2)  # Square of the amplitude of the coherent PSF
    ipsf = ipsf[size//2 - psfsize//2:size//2 + psfsize//2 + 1, size//2 - psfsize//2:size//2 + psfsize//2 + 1]  # Crop to desired size
    # print(ipsf.shape)
    ipsf /= np.sum(ipsf)  # Normalize

    return ipsf

# Optical Transfer Function (OTF) for coherent and incoherent imaging systems
def otf_coherent(images, NA=1.2, wavelength=500, pixelsize=100, show_spectrum=False):
    """
    Computes the Optical Transfer Function (OTF) for a coherent imaging system
    and applies it as a low-pass filter to the given images in the Fourier domain.

    Parameters:
        images (numpy array): Single image (H, W) or batch of images (N, H, W).
        NA (float): Numerical Aperture of the system.
        wavelength (float): Wavelength of the light in nanometers.
        show_spectrum (bool): If True, displays the filtered spectrum.
    
    Returns:
        numpy array: The filtered images.
    """
    data_type = images.dtype
    # Ensure input is at least 3D (N, H, W) for batch processing
    single_image = False
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Convert (H, W) to (1, H, W)
        single_image = True

    N, H, W = images.shape
    k0 = 2 * np.pi / wavelength
    cutoff = NA * k0  # in radians/nanometer
    
    # Generate spatial frequency grids in radians/nanometer
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=pixelsize)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=pixelsize)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2

    # Coherent Transfer Function (binary circular mask)
    CTF = np.zeros_like(K_squared)
    CTF[K_squared <= cutoff**2] = 1
    

    output_images = np.zeros_like(images, dtype=data_type)
    for i in range(N):
        intensity = images[i]
        amplitude = np.sqrt(intensity)

        fft_amp = np.fft.fftshift(np.fft.fft2(amplitude))
        fft_filtered = fft_amp * CTF

        # Show filtered spectrum
        if show_spectrum:
            plt.figure()
            plt.imshow(np.log(np.abs(fft_filtered) + 1), extent=[kx[0], kx[-1], ky[0], ky[-1]], cmap='gray')
            plt.title('Filtered Spectrum (log scale)')
            plt.xlabel('kx (rad/nm)')
            plt.ylabel('ky (rad/nm)')
            plt.colorbar()
            plt.tight_layout()
            # plt.show()

        # Inverse FFT and intensity reconstruction
        amp_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
        intensity_filtered = np.abs(amp_filtered)**2

        output_images[i] = normalize(intensity_filtered)
    
    return output_images[0] if single_image else output_images


def otf_incoherent(images, NA=1.2, wavelength=500, pixelsize=100, show_spectrum=False, compare_analytic=False):
    """
    Computes the Optical Transfer Function (OTF) for an incoherent imaging system
    and applies it as a low-pass filter to the given images in the Fourier domain.

    Parameters:
        images (numpy array): Single image (H, W) or batch of images (N, H, W).
        NA (float): Numerical Aperture of the system.
        wavelength (float): Wavelength of the light in nanometers.
        show_spectrum (bool): If True, displays the filtered spectrum.
    
    Returns:
        numpy array: The filtered images.
    """
    data_type = images.dtype
    # Ensure input is at least 3D (N, H, W) for batch processing
    single_image = False
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Convert (H, W) to (1, H, W)
        single_image = True

    N, H, W = images.shape
    k0 = 2 * np.pi / wavelength
    cutoff = NA * k0  # in radians/nanometer
    
    # Generate spatial frequency grids in radians/nanometer
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=pixelsize)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=pixelsize)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2

    # Coherent Transfer Function (binary circular mask)
    CTF = np.zeros_like(K_squared)
    CTF[K_squared <= cutoff**2] = 1

    # Incoherent Transfer Function (OTF)
    cpsf = np.fft.ifft2(np.fft.ifftshift(CTF))
    ipsf = np.abs(cpsf)**2  # Square of the amplitude of the coherent PSF
    OTF = np.abs(np.fft.fftshift(np.fft.fft2(ipsf)))  # Fourier transform of the incoherent PSF
    OTF = OTF / np.max(OTF)  # Normalize

    if compare_analytic:
        f = np.sqrt(K_squared) / (2 * np.pi)  # Convert rad/nm → cyc/nm
        fc = 2*NA / wavelength  # Proper analytic cutoff in cyc/nm

        analytic_otf = np.zeros_like(f)
        mask = f <= fc
        analytic_otf[mask] = (2 / np.pi) * (np.arccos(f[mask] / fc) - (f[mask] / fc) * np.sqrt(1 - (f[mask] / fc)**2))
        analytic_otf = normalize(analytic_otf)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(OTF, cmap='viridis'); plt.title('Numerical OTF')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(analytic_otf, cmap='viridis'); plt.title('Analytic OTF')
        plt.colorbar()
        plt.suptitle("OTF Comparison (cyc/nm)")
        plt.tight_layout()

        plt.figure()
        plt.plot(kx, OTF[512, :].T, label='Numerical OTF', color='yellow', alpha=0.5)
        # plt.plot(kx, analytic_otf[512, :].T, label='Analytic OTF', color='green', alpha=0.5)
        plt.title('OTF Comparison (1D)')
        plt.xlabel('kx (rad/nm)')
        plt.ylabel('OTF')
        plt.legend()
        plt.tight_layout()

    output_images = np.zeros_like(images, dtype=data_type)
    for i in range(N):
        intensity = images[i]
        fft_intensity = np.fft.fftshift(np.fft.fft2(intensity))  # Compute FFT and shift zero frequency to center

        fft_filtered = fft_intensity * OTF  # Apply OTF
        intensity_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))  # Compute inverse FFT

        # Show filtered spectrum
        if show_spectrum:
            plt.figure()
            plt.imshow(np.log(np.abs(fft_filtered) + 1), extent=[kx[0], kx[-1], ky[0], ky[-1]], cmap='gray')
            plt.title('Filtered Spectrum (log scale)')
            plt.xlabel('kx (rad/nm)')
            plt.ylabel('ky (rad/nm)')
            plt.colorbar()
            plt.tight_layout()
            # plt.show()

        # print(np.imag(filtered_intensity).min(), np.imag(filtered_intensity).max(), np.imag(filtered_intensity).mean())

        output_images[i] = normalize(intensity_filtered)  # Normalize
    
    return output_images[0] if single_image else output_images
    
def display_fourier(image):
    """Display the Fourier transform of an image."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Spacial domain')
    
    ft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log(np.abs(ft) + 1)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Frequency domain')
    plt.axis('off')
    # plt.show()

# FRC curve
def frc(image1, image2, num_bins=50, pixelsize=25):
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
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=pixelsize/1000))
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=pixelsize/1000))
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

# FRC
def radial_profile(data):
    """Compute radial average (for FRC curve)."""
    y, x = np.indices(data.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)

    return radialprofile



def compute_true_frc(sub1, sub2):
    """Compute true FRC curve and cutoff frequency."""
    fft1 = np.fft.fftshift(np.fft.fft2(sub1))
    fft2_ = np.fft.fftshift(np.fft.fft2(sub2))

    numerator = np.real(fft1 * np.conj(fft2_))
    denom = np.sqrt(np.abs(fft1)**2 * np.abs(fft2_)**2)

    frc_map = numerator / (denom + 1e-8)  # Prevent divide-by-zero
    frc_curve = radial_profile(frc_map)

    return frc_curve

def find_cutoff(frc_curve, threshold=1/7):
    """Find frequency where FRC crosses below threshold."""
    for idx, val in enumerate(frc_curve):
        if val < threshold:
            return idx
    return len(frc_curve) - 1  # If it never drops below

def frc_map(img1, img2, block_size, pixel_size_nm=10):
    """Compute the FRC map for the image."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    h, w = img1.shape

    n_blocks_y = h // block_size
    n_blocks_x = w // block_size

    resolution_map = np.zeros((n_blocks_y, n_blocks_x))

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            y0, y1 = i * block_size, (i+1) * block_size
            x0, x1 = j * block_size, (j+1) * block_size
            sub1 = img1[y0:y1, x0:x1]
            sub2 = img2[y0:y1, x0:x1]

            frc_curve = compute_true_frc(sub1, sub2)
            cutoff = find_cutoff(frc_curve)

            # Convert cutoff to resolution (nm):
            # max frequency = block_size/2 (Nyquist)
            # Resolution (nm) = (pixel_size_nm * block_size) / (2 * cutoff)
            if cutoff == 0:
                res = np.inf  # if never crossed threshold
            else:
                res = (pixel_size_nm * block_size) / (2 * cutoff)
            resolution_map[i, j] = res

    return resolution_map

def analyze_resolution(resolution_map):
    """Print statistics for the resolution map."""
    valid = resolution_map[np.isfinite(resolution_map)]

    print(f"--- Resolution Map Statistics ---")
    print(f"N-blocks: {valid.size}")
    print(f"Mean Resolution (nm): {np.mean(valid):.2f}")
    print(f"Std-Dev (nm): {np.std(valid):.2f}")
    print(f"Min Resolution (nm): {np.min(valid):.2f}")
    print(f"Max Resolution (nm): {np.max(valid):.2f}")

def plot_resolution_map(resolution_map):
    """Plot the resolution map."""
    plt.figure(figsize=(8, 6))
    plt.imshow(resolution_map, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Resolution (nm)')
    plt.title('FRC Resolution Map')
    plt.show()

### OLD CODE ###

'''
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

        print(filtered_image.min(), filtered_image.max(), filtered_image.mean())

        # Normalize the filtered image
        output_images[i] = normalize(filtered_image)

    return output_images[0] if single_image else output_images
'''