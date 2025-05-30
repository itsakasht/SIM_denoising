import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split_image_into_blocks(image, block_size):
    """Split image into non-overlapping blocks."""
    h, w = image.shape
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.shape == (block_size, block_size):
                blocks.append(block)
    return blocks

def compute_frc(block1, block2, pixel_size_nm=25, block_size=32, threshold=1/7):
    """Compute FRC between two blocks and return resolution in nm."""
    F1 = np.fft.fftshift(np.fft.fft2(block1))
    F2 = np.fft.fftshift(np.fft.fft2(block2))
    
    numerator = np.real(F1 * np.conj(F2))
    denominator = np.abs(F1) * np.abs(F2) + 1e-8  # avoid zero division
    
    frc_curve = radial_profile(numerator / denominator)
    
    return threshold_resolution(frc_curve, pixel_size_nm=pixel_size_nm, block_size=block_size, threshold=threshold)

def radial_profile(data):
    """Compute the radial average of a 2D array."""
    y, x = np.indices((data.shape))
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8) # avoid division by zero
    return radialprofile

def threshold_resolution(frc_curve, pixel_size_nm=25, block_size=64, threshold=1/7):
    """Find spatial resolution where FRC curve drops below threshold."""
    n_freqs = len(frc_curve)
    freqs = np.linspace(0, 0.5, n_freqs) # from 0 to Nyquist (0.5 cycles/pixel)

    for idx, val in enumerate(frc_curve):
        if val < threshold and freqs[idx] > 0:
            freq = freqs[idx] # cycles per pixel
            resolution_nm = (1 / freq) * pixel_size_nm
            return resolution_nm # Return resolution in nm

    # If never drops below threshold (best possible)
    return (1 / freqs[-1]) * pixel_size_nm

def create_frc_map(image1, image2, block_size, pixel_size_nm, threshold=1/7):
    """Create FRC map and list of resolutions."""
    blocks1 = split_image_into_blocks(image1, block_size)
    blocks2 = split_image_into_blocks(image2, block_size)

    if len(blocks1) != len(blocks2):
        raise ValueError("Images must have same number of blocks.")

    frc_values = []

    for b1, b2 in zip(blocks1, blocks2):
        res_nm = compute_frc(b1, b2, pixel_size_nm=pixel_size_nm, block_size=block_size, threshold=threshold)
        frc_values.append(res_nm)

    side_blocks = image1.shape[0] // block_size
    frc_array = np.array(frc_values).reshape((side_blocks, side_blocks))

    if 1==0: # Debugging mode
        # Pick a random block to debug
        rand_idx = np.random.randint(0, len(blocks1))
        block1 = blocks1[rand_idx]
        block2 = blocks2[rand_idx]

        F1 = np.fft.fftshift(np.fft.fft2(block1))
        F2 = np.fft.fftshift(np.fft.fft2(block2))

        numerator = np.abs(F1 * np.conj(F2))
        denominator = np.abs(F1) * np.abs(F2) + 1e-8  # avoid zero division
        frc_curve = radial_profile(numerator / denominator)

        # Plot the FRC curve
        n_freqs = len(frc_curve)
        freqs = np.linspace(0, 0.5, n_freqs)

        plt.figure(figsize=(6,4))
        plt.plot(freqs, frc_curve, label='FRC curve')
        plt.axhline(1/7, color='red', linestyle='--', label=f'{threshold} threshold')
        plt.xlabel('Spatial Frequency (cycles/pixel)')
        plt.ylabel('FRC value')
        plt.title('FRC Curve of Random Block')
        plt.legend()
        plt.grid()
        plt.show()

    return frc_array, frc_values

def plot_frc_map(frc_map):
    """Display FRC map with colormap."""
    plt.figure(figsize=(8,6))
    plt.imshow(frc_map, cmap='coolwarm' , origin='lower')
    plt.colorbar(label='Resolution (nm)')
    plt.title('FRC Resolution Map')
    # plt.show()

def generate_results_table(frc_values):
    """Generate statistics table."""
    frc_values = np.array(frc_values)
    data = {
        "N-blocks": [len(frc_values)],
        "Mean (nm)": [np.mean(frc_values)],
        "Std-Dev (nm)": [np.std(frc_values)],
        "Min FRC (nm)": [np.min(frc_values)],
        "Max FRC (nm)": [np.max(frc_values)],
    }
    return pd.DataFrame(data)