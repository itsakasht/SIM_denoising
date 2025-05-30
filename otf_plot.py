import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
wavelength = 480  # in nanometers
NA = 1.2
pixelsize = 25  # in nanometers
W, H = 1024, 1024
pad_factor = 1  # Zero padding factor

# Define padded shape
W_pad = W * pad_factor
H_pad = H * pad_factor

# Spatial frequency grids in radians/nm
kx = np.fft.fftshift(np.fft.fftfreq(W_pad, d=pixelsize)) * 2 * np.pi
ky = np.fft.fftshift(np.fft.fftfreq(H_pad, d=pixelsize)) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2

# Create circular CTF (cutoff spatial frequency)
cutoff = NA * 2 * np.pi / wavelength  # in radians/nm
CTF_padded = np.zeros_like(K_squared)
CTF_padded[K_squared <= cutoff**2] = 1

# Compute Coherent PSF and Incoherent PSF
cpsf = np.fft.ifft2(np.fft.ifftshift(CTF_padded))
cpsf = np.fft.fftshift(cpsf)  # Center the PSF
ipsf = np.abs(cpsf) ** 2

OTF = np.abs(np.fft.fftshift(np.fft.fft2(ipsf)))  # OTF in the spatial domain
OTF = OTF / np.max(OTF)  # Normalize OTF


# Extract central crop for better visualization
mid_y, mid_x = np.array(cpsf.shape) // 2
window = 50

x = np.linspace(-window//2, window//2, window)
y = np.linspace(-window//2, window//2, window)
X, Y = np.meshgrid(x, y)

Z_coherent = np.abs(cpsf[mid_y - window//2 : mid_y + window//2,
                         mid_x - window//2 : mid_x + window//2])
Z_incoherent = ipsf[mid_y - window//2 : mid_y + window//2,
                    mid_x - window//2 : mid_x + window//2]

# Plot PSF
fig = plt.figure(figsize=(14, 6))

# Coherent PSF
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z_coherent, cmap='inferno', edgecolor='none')
ax1.set_title('Coherent PSF (Amplitude)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Amplitude')

# Incoherent PSF
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z_incoherent, cmap='viridis', edgecolor='none')
ax2.set_title('Incoherent PSF (Intensity)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Intensity')

plt.tight_layout()

# otf_window = 512

# Z_coherent_otf = np.abs(CTF_padded[mid_y - otf_window//2 : mid_y + otf_window//2,
#                          mid_x - otf_window//2 : mid_x + otf_window//2])
# Z_incoherent_otf = OTF[mid_y - otf_window//2 : mid_y + otf_window//2,
#                     mid_x - otf_window//2 : mid_x + otf_window//2]

# X_otf = KX[mid_y - otf_window//2 : mid_y + otf_window//2,
#            mid_x - otf_window//2 : mid_x + otf_window//2]
# Y_otf = KY[mid_y - otf_window//2 : mid_y + otf_window//2,
#            mid_x - otf_window//2 : mid_x + otf_window//2]

Z_coherent_otf = np.abs(CTF_padded)
Z_incoherent_otf = OTF
X_otf = KX
Y_otf = KY

print("OTF shape:", Z_coherent_otf.shape, Z_incoherent_otf.shape)

# Plot OTF
fig = plt.figure(figsize=(14, 6))

# Coherent PSF
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_otf, Y_otf, Z_coherent_otf, cmap='inferno', edgecolor='none')
ax1.set_title('Coherent OTF (Amplitude)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Amplitude')

# Incoherent PSF
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_otf, Y_otf, Z_incoherent_otf, cmap='viridis', edgecolor='none')
ax2.set_title('Incoherent OTF (Ampltiude)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Intensity')

plt.tight_layout()

plt.show()