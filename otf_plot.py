import numpy as np
import matplotlib.pyplot as plt


NA = 1.2
wavelength = 400
pixelsize = 100
H = 255
W = 255
fc = NA / wavelength  # Cutoff frequency
fx = np.fft.fftshift(np.fft.fftfreq(W, pixelsize))  # Normalized frequency coordinates
fy = np.fft.fftshift(np.fft.fftfreq(H, pixelsize))

FX, FY = np.meshgrid(fx, fy)
f = np.sqrt(FX**2 + FY**2)  # Spatial frequency magnitude

# Compute OTF based on the diffraction-limited incoherent system
OTF = np.zeros_like(f)
mask = f <= fc
OTF[mask] = (2 / np.pi) * (np.arccos(f[mask] / fc) - (f[mask] / fc) * np.sqrt(1 - (f[mask] / fc) ** 2))
# Example 2D matrix

# Create a 3D plot
fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface
ax.plot_surface(FX, FY, OTF, cmap='viridis')
ax.set_title('Incoherent OTF')

r = 64
y, x = np.ogrid[:H, :W]
center = (H // 2, W // 2)
mask = (x - center[1])**2 + (y - center[0])**2 <= r**2

# Create a 3D plot
fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface
ax.plot_surface(x, y, mask, cmap='viridis')
ax.set_title('Circular OTF')
# Show the plot
plt.show()