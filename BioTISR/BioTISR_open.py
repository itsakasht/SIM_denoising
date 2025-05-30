import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski
import bm3d
import tifffile as tiff

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import modules.optics as op
from BioTISR.read_mrc import read_mrc

# Set plotting styles
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

folder_path = '..//BioTISR//BioTISR_Microtubules//Cell_001//'
header, data = read_mrc(os.path.join(folder_path, 'RawSIMData_level_03.mrc'))
data[data < 0] = 0
data = op.normalize(data)
data = data.transpose(2, 0, 1)
data = np.rot90(data, k=1, axes=(1, 2))

data_shape = np.shape(data)
print(np.min(data), np.max(data), data_shape)
op.display_fourier(data[0])
op.display_fourier(data[3])
op.display_fourier(data[6])

plt.show()
