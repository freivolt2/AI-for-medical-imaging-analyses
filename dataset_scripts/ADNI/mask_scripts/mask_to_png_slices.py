"""This script converts mean.nii created by 'create_mean_mri.py' to png slices."""

import matplotlib.pyplot as plt

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

dir = 'outputDirectory'

meanScan = ps.readNiftiFile('mean.nii')

for i in range(120):
    slice = meanScan[:, :, i]
    plt.imshow(slice, cmap='gray')
    plt.axis('off')
    plt.savefig(dir + 'slice' + str(i) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
