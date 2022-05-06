"""This script was used to convert png binary slices created by 'mask_slices_to_binary.py' to mask in .nii format."""

from os import listdir
from os.path import isfile, join

import nibabel as nib
import numpy as np
import skimage.transform as skTrans
from scipy import ndimage
from skimage import color
from skimage import io

dir = ''  # Directory to binary slices.
onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

files = [0] * 120
for file in onlyfiles:
    number = file.replace('slice', '')
    number = int(number.replace('.jpg', ''))
    files[number] = file

mask = []
for f in files:
    img = color.rgb2gray(io.imread(dir + '/' + f))
    img = skTrans.resize(img, (120, 160), order=1, preserve_range=True)

    img[img >= 0.5] = 1
    img[img < 0.5] = 0

    mask.append(img.T)

mask = np.asarray(mask)
mask = np.asarray(ndimage.rotate(mask, 90, axes=(2, 0)))

mask[mask >= 0.5] = 1
mask[mask < 0.5] = 0

img = nib.Nifti1Image(mask, np.eye(4))
nib.save(img, 'mask.nii')
