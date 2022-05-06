"""This script converts png slices created by 'mask_to_png_slices.py' to binary png slices."""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as skTrans
from skimage import color
from skimage import io

dirIn = 'inputDirectory'
dirOut = 'outputDirectory'
THRESHOLD = 0.5

files = [f for f in listdir(dirIn) if isfile(join(dirIn, f))]

fileNames = [0] * 120
for file in files:
    number = file.replace('slice', '')
    number = int(number.replace('.jpg', ''))
    fileNames[number] = file

for f in fileNames:
    img = color.rgb2gray(io.imread(dirIn + '/' + f))
    img = skTrans.resize(img, (120, 160), order=1, preserve_range=True)
    img[img >= THRESHOLD] = 1
    img[img < THRESHOLD] = 0

    img = np.asarray(img)

    where_0 = np.where(img == 0)
    where_1 = np.where(img == 1)

    img[where_0] = 1
    img[where_1] = 0

    plt.imshow(img, cmap='gray', origin="lower")
    plt.axis('off')
    plt.savefig(dirOut + f, bbox_inches='tight', pad_inches=0)
    plt.clf()
