"""This script creates mean .nii file from DIRECTORY."""

import os

import nibabel as nib
import numpy as np

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

# Directories to data
DIRECTORY = ""

paths = [
    os.path.join(os.getcwd(), DIRECTORY, x)
    for x in os.listdir(DIRECTORY)
]

scans = np.array([ps.processScan(path) for path in paths])
meanScan = np.mean(scans, axis=0)

img = nib.Nifti1Image(meanScan, np.eye(4))
nib.save(img, 'mean.nii')
