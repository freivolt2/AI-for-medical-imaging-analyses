"""By this script we created Grad-CAM activations projected on mean MRI images showed in the paper."""

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from GradCamVisualiser import GradCamVisualiser
from ProcessingFunctions import ProcessingFunctions as ps

directory = 'directory.nii'  # Derectory of folder where .nii files from which will be mean image created.

model = load_model('placeholder.h5', custom_objects={'AdamW': AdamW})

SAGITAL = 'sagital'
CORONAL = 'coronal'
AXIAL = 'axial'
VIS_DIR = ''  # Output directory. This directory must contain folders /axial, /sagital, /coronal

CLASS = 'CLASS1'

heatmaps = []
scans = []
for filename in os.listdir(directory):
    fullDirectory = os.path.join(directory, filename)
    scan = ps.processScan(fullDirectory)
    gradcam = GradCamVisualiser(model, scan, CLASS)
    scans.append(scan)
    heatmaps.append(gradcam.generateHeatMap())

scans = np.asarray(scans)
heatmaps = np.asarray(heatmaps)

meanScan = np.mean(scans, axis=0)
meanHeatmap = np.mean(heatmaps, axis=0)
meanHeatmap = meanHeatmap.astype(np.uint8)


def getSliceAndHeatMapForSlice(sliceIndex, plane, scan, heatMap):
    if plane == SAGITAL:
        return scan[sliceIndex, :, :], heatMap[sliceIndex, :, :]
    elif plane == CORONAL:
        return scan[:, sliceIndex, :], heatMap[:, sliceIndex, :]
    else:
        return scan[:, :, sliceIndex], heatMap[:, :, sliceIndex]


def visualiseSlices(numberOfSlices, plane, scan, heatMap):
    for i in range(numberOfSlices):
        slice, sliceHeatMap = getSliceAndHeatMapForSlice(i, plane, scan, heatMap)
        plt.imshow(slice, cmap='gray')
        plt.imshow(sliceHeatMap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(VIS_DIR % plane + str(i), bbox_inches='tight', pad_inches=0)
        plt.clf()


scanShape = np.shape(meanScan)
visualiseSlices(scanShape[0], AXIAL, meanScan, meanHeatmap)
visualiseSlices(scanShape[1], CORONAL, meanScan, meanHeatmap)
visualiseSlices(scanShape[2], SAGITAL, meanScan, meanHeatmap)
