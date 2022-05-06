import random
from random import randrange

import nibabel as nib
import numpy as np
import skimage.transform as skTrans
from scipy import ndimage

from Scan import Scan

"""Function used mainly during preprocessing."""
class ProcessingFunctions:

    @staticmethod
    def readNiftiFile(filePath):
        """Read and load volume."""
        # Read file
        volume = nib.load(filePath)
        # Get raw data
        volume = volume.get_fdata()
        # Replace Nan values with zeroes
        volume[np.isnan(volume)] = 0
        return volume

    @staticmethod
    def normalize(volume):
        """Normalizes volume to [0 1] range."""
        volume = (volume - np.min(volume)) / np.ptp(volume)
        volume = volume.astype("float32")
        return volume

    @staticmethod
    def resizeVolume(volume):
        """Resizes volume to 120x160x120 size."""
        return skTrans.resize(volume, (120, 160, 120), order=1, preserve_range=True)

    @staticmethod
    def processScan(path):
        """Reads, resizes and normalizes volume."""
        volume = ProcessingFunctions.readNiftiFile(path)
        volume = ProcessingFunctions.normalize(volume)
        volume = ProcessingFunctions.resizeVolume(volume)
        return volume

    @staticmethod
    def scaleVolume(volume, scaleFactor):
        """Scales volume by given 'scaleFactor'."""
        volume = ProcessingFunctions.squeeze(volume)
        h, w, d = volume.shape[:3]
        zoom_tuple = (scaleFactor, scaleFactor, scaleFactor)

        # Zooming out
        if scaleFactor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * scaleFactor))
            zw = int(np.round(w * scaleFactor))
            zd = int(np.round(d * scaleFactor))

            top = (h - zh) // 2
            left = (w - zw) // 2
            depth = (d - zd) // 2

            # Zero-padding
            out = np.zeros_like(volume)
            out[top:top + zh, left:left + zw, depth:depth + zd] = ndimage.zoom(volume, zoom_tuple)

        # Zooming in
        elif scaleFactor > 1:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / scaleFactor))
            zw = int(np.round(w / scaleFactor))
            zd = int(np.round(d / scaleFactor))

            top = (h - zh) // 2
            left = (w - zw) // 2
            depth = (d - zd) // 2

            out = ndimage.zoom(volume[top:top + zh, left:left + zw, depth:depth + zd], zoom_tuple)

            # Trimming off extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            trim_depth = ((out.shape[2] - d) // 2)

            out = out[trim_top:trim_top + h, trim_left:trim_left + w, trim_depth:trim_depth + d]
        else:
            out = volume

        return ProcessingFunctions.expandDims(out, 3)

    @staticmethod
    def shiftVolume(volume, rangeFrom, rangeTo):
        """Shifts volume by given ranges."""
        dx = randrange(rangeFrom, rangeTo)
        dy = randrange(rangeFrom, rangeTo)
        dz = randrange(rangeFrom, rangeTo)

        volume = np.roll(volume, dy, axis=0)
        volume = np.roll(volume, dx, axis=1)
        volume = np.roll(volume, dz, axis=2)

        if dy > 0:
            volume[:dy, :, :] = 0
        elif dy < 0:
            volume[dy:, :, :] = 0

        if dx > 0:
            volume[:, :dx, :] = 0
        elif dx < 0:
            volume[:, dx:, :] = 0

        if dz > 0:
            volume[:, :, :dz] = 0
        elif dz < 0:
            volume[:, :, dz:] = 0

        return volume

    @staticmethod
    def rotateVolume(volume):
        """Rotate volume by random axis from -5 to 5 degrees."""
        angles = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        axes = [(1, 0), (2, 0), (2, 1)]

        angle = random.choice(angles)
        axis = random.choice(axes)

        return ndimage.rotate(volume, angle, axes=axis, reshape=False)

    @staticmethod
    def augmentVolume(volume):
        """Randomly scales ([0.8 1.2]), rotates ([-5 5]) and shifts ([-5 5]) volume"""
        scales = [0.8, 0.9, 1.1, 1.2]
        scale = random.choice(scales)

        augmented_volume = ProcessingFunctions.scaleVolume(volume, scale)
        augmented_volume = ProcessingFunctions.rotateVolume(augmented_volume)
        augmented_volume = ProcessingFunctions.shiftVolume(augmented_volume, -5, 5)

        return augmented_volume

    @staticmethod
    def augmentVolumes(volumes, augNum):
        """Returns array with extra augmented volumes.
        Parameter augNum declares the number of volumes we want to augment"""
        augumentedVolumes = []
        for i in range(augNum):
            scan = volumes[i].getScan()
            label = volumes[i].getLabel()
            augumentedScan = ProcessingFunctions.augmentVolume(scan)
            augumentedVolumes.append(Scan(augumentedScan, label))
        return np.concatenate((volumes, np.asarray(augumentedVolumes)))

    @staticmethod
    def squeeze(volume):
        """Remove dimension of size 1 from volume"""
        return np.squeeze(volume)

    @staticmethod
    def expandDims(volume, axis=0):
        """Expands dimensions of volume."""
        return np.expand_dims(volume, axis=axis)

    @staticmethod
    def expandDimsList(volumes):
        """Expand dimensions of volumes."""
        expandedVolumes = list()
        for i in range(len(volumes)):
            expandedVolumes.append(ProcessingFunctions.expandDims(volumes[i], 3))
        return np.array(expandedVolumes)

    @staticmethod
    def shuffleDataset(volumes):
        """Randomly shuffles volumes."""
        p = np.random.permutation(len(volumes))
        return volumes[p]

    @staticmethod
    def splitDataset(volumes, percentage):
        """Splits volumes to two arrays by given attribute 'percentage'."""
        numOfsamples = len(volumes)
        perc = (numOfsamples * percentage) // 100
        return volumes[:perc], volumes[perc:]

    @staticmethod
    def applyMaskToVolumes(volumes, mask):
        """Applies mask to volumes."""
        volumes = np.asarray(volumes)
        mask = np.asarray(mask)
        return np.asarray([ProcessingFunctions.applyMask(volume, mask) for volume in volumes])

    @staticmethod
    def applyMask(volume, mask):
        """Applies mask to volume."""
        return volume * mask
