import nibabel as nib
import numpy as np
import skimage.transform as skTrans
import tensorflow as tf


class ProcessingFunctions:

    @staticmethod
    def readNiftiFile(filePath):
        """Read and load volume"""
        # Read file
        volume = nib.load(filePath)
        # Get raw data
        volume = volume.get_fdata()
        # Replace Nan values with zeroes
        volume[np.isnan(volume)] = 0
        return volume

    @staticmethod
    def normalize(volume):
        """Normalize the volume to [0 1] range"""
        volume = (volume - np.min(volume)) / np.ptp(volume)
        volume = volume.astype("float32")
        return volume

    @staticmethod
    def resizeVolume(volume):
        """Resize volume to 120x160x120"""
        return skTrans.resize(volume, (120, 160, 120), order=1, preserve_range=True)

    @staticmethod
    def processScan(path):
        """Read and resize volume"""
        volume = ProcessingFunctions.readNiftiFile(path)
        volume = ProcessingFunctions.normalize(volume)
        volume = ProcessingFunctions.resizeVolume(volume)
        return volume

    @staticmethod
    def squeeze(volume):
        """Remove dimension of size 1 from volume"""
        return tf.squeeze(volume)

    @staticmethod
    def expandDims(volume, axis=0):
        """AExpand dimensions of volume"""
        return tf.expand_dims(volume, axis=axis)

    @staticmethod
    def expandDimsList(volumes):
        """Expand dimensions of volumes"""
        expandedVolumes = list()
        for i in range(len(volumes)):
            expandedVolumes.append(ProcessingFunctions.expandDims(volumes[i], 3))
        return np.array(expandedVolumes)

    @staticmethod
    def shuffleDataset(volumes, labels):
        """Shuffle volumes and their labels"""
        assert len(volumes) == len(labels)
        p = np.random.permutation(len(volumes))
        return volumes[p], labels[p]
