import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam as gc
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

"""This class is using Grad-CAM visualization technique to project activations of neural network learned onto a scan."""
class GradCamVisualiser:
    CLASS1 = 'CLASS1'
    CLASS2 = 'CLASS2'
    AXIAL = 'axial'
    SAGITAL = 'sagital'
    CORONAL = 'coronal'
    VIS_DIR = ''  # Output directory. This directory must contain folders /axial, /sagital, /coronal

    def __init__(self, model, scan, label):
        self._model = model
        self._scan = scan
        self._label = label
        self._score = self._getScore()
        self._heatMap = self.generateHeatMap()

    def _getScore(self):
        if self._label == self.CLASS2:
            return CategoricalScore([0])
        elif self._label == self.CLASS1:
            return CategoricalScore([1])

    def generateHeatMap(self):
        gradcam = gc(self._model, model_modifier=ReplaceToLinear(), clone=True)
        cam = gradcam(self._score, np.array([ps.expandDims(self._scan, 3)]), penultimate_layer=-1)
        return np.uint8(cm.jet(cam[0])[..., :3] * 255)

    def _getSliceAndHeatMapForSlice(self, sliceIndex, plane):
        if plane == self.SAGITAL:
            return self._scan[sliceIndex, :, :], self._heatMap[sliceIndex, :, :]
        elif plane == self.CORONAL:
            return self._scan[:, sliceIndex, :], self._heatMap[:, sliceIndex, :]
        else:
            return self._scan[:, :, sliceIndex], self._heatMap[:, :, sliceIndex]

    def _visualiseSlices(self, numberOfSlices, plane):
        for i in range(numberOfSlices):
            slice, sliceHeatMap = self._getSliceAndHeatMapForSlice(i, plane)
            plt.imshow(slice, cmap='gray')
            plt.imshow(sliceHeatMap, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.savefig(self.VIS_DIR % (self._label, plane) + str(i), bbox_inches='tight', pad_inches=0)
            plt.clf()

    def visualise(self):
        scanShape = np.shape(self._scan)
        self._visualiseSlices(scanShape[0], self.AXIAL)
        self._visualiseSlices(scanShape[1], self.CORONAL)
        self._visualiseSlices(scanShape[2], self.SAGITAL)
