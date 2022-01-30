from os import walk

import numpy as np
from keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

model = load_model('cnn_scratch.h5', custom_objects={'AdamW': AdamW})

predAD = []
predNC = []
predOasisAD = []
predOasisNC = []

# for _, _, files in walk('E:/ADNI/GradWrap/AD/MNI/'):
#     for file in files:
#         print(file)
#         scan_orig = ps.processScan('E:/ADNI/GradWrap/AD/MNI/' + file)
#         scan = np.array([ps.expandDims(scan_orig, 3)])
#         predAD.append(model.predict(scan))
#
# for _, _, files in walk('E:/ADNI/GradWrap/NC/MNI/'):
#     for file in files:
#         print(file)
#         scan_orig = ps.processScan('E:/ADNI/GradWrap/NC/MNI/' + file)
#         scan = np.array([ps.expandDims(scan_orig, 3)])
#         predNC.append(model.predict(scan))

for _, _, files in walk('D:/test_dataset/AD10/MNI'):
    for file in files:
        print(file)
        scan_orig = ps.processScan('D:/test_dataset/AD10/MNI/' + file)
        scan = np.array([ps.expandDims(scan_orig, 3)])
        predOasisAD.append(model.predict(scan))

for _, _, files in walk('D:/test_dataset/NC10/MNI'):
    for file in files:
        print(file)
        scan_orig = ps.processScan('D:/test_dataset/NC10/MNI/' + file)
        scan = np.array([ps.expandDims(scan_orig, 3)])
        predOasisNC.append(model.predict(scan))

daco = 0