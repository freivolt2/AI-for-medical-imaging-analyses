import numpy as np
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

model = load_model('cnn_scratch.h5', custom_objects={'AdamW': AdamW})

scan_orig = ps.processScan(
    'D:/ADNI/GradWrap/NC/MNI/wmADNI_002_S_0295_MR_MPR__GradWarp_Br_20070319114002547_S13408_I45110.nii')
scan = np.array([ps.expandDims(scan_orig, 3)])
predNC = model.predict(scan)

scan_orig = ps.processScan(
    'D:/ADNI/GradWrap/AD/MNI/wmADNI_002_S_0619_MR_MPR-R__GradWarp_Br_20070412095550910_S15145_I48758.nii')
scan = np.array([ps.expandDims(scan_orig, 3)])
predAD = model.predict(scan)
