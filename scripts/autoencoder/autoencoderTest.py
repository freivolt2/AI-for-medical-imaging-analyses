import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

# File path to testing file
file = 'D:/ADNI/GradWrap/AD/MNI/wmADNI_002_S_0619_MR_MPR-R__GradWarp_Br_20070412095550910_S15145_I48758.nii'

model = load_model('autoencoder.h5', custom_objects={'AdamW': AdamW})

scan_orig = ps.processScan(file)
scan = np.array([ps.expandDims(scan_orig, 3)])
prediction = ps.squeeze(model.predict(scan))

# Show original and decoded images
plt.imshow(scan_orig[:, :, 60], cmap='gray', origin="lower")
plt.savefig('original', bbox_inches='tight', pad_inches=0)

plt.imshow(prediction[:, :, 60], cmap='gray', origin="lower")
plt.savefig('decoded', bbox_inches='tight', pad_inches=0)
