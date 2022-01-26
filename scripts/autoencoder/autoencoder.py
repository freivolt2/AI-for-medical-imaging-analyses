import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Directories to data
AD = "ADNI/AD"
NC = "ADNI/NC"
pMCI = "ADNI/pMCI"
sMCI = "ADNI/sMCI"

##LOAD DATA FILE PATHS##################################################################################################

print("\nLoading file paths to data...\n")

ncPaths = [
    os.path.join(os.getcwd(), NC, x)
    for x in os.listdir(NC)
]
adPaths = [
    os.path.join(os.getcwd(), AD, x)
    for x in os.listdir(AD)
]
pMciPaths = [
    os.path.join(os.getcwd(), pMCI, x)
    for x in os.listdir(pMCI)
]
sMciPaths = [
    os.path.join(os.getcwd(), sMCI, x)
    for x in os.listdir(sMCI)
]

print("Number of NC scans: ", str(len(ncPaths)))
print("Number of AD scans: ", str(len(adPaths)))
print("Number of pMCI scans: ", str(len(pMciPaths)))
print("Number of sMCI scans: ", str(len(sMciPaths)))

print("\nData file paths loading finished.\n")

##LOAD AND PREPROCESS DATA##############################################################################################

print("Loading and preprocessing data...")

ncScans = np.array([ps.processScan(path) for path in ncPaths])
adScans = np.array([ps.processScan(path) for path in adPaths])
pMciScans = np.array([ps.processScan(path) for path in pMciPaths])
sMciScans = np.array([ps.processScan(path) for path in sMciPaths])

print("Data lading and preprocessing finished.\n")

##SPLIT DATA############################################################################################################

print("Splitting data...")

numOfNc = len(ncScans)
numOfAd = len(adScans)
numOfPMci = len(pMciScans)
numOfSMci = len(sMciScans)

nc80 = (numOfNc * 80) // 100
ad80 = (numOfAd * 80) // 100
pMci80 = (numOfPMci * 80) // 100
sMci80 = (numOfSMci * 80) // 100

train_data = np.concatenate((ncScans[:nc80], adScans[:ad80], pMciScans[:pMci80], sMciScans[:sMci80]))
val_data = np.concatenate(
    (ncScans[nc80:numOfNc], adScans[ad80:numOfAd], pMciScans[pMci80:numOfPMci], sMciScans[sMci80:numOfSMci]))

train_data = ps.expandDimsList(train_data)
val_data = ps.expandDimsList(val_data)

print("Data splitting finished.\n")

########################################################################################################################

def getModel(width=120, height=160, depth=120):
    """Method builds a 3D convolutional neural autoencoder"""

    L1L2 = regularizers.l1_l2(l1=1e-4, l2=1e-4)

    input = keras.Input((width, height, depth, 1))

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(input)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    encoded = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(encoded)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(x)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', kernel_regularizer=L1L2, padding='same')(x)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    output = layers.Convolution3D(1, kernel_size=3, padding='same')(x)

    return keras.Model(input, output, name='autoencoder')

##BUILD MODEL###########################################################################################################

print("Building model...")

optimizer = AdamW(learning_rate=1e-4, beta_1=0.9, beta_2=0.9, weight_decay=1e-5)
autoencoder = getModel(width=120, height=160, depth=120)
autoencoder.compile(optimizer=optimizer, loss='mse')
autoencoder.summary()

print("Building model finished.\n")

##TRAIN#################################################################################################################

print("Training...")

autoencoder.fit(train_data, train_data,
                epochs=200,
                batch_size=10,
                shuffle=True,
                validation_data=(val_data, val_data))

print("Training finished.\n")

##SAVE MODEL############################################################################################################
print("Saving model...")

autoencoder.save('autoencoder.h5')

print("Model saved.")
