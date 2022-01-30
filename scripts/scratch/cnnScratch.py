import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Directories to data
NC = "ADNI/NC"
AD = "ADNI/AD"

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

print("Number of NC scans: ", str(len(ncPaths)))
print("Number of AD scans: ", str(len(adPaths)))

print("\nData file paths loading finished.\n")

##LOAD AND PREPROCESS DATA##############################################################################################

print("Loading and preprocessing data...")

ncScans = np.array([ps.processScan(path) for path in ncPaths])
adScans = np.array([ps.processScan(path) for path in adPaths])

ncLabels = np.array([0 for _ in range(len(ncScans))])
adLabels = np.array([1 for _ in range(len(adScans))])

scans = np.concatenate((ncScans, adScans))
labels = np.concatenate((ncLabels, adLabels))

scans = ps.expandDimsList(scans)
labels = to_categorical(labels, num_classes=2)

scans, labels = ps.shuffleDataset(scans, labels)

print("Data lading and preprocessing finished.\n")

##SPLIT DATA############################################################################################################

print("Splitting data...")

numOfsamples = len(scans)
perc90 = (numOfsamples * 90) // 100

scansTest = scans[perc90:]
labelsTest = labels[perc90:]

scans = scans[:perc90]
labels = labels[:perc90]

print("Data splitting finished.\n")


########################################################################################################################

def getModel(width=120, height=160, depth=120):
    """Build a 3D convolutional neural network model."""

    L1L2 = regularizers.l1_l2(l1=1e-4, l2=1e-4)

    inputs = (width, height, depth, 1)
    model = models.Sequential(name='cnn_scratch')

    model.add(layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=L1L2,
                                   input_shape=inputs))
    model.add(layers.MaxPool3D(pool_size=2))
    model.add(layers.GaussianDropout(rate=0.8))

    model.add(layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=L1L2))
    model.add(layers.MaxPool3D(pool_size=2))
    model.add(layers.GaussianDropout(rate=0.8))

    model.add(layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=L1L2))
    model.add(layers.MaxPool3D(pool_size=2))
    model.add(layers.GaussianDropout(rate=0.8))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='selu'))
    model.add(layers.Dense(16, activation='selu'))
    model.add(layers.Dense(2, activation="softmax"))

    return model

##BUILD MODEL###########################################################################################################

print("Building model...")

adam = AdamW(learning_rate=1e-4, beta_1=0.9, beta_2=0.9, weight_decay=1e-5)
model = getModel(width=120, height=160, depth=120)
model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
model.summary()

print("Building model finished.\n")

##TRAIN MODEL###########################################################################################################

print("Training model...")

model.fit(scans, labels,
          batch_size=10,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_split=0.20)

print("Training model finished.\n")

##TEST MODEL############################################################################################################

print("Testing model...")

results = model.evaluate(scansTest, labelsTest, batch_size=10)
print("test loss, test acc:", results)

print("Testing finished.")

##SAVE MODEL############################################################################################################

print("Saving model...")

model.save('cnn_scratch.h5')

print("Model saved.")
