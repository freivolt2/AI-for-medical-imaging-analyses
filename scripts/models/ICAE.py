import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

EPOCHS = 1
BATCH_SIZE = 10

# Directories to data
AD = ""
NC = ""
pMCI = ""
sMCI = ""

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

def getInceptionModule(layerIn, reg, add=''):
    layer1 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv1_incept_layer1' + add)(layerIn)

    layer2 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv1_incept_layer2' + add)(layerIn)
    layer2 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv2_incept_layer2' + add)(layer2)

    layer3 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv1_incept_layer3' + add)(layerIn)
    layer3 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv2_incept_layer3' + add)(layer3)
    layer3 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv3_incept_layer3' + add)(layer3)

    layer4 = layers.MaxPool3D(pool_size=3, strides=1, padding='same', name='pool1_incept_layer4' + add)(layerIn)
    layer4 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg,
                                  name='conv1_incept_layer4' + add)(layer4)

    return layers.concatenate([layer1, layer2, layer3, layer4], axis=-1)


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

    x = getInceptionModule(x, L1L2)
    x = layers.MaxPool3D(pool_size=2)(x)
    encoded = layers.GaussianDropout(rate=0.8)(x)

    x = getInceptionModule(encoded, L1L2, '_2')
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

from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

caeModel = load_model('autoencoder.h5', custom_objects={'AdamW': AdamW})

conW = caeModel.get_layer('conv3d').get_weights()
poolW = caeModel.get_layer('max_pooling3d').get_weights()
gausW = caeModel.get_layer('gaussian_dropout').get_weights()
con1W = caeModel.get_layer('conv3d_1').get_weights()
pool1W = caeModel.get_layer('max_pooling3d_1').get_weights()
gaus1W = caeModel.get_layer('gaussian_dropout_1').get_weights()

autoencoder.get_layer('conv3d').set_weights(conW)
autoencoder.get_layer('max_pooling3d').set_weights(poolW)
autoencoder.get_layer('gaussian_dropout').set_weights(gausW)
autoencoder.get_layer('conv3d_1').set_weights(con1W)
autoencoder.get_layer('max_pooling3d_1').set_weights(pool1W)
autoencoder.get_layer('gaussian_dropout_1').set_weights(gaus1W)

autoencoder.compile(optimizer=optimizer, loss='mse')
autoencoder.summary()

print("Building model finished.\n")

##TRAIN#################################################################################################################

print("Training...")

autoencoder.fit(train_data, train_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(val_data, val_data))

print("Training finished.\n")

##SAVE MODEL############################################################################################################
print("Saving model...")

autoencoder.save('inception_autoencoder.h5')

print("Model saved.")
