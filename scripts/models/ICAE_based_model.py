import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

from scripts.common.ProcessingFunctions import ProcessingFunctions as ps
from scripts.common.Scan import Scan

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

EPOCHS = 1
BATCH_SIZE = 10
AUGMENT = 50

# Directories to data
CLASS1 = ""
CLASS2 = ""

##LOAD DATA FILE PATHS##################################################################################################

print("\nLoading file paths to data...\n")

c1Paths = [
    os.path.join(os.getcwd(), CLASS1, x)
    for x in os.listdir(CLASS1)
]
c2Paths = [
    os.path.join(os.getcwd(), CLASS2, x)
    for x in os.listdir(CLASS2)
]

print("Number of CLASS1 scans: ", str(len(c1Paths)))
print("Number of CLASS2 scans: ", str(len(c2Paths)))

print("\nData file paths loading finished.\n")

##LOAD AND PREPROCESS DATA##############################################################################################

print("Loading and preprocessing data...")

c1Scans = np.array([ps.processScan(path) for path in c1Paths])
c2Scans = np.array([ps.processScan(path) for path in c2Paths])

c2Scans = ps.expandDimsList(c2Scans)
c1Scans = ps.expandDimsList(c1Scans)

c1Scans = np.array([Scan(scan, 0) for scan in c1Scans])
c2Scans = np.array([Scan(scan, 1) for scan in c2Scans])

c2Scans = ps.shuffleDataset(np.asarray(c2Scans))
c1Scans = ps.shuffleDataset(np.asarray(c1Scans))

print("Data lading and preprocessing finished.\n")

##SPLIT AND AUGMENT DATA################################################################################################

print("Splitting data...")

scanC1Test, c1Scans = ps.splitDataset(c1Scans, 10)
scanC2Test, c2Scans = ps.splitDataset(c2Scans, 10)

numOfC1 = len(c1Scans)
numOfC2 = len(c2Scans)
a = (((numOfC1 + numOfC2) * 20) // 100) // 2

scanC1Train, scanNcVal = c1Scans[a:], c1Scans[:a]
scanC2Train, scanAdVal = c2Scans[a:], c2Scans[:a]

scanC1Train = ps.augmentVolumes(scanC1Train, (len(scanC1Train) * AUGMENT) // 100)
scanC2Train = ps.augmentVolumes(scanC2Train, abs(len(scanC2Train) - len(scanC1Train)))

scansTest = np.concatenate((scanC2Test, scanC1Test))
scansTrain = np.concatenate((scanC2Train, scanC1Train))
scansVal = np.concatenate((scanAdVal, scanNcVal))

scansTest = ps.shuffleDataset(scansTest)
scansTrain = ps.shuffleDataset(scansTrain)
scansVal = ps.shuffleDataset(scansVal)

labelsTest = [scan.getLabel() for scan in scansTest]
labelsTrain = [scan.getLabel() for scan in scansTrain]
labelsVal = [scan.getLabel() for scan in scansVal]

labelsTest = to_categorical(labelsTest, num_classes=2)
labelsTrain = to_categorical(labelsTrain, num_classes=2)
labelsVal = to_categorical(labelsVal, num_classes=2)

scansTest = np.asarray([scan.getScan() for scan in scansTest])
scansTrain = np.asarray([scan.getScan() for scan in scansTrain])
scansVal = np.asarray([scan.getScan() for scan in scansVal])

print("Data splitting finished.\n")


########################################################################################################################

def getInceptionModule(layerIn):
    layer1 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', name='conv1_incept_layer1')(layerIn)

    layer2 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', name='conv1_incept_layer2')(layerIn)
    layer2 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', name='conv2_incept_layer2')(layer2)

    layer3 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', name='conv1_incept_layer3')(layerIn)
    layer3 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', name='conv2_incept_layer3')(layer3)
    layer3 = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', name='conv3_incept_layer3')(layer3)

    layer4 = layers.MaxPool3D(pool_size=3, strides=1, padding='same', name='pool1_incept_layer4')(layerIn)
    layer4 = layers.Convolution3D(10, kernel_size=1, activation='relu', padding='same', name='conv1_incept_layer4')(layer4)

    return layers.concatenate([layer1, layer2, layer3, layer4], axis=-1)

def getModel(width=120, height=160, depth=120):
    """Build a 3D convolutional neural network model."""

    L1L2 = regularizers.l1_l2(l1=1e-4, l2=1e-4)

    inputs = Input((width, height, depth, 1))

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=L1L2)(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Convolution3D(10, kernel_size=3, activation='relu', padding='same', kernel_regularizer=L1L2)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = getInceptionModule(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.GaussianDropout(rate=0.8)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return Model(inputs, outputs, name="cnn_inception")


##BUILD MODEL###########################################################################################################

print("Building model...")

adam = AdamW(learning_rate=1e-4, beta_1=0.9, beta_2=0.9, weight_decay=1e-5)

model = getModel(width=120, height=160, depth=120)

from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

icaeModel = load_model('inception_autoencoder_bf_MNI_ss_transfer.h5', custom_objects={'AdamW': AdamW})

con1W = icaeModel.get_layer('conv3d').get_weights()
pool1W = icaeModel.get_layer('max_pooling3d').get_weights()
gaus1W = icaeModel.get_layer('gaussian_dropout').get_weights()
con2W = icaeModel.get_layer('conv3d_1').get_weights()
pool2W = icaeModel.get_layer('max_pooling3d_1').get_weights()
gaus2W = icaeModel.get_layer('gaussian_dropout_1').get_weights()

conIncept1W = icaeModel.get_layer('conv1_incept_layer1').get_weights()
conIncept2W = icaeModel.get_layer('conv1_incept_layer2').get_weights()
conIncept3W = icaeModel.get_layer('conv2_incept_layer2').get_weights()
conIncept4W = icaeModel.get_layer('conv1_incept_layer3').get_weights()
conIncept5W = icaeModel.get_layer('conv2_incept_layer3').get_weights()
conIncept6W = icaeModel.get_layer('conv3_incept_layer3').get_weights()
poolIncept1W = icaeModel.get_layer('pool1_incept_layer4').get_weights()
conIncept7W = icaeModel.get_layer('conv1_incept_layer4').get_weights()
con = icaeModel.get_layer('concatenate').get_weights()

pool3W = icaeModel.get_layer('max_pooling3d_2').get_weights()
gaus3W = icaeModel.get_layer('gaussian_dropout_2').get_weights()

model.get_layer('conv3d').set_weights(con1W)
model.get_layer('max_pooling3d').set_weights(pool1W)
model.get_layer('gaussian_dropout').set_weights(gaus1W)
model.get_layer('conv3d_1').set_weights(con2W)
model.get_layer('max_pooling3d_1').set_weights(pool2W)
model.get_layer('gaussian_dropout_1').set_weights(gaus2W)

model.get_layer('conv1_incept_layer1').set_weights(conIncept1W)
model.get_layer('conv1_incept_layer2').set_weights(conIncept2W)
model.get_layer('conv2_incept_layer2').set_weights(conIncept3W)
model.get_layer('conv1_incept_layer3').set_weights(conIncept4W)
model.get_layer('conv2_incept_layer3').set_weights(conIncept5W)
model.get_layer('conv3_incept_layer3').set_weights(conIncept6W)
model.get_layer('pool1_incept_layer4').set_weights(poolIncept1W)
model.get_layer('conv1_incept_layer4').set_weights(conIncept7W)
model.get_layer('concatenate').set_weights(con)

model.get_layer('max_pooling3d_2').set_weights(pool3W)
model.get_layer('gaussian_dropout_2').set_weights(gaus3W)

model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
model.summary()

print("Building model finished.\n")

##TRAIN MODEL###########################################################################################################

print("Training model...")

model.fit(scansTrain, labelsTrain,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          verbose=1,
          validation_data=(scansVal, labelsVal))

print("Training model finished.\n")

##TEST MODEL############################################################################################################

print("Testing model...")

results = model.evaluate(scansTest, labelsTest, batch_size=10)
print("test loss, test acc:", results)

print("Testing finished.")

##SAVE MODEL############################################################################################################

print("Saving model...")

model.save('icae_transfer.h5')

print("Model saved.")
