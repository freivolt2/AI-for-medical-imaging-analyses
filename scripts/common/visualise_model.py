"""This scropt visualises architectures."""

import tensorflow as tf
import visualkeras
from PIL import ImageFont
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW

model = load_model('placeholder.h5', custom_objects={'AdamW': AdamW})

from collections import defaultdict

color_map = defaultdict(dict)
color_map[layers.InputLayer]['fill'] = 'wheat'
color_map[layers.Conv3D]['fill'] = 'yellow'
color_map[layers.MaxPooling3D]['fill'] = 'orangered'
color_map[layers.UpSampling3D]['fill'] = 'dodgerblue'
color_map[layers.GaussianDropout]['fill'] = 'silver'

font = ImageFont.truetype('arial.ttf')

vis = visualkeras.layered_view(model, spacing=20, max_z=10, scale_xy=2, legend=True, color_map=color_map, font=font,
                               to_file="placeholder.png")
tf.keras.utils.plot_model(model, to_file='placeholder.png', show_layer_names=False)
