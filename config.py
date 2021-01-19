"""
    Configuration parameters
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
tf.device('/gpu:1')
dataset_dir = '/media/pau/extern_wd/Datasets/ToyotaSmartHome'
variant_dir = 'json_rot_ds'
weights_dir = '/media/pau/extern_wd/weights/'

random_seed = 42  # For reproducible experiments.

# data_dim = 39  # 13 joints, 3D = 39
data_dim = 42  # 13 joints, 3D = 39 + 3 distances.
num_classes = 35  # Number of possible labels.
batch_size = 200  # Originally in git.
n_neuron = 512  # Neurons per layer (for SmartHome)
n_dropout = 0.5  # Dropout according to paper.
name = 'pau_notrans_rot_wc_ds'
timesteps = 30  # Time steps for SmartHome.

epochs = 150

use_distances = True
num_distances = 3
sample_weights = False