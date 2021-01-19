import numpy as np
import keras
import pandas as pd
import os
import config
import json

use_distances = config.use_distances
num_distances = config.num_distances

weights_CV = {0: 0.2312, 1: 0.0345, 2: 0.0310, 3: 0.0040, 4: 0.1290, 5: 0.0154, 6: 0.0332, 7: 0.0166, 8: 0.0131,
              9: 0.0179, 10: 0.0334, 11: 0.1423, 12: 0.0117, 13: 0.0098, 14: 0.0298, 15: 0.0259, 16: 0.1981,
              17: 0.0219, 18: 0.0012}

weights_CS = {0: 0.0000, 1: 0.0130, 2: 0.0117, 3: 0.0299, 4: 0.0085, 5: 0.0424, 6: 0.1222, 7: 0.0000, 8: 0.0152,
              9: 0.0155, 10: 0.0022, 11: 0.0536, 12: 0.0090, 13: 0.0196, 14: 0.0104, 15: 0.0063, 16: 0.0280,
              17: 0.0103, 18: 0.0000, 19: 0.0783, 20: 0.0664, 21: 0.0694, 22: 0.0848, 23: 0.0186, 24: 0.0898,
              25: 0.0000, 26: 0.0407, 27: 0.0053, 28: 0.0048, 29: 0.0156, 30: 0.0150, 31: 0.0954, 32: 0.0104,
              33: 0.0011, 34: 0.0066}


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, path_video_files, batch_size=32, is_test=True):
        'Initialization'
        self.batch_size = batch_size
        self.path = config.dataset_dir + '/%s/' % config.variant_dir
        self.list_IDs = [i.strip() for i in open(path_video_files).readlines()]
        self.n_classes = config.num_classes
        self.step = 30
        self.dim = 39 if not use_distances else (39+num_distances)
        self.is_test = is_test
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        tup = generate_data(list_IDs_temp, self.path, self.batch_size, self.step, self.dim, self.is_test)
        return tup

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.list_IDs)


def load_detection_from_json(json_file, detection_index):
    try:
        fh = open(json_file, 'r')
        skel_data = json.load(fh)
        fh.close()
    except:
        print('ERROR: ', json_file)

    skel_frames = skel_data['frames']
    num_joints = skel_data['njts']

    dims = (3*num_joints) if not use_distances else (3*num_joints+num_distances)

    if len(skel_frames) > 0:
        skel_arr = np.zeros(shape=(len(skel_frames), dims))

        for f, skel_frame in enumerate(skel_frames):
            if len(skel_frame) > 0:
                skeleton = skel_frame[detection_index]
                skel_3d = np.zeros((dims,))
                joints = np.array(skeleton['pose3d'][:3*num_joints]).reshape((3, num_joints)).reshape((3 * num_joints,))
                distances = np.array(skeleton['pose3d'][-3:])
                skel_3d[:3*num_joints] = joints
                if use_distances:
                    skel_3d[3*num_joints:] = distances
                skel_arr[f, :] = skel_3d
    else:
        print('Empty skeleton array (%s)' % json_file)
        skel_arr = []

    return skel_arr


def name_to_int_CV(name):
    integer = 0
    if name == "Cutbread":
        integer = 1
    elif name == "Drink.Frombottle":
        integer = 2
    elif name == "Drink.Fromcan":
        integer = 3
    elif name == "Drink.Fromcup":
        integer = 4
    elif name == "Drink.Fromglass":
        integer = 5
    elif name == "Eat.Attable":
        integer = 6
    elif name == "Eat.Snack":
        integer = 7
    elif name == "Enter":
        integer = 8
    elif name == "Getup":
        integer = 9
    elif name == "Leave":
        integer = 10
    elif name == "Pour.Frombottle":
        integer = 11
    elif name == "Pour.Fromcan":
        integer = 12
    elif name == "Readbook":
        integer = 13
    elif name == "Sitdown":
        integer = 14
    elif name == "Takepills":
        integer = 15
    elif name == "Uselaptop":
        integer = 16
    elif name == "Usetablet":
        integer = 17
    elif name == "Usetelephone":
        integer = 18
    elif name == "Walk":
        integer = 19
    if integer == 0:
        print('ERROR: Returned zero in name_to_int_CV().')
    return integer


def name_to_int(name):
    integer = 0
    if name == "Cook":
        integer = 1
    elif name == "Cook.Cleandishes":
        integer = 2
    elif name == "Cook.Cleanup":
        integer = 3
    elif name == "Cook.Cut":
        integer = 4
    elif name == "Cook.Stir":
        integer = 5
    elif name == "Cook.Usestove":
        integer = 6
    elif name == "Cutbread":
        integer = 7
    elif name == "Drink":
        integer = 8
    elif name == "Drink.Frombottle":
        integer = 9
    elif name == "Drink.Fromcan":
        integer = 10
    elif name == "Drink.Fromcup":
        integer = 11
    elif name == "Drink.Fromglass":
        integer = 12
    elif name == "Eat.Attable":
        integer = 13
    elif name == "Eat.Snack":
        integer = 14
    elif name == "Enter":
        integer = 15
    elif name == "Getup":
        integer = 16
    elif name == "Laydown":
        integer = 17
    elif name == "Leave":
        integer = 18
    elif name == "Makecoffee":
        integer = 19
    elif name == "Makecoffee.Pourgrains":
        integer = 20
    elif name == "Makecoffee.Pourwater":
        integer = 21
    elif name == "Maketea.Boilwater":
        integer = 22
    elif name == "Maketea.Insertteabag":
        integer = 23
    elif name == "Pour.Frombottle":
        integer = 24
    elif name == "Pour.Fromcan":
        integer = 25
    elif name == "Pour.Fromcup":
        integer = 26
    elif name == "Pour.Fromkettle":
        integer = 27
    elif name == "Readbook":
        integer = 28
    elif name == "Sitdown":
        integer = 29
    elif name == "Takepills":
        integer = 30
    elif name == "Uselaptop":
        integer = 31
    elif name == "Usetablet":
        integer = 32
    elif name == "Usetelephone":
        integer = 33
    elif name == "Walk":
        integer = 34
    elif name == "WatchTV":
        integer = 35
    if integer == 0:
        print('ERROR: Returned zero in name_to_int() [CS].')
    return integer


def generate_data(list_IDs_temp, skel_path, batch_size, step, dim, is_test):
    """Generates data containing batch_size samples"""
    # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((batch_size, step, dim))

    # Generate data
    f = 'Cook.Stir_p15_r03_v16_c03.json'
    for i, ID in enumerate(list_IDs_temp):
        # Store sample

        # unpadded_file = np.load(self.path + os.path.splitext(ID)[0] + '.npz')['arr_0']
        unpadded_file = load_detection_from_json(skel_path + ID, 0)

        if len(unpadded_file) > 0:
            f = ID
        if len(unpadded_file) == 0:
            print('unpadded file len is zero.')
            unpadded_file = load_detection_from_json(skel_path + f, 0)
            list_IDs_temp[i] = f
        # origin = unpadded_file[0, 12:15]  # Extract left hip of the first frame
        # [row, col] = unpadded_file.shape
        # origin = np.tile(origin, (row, 13))  # making equal dimension
        # unpadded_file = unpadded_file - origin  # translation
        extra_frames = (len(unpadded_file) % step)
        l = 0
        if len(unpadded_file) < step:
            extra_frames = step - len(unpadded_file)
            l = 1
        if (extra_frames < (step / 2)) & (l == 0):
            padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
        else:
            [row, col] = unpadded_file.shape
            alpha = int(len(unpadded_file) / step) + 1
            req_pad = np.zeros(((alpha * step) - row, col))
            padded_file = np.vstack((unpadded_file, req_pad))
        splitted_file = np.split(padded_file, step)
        splitted_file = np.asarray(splitted_file)
        row, col, width = splitted_file.shape
        sampled_file = []
        for k in range(0, step):
            c = 0
            if not is_test:
                c = np.random.choice(col, 1)
            sampled_file.append(splitted_file[k, c, :])
        sampled_file = np.asarray(sampled_file)
        X[i,] = np.squeeze(sampled_file)

        # Store class
        # y[i] = self.labels[ID]

    # y = np.array([int(i[-3:]) for i in list_IDs_temp]) - 1
    if config.num_classes == 35:
        ids = np.array([int(name_to_int(i.split('_')[0])) for i in list_IDs_temp]) - 1
    else:
        ids = np.array([int(name_to_int_CV(i.split('_')[0])) for i in list_IDs_temp]) - 1
    y = np.zeros((batch_size, config.num_classes))
    w = np.zeros((batch_size,))

    for i in range(len(list_IDs_temp)):
        y[i, ids[i]] = 1
        if config.num_classes == 35:
            w[i] = weights_CS[ids[i]]
        else:
            w[i] = weights_CV[ids[i]]
    if config.sample_weights:
        return X, y, w
    else:
        return X, y
