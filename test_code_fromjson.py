import numpy as np
import config
import os
from readers.smarthome_skeleton_fromjson_sampling import load_detection_from_json

batch_size = 18000
dim = 42  # 39 w/o dists.
step = 30
n_classes = 35

list_IDs = [i.strip() for i in open(config.dataset_dir + '/splits/train_CS.txt').readlines()]
# list_IDs_temp = [list_IDs[k] for k in range(1000, 2000)]
list_IDs_temp = list_IDs
path = config.dataset_dir + '/json_rot_ds/'
X = np.empty((batch_size, step, dim))
f = 'Cook.Stir_p15_r03_v16_c03.json'

for i, ID in enumerate(list_IDs_temp):
    print(i)
    # Store sample
    # We try to load the arr_0 of the NPZ file named ID
    # unpadded_file = np.load(path + os.path.splitext(ID)[0] + '.npz')['arr_0']
    unpadded_file = load_detection_from_json(path + ID, 0)
    if len(unpadded_file)>0:   # if the length of arr_0 is greater than 0, continue with f = that ID
       f = ID
    if len(unpadded_file)==0:  # if the length of that arr_0 is 0, then use a default "Cook_p15..."
        print('Default.')
        unpadded_file = load_detection_from_json(path + f, 0)
        list_IDs_temp[i] = f
    # origin = unpadded_file[0, 3:6]   # Extract hip of the first frame (well, 'hip' are originally joints 4,5)
    # [row, col] = unpadded_file.shape  # Rows are 'frames', columns are 'joints'x3 (x,y,z).
    # origin = np.tile(origin, (row, 13))  # making equal dimension; that is,
    # unpadded_file = unpadded_file - origin  # translation of all (x,y,z)
    extra_frames = (len(unpadded_file) % step)  # how many frames remaining after taking 'step' by 'step' frames.
    l = 0
    if len(unpadded_file)<step:  # if the file is shorter than step (e.g. 30)
        extra_frames = step - len(unpadded_file)  # then extra frames necessary are the remainder to 30.
        l = 1
    if (extra_frames < (step/2)) & (l == 0):  # However, if the extra_frames calculated in line 30, are less than 15.
       padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]  # just discard those extra frames.
    else:
       [row, col] = unpadded_file.shape
       alpha = int(len(unpadded_file)/step) + 1  # Otherwise, add padding for one 'step' more.
       req_pad = np.zeros(((alpha * step)-row, col))
       padded_file = np.vstack((unpadded_file, req_pad))
    splitted_file = np.split(padded_file, step)  # Break down the padded data into a list of 'chunks' of 'step' frames.
    splitted_file = np.asarray(splitted_file)  # Make the list into an array.
    row, col, width = splitted_file.shape
    sampled_file = []
    for k in range(0,step):
        c = np.random.choice(col,1)
        sampled_file.append(splitted_file[k,c,:])
    sampled_file = np.asarray(sampled_file)
    X[i,] = np.squeeze(sampled_file)
