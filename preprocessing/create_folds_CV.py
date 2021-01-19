"""
    Create folds files for cross-subject experiments.
"""
import config
from glob import glob
import os
import re
import random
from readers.smarthome_skeleton_fromjson_sampling import name_to_int_CV

# According to paper (Toyota SmartHome: real-world activities of daily living)
train_cameras = [1,3,4,6,7]
validation_cameras = [5]
test_cameras = [2]

sequences = glob(config.dataset_dir + '/json/*.json')
print('OK: Found %d sequences in total.' % len(sequences))

train_cameras_str = ['c%02d' % n for n in train_cameras]
valid_cameras_str = ['c%02d' % n for n in validation_cameras]
test_cameras_str = ['c%02d' % n for n in test_cameras]

cam1_sequences = []
train_sequences = []
valid_sequences = []
test_sequences = []

for sequence in sequences:
    sequence_path, sequence_file = os.path.split(sequence)
    sequence_name, extension = os.path.splitext(sequence_file)
    sequence_name_chunks = re.split('_', sequence_name)

    camera = sequence_name_chunks[4]  # e.g. 'c01'
    name = sequence_name_chunks[0]
    if name_to_int_CV(name) > 0:
        if camera in train_cameras_str:
            if camera in 'c01':
                cam1_sequences.append(sequence)
            train_sequences.append(sequence)
        elif camera in valid_cameras_str:
            valid_sequences.append(sequence)
        elif camera in test_cameras_str:
            test_sequences.append(sequence)
        else:
            print('ERROR: Found person number not in either set (train/test) !!')

num_train_samples = len(train_sequences)
num_valid_samples = len(valid_sequences)
num_test_samples = len(test_sequences)

print('DONE: %d train; %d validation; %d test sequences.' % (num_train_samples, num_valid_samples, num_test_samples))


def save_split_to_file(name, samples_list):
    filepath = '%s/splits/' % config.dataset_dir
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = os.path.join(filepath, name)
    if not os.path.exists(filename):
        fh = open(filename, 'w')
        for sample in samples_list:
            fh.write('%s\n' % os.path.split(sample)[1])
        fh.close()
    else:
        print('ERROR: Split file already exists under "%s"' % filename)


save_split_to_file('train_CV_1.txt', cam1_sequences)
save_split_to_file('train_CV.txt', train_sequences)
save_split_to_file('validation_CV.txt', valid_sequences)
save_split_to_file('test_CV.txt', test_sequences)
print('SAVED.')