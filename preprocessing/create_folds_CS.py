"""
    Create folds files for cross-subject experiments.
"""
import config
from glob import glob
import os
import re
import random

# According to paper (Toyota SmartHome: real-world activities of daily living)
train_subjects = [3,4,6,7,9,12,13,15,17,19,25]
test_subjects = [2,10,11,14,16,18,20]

sequences = glob(config.dataset_dir + '/json/*.json')
print('OK: Found %d sequences in total.' % len(sequences))

train_subjects_str = ['p%02d' % n for n in train_subjects]
test_subjects_str = ['p%02d' % n for n in test_subjects]

train_sequences = []
test_sequences = []

for sequence in sequences:
    sequence_path, sequence_name = os.path.split(sequence)
    sequence_name_chunks = re.split('_', sequence_name)
    person = sequence_name_chunks[1]  # e.g. 'p18'
    if person in train_subjects_str:
        train_sequences.append(sequence)
    elif person in test_subjects_str:
        test_sequences.append(sequence)
    else:
        print('ERROR: Found person number not in either set (train/test) !!')

num_train_samples = len(train_sequences)
num_test_samples = len(test_sequences)
print('OK: Found %d train sequences, and %d test sequences' % (num_train_samples, num_test_samples))

# NOTE: Pick 5% of samples for validation
random.seed(config.random_seed)
valid_sequences = random.sample(train_sequences, k=int(num_train_samples*0.05))
train_sequences = [seq for seq in train_sequences if not seq in valid_sequences]

num_train_samples = len(train_sequences)
num_valid_samples = len(valid_sequences)

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


save_split_to_file('train_CS.txt', train_sequences)
save_split_to_file('validation_CS.txt', valid_sequences)
save_split_to_file('test_CS.txt', test_sequences)
print('SAVED.')