"""
    This script calculates the weights for each class, based on the maximum weight (max num. samples)
"""
import sys
import config
from readers.smarthome_skeleton_fromjson_sampling import name_to_int_CV
import numpy as np
splits_dir = config.dataset_dir + '/splits'
train_samples_file = splits_dir + '/train_CV.txt'

config.num_classes = 19

fh = open(train_samples_file, 'r')
samples = fh.readlines()
fh.close()

class_samples = np.zeros((config.num_classes,))
for sample in samples:
    c = name_to_int_CV(sample.split('_')[0])
    class_samples[c-1] += 1

class_weights = np.max(class_samples) / class_samples
# loss_weights = class_weights / np.max(class_weights)
loss_weights = (np.sum(class_samples) - class_samples) / (np.sum(class_samples))


sys.stdout.write('class_weights = ')
for c, w in enumerate(class_weights):
    if c == 0:
        sys.stdout.write('{')
    else:
        sys.stdout.write(', ')
    sys.stdout.write('%d: ' % c)
    sys.stdout.write('%.2f' % w)
sys.stdout.write('}\n')

sys.stdout.write('loss_weights = ')
for c, w in enumerate(loss_weights):
    if c == 0:
        sys.stdout.write('{')
    else:
        sys.stdout.write(', ')
    sys.stdout.write('%d: ' % c)
    sys.stdout.write('%.4f' % w)
sys.stdout.write('}\n')