"""
    Test load_model and obtain results (per class accuracies)
"""
import numpy as np
from keras.models import Model, load_model
from readers.smarthome_skeleton_fromjson_sampling import DataGenerator
from tqdm import tqdm
import config as cfg

print('Loading model ...')
wdir = cfg.weights_dir + '/weights_' + cfg.name
model = load_model('%s/epoch_150_101.hdf5' % wdir)
# model = load_model('/media/paal/sedebe2/PESOS_LSTM/weights_pau_notrans_rot/epoch_150_148.hdf5')
# model = load_model('/media/paal/sedebe2/PESOS_LSTM/weights_pau_test_rot/epoch_150_148.hdf5')
# model = load_model('/media/paal/Data/LSTM_weights/weights_crossview_notrans_rot_wc_nods/epoch_185.hdf5')
# model = load_model('./weights_crossview_nt_rot_sw/epoch_290.hdf5')
print('Done.')

splits_dir = cfg.dataset_dir + '/splits'

if cfg.num_classes == 19:  # Cross-view
    print('Assuming cross-view.')
    test_generator = DataGenerator(splits_dir + '/test_CV.txt', batch_size=1)
elif cfg.num_classes == 35:  # Cross-subject
    print('Assuming cross-subject.')
    test_generator = DataGenerator(splits_dir + '/test_CS.txt', batch_size=1)

num_tests = len(test_generator)
print('Testing %d samples.' % num_tests)

nc = test_generator.n_classes
print('number of classes: %d' % nc)
conf_mat = np.zeros((nc, nc))

for i in tqdm(range(num_tests)):
    sample = test_generator[i]
    x, y, wg = sample
    pred = model.predict(x)
    p = np.argmax(pred)
    t = np.argmax(y)
    conf_mat[t, p] += 1

np.savetxt("confusion_matrix_lstm_CS_nt_rot_wc_298_THIRD.csv", conf_mat, delimiter=";")
print('FINISHED.')
