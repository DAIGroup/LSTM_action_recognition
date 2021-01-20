from argparse import ArgumentParser
import sys
import re
import config
from readers.smarthome_skeleton_fromjson_sampling import DataGenerator
import keras
from keras.models import load_model
from model_scripts.models import build_model_without_TS
from keras.callbacks import CSVLogger, Callback
import config as cfg
import os.path


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):
        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


# lr0 = 0.005
lr0 = 0.00005

model = build_model_without_TS(cfg.n_neuron, cfg.n_dropout, cfg.batch_size, cfg.timesteps, cfg.data_dim, cfg.num_classes)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr0, clipnorm=1),
              metrics=['accuracy'])

ep = ''
if len(sys.argv) == 2:
    model_file = sys.argv[1]
    model = load_model(model_file)
    ep = '%s_' % re.split('_', os.path.split(model_file)[1].replace('.hdf5', ''))[1]
print(ep)

splits_dir = config.dataset_dir + '/splits'
train_generator = DataGenerator(splits_dir + '/train_CS.txt', batch_size=cfg.batch_size, is_test=False)
val_generator = DataGenerator(splits_dir + '/validation_CS.txt', batch_size=cfg.batch_size)
test_generator = DataGenerator(splits_dir + '/test_CS.txt', batch_size=cfg.batch_size)

wdir = cfg.weights_dir + '/weights_' + cfg.name
if not os.path.exists(wdir):
    os.makedirs(wdir)

model_checkpoint = CustomModelCheckpoint(model, wdir + '/epoch_' + ep)
csvlogger = CSVLogger(cfg.name + '_smarthomes%s.csv' % ep)

# This comes from 'preprocessing/calculate_class_weights.py'
class_weights = {0: 0, 1: 11.48, 2: 10.29, 3: 26.44, 4: 7.47, 5: 37.46,
                 6: 107.88, 7: 0, 8: 13.42, 9: 13.69, 10: 1.90, 11: 47.32,
                 12: 7.93, 13: 17.29, 14: 9.17, 15: 5.58, 16: 24.74, 17: 9.08,
                 18: 0, 19: 69.15, 20: 58.63, 21: 61.30, 22: 74.92, 23: 16.45,
                 24: 79.32, 25: 0, 26: 35.96, 27: 4.68, 28: 4.20, 29: 13.76,
                 30: 13.29, 31: 84.28, 32: 9.17, 33: 1.00, 34: 5.86}


model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    class_weight=class_weights,
                    use_multiprocessing=False,
                    epochs=cfg.epochs,
                    callbacks=[csvlogger, model_checkpoint],
                    workers=6)
