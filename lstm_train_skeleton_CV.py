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

# Make sure number of classes is correct.
config.num_classes = 19


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
lr0 = 0.0005

model = build_model_without_TS(cfg.n_neuron, cfg.n_dropout, cfg.batch_size, cfg.timesteps, cfg.data_dim, cfg.num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=lr0, clipnorm=1),
              metrics=['accuracy'])

ep = ''
if len(sys.argv) == 2:
    model_file = sys.argv[1]
    model = load_model(model_file)
    ep = '%s_' % re.split('_', os.path.split(model_file)[1].replace('.hdf5', ''))[1]
print(ep)

splits_dir = config.dataset_dir + '/splits'
train_generator = DataGenerator(splits_dir + '/train_CV.txt', batch_size=cfg.batch_size, is_test=False)
val_generator = DataGenerator(splits_dir + '/validation_CV.txt', batch_size=cfg.batch_size)
test_generator = DataGenerator(splits_dir + '/test_CV.txt', batch_size=cfg.batch_size)

wdir = cfg.weights_dir + '/weights_' + cfg.name
if not os.path.exists(wdir):
    os.makedirs(wdir)

model_checkpoint = CustomModelCheckpoint(model, wdir + '/epoch_' + ep)
csvlogger = CSVLogger(cfg.name + '_smarthomes%s.csv' % ep)

# This has been calculated by 'preprocessing/calculate_class_weights_CV.py'.
class_weights = {0: 103.04, 1: 15.36, 2: 13.82, 3: 1.76, 4: 57.51, 5: 6.87, 6: 14.81, 7: 7.40, 8: 5.82, 9: 7.98,
                 10: 14.90, 11: 63.41, 12: 5.23, 13: 4.35, 14: 13.30, 15: 11.56, 16: 88.32, 17: 9.77, 18: 1.00}

model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    # class_weight=class_weights,
                    use_multiprocessing=False,
                    epochs=cfg.epochs,
                    callbacks=[csvlogger, model_checkpoint],
                    workers=6)
