# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:41:18 2020

@author: sxj146830
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from keras.optimizers import Adadelta, Adam
from keras.callbacks import ModelCheckpoint

from generate_sequence import data_generator
import list_of_data_angles
import list_of_data_angles_markers
from mobnet_model import model
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from keras import backend as K
import os
import params

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def cc_loss(ytrue, ypred):
    ypred = K.clip(ypred, 1e-14, 1)
    ytrue = K.clip(ytrue, 1e-14, 1)
    cce = -K.sum(K.sum(ytrue * K.log(ypred)))  # /(64*112)
    mae = K.abs(ytrue - ypred)
    return cce + mae


def fl_loss(ytrue, ypred):
    gamma = 1
    ypred = K.clip(ypred, 1e-14, 1)
    ytrue = K.clip(ytrue, 1e-14, 1)
    cce = -K.sum(K.sum(ytrue * ((1 - ypred) ** gamma) * K.log(ypred)))  # /(64*112)
    return cce

hist = {}
model = model()
save_model_loc = 'model_fusion_eyedrop20/'
os.mkdir(save_model_loc)
filepath = save_model_loc + 'model-{epoch:02d}-{val_loss:.2f}.hdf5'
cb = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', save_weights_only=True)
# images driver and head pose
train_images, train_bb, train_angles, train_hp, val_images, val_bb, val_angles, val_hp = list_of_data_angles.list_of_data()
train_images_mark, train_bb_mark, train_angles_mark, train_hp_mark, val_images_mark, val_bb_mark, val_angles_mark, val_hp_mark = list_of_data_angles_markers.list_of_data()
train_images, train_bb, train_angles, train_hp = list_of_data_angles.rep_data_for_balance(train_images, train_bb, train_angles, train_hp)
print('total training points ' + str(len(train_images)))
print('total validation points ' + str(len(val_images)))
print('total training points marker' + str(len(train_images_mark)))
print('total validation points  marker' + str(len(val_images_mark)))

# end-to-end
loss_weights_sequence = [1, 1, 1, 1, 1, 1]

lr_seq = 1e-3
batch_size_all = 16
prev_epoch = 0
n_epochs = 1

opt = Adam(lr=lr_seq)
model.model.compile(loss=[cc_loss] * 6, loss_weights=loss_weights_sequence, optimizer=opt)
batch_size = batch_size_all
# print(model.model.layers[87].get_weights())
hist1 = model.model.fit_generator(data_generator(train_images, train_hp, train_bb, train_angles, batch_size),
                           steps_per_epoch=len(train_images) // batch_size,
                           epochs=n_epochs,
                           validation_data=data_generator(val_images, val_hp, val_bb, val_angles, batch_size),
                           validation_steps = 100, # len(val_images) // batch_size,
                           workers=8,
                           max_queue_size=16,
                           shuffle=True,
                           use_multiprocessing=True,
                           initial_epoch=prev_epoch,
                           callbacks=[cb]
                           )
hist['start'] = hist1.history
loss_weights_sequence = [1, 1, 1, 1, 1, 1]
# print(model.model.layers[87].get_weights())

lr_seq = 1e-3
batch_size_all = 16
prev_epoch = n_epochs
n_epochs += 1

opt = Adam(lr=lr_seq)
model.model.compile(loss=[cc_loss] * 6, loss_weights=loss_weights_sequence, optimizer=opt)
batch_size = batch_size_all
hist1 = model.model.fit_generator(data_generator(train_images_mark, train_hp_mark, train_bb_mark, train_angles_mark, batch_size),
                           steps_per_epoch=len(train_images_mark) // batch_size,
                           epochs=n_epochs,
                           validation_data=data_generator(val_images_mark, val_hp_mark, val_bb_mark, val_angles_mark, batch_size),
                           validation_steps = 100, # len(val_images) // batch_size,
                           workers=8,
                           max_queue_size=16,
                           shuffle=True,
                           use_multiprocessing=True,
                           initial_epoch=prev_epoch,
                           callbacks=[cb]
                           )
hist['start_markers'] = hist1.history

prev_epoch = n_epochs
n_epochs += 50
# model.unfreeze_down()
opt = Adam(lr=lr_seq)
model.model.compile(loss=[cc_loss] * 6, loss_weights=loss_weights_sequence, optimizer=opt)
batch_size = batch_size_all
hist1 = model.model.fit_generator(data_generator(train_images_mark, train_hp_mark, train_bb_mark, train_angles_mark, batch_size),
                           steps_per_epoch=len(train_images_mark) // batch_size,
                           epochs=n_epochs,
                           validation_data=data_generator(val_images_mark, val_hp_mark, val_bb_mark, val_angles_mark, batch_size),
                           validation_steps = 100, # len(val_images) // batch_size,
                           workers=8,
                           max_queue_size=16,
                           shuffle=True,
                           use_multiprocessing=True,
                           initial_epoch=prev_epoch,
                           callbacks=[cb]
                           )

hist['end2end_marker'] = hist1.history

# fine-tune final prediction
loss_weights_sequence = [0, 0, 0, 0, 0, 1]
lr_seq = 1e-4
prev_epoch = n_epochs
n_epochs += 50
opt = Adam(lr=lr_seq)
model.model.compile(loss=[cc_loss] * 6, loss_weights=loss_weights_sequence, optimizer=opt)
batch_size = batch_size_all
hist1 = model.model.fit_generator(data_generator(train_images_mark, train_hp_mark, train_bb_mark, train_angles_mark, batch_size),
                           steps_per_epoch=len(train_images_mark) // batch_size,
                           epochs=n_epochs,
                           validation_data=data_generator(val_images_mark, val_hp_mark, val_bb_mark, val_angles_mark, batch_size),
                           validation_steps = 100, # len(val_images) // batch_size,
                           workers=8,
                           max_queue_size=16,
                           shuffle=True,
                           use_multiprocessing=True,
                           initial_epoch=prev_epoch,
                           callbacks=[cb]
                           )
hist['finetune_marker'] = hist1.history

# fine-tune final prediction
loss_weights_sequence = [0, 0, 0, 0, 0, 1]
lr_seq = 1e-4
prev_epoch = n_epochs
n_epochs += 10
opt = Adam(lr=lr_seq)
model.model.compile(loss=[cc_loss] * 6, loss_weights=loss_weights_sequence, optimizer=opt)
batch_size = batch_size_all
hist1 = model.model.fit_generator(data_generator(train_images, train_hp, train_bb, train_angles, batch_size),
                           steps_per_epoch=len(train_images) // batch_size,
                           epochs=n_epochs,
                           validation_data=data_generator(val_images, val_hp, val_bb, val_angles, batch_size),
                           validation_steps =  100, # len(val_images) // batch_size,
                           workers=8,
                           max_queue_size=16,
                           shuffle=True,
                           use_multiprocessing=True,
                           initial_epoch=prev_epoch,
                           callbacks=[cb]
                           )
hist['finetune'] = hist1.history

sio.savemat('logs/history_fusion_maeonly', hist)
model.model.save_weights('logs/model_fusion_maeonly.h5')
# model.save('logs/model_multi_class_full.h5')

