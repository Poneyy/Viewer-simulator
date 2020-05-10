from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed

from keras.callbacks import ModelCheckpoint

import argparse
from RNN_utils import *

from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/data.txt')
ap.add_argument('-batch_size', type=int, default=128)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=128)
ap.add_argument('-generate_length', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")



# Generate some sample before training to know how bad it is!
# generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char)

if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
  nb_epoch = 0

#saving definition
filepath='checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch+1)
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Training if there is no trained weights specified
if args['mode'] == 'train' or WEIGHTS == '':
  print('Debut du training')
  while True:
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    filepath='checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch+1)
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1, callbacks=callbacks_list)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    # if nb_epoch % 2 == 0:
      # model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# fit the model
# Else, loading the trained weights and performing generation only
elif WEIGHTS != '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  #ajout de test
  # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
  ##############
  print('\n\n')
  for i in range(10):
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print('\n\n')
else:
  print('\n\nNothing to do!')