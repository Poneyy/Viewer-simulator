from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
import pickle

from RNN_utils import *

from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# arg conf
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/data.txt')
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=128)
ap.add_argument('-generate_length', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
SEQ_LENGTH = args['seq_length']
HIDDEN_DIM = args['hidden_dim']
WEIGHTS = args['weights']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']
# training data
ix_to_char = pickle.load(open("ix_to_char.pickle", "rb"))
VOCAB_SIZE = pickle.load(open("VOCAB_SIZE.pickle", "rb"))
# VOCAB_SIZE, ix_to_char = load_data_gen(DATA_DIR, SEQ_LENGTH)
# nouveau modele
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

if WEIGHTS != '':
    model.load_weights(WEIGHTS)
    print('\n\n')
    print(generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char))
    print('\n\n')
else:
    print('J\'ai besoin d\'un modele')