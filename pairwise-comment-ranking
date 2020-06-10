#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from keras.layers import Bidirectional, CuDNNGRU, SpatialDropout1D, GlobalAveragePooling1D, concatenate, Dropout, Dense
from keras.models import Model
from fastText import load_model
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF

classes = ['requires_reply']

path = './'
training_filename = ''
test_filename = ''
fasttext_filename = 'cc.de.300.bin'

ft_model = load_model(path+fasttext_filename)
n_features = ft_model.get_dimension()
print('Dimensions '+str(n_features))

train = pd.read_csv(path + training_filename)
test = pd.read_csv(path + test_filename)

def get_session(gpu_fraction=0.9):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

def text_to_vector(text):
    words = text.split()
    window = words[-window_length:]
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df[comment_field].values):
        x[i, :] = text_to_vector(comment)

    return x

def get_model_multi_input(num_filters=64):
    
    inp_comment_1 = Input(shape=(window_length, n_features), name='inp_comment_1')
    inp_comment_2 = Input(shape=(window_length, n_features), name='inp_comment_2')

    comment_1 = SpatialDropout1D(0.1)(inp_comment_1)
    comment_2 = SpatialDropout1D(0.1)(inp_comment_2)

    shared_bidirectional_cudnngru = Bidirectional(CuDNNGRU(num_filters, return_sequences=True))

    comment_1 = shared_bidirectional_cudnngru(comment_1)
    comment_2 = shared_bidirectional_cudnngru(comment_2)

    avg_pool_comment_1 = GlobalAveragePooling1D()(comment_1)
    avg_pool_comment_2 = GlobalAveragePooling1D()(comment_2)

    conc = concatenate([avg_pool_comment_1, avg_pool_comment_2])
    conc = Dropout(0.1)(conc)
    x = Dense(len(classes), activation="sigmoid")(conc)
    return Model(inputs=[inp_comment_1, inp_comment_2], outputs=x)

model = get_model_multi_input()
model.compile(optimizer='adam',
                  loss={'x': 'binary_crossentropy',
                        'parent_output': 'binary_crossentropy', 'previous_output': 'binary_crossentropy', 'main_output': 'binary_crossentropy'},)

model.fit_generator(training_generator, steps_per_epoch=training_steps_per_epoch, epochs=epochs,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch, verbose=1,
                        callbacks=[earlyStopping, model_checkpoint, metrics],
                        max_queue_size=10)
