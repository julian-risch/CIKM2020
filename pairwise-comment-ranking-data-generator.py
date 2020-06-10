import sys, os, re, csv, codecs, numpy as np, pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM,CuDNNGRU, GRU, Embedding, SpatialDropout1D, Dropout, Activation, BatchNormalization,     ELU, concatenate, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, maximum, average, Concatenate, Reshape,     Flatten, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from fastText import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from collections import OrderedDict
import logging
from sklearn.metrics import roc_auc_score, classification_report, log_loss, f1_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_session(gpu_fraction=0.9):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

commented = pd.read_csv('/mnt/data/datasets/newspapers/guardian/comments_commented_on_by_author.csv')
commented = commented[['article_id', 'comment_id', 'comment_text']]
commented['class'] = True
not_commented = pd.read_csv('/mnt/data/datasets/newspapers/guardian/comments_not_commented_on_by_author.csv')
not_commented = not_commented[['article_id', 'comment_id', 'comment_text']]
not_commented['class'] = False


all_data = commented.append(not_commented)


def normalize(s):
    # transform to lowercase characters
    s = str(s)
    s = s.lower()
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|\n])', ' ', s)
    return s


def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x


def df_to_data(df, datacolumn):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df[datacolumn].values):
        x[i, :] = text_to_vector(comment)

    return x

def probabilities_to_classes(a):
    if a[0]>a[1]:
        return [1,0]
    else:
        return [0,1]


commented = commented.sort_values('article_id')
not_commented = not_commented.sort_values('article_id')

print('Loading FT model')
ft_model = load_model('/mnt/data/embeddings/fasttext_guardian_comments/guardian-twokenized-lower-50.bin')
n_features = ft_model.get_dimension()
print(n_features)

window_length = 100


train_commented, test_commented, train_not_commented, test_not_commented =train_test_split(commented, not_commented, test_size=0.1, shuffle=True)

batch_size = 64
epochs = 50
dropout = 0.2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe_positive, dataframe_negative, batch_size=32, dim=(100,50), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        assert(len(dataframe_positive) == len(dataframe_negative))
        self.df_positive = dataframe_positive
        self.df_negative = dataframe_negative
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df_positive) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df_positive))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [np.empty((self.batch_size, *self.dim, self.n_channels)), np.empty((self.batch_size, *self.dim, self.n_channels))]
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        positive_list = self.df_positive.iloc[list_IDs_temp]
        negative_list = self.df_negative.iloc[list_IDs_temp]
        
        swap_indices = np.random.choice(self.batch_size, self.batch_size//2, replace=False)
        all_indices = np.arange(0, self.batch_size)
        first_output = positive_list.iloc[swap_indices].append(negative_list.iloc[np.setdiff1d(all_indices, swap_indices)]).sort_values('article_id')
        second_output = negative_list.iloc[swap_indices].append(positive_list.iloc[np.setdiff1d(all_indices, swap_indices)]).sort_values('article_id')
        
        X[0] = (df_to_data(first_output, 'comment_text'))
        y = ~(first_output['class'].values)
        
        X[1] = (df_to_data(second_output, 'comment_text'))

        

        return X, y


# Parameters
params = {'dim': (window_length, n_features),
          'batch_size': 64,
          'shuffle': True}

training_generator = DataGenerator(train_commented,train_not_commented, **params)
validation_generator = DataGenerator(test_commented, test_not_commented , **params)


print('Build model extra input...')
inp = Input(shape=(window_length, n_features), name='word1')
inp2 = Input(shape=(window_length, n_features), name='word2')

spatial_dropout =  SpatialDropout1D(dropout)
bi_gru = Bidirectional(CuDNNGRU(16, return_sequences=True))
max_pooling = GlobalMaxPool1D()

w1 = spatial_dropout(inp)
w2 = spatial_dropout(inp2)

w1 = bi_gru(w1)
w2 = bi_gru(w2)

w1 = max_pooling(w1)
w2 = max_pooling(w2)

merged = concatenate([w1, w2], axis=-1)

x = Dense(1, activation="sigmoid")(merged)

model = Model(inputs=[inp, inp2], outputs=x)

print(model.summary())
optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=2, verbose=0, mode='auto', restore_best_weights=True)

#input_data = [train_commented_x, train_not_commented_x]
#validation_data = ([test_commented_x, test_not_commented_x], y_test)
model.fit_generator(generator=training_generator, validation_data=validation_generator, verbose=1, callbacks=[early_stopping], epochs=epochs)

#y_val_pred = model.predict(x_val, verbose=1, batch_size=1024)
