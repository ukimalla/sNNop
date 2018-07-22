import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, LSTMCell, Flatten
import pandas as pd
import math
from preprocessing import clean_data, generate_key_map

"""
Data Import
"""
# Importing data
training_data = pd.read_csv("data/701.34067282.csv")
test_data = pd.read_csv("data/5328.28936538.csv")

# Generating key maps
keyNames, v_map = generate_key_map(training_data)
v_len = len(v_map)


"""
Testing Parameters
"""

# Audio Parameters
RATE = 8100
CHUNK = 512
KPS = (300/60) / 60 # Keys per second = 300kpm / 60

# Learning hyper-parameters
learning_rate = 0.01
n_hidden = 512
total_epoch = 500
n_step = CHUNK  # the length of the input sequence
# n_input = n_class = v_len  # the size of each input
n_input = 1
n_class = v_len
batch_size = 150


"""
Pre-processing
"""

# Removes unnecessary "NONE" chunks and encodes y values
X_train, y_train = clean_data(training_data, v_map, chunk_size=CHUNK)
number_of_batches = int(math.ceil(len(X_train) / batch_size))

"""
Model
"""


model = Sequential()
model.add(Conv1D(filters=32, kernel_size=30, strides=1, padding="causal", input_shape=(512, 1)))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(LSTMCell(512, dropout=0.25))
model.add(Dense(n_class, activation="softmax"))


model.compile(optimizer=keras.optimizers.adam, loss=keras.losses.sparse_categorical_crossentropy(
    y_true=y_train, y_pred=model.predict(X_train)))

model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=100)









