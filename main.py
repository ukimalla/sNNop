"""
  Autocompletion of the last character of words
  Given the first three letters of a four-letters word, learn to predict the last letter
"""
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

keyNames = ['cmd_r', 'cmd', 'ctrl', 'alt_r', 'caps_lock', 'up', 'down', 'right', 'comma', 'alt_l', 'tab', 'shift', "'",
            'space', '-', '/', '.', '1', '0', '3', '2', 'backspace', '4', '7', '6', '9', '5', ';', '=', 'v', 'cmd combo'
            ,'8', 'shift_r', '[', ']', str.rstrip("\ "), 'a', '`', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm'
            ,'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'enter', 'y', 'x', 'z', 'left', 'NONE', '!', '@', '#',
            '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', '<', '>', '?', '~']

# index array of characters in vocab
v_map = {n: i for i, n in enumerate(keyNames)}
v_len = len(v_map)

training_data = pd.read_csv("data/E.csv")

CHUNK = 512

# training data (character sequences)
test_data = pd.read_csv("data/406.786712543.csv")

def make_batch(seq_data):
    input_batch = []
    target_batch = []

    input_batch = seq_data.iloc[:, :-1].values

    blankCounter = 0
    for i, seq in enumerate(seq_data):

    input_batch = np.reshape(input_batch, (len(input_batch), CHUNK, 1))
    for seq in seq_data.iloc[:, -1:].values:
        target = v_map[seq[0]]
        target_batch.append(target)

    return input_batch, target_batch

learning_rate = 0.01
n_hidden = 10
total_epoch = 100
n_step = CHUNK  # the length of the input sequence
# n_input = n_class = v_len  # the size of each input
n_input = 1
n_class = v_len

"""
  Phase 1: Create the computation graph
"""
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

#                output (pred. of forth letter)
#                 | (W,b)
#                outputs (hidden)
#       |    |    |
# RNN: [t1]-[t2]-[t3]
#       x1   x2   x3

# Create an LSTM cell
cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# Apply dropout for regularization
# cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)

# Create the RNN
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# outputs : [batch_size, max_time, cell.output_size]

# Transform the output of RNN to create output values
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
# [batch_size, cell.output_size]
model = tf.matmul(outputs, W) + b
# [batch_size, n_classes]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

"""
  Phase 2: Train the model
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    input_batch, target_batch = make_batch(training_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={X: input_batch, Y: target_batch})
        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))

    print('Optimization finished')

    """
      Make predictions
    """
    seq_data = training_data  # test_data
    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

    input_batch, target_batch = make_batch(seq_data)

    predict, accuracy_val = sess.run([prediction, accuracy],
                                     feed_dict={X: input_batch, Y: target_batch})

    predicted = []
    real_keyList = []

    test_keys = test_data.iloc[:, -1:].values
    for idx, val in enumerate(predict):
        real_key = test_keys[idx][0]
        if str(real_key) != 'NONE':
            real_keyList.append(real_key)

        pred_key = keyNames[predict[idx]]
        if str(pred_key) != 'NONE':
            predicted.append(pred_key)

    print('\n=== Predictions ===')
    print('Real Key:', real_keyList)
    print('Predicted:', predicted)
    print('Accuracy:', accuracy_val)

