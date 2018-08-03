"""
    Project sNNop
"""
import os
import pandas as pd
import math
import tensorflow as tf
from preprocessing import clean_data, generate_key_map
# from convNet import conv_net
from LSTM_layers import lstm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Data Import
"""
# Importing data
training_data = pd.read_csv("data/701.34067282.csv")
test_data = pd.read_csv("data/5328.28936538.csv")

# Generating key maps
keyNames, v_map = generate_key_map(training_data)
v_len = len(v_map)

# Audio Parameters
RATE = 8100
CHUNK = 512
KPS = (300/60) / 60 # Keys per second = 300kpm / 60


# Learning hyper-parameters
learning_rate = 0.01
n_hidden = 512
total_epoch = 1
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
  Phase 1: Create the computation graph
"""

# Construct model
# pred = conv_net(x, weights, biases, keep_prob)

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

# pred = conv_net(x, weights, biases, keep_prob)

model = lstm(X=X, layers=[128, 128], n_class=n_class)


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

input_batch_tf = tf.convert_to_tensor(X_train)
target_batch_tf = tf.convert_to_tensor(y_train)


input_qbatch, target_qbatch = tf.train.batch([input_batch_tf, target_batch_tf],
                                           batch_size=batch_size,
                                           num_threads=4,
                                           capacity=50000,
                                           allow_smaller_final_batch=True,
                                           enqueue_many=True)

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


"""
  Phase 2: Train the model
"""
with tf.Session(config=config) as sess:
    epoch_loss = 0
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(total_epoch):
        for batch in range(number_of_batches):
            print("Epoch: ", epoch, " Batch: ", batch, " of", number_of_batches)
            input_batch, target_batch = sess.run([input_qbatch, target_qbatch])

            _, loss = sess.run([optimizer, cost],
                               feed_dict={X: input_batch, Y: target_batch})
            epoch_loss += loss

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(epoch_loss))
        epoch_loss = 0

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    print('Optimization finished')

    """
      Make predictions
    """
    for batch in range(number_of_batches):
        print(" Batch: ", batch, " of", number_of_batches)
        input_batch, target_batch = sess.run([input_qbatch, target_qbatch])

        prediction = tf.cast(tf.argmax(model, 1), tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))


        predict, accuracy_val = sess.run([prediction, accuracy],
                                         feed_dict={X: input_batch, Y: target_batch})

        predicted = []
        real_keyList = []

        test_keys = training_data.iloc[:, -1:].values

        for idx, val in enumerate(test_keys):
            real_key = test_keys[idx][0]
            if str(real_key) != 'NONE':
                real_keyList.append(real_key)

        for idx, val in enumerate(predict):
            pred_key = keyNames[predict[idx]]
            if str(pred_key) != 'NONE':
                predicted.append(pred_key)

        print('\n=== Predictions ===')
        print('Real Key:', real_keyList)
        print('Predicted:', predicted)
        print('Accuracy:', accuracy_val)
