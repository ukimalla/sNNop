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
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""------------
    Data Import
------------"""
# Importing data
training_data = pd.read_csv("data/302.634269701.csv")
training_data2 = pd.read_csv("data/701.34067282.csv")
training_data3 = pd.read_csv("data/502.730691378.csv")
training_data4 = pd.read_csv("data/8397.60171287.csv")
training_data5 = pd.read_csv("data/210.547412155.csv")
training_data6 = pd.read_csv("data/1755.15627725.csv")
training_data7 = pd.read_csv("data/5142.83004162.csv")
training_data8 = pd.read_csv("data/4660.65770671.csv")
training_data9 = pd.read_csv("data/5328.28936538.csv")
training_data10 = pd.read_csv("data/3664.49261897.csv")


test_data = pd.read_csv("data/1732.94239451.csv")


'''
Keymaps
'''
# Generating key maps
keyNames, v_map = generate_key_map([training_data, training_data4, training_data5,
                                    training_data6, training_data7, training_data9,
                                    test_data])

v_len = len(v_map)


"""----------------
    Pre-processing
----------------"""

# Removes unnecessary "NONE" chunks and encodes y values
# X_train, y_train = clean_data(training_data, v_map, chunk_size=CHUNK) 
data = np.load("saveData.npz")
# X_train = data['x'][:3600]100
# y_train = data['y'][:3600]

X_train, y_train = clean_data(training_data, v_map, 512)
X_train2, y_train2 = clean_data(training_data4, v_map, 512)
X_train3, y_train3 = clean_data(training_data4, v_map, 512)
X_train4, y_train4 = clean_data(training_data4, v_map, 512)
X_train5, y_train5 = clean_data(training_data5, v_map, 512)
X_train6, y_train6 = clean_data(training_data6, v_map, 512)
X_train7, y_train7 = clean_data(training_data7, v_map, 512)
X_train8, y_train8 = clean_data(training_data4, v_map, 512)
X_train9, y_train9 = clean_data(training_data9, v_map, 512)
X_train10, y_train10 = clean_data(training_data4, v_map, 512)


X_train = np.vstack((X_train, X_train2, X_train3, X_train4, X_train5,
                     X_train6, X_train7, X_train8, X_train9, X_train10))
y_train = np.concatenate((y_train, y_train2, y_train3, y_train4, y_train5,
                          y_train6, y_train7, y_train8, y_train9, y_train10))

len(X_train)
len(y_train)

X_test, y_test = clean_data(test_data, v_map, 512)

X_test = X_test[:1300]
y_test = y_test[:1300]

X_train = X_train[:12400]
y_train = y_train[:12400]
#
# X_train /= 32768
# y_train /= 32768
# X_test /= 32768
# y_test /= 32768

len(X_test)

'''-----------
    Parameters
----------'''

# Audio Parameters
RATE = 8100
CHUNK = 512
KPS = (300/60) / 60 # Keys per second = 300kpm / 60

# Learning hyper-parameters
learning_rate = 0.001
n_hidden = 512
total_epoch = 10000
n_step = CHUNK
# n_input = n_class = v_len  # the size of each input
n_input = 1
n_class = v_len
batch_size = 300
number_of_batches = int(math.ceil(len(X_train) / batch_size))

# RNN Parameters
rnn_hidden_layers = [128, 256]
num_time_steps = 25


"""------------------------------------
  Phase 1: Create the computation graph
------------------------------------"""
# Construct model3500
# pred = conv_net(x, weights, biases, keep_prob)

X = tf.placeholder(tf.float32, [None, n_step, 1], name="X")
Y = tf.placeholder(tf.int32, [None], name="Y")

# 1st Convolutional Layer
n_hidden = 512
conv1 = tf.layers.conv1d(inputs=X, filters=16, kernel_size=10, use_bias=True, name="conv1")
weights = tf.get_default_graph().get_tensor_by_name(os.path.split(conv1.name)[0]+'/kernel:0')
tf.summary.histogram("conv1_w", weights)
# Max Pool
maxpool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=2, name="maxpool1")


# 2nd Convolutional Layer
conv2 = tf.layers.conv1d(inputs=maxpool1, filters=32, kernel_size=10, use_bias=True, name="conv2")
weights2 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv2.name)[0]+'/kernel:0')
tf.summary.histogram("conv2_w", weights2)
# Max Pool
maxpool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4, strides=1, name="maxpool2")

# 3rd Convolutional Layer
conv3 = tf.layers.conv1d(inputs=maxpool2, filters=64, kernel_size=10, use_bias=True, name="conv3")
weights3 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv1.name)[0]+'/kernel:0')
tf.summary.histogram("conv3_w", weights3)
# Max Pool
maxpool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=4, strides=2, name="maxpool3")


mp_flat_dim = maxpool3.get_shape().as_list()[1] * maxpool3.get_shape().as_list()[2]

n_hidden = 512
seq_batch = tf.reshape(maxpool3, [-1, num_time_steps, mp_flat_dim], name="flatten")


# Flattening the convolutional layers
# flatten = tf.layers.Flatten()(seq_batch)32, 64, 128


# RNN Layers32, 64, 128
# initial_state = state = tf.Variable(tf.random_normal(flatten.shape, tf.float32))
outputs, states = lstm(seq_batch, layers=rnn_hidden_layers, n_class=v_len)
W = tf.Variable(tf.random_normal([rnn_hidden_layers[-1], n_class]))
tf.summary.histogram("dense_W", W)

b = tf.Variable(tf.random_normal([n_class]))
tf.summary.histogram("dense_b", b)


#                output (pred. of forth letter)
#                 | (W,b)
#                outputs (hidden)
#       |    |    |
# RNN: [t1]-[t2]-[t3]
#       x1   x2   x3

rnn_outputs_flat = tf.reshape(outputs, [-1, rnn_hidden_layers[-1]])
model = tf.matmul(rnn_outputs_flat, W) + b

# Cost Function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y), name="cost")


with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


input_batch_tf = tf.convert_to_tensor(X_train)
target_batch_tf = tf.convert_to_tensor(y_train)

input_qbatch, target_qbatch = tf.train.batch([X_train, y_train],
                                             batch_size=batch_size,
                                             num_threads=4,
                                             capacity=5000,
                                             allow_smaller_final_batch=False,
                                             enqueue_many=True)


tf.summary.scalar("cost", cost)
audio = tf.cast(tf.reshape(input_qbatch, (batch_size, CHUNK, 1)), dtype=tf.float32)

tf.summary.audio("audioSample", audio,
                 sample_rate=8100, max_outputs=10)
saver = tf.train.Saver()

prediction = tf.cast(tf.argmax(tf.nn.softmax(model, axis=1), axis=1), tf.int32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

"""-----------------------
  Phase 2: Train the model
--------------------------"""
with tf.Session() as sess:
    writer = tf.summary.FileWriter("/home/ukimalla/PycharmProjects/nnSnoop/tensorboard/7")
    merged_summary = tf.summary.merge_all()

    epoch_accuracy = 0
    epoch_loss = 0
    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)

    # Coordinate the loading of image files.bar graphinput_batch_tf
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(total_epoch):
        for batch in range(number_of_batches):
            # print("Epoch: ", epoch, " Batch: ", batch, " of", number_of_batches)
            input_batch, target_batch = sess.run([input_qbatch, target_qbatch])

           # print(np.shape(input_batch))bar graph
            # target_batch = tf.reshape(target_batch, [in   t(batch_size/num_time_steps), num_time_steps, maxpool2.get_shape().as_list()[-2] * maxpool2.get_shape().as_list()[-1]])

            # target_batch = sess.run(target_batch)y_train

            batch_accuracy, _, loss, s = sess.run([accuracy, optimizer, cost, merged_summary], feed_dict={X: input_batch, Y: target_batch})
            writer.add_summary(s, batch)

            epoch_loss += loss
            epoch_accuracy += batch_accuracy

            sess.run(audio)

        if epoch % 10 == 0:
            test_input = X_test
            test_target = y_test

            save_path = saver.save(sess, "/home/ukimalla/PycharmProjects/nnSnoop//tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

            predict, accuracy_val = sess.run([prediction, accuracy],
                                             feed_dict={X: test_input, Y: test_target})

            print(predict)

            predicted = []
            real_keyList = []

            for idx, val in enumerate(test_target):
                real_key = keyNames[val]
                # if str(real_key) != 'NONE':
                real_keyList.append(real_key)

            for idx, val in enumerate(predict):
                pred_key = keyNames[val]
                # if str(pred_key) != 'NONE':y_train
                predicted.append(pred_key)

            print('\n=== Predictions ===')
            print('Real Key:', real_keyList)
            print('Predicted:', predicted)
            print('Accuracy:', accuracy_val)

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(epoch_loss),
              'accuracy=', '{:.6f}'.format(epoch_accuracy/number_of_batches))

        epoch_loss = 0
        epoch_accuracy = 0




    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    print('Optimization finished')

    """
      Make predictions
    """

    # for batch in range(number_of_batches):
    # print(" Batch: ", batch, " of", number_of_batches)
    input_batch, target_batch = sess.run([input_batch_tf, target_batch_tf])



    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))


    predict, accuracy_val = sess.run([prediction, accuracy],
                                     feed_dict={X: input_batch, Y: target_batch})

    predicted = []
    real_keyList = []

    test_keys = training_data.iloc[:, -1:].valuesy_train

    for idx, val in enumerate(test_keys):
        real_key = test_keys[idx][0]
        #if str(real_key) != 'NONE':
        real_keyList.append(real_key)

    for idx, val in enumerate(predict):
        pred_key = keyNames[predict[idx]]
        #if str(pred_key) != 'NONE':
        predicted.append(pred_key)




    print('\n=== Predictions ===')
    print('Real Key:', real_keyList)
    print('Predicted:', predicted)
    print('Accuracy:', accuracy_val)