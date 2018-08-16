'''
CNN to classify MNIST handwritten digits
'''

import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# # Read in MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("data/mnist", one_hot=True)
#
# # Parameters
# learning_rate = 0.001
# training_iters = 200000
# batch_size = 128
# display_step = 10
#
# # Network Parameters
# n_input = 784  # input image shape = 28*28 grey scale
# n_classes = 10  # 10 classes (0-9 digits)
# dropout = 0.75  # probability to keep units during dropout
#
# # tf Graph input
# x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Wrappers
def reshape(x, xdim, ydim):
    return tf.reshape(x, shape=[-1, xdim, ydim, 1])


def conv1d(x, W, b, stride=1):
    x = tf.nn.conv1d(x, W, strides=stride, padding='SAME')
    return tf.nn.relu(x + b)


def maxpool1d(x, size=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, size, 1, 1], strides=[1, stride, 1, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = reshape(x, 28, 28)

    # Convolution Layer
    conv1 = conv1d(x, weights['wc1'], biases['bc1'])
    print("Conv 1 = ", conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool1d(conv1, size=2, stride=2)
    print("Conv 1 = ", conv1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    out = tf.nn.dropout(fc1, dropout)

    tf.summary.histogram("wc1_hist", weights['wc1'])
    tf.summary.histogram("wdc1_hist", weights['wd1'])

    # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # fully connected, ? inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([14*14*32, 1024]))
    # 1024 inputs, 10 outputs (class prediction)
    # 'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024]))
    # 'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
# pred = conv_net(x, weights, biases, keep_prob)
#
# # Define loss and optimizer
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     step = 1
#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         # Run optimization op (backprop)
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#
#         if step % display_step == 0:
#             # Calculate batch loss and accuracy
#             l, acc = sess.run([loss, accuracy], feed_dict={x: batch_x,
#                                                            y: batch_y,
#                                                            keep_prob: 1.})
#             print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(l) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")
#
#     # Calculate accuracy for 256 mnist test images
#
#     print("Testing Accuracy:", \
#           sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#                                         y: mnist.test.labels[:256], keep_prob: 1.}))
#
#
#     correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
#     accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
#
#     n_batches = int(mnist.test.num_examples/batch_size)
#     total_correct_pred = 0
#
#     for i in range(n_batches):
#         X_batch, Y_batch = mnist.test.next_batch(batch_size)
#         accuracy_batch = sess.run(accuracy, feed_dict={x: X_batch, y:Y_batch, keep_prob: 1.})
#         total_correct_pred += accuracy_batch
#
#     print('Accuracy {0}'.format(total_correct_pred/mnist.test.num_examples))
