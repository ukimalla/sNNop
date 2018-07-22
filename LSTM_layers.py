import tensorflow as tf


def basic_lstm(X, n_hidden, n_class):

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
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)

    # Create the RNN
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs : [batch_size, max_time, cell.output_size]

    # Transform the output of RNN to create output values
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]
    # [batch_size, cell.output_size]
    model = tf.matmul(outputs, W) + b
    # [batch_size, n_classes]

    return model
