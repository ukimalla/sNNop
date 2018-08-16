import tensorflow as tf


def lstm(X, layers: [], n_class):



    # Create an LSTM cell
    rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size), output_keep_prob=0.80) for size in layers]


    # Apply dropout for regularization
    cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # Create the RNN
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs : [batch_size, max_time, cell.output_size]


    # Transform the output of RNN to create output values
    # outputs = tf.transpose(outputs, [1, 0, 2])
    # outputs = outputs
    # # [batch_size, cell.output_size]
    # model = tf.matmul(outputs, W) + b
    # [batch_size, n_classes]

    return outputs, states