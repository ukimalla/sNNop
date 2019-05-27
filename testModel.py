import tensorflow as tf
import numpy as np
import pandas as pd


tf.reset_default_graph()

model = tf.get_variable("model", shape=[-1, ])
saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore("cd PycharmProjects/nnSnoop/tmp/model.ckpt")

