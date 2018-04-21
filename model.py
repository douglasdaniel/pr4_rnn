#!/usr/bin/env python3

import tensorflow as tf
import pdb
from math import sqrt
import numpy as np

LEARN_RATE = 1e-3
num_units = 5
seq_length = 10


def model(data):
    batch_size = data.shape[0]
    
    cnn_output = []

    # Instantiate a LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units,reuse=tf.AUTO_REUSE)

    for sequence in tf.unstack(data, batch_size):
        input_layer = sequence

        # Convolutional Layer 1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="valid",
            activation=tf.nn.relu)
        
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
        
        # Convolutional Layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
        
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[3,3],strides=2)
        
        # Convolutional Layer 3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=64,
            kernel_size=[5, 5],
            padding="valid",
            activation = tf.nn.relu)
        
        # Pooling Layer 3
        pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[3,3],strides=2)
        
        # Fully-Connected Layer
        fulcon = tf.reshape(pool3, [-1, pool3.shape[1]*pool3.shape[2]*64])

        cnn_output.append(fulcon)

    cnn_output = tf.stack(cnn_output)
    rnn_input = cnn_output
        
    # Define RNN network
    h_val, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_input, dtype=tf.float32)

    # Collection of all the final output
    final_output = tf.zeros(shape=[batch_size, 0, 14])
    w_fc  = tf.Variable(tf.random_normal([num_units,14]))
    b_fc = tf.Variable(tf.random_normal([14]))
    for i in np.arange(seq_length):
        temp = tf.reshape(h_val[:,i,:], [batch_size, num_units])
        output = tf.matmul(temp, w_fc) + b_fc
        output = tf.reshape(output, [-1,1,14])
        final_output = tf.concat([final_output, output], axis=1)
    final_output = tf.reshape(final_output, [-1,10,7,2])
        
    return final_output




