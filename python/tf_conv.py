# -*- coding: utf-8 -*-

import tensorflow as tf

input = tf.Variable(tf.random_normal([1,3,3,5])) # 3x3x5 image


# to understand padding argument
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op_pad_valid = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
print(op_pad_valid) # Tensor("Conv2D_8:0", shape=(1, 1, 1, 1), dtype=float32)

op_pad_same = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')
print(op_pad_same) # Tensor("Conv2D_11:0", shape=(1, 3, 3, 1), dtype=float32)