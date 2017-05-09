# -*- coding: utf-8 -*-

import tensorflow as tf

#x = tf.constant([[2,3]])
x = tf.Variable(tf.random_normal([2,3,4,5]))

print(tf.shape(x))  # Tensor("Shape_1:0", shape=(4,), dtype=int32)
print(x.get_shape()) # (2, 3, 4, 5)
print(x.get_shape().as_list()) # [2, 3, 4, 5] ==> standard python list

x_shape = x.get_shape().as_list()
print(x_shape[1:])  # [3, 4, 5]
print(x_shape[-1:]) # [5]


