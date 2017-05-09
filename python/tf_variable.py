# -*- coding: utf-8 -*-

'''
Reference:
    @https://www.tensorflow.org/programmers_guide/variables
'''

import tensorflow as tf

## how to create variable

# Create two separable variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

print(weights)  # Tensor("weights/read:0", shape=(784, 200), dtype=float32)
print(biases)   # Tensor("biases/read:0", shape=(200,), dtype=float32)

# create a variable with a specific device
with tf.device('/cpu:0'):
    v = tf.Variable(tf.zeros([100]), name='v')
    print(v) # Tensor("v/read:0", shape=(100,), dtype=float32, device=/device:CPU:0)

