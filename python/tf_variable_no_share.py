# -*- coding: utf-8 -*-

'''
Reference:
    @https://www.tensorflow.org/programmers_guide/variables
'''

import tensorflow as tf


def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')

    print(conv1_weights)
    print(conv1_biases)
    print(conv2_weights)
    print(conv2_biases)
    
    return tf.nn.relu(conv2 + conv2_biases)

# create 4 different variables for weights and biases
image_1 = tf.placeholder(tf.float32, shape=[None, 784, 200, 32])
my_image_filter(image_1)
'''

Tensor("conv1_weights/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv1_biases/read:0", shape=(32,), dtype=float32)
Tensor("conv2_weights/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv2_biases/read:0", shape=(32,), dtype=float32)
'''


# create 4 different variables for weights and biases
image_2 = tf.placeholder(tf.float32, shape=[None, 784, 200, 32])
my_image_filter(image_2)
'''
Tensor("conv1_weights_1/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv1_biases_1/read:0", shape=(32,), dtype=float32)
Tensor("conv2_weights_1/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv2_biases_1/read:0", shape=(32,), dtype=float32)
'''



