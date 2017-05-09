# -*- coding: utf-8 -*-
'''
Reference:
    @https://www.tensorflow.org/programmers_guide/variables
'''

import tensorflow as tf


def conv_relu(input, kernel_shape, bias_shape):
    # create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    print(weights)
    
    # create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    print(biases)
    
    # apply conv
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    
    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
        
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        relu2 = conv_relu(relu1, [5, 5, 32, 32], [32]) 
        
        return relu2

image_1 = tf.placeholder(tf.float32, shape=[None, 784, 200, 32])
result1 = my_image_filter(image_1)
'''
Tensor("conv1/weights/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv1/biases/read:0", shape=(32,), dtype=float32)
Tensor("conv2/weights/read:0", shape=(5, 5, 32, 32), dtype=float32)
Tensor("conv2/biases/read:0", shape=(32,), dtype=float32)
'''

# if we continue apply my_image_filter function to image2, then it will generate an error
# due to duplicate variable names.
# to solve this the duplicate problem, we must add tf.get_variable_scope().reuse_variables()
tf.get_variable_scope().reuse_variables()
image_2 = tf.placeholder(tf.float32, shape=[None, 784, 200, 32])
result2 = my_image_filter(image_2)

