"""
  - This tutorial uses multilayer convolutional network to train and predict digits
  (0 to 9) in MNIST data. Surely, this model gives a better result than Softmax
  Regression model.
"""

# Import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

# Create weight def
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# Create bias def
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Create convolution def
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding='SAME')

# Create pooling def
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  # Load input data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot = True)

  # Reshape input x to a 4d tensor
  x = tf.placeholder(tf.float32, shape=[None, 784])
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # 1st convolutional layer
  # Create weight & bias tensor
  W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5 patch, 1 input, 32 outputs (or 32 neurons)
  b_conv1 = bias_variable([32]) # 32 output channels
  # Convolve x_image with weight tensor, add bias and apply max pool
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # 2nd convolutional layer
  # Create weight & bias tensor
  W_conv2 = weight_variable([5, 5, 32, 64]) # 5x5 patch, 32 inputs, 64 outputs (or 64 neurons)
  b_conv2 = bias_variable([64])
  # Convolve h_pool1 with weight tensor W_conv2, add bias and apply max pool
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Create densely connected layer
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Calculate dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Add readout layer
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Launch the model in session
  sess = tf.InteractiveSession()

  # Initialize
  tf.global_variables_initializer().run()

  # Train
  for i in range(20000):
      batch = mnist.train.next_batch(50)

      if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict = {x:batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))

      train_step.run(feed_dict = {x: batch[0],
                                  y_: batch[1],
                                  keep_prob: 0.5})

  print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
                                                    y_: mnist.test.labels,
                                                    keep_prob: 1.0}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
                      type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
