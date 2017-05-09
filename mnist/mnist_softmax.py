"""
  - This tutorial uses Softmax Regresion model to train and predict digits
  (0 to 9) in MNIST data
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

def main(_):

  # Load input data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784]) # input
  W = tf.Variable(tf.zeros([784, 10]))        # weight
  b = tf.Variable(tf.zeros([10]))             # bias
  y = tf.matmul(x, W) + b                     # model output

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Launch the model in session
  sess = tf.InteractiveSession()

  # Initialize
  tf.global_variables_initializer().run()

  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test the trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))   # type bool
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
                      type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)