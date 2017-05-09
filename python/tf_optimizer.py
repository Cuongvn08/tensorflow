# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# create placeholder for input and output
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

# build model: y = ax + b
a = tf.Variable(1.0, name='a')
b = tf.Variable(2.0, name='b')
pred = tf.add(tf.multiply(x,a), b)

# compute loss and optimizer
loss = tf.square(pred - y) # L2
op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#op = tf.train.AdamOptimizer(0.001).minimize(loss)
#op = tf.train.RMSPropOptimizer(0.001).minimize(loss)

# train
loss_array = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        x_feed = np.random.rand()
        y_feed = x_feed*5 + 10
        
        rLoss,_ = sess.run([loss, op], feed_dict={x: x_feed, y: y_feed})
        print(rLoss)

        loss_array.append(rLoss)
        
# show loss
plt.plot(loss_array)
plt.show()
