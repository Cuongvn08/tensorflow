# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

'''
    Tutorial: http://stackoverflow.com/questions/34240703/difference-between-
    tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
'''

# y_hat = W*x + b
# y_hat is 2x3 array whereas:
# 1. The rows correspond to the training instances.
# 2. The columns correspond to classes.
arr = np.array([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]])
y_hat = tf.convert_to_tensor(arr)
print(arr)
'''
[[ 0.5  1.5  0.1]
 [ 2.2  1.3  1.7]]
'''

y_hat_softmax = tf.nn.softmax(y_hat)

with tf.Session() as sess:
    tf.initialize_all_variables()
    
    sess.run(y_hat_softmax)    
    print(y_hat_softmax.eval())
    '''
    [[ 0.227863    0.61939586  0.15274114]
    [ 0.49674623  0.20196195  0.30129182]]
    '''
    
    argmax = tf.argmax(y_hat_softmax,1)
    print(argmax.eval())
    '''
    [1 0]
    '''