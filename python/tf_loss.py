# -*- coding: utf-8 -*-

'''
1. Difference:
- tf.nn.sparse_softmax_cross_entropy_with_logits: 
    + label shape must be [batch_size]
    + dtype is int32 or int64
    + Each label is an integer in range [0, num_classes - 1]
    + E.g. num_classes = 10, batch_size = 5
        [1]
        [2]
        [9]
        [0]
        [4]

- tf.nn.softmax_cross_entropy_with_logits:
    + label shape must be [batch_size, num_classes]
    + dtype is float32, float64
    + E.g. num_classes = 10, batch_size = 5
        [0][1][0][0][0][0][0][0][0][0]
        [0][0][0][0][0][0][0][0][0][1]
        [0][0][0][0][0][1][0][0][0][0]
        [0][0][0][1][0][0][0][0][0][0]
        [0][0][0][0][1][0][0][0][0][0]
        ==> each row is one-hot encoded        
'''