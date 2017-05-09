# -*- coding: utf-8 -*-
"""
@overview: http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
@summary: prevent overfitting with dropout and regularization
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


# loading data
with tf.name_scope('load_data'):
    pickle_file = 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        print('')        
        '''
        Training set (20000, 28, 28) (20000,)
        Validation set (1000, 28, 28) (1000,)
        Test set (1000, 28, 28) (1000,)    
        '''
    
    
# reformat data
with tf.name_scope('reformat_data'):
    image_size = 28
    num_labels = 10
    
    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
    
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)  
    print('')
    '''
    Training set (20000, 784) (20000, 10)
    Validation set (1000, 784) (1000, 10)
    Test set (1000, 784) (1000, 10)
    '''  
    
################################################################################    
# logistic regression with L2 loss function
with tf.name_scope('logistic_regression_with_L2_loss'):
    print('')
    print('logistic_regression_with_L2_loss')
    
    # build model
    train_subset = 10000
    beta = 0.01
    graph = tf.Graph()
    
    with graph.as_default():
        # Input data.
        # They're all constants.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
      
        # Variables    
        # They are variables we want to update and optimize.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))
      
        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases 
        # Original loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, 
                                                                      labels = tf_train_labels) )
        # Loss function using L2 Regularization
        regularizer = tf.nn.l2_loss(weights)
        loss = tf.reduce_mean(loss + beta * regularizer)
        
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax( tf.matmul(tf_valid_dataset, weights) + biases )
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
    # train model
    num_steps = 801
    
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])
    
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases. 
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('Loss at step {}: {}'.format(step, l))
                print('Training accuracy: {:.1f}'.format(accuracy(predictions, 
                                                             train_labels[:train_subset, :])))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                
                # You don't have to do .eval above because we already ran the session for the
                # train_prediction
                print('Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(), 
                                                               valid_labels)))
        print('Test accuracy: {:.1f}'.format(accuracy(test_prediction.eval(), test_labels)))     
        '''
        Loss at step 0: 48.195762634277344
        Training accuracy: 12.7
        Validation accuracy: 15.1
        Loss at step 100: 11.978610038757324
        Training accuracy: 72.9
        Validation accuracy: 75.2
        Loss at step 200: 4.531860828399658
        Training accuracy: 78.5
        Validation accuracy: 78.8
        Loss at step 300: 2.002777099609375
        Training accuracy: 81.7
        Validation accuracy: 80.5
        Loss at step 400: 1.144658088684082
        Training accuracy: 83.1
        Validation accuracy: 81.3
        Loss at step 500: 0.8502261638641357
        Training accuracy: 83.9
        Validation accuracy: 81.9
        Loss at step 600: 0.7480019927024841
        Training accuracy: 84.1
        Validation accuracy: 82.4
        Loss at step 700: 0.7121829390525818
        Training accuracy: 84.3
        Validation accuracy: 82.7
        Loss at step 800: 0.6995388269424438
        Training accuracy: 84.3
        Validation accuracy: 82.8
        Test accuracy: 87.2        
        '''

################################################################################
# neural network with L2 regularization
with tf.name_scope('neural_network_with_L2_regularization'):
    print('')
    print('neural_network_with_L2_regularization')
    
    # build model
    num_nodes= 1024
    batch_size = 128
    beta = 0.01

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
    
        # Variables.
        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
        biases_1 = tf.Variable(tf.zeros([num_nodes]))
        weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))
    
        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        # Normal loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_2, 
                                                                      labels = tf_train_labels))
        # Loss function with L2 Regularization with beta=0.01
        regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
        loss = tf.reduce_mean(loss + beta * regularizers)
    
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
        # Predictions for the training
        train_prediction = tf.nn.softmax(logits_2)
        
        # Predictions for validation 
        logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        
        valid_prediction = tf.nn.softmax(logits_2)
        
        # Predictions for test
        logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        
        test_prediction =  tf.nn.softmax(logits_2)

    # train model
    num_steps = 3001
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step {}: {}".format(step, l))
                print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
        print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))
        '''
        Minibatch loss at step 0: 3504.3623046875
        Minibatch accuracy: 4.7
        Validation accuracy: 38.1
        Minibatch loss at step 500: 21.357357025146484
        Minibatch accuracy: 88.3
        Validation accuracy: 84.5
        Minibatch loss at step 1000: 0.9844450950622559
        Minibatch accuracy: 80.5
        Validation accuracy: 84.8
        Minibatch loss at step 1500: 0.6801353693008423
        Minibatch accuracy: 86.7
        Validation accuracy: 84.6
        Minibatch loss at step 2000: 0.6293468475341797
        Minibatch accuracy: 89.1
        Validation accuracy: 84.9
        Minibatch loss at step 2500: 0.6685857772827148
        Minibatch accuracy: 86.7
        Validation accuracy: 83.9
        Minibatch loss at step 3000: 0.6567754745483398
        Minibatch accuracy: 86.7
        Validation accuracy: 84.5
        Test accuracy: 89.4        
        '''

    # train model with the restricted training data (just a few batches)
    # to demonstrate an extreme case of overfitting
    num_steps = 3001
    
    train_dataset_2 = train_dataset[:500, :]
    train_labels_2 = train_labels[:500]
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels_2.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset_2[offset:(offset + batch_size), :]
            batch_labels = train_labels_2[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step {}: {}".format(step, l))
                print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
        print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))    
        '''
        Minibatch loss at step 0: 3469.40673828125
        Minibatch accuracy: 9.4
        Validation accuracy: 34.2
        Minibatch loss at step 500: 21.004220962524414
        Minibatch accuracy: 99.2
        Validation accuracy: 78.2
        Minibatch loss at step 1000: 0.4954145848751068
        Minibatch accuracy: 99.2
        Validation accuracy: 79.1
        Minibatch loss at step 1500: 0.31954947113990784
        Minibatch accuracy: 99.2
        Validation accuracy: 79.3
        Minibatch loss at step 2000: 0.30957943201065063
        Minibatch accuracy: 99.2
        Validation accuracy: 79.4
        Minibatch loss at step 2500: 0.29615533351898193
        Minibatch accuracy: 99.2
        Validation accuracy: 79.4
        Minibatch loss at step 3000: 0.27402424812316895
        Minibatch accuracy: 100.0
        Validation accuracy: 79.1
        Test accuracy: 84.0        
        '''
        '''
        Remark: there is high training accuracy but low validation accuracy.
        This is overfitting problem.
        '''
        
################################################################################
# add dropout to prevent overfitting
'''
Remember: Dropout should only be introduced during training, not evaluation, 
otherwise your evaluation results would be stochastic as well
'''

with tf.name_scope('add_dropout'):
    print('')
    print('add_dropout')
    
    # build model
    num_nodes= 1024
    batch_size = 128
    beta = 0.01
    
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
    
        # Variables.
        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
        biases_1 = tf.Variable(tf.zeros([num_nodes]))
        weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))
        
        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        # Dropout on hidden layer: RELU layer
        keep_prob = tf.placeholder("float")
        relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)
        
        logits_2 = tf.matmul(relu_layer_dropout, weights_2) + biases_2
        # Normal loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_2, 
                                                                      labels = tf_train_labels))
        # Loss function with L2 Regularization with beta=0.01
        regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
        loss = tf.reduce_mean(loss + beta * regularizers)
    
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
        # Predictions for the training
        train_prediction = tf.nn.softmax(logits_2)
        
        # Predictions for validation 
        logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        
        valid_prediction = tf.nn.softmax(logits_2)
        
        # Predictions for test
        logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
        relu_layer= tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        
        test_prediction =  tf.nn.softmax(logits_2)    
        
        
    # train model
    num_steps = 3001
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step {}: {}".format(step, l))
                print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
        print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))
        '''
        Minibatch loss at step 0: 3665.089111328125
        Minibatch accuracy: 16.4
        Validation accuracy: 31.4
        Minibatch loss at step 500: 21.575355529785156
        Minibatch accuracy: 80.5
        Validation accuracy: 84.0
        Minibatch loss at step 1000: 1.0815726518630981
        Minibatch accuracy: 78.9
        Validation accuracy: 83.9
        Minibatch loss at step 1500: 0.7175045609474182
        Minibatch accuracy: 87.5
        Validation accuracy: 84.9
        Minibatch loss at step 2000: 0.6822401285171509
        Minibatch accuracy: 89.1
        Validation accuracy: 84.7
        Minibatch loss at step 2500: 0.7874009609222412
        Minibatch accuracy: 83.6
        Validation accuracy: 83.9
        Minibatch loss at step 3000: 0.7098387479782104
        Minibatch accuracy: 86.7
        Validation accuracy: 84.3
        Test accuracy: 88.7        
        '''
        
        