
##### 1. SESSION ####
from unittest import result

import tensorflow as tf

# Create a constant op that produces a 1x2 matrix
mtx_1 = tf.constant([[3.0, 3.0]])

# Create another constant that produces a 2x1 matrix
mtx_2 = tf.constant([[2.0], [2.0]])

# Multiply these two matrices
product = tf.matmul(mtx_1, mtx_2)

# Show graph
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

"""
    In case of having multi gpu: Devices are specified with strings.
    The currently supported devices are:
    "/cpu:0": The CPU of your machine.
    "/gpu:0": The GPU of your machine, if you have one.
    "/gpu:1": The second GPU of your machine.

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        mtx_1 = tf.constant([[3.0, 3.0]])
        mtx_2 = tf.constant([[2.0], [2.0]])
        product = tf.matmul(mtx_1, mtx_2)
"""


#### 2. VARIABLES ####

# Create new variable
state = tf.Variable(0, name = "counter")

# Add 1 to variable
update = tf.assign(state, tf.add(state, tf.constant(1)))

# Initialize variable
init_op = tf.initialize_all_variables()

# Launch graph and run ops
with tf.Session() as sess:

    # Run init op
    sess.run(init_op)

    # Print the initial value of state
    print(sess.run(state))

    # Run the op that updates and print state
    for i in range(3):
        sess.run(update)
        print(sess.run(state))


#### 3. FETCHES ####
input_1 = tf.constant([3.0])
input_2 = tf.constant([2.0])
input_3 = tf.constant([5.0])

add = tf.add(input_2, input_3)
mul = tf.mul(input_1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)


#### 4. FEEDS ####
# placeholder is a variable whose data will be feed or assigned in execution
inp_1 = tf.placeholder(tf.float32)
inp_2 = tf.placeholder(tf.float32)
output = tf.mul(inp_1, inp_2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={inp_1:[7.], inp_2:[2.]}))

