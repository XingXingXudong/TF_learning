# coding: utf-8


"""create a linear model which try to fit the line y = x + 2 using SGD optimizer to minimize
   root-mean-square(RMS) loss function
"""

import tensorflow as tf
import numpy as np


# number of epoch
num_epoch = 100

# traning data x and label y
x = np.array([0., 1., 2., 3.], dtype=np.float32)
y = np.array([2., 3., 4., 5.], dtype=np.float32)

# convert x and y to 4x1 matrix
x = np.reshape(x, [4, 1])
y = np.reshape(y, [4, 1])

# test set(using a little trick)
x_test = x + 0.5
y_test = y + 0.5

# This part of the script builds the TensorFlow graph using the Python API

# Fist declare placeholders for input x and label y
# Placesholders are TensorFlow Variables requiring to be explicitly fed by some input data

x_ph = tf.placeholder(tf.float32, shape=[None, 1])
y_ph = tf.placeholder(tf.float32, shape=[None, 1])

# Variables (if not specified) will be learnt as the GradientDescentOptimizer is run
# Declare weight variable initialized using a truncated_normal law
W = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
# Declare bias variable initialized to a constant 0.1
b = tf.Variable(tf.constant(0.1, shape=[1]))

# Initialize variables just declared
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# In this part of the script, we build operators storing operations on the previous varibales and
# placeholders. Model: y = W * x + b
y_pred = x_ph * W + b

# loss function
loss = tf.multiply(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_ph))), 1./2)
# creat traning graph
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# This part of the script run the TensorFlow graph (variables and operations operator) just built
with tf.Session() as sess:
    # initialize all the variables by running the initializer operator
    sess.run(init)
    for epoch in range(num_epoch):
        # Run sequentially the train_op and loss operators with x_ph and y_ph placeholders fed by
        # variables x and y
        _, loss_val = sess.run([train_op, loss], feed_dict={x_ph: x, y_ph: y})
        print('epoch %d: loss if %.4f' % (epoch, loss_val))
    # see what model do in the test set by evaluating the y_pred operator using the x_test_data
    test_val = sess.run(y_pred, feed_dict={x_ph: x_test})
    print('ground truth y is: %s' % y_test.flatten())
    print('prdict y is      : %s' % test_val.flatten())
