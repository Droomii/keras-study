# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:52:32 2020

@author: DATALAB_3
"""

import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

# read dataset
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# x is placeholder for the 28 x 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784]) # 28 * 28 = 784

# y_ is called "y bar" and is a 10 element vector, containing the predicted prob. of each
# digit(0-9) class, such as [0.14,0.8,0,0,0,0 ...]
y_ = tf.placeholder(tf.float32, [None, 10])

# define weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10])) 


# defining model - matrix multiply
# softmax is an exponential function - highlights largest value, suppresses smaller values
# which clarifies the result
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
# reduce mean is called to return the mean of these differences,
# which we want to be low.
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient decent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize global var
init = tf.global_variables_initializer()

# create an interactive session that can span multiple code bocks. Don't
# forget to explicitly close the session with sess.close()
sess = tf.Session()

sess.run(init)

for i in range(1000):
    
    # get 100 random data points from the data.
    # batch_xs = image, batch_ys = digit(0-9) class
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
    # perform optimization with the data above
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# Evaluate how well the model did. Do this by comparing the digit with the highest prob. in
# actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()