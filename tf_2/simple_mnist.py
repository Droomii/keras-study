# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:52:32 2020

@author: DATALAB_3
"""

import tensorflow as tf
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
b = tf.Variable = tf.Variable(tf.zeros([10]))

# defining model
y = tf.nn.softmax(tf.matmul(x, W) + b)
