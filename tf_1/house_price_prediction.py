# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:39:03 2020

@author: DATALAB_3
"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# generating some house sizes between 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)


# generate house prices from house size with a random noise added.
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# normalize values
def normalize(array):
    return (array - array.mean()) / array.std()

# use 70% of data as training data
num_train_samples = math.floor(num_house * 0.7)

#define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

# normalize training data
train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)


#define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])

# normalize test data
test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

# set up placeholders that get updated as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Define the variables holding the size_factor and price we set during training
# We initialize them to some random values based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# predicted price = (size_factor * house_size) + price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2)) / (2 * num_train_samples)

# learning rate
learning_rate = 0.1

# gradient descent optimizer - minimizing loss defined in the operation 'cost'
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initializing variables
init = tf.global_variables_initializer()

# Launch graph in session
with tf.Session() as sess:
    sess.run(init)
    
    # display training progress settings
    display_every = 2
    num_training_iter = 50
