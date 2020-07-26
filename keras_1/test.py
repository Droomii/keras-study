# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello Tensorflow!!")
print(sess.run(hello))

a = tf.constant(1)
b = tf.constant(2)

print("a + b = {}".format(sess.run(a + b)))