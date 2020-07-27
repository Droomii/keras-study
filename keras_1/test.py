# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello Tensorflow!!")
print(sess.run(hello))