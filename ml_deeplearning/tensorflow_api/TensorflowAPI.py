#encoding:utf-8
#@Time : 2017/11/16 14:45
#@Author : JackNiu

import tensorflow  as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float(
    'flag_float', 0.01, 'input a float')
tf.app.flags.DEFINE_integer(
    'flag_int', 400, 'input a int')
tf.app.flags.DEFINE_boolean(
    'flag_bool', True, 'input a bool')
tf.app.flags.DEFINE_string(
    'flag_string', 'yes', 'input a string')

print(FLAGS.flag_float)
print(FLAGS.flag_int)
print(FLAGS.flag_bool)
print(FLAGS.flag_string)

sl= slim.dataset.Dataset()

