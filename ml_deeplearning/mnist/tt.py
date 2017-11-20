#encoding:utf-8
#@Time : 2017/11/3 13:52
#@Author : JackNiu

import tensorflow as tf

#
v1 = tf.Variable(tf.constant(2.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(3.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()
with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "./model/qlj/model_1.ckpt")
    print(sess.run(result))
