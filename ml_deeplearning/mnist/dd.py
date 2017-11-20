#encoding:utf-8
#@Time : 2017/11/3 13:52
#@Author : JackNiu

import tensorflow as tf

#
v1=tf.Variable(tf.constant(2.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(3.0,shape=[1]),name="v2")
result = v1+v2
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(result))
    saver.save(sess,"./model/qlj/model_1.ckpt")