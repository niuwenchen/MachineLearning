#encoding:utf-8
#@Time : 2017/7/12 17:27
#@Author : JackNiu

import tensorflow as tf


def withTest():
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="A_add")
        b = tf.mul(a, 3, name="A_mul")
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="B_add")
        d = tf.mul(c, 6, name="B_mul")
    e = tf.add(b, d, name="output")
    sess = tf.Session()
    output = sess.run(e)
    writer = tf.train.SummaryWriter('./name_scope_1', sess.graph)
    writer.close()
    sess.close()


def zhangliang():
    s_1 = [[[1,2],[3,4],[5,6]]]
    shape=tf.shape(s_1)
    print(shape)


# withTest()
zhangliang()