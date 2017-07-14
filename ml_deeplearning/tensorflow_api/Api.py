#encoding:utf-8
#@Time : 2017/7/14 14:12
#@Author : JackNiu
import tensorflow as tf
from numpy import *


def constanttest():
    v1= tf.zeros([3,4],dtype=tf.int32)
    v2= tf.zeros_like(v1,dtype=tf.float32)
    v3=tf.ones([3,4],dtype=tf.float32,name="ones")
    v4= tf.ones_like(v1,dtype=tf.int64)
    x = tf.placeholder(dtype=tf.int32,shape=[3,3])
    v5= tf.fill(tf.shape(x),9)
    v6 = tf.constant([1,2,3,4,5])
    v7= tf.constant([1,2,3,4,5],shape=[3,4])
    with tf.Session() as sess:

        print(sess.run(v6))

def Sequencestest():
    sess = tf.Session()
    s1=tf.linspace(10.0, 12.0, 10, name="linspace")
    s2=tf.range(0, 10, 0.1)

    with sess.as_default():
        print(sess.run(s1))
        print(sess.run(s2))


def RandomtensorTest():
    # 均值  方差
    r1=norm = tf.random_normal([2, 3], mean=1, stddev=1)
    r2=tf.constant([[1, 2], [3, 4], [5, 6]])
    shuff = tf.random_shuffle(r2)
    r3=tf.random_normal(tf.shape(r2),seed=1234)
    var = tf.Variable(tf.random_uniform([2, 3]), name="xx")
    # sess= tf.Session()
    # init =tf.initialize_all_variables()
    # sess.run(init)
    # print(sess.run(var))

    tf.set_random_seed(100)
    a = tf.random_uniform([2])
    b = tf.random_normal([2])

    print("Session 1")
    with tf.Session() as sess1:
        print (sess1.run(a))  # generates 'A1'
        print (sess1.run(a))  # generates 'A2'
        print (sess1.run(b))  # generates 'B1'
        print (sess1.run(b))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(a))  # generates 'A3'
        print(sess2.run(a))  # generates 'A4'
        print(sess2.run(b)) # generates 'B3'
        print(sess2.run(b))  # generates 'B4'


def variableTest():
    w = tf.Variable(1, dtype=tf.int32 ,name="w")
    a=tf.constant([1,2,3,4],shape=[2,2])
    b= tf.constant([2,3,4,5,6,7],shape=[2,3],dtype=tf.float32)
    y=w.count_up_to(10)
    # y = tf.matmul(a,b) #用来做矩阵乘法
    z = tf.sigmoid(b)
    # f=w.assign(w.initialized_value()+1.0)

    with tf.Session() as sess:
        init=tf.initialize_all_variables()
        sess.run(init)
        # print(sess.run(w))
        # print(sess.run(q))
        # sess.run(w.initializer)
        # print(sess.run(w))
        # w.initialized_value()
        w.count_up_to(10)
        print(sess.run(w))
        print(sess.run(y))
        print(sess.run(y))
        print(sess.run(y))
        print(sess.run(tf.all_variables()))
        print(sess.run(tf.trainable_variables()))
        print(sess.run(tf.assert_variables_initialized([w])))


def variableTest2():
    v = tf.Variable([1, 2])
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
        print(v.dtype)
        print(v.get_shape())
        print(v.graph)


def saverTest():
    pass

variableTest()