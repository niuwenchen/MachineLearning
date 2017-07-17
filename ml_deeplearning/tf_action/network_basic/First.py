#encoding:utf-8
#@Time : 2017/7/17 11:07
#@Author : JackNiu

import tensorflow as tf
a= tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a+b
print(a.graph is tf.get_default_graph())

# 不同的计算图上的张量和运算都不会共享
# g1= tf.Graph()
g1=tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量v，并设置初始值为0
    v=tf.get_variable('v',initializer=tf.zeros_initializer(shape=[2,3]))

g2=tf.Graph()
with g2.as_default():
    v = tf.get_variable('v',initializer=tf.zeros_initializer(shape=[2,3]))

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中
        print(sess.run(tf.get_variable("v")))

# 会话
# TF会自动生成一个默认的计算图，如果没有特殊指定，运算会自动加入这个计算图中，TF的会话
#也有类似的机制，但TF不会自动生成默认的会话，而是需要手动指定，当默认的会话被指定至哦户
# 可以通过tf.Tensor.eval函数计算一个张量的取值
# sess= tf.Session()
# print(sess.run(result))
# print(result.eval(session=sess))
# print(a.eval(session=sess))

# 在交互式环境下直接构建默认会话的函数，tf.InteractiveSession,自动将生成的会话注册为默认会话
sess1= tf.InteractiveSession()
print(result.eval(session=sess1))
sess1.close()

tf.train.exponential_decay(learning_rate=0.1,
                           global_step=1000,decay_steps=100,decay_rate=0.96,staircase=True)