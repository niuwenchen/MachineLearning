#encoding:utf-8
#@Time : 2017/7/13 13:21
#@Author : JackNiu
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[], name="input_b")
    const = tf.constant(3, dtype=tf.float32, name="static_value")

    with tf.name_scope("Transformation"):
        with tf.name_scope("A"):
            A_mul = tf.mul(in_1, const)
            A_out = tf.sub(A_mul, in_1)
        with tf.name_scope("B"):
            B_mul = tf.mul(in_2, const)
            B_out = tf.sub(B_mul, in_2)
        with tf.name_scope("C"):
            C_div = tf.mul(A_out, B_out)
            C_out = tf.sub(C_div, const)
        with tf.name_scope("D"):
            D_div = tf.div(B_out, A_out)
            D_out = tf.add(D_div, const)
    out = tf.maximum(C_out, D_out)
    writer = tf.train.SummaryWriter('./name_scope_2', graph=graph)
    writer.close()