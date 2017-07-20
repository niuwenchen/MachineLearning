#encoding:utf-8
#@Time : 2017/7/19 22:40
#@Author : JackNiu

import tensorflow as tf

v1=tf.Variable(tf.constant(1.0,shape=[1],name="other-v1"))
v2=tf.Variable(tf.constant(2.0,shape=[1],name="v2"))
result = v1-v2

init_op = tf.initialize_all_variables()
# 这种交换成功是由条件的，先进行合理的保存other-v1->v1,v2->v2,再加载 other->v2,v2->v1
saver = tf.train.Saver({"other-v1":v2,"v2":v1})
with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(result))
    # saver.save(sess, './model/test.ckpt')
    saver.restore(sess,'./model/test.ckpt')
    # print(sess.run(v1))
    print(sess.run(v2))