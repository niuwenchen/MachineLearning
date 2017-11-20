#encoding:utf-8
#@Time : 2017/11/3 13:57
#@Author : JackNiu


import tensorflow as tf

saver = tf.train.import_meta_graph("./model/qlj/model_1.ckpt.meta")
with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "./model/qlj/model_1.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

