#encoding:utf-8
#@Time : 2017/7/19 23:36
#@Author : JackNiu
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename='model/combined_model.pb'
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def =tf.GraphDef()
        # 读取保存的图结构，加法操作Op
        graph_def.ParseFromString(f.read())
    # 将graph_def 中保存的图加载到当前的图中，return_elements=["add:0"]
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    # 运行这个Op
    print(sess.run(result))