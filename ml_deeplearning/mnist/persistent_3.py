#encoding:utf-8
#@Time : 2017/7/19 23:29
#@Author : JackNiu
import tensorflow as tf
from tensorflow.python.framework import  graph_util

v1= tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2= tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
result = v1+v2

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前及算算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    # 保存的是图结构，这个图结构包括几部分，两个变量一个加法操作，实际是这个加法操作
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量及其取值转换为常量，同时将图中不必要的节点独钓
    output_graph_def=graph_util.convert_variables_to_constants(
        sess,graph_def,['add']
    )
    # 将导出的模型存入文件
    with  tf.gfile.GFile('model/combined_model.pb','wb') as f:
        f.write(output_graph_def.SerializeToString())

#输出 ： Converted 2 variables to const ops.
