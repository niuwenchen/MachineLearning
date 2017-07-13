#encoding:utf-8
#@Time : 2017/7/13 14:03
#@Author : JackNiu
'''
输入采用占位符，而非tf.constant节点
模型不再接受两个离散标量，而改为接受一个任意长度的向量
使用该数据流图时，将随时间计算所有输出的总和
将采用名称作用域对数据流图进行合理划分
每次运行时，都将数据流图的输出，所有输出的累加以及输出的均值保存到磁盘，供TB使用
TB的汇总数据有一个专属的名称作用域，用于容纳tf.scalar_summary() Op

'''
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('variables'):
        # 记录数据流图运行次数的variable
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")
        # 指定variable只能手工设置
    with tf.name_scope("Transformation"):
        # 独立的输入层
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
        # 独立的中间层
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        # 独立的输出层
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")
    with tf.name_scope("update"):
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

        # 为输出节点汇总数据
        with tf.name_scope('summaries'):
            avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
            tf.scalar_summary(b'Output', output, name="output_summary")
            tf.scalar_summary(b'Sum of outputs over time', update_total, name="total_summary")
            tf.scalar_summary(b'Average of outputs over time', avg, name="average_summary")
    with tf.name_scope('global_ops'):
        # 初始化ＯＰ
        init = tf.initialize_all_variables()
        # 　将所有汇总工作合并到一个Ｏｐ中
        merged_summaries = tf.merge_all_summaries()

sess = tf.Session(graph=graph)
writer = tf.train.SummaryWriter('./improver_graph', graph)
sess.run(init)


def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


run_graph([2, 8])
run_graph([3, 4])
run_graph([2, 8])
run_graph([3, 4])
writer.flush()
writer.close()
sess.close()
