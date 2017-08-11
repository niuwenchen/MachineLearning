#encoding:utf-8
#@Time : 2017/7/24 16:32
#@Author : JackNiu

'''
使用TF-Slim工具简洁的实现一个卷积层
slim.arg_scope 函数可以用于设置默认的参数取值，slim.arg_scope函数的第一个参数是一个函数列表
在这个列表中的函数将使用默认的参数取值。
'''

import tensorflow  as tf
import tensorflow.contrib.slim as slim


with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="SAME"):
    # 假设输入图片经过之前的神经网络结果保存在变量net中
    net=0
    with tf.variable_scope("Mixed_7c"):
        # 给Inception模块中的每一条路径申明一个命名空间
        with tf.variable_scope("Branch_0"):
            branch_0 = slim.conv2d(net,320,[1,1],scope="Conv2d_0a_1x1")

        # 第二条路径
        with tf.variable_scope('Branch_1'):
            branch_1= slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1x1")
            # tf.concat 函数可以将多个矩阵拼接起来
            branch_1= tf.concat(3,[
                slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1x3"),
                slim.conv2d(branch_1, 384, [3, 1], scope="Conv2d_0c_3x1")])


        # 第三条路径
        with tf.variable_scope('Branch_2'):
            branch_2= slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1x1")
            # tf.concat 函数可以将多个矩阵拼接起来
            branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3*3')

            branch_2= tf.concat(3,[
                slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1x3"),
                slim.conv2d(branch_2, 384, [3, 1], scope="Conv2d_0d_3x1")])

        # 第四条路径
        with tf.variable_scope("Branch_3"):
            branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3*3")
            branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

        #当前Inception模块的最后结果是由上面四个计算结果拼接得到的
        net=tf.conncat(3,[branch_0,branch_1,branch_2,branch_3])
