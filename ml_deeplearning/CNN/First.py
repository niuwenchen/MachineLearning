#encoding:utf-8
#@Time : 2017/7/21 15:12
#@Author : JackNiu

import tensorflow as tf
# 卷积层的参数个数之和过滤器的尺寸，深度以及当前节点矩阵的深度有关，这里什么的参数变量是一个4维矩阵，
# 前面两维代表了过滤器的尺寸，第三个参数代表当前层的深度，第四个维度表示过滤器的深度


filter_weights=tf.get_variable('weights',[5,5,3,16],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases',[16],
                         initializer=tf.constant_initializer(0.1))

'''
tf.nn.conv2d 提供了一个非常简单的函数来实现卷积层前向传播的算法，这个函数的第一个输入为当前层的节点矩阵
注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一位对应一个输入batch，input[1,:,:,:]代表第一张图片

tf.nn.conv2d第二个参数提供了卷积层的权重，第三个参数为不同维度上的步长，第一维和最后一维的数字要求一定是1，
这是因为卷积层的步长只对矩阵的长和宽有效，最后一个参数是填充的方法，SAME和VALID，SAME全0，VALID表示不添加

'''
conv = tf.nn.conv2d(input,filter_weights,strides=[1,1,1,1],padding="SAME")

bias = tf.nn.bias_add(conv,biases)

#
actived_conv = tf.nn.relu(bias)

# tf.nn.max_pool实现了最大化池化层的前向传播过程，
pool =tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#  第二个参数为过滤器的尺寸，虽然给出的是一个长度为4的一维数组，但是这个数组的第一个和最后一个数必须为1.
# 这意味着池化层的过滤器是不可以跨不同输入样例或节点矩阵深度的，在实际引用中[1,2,2,1],[1,3,3,1]
# 难道2，2 就是2*2中选出最大值？

