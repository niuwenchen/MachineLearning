#encoding:utf-8
#@Time : 2017/7/20 9:47
#@Author : JackNiu

import  tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

# 训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值
# 可以在训练时使用变量自身，在测试时使用变量的滑动平均值

def get_weight_variable(shape,regularizer):
    weighs = tf.get_variable(
        name="weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    # 正则化函数，将当前变量的正则化损失加入名字为losses的集合,保存为中间变量
    if regularizer:
        tf.add_to_collection("losses",regularizer(weighs))
    return weighs

# 定义前向传播过程
def inference(input_tensor,avg_class,regularizer):
    with tf.variable_scope("layer1"):
        weights =get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        if avg_class:
            layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights))+avg_class.average(biases))
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        if avg_class:
            layer2 = tf.matmul(layer1, avg_class.average(weights)) +  avg_class.average(biases)
        layer2 = tf.matmul(layer1, weights) + biases
    #返回最后前向传播的结果
    return layer2

