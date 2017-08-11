#encoding:utf-8
#@Time : 2017/7/20 9:47
#@Author : JackNiu

import  tensorflow as tf
# 配置神经网络的参数
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

IMAGE_SIZE = 28
NUM_CHANNELS=1
NUM_LABELS=10

# 第一层卷积层的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5

# 第二层卷积层的尺寸和深度
CONV2_DEEP=65
CONV2_SIZE=5

#全连接层的节点个数
FC_SIZE=512




def get_weight_variable(shape,regularizer):
    weighs = tf.get_variable(
        name="weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    # 正则化函数，将当前变量的正则化损失加入名字为losses的集合,保存为中间变量
    if regularizer:
        tf.add_to_collection("losses",regularizer(weighs))
    return weighs

# 定义卷积神经网络的前向传播过程，新的参数train,用于区分训练和测试过程
# dropout 方法可以进一步提升模型可靠性并防止过拟合
# 只在训练时使用
# 定义前向传播过程
def inference(input_tensor,train,regularizer):
    # 声明第一层卷积层的变来并实现前向传播过程，这里和LeNet_5不太一样
    # 使用全0填充，32的深度
    with tf.variable_scope("layer1-conv1"):
        conv1_weights= tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=
                                       tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases= tf.get_variable("biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1= tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1= tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))


    #实现第二层池化层的前向传播过程，最大池化层，使用全0填充并且移动的步长为2
    with tf.name_scope("layer2_pool1"):
        # ksize代表过滤器的尺寸。第一个和最后一个参数必须为1
        pool1= tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 14*14*32
    #声明第三层卷积层的变量并且实现前向传播过程
    with tf.variable_scope("layer3-conv1"):
        conv2_weights= tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=
                                       tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程，最大池化层，使用全0填充并且移动的步长为2
    with tf.name_scope("layer4_pool2"):
        # ksize代表过滤器的尺寸。第一个和最后一个参数必须为1
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    #将第四层池化层的输出转换为第五层全连接层的输入, 7*7*64
    #将这个输入向量变成一列向量,get_shape(
    # 因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数
    pool_shape = pool2.get_shape().as_list()
    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积
    nodes= pool_shape[1]*pool_shape[2]*pool_shape[3]
    #通过tf.reshape() 将输出变成一个batch的向量
    #batch 说明:
    reshaped= tf.reshape(pool2,[pool_shape[0],nodes])

    # 声明第五层全连接层的变量并实现前向传播过程，这一层的输入是在拉直之后的一组向量，
    # 向量长度为3136，输出是一组长度为512 的向量，
    # dropout会随机将部分节点的输出改为0，dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好
    # dropout一般只在全连接层使用而不是卷积层或池化层使用
    with tf.variable_scope("layer5-fc1"):
        fc1_weighs = tf.get_variable("weight",[nodes,FC_SIZE],initializer=
                                     tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc1_weighs))
        fc1_biases= tf.get_variable("bias",[FC_SIZE],initializer=
                                    tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weighs)+fc1_biases)
        if train: fc1= tf.nn.dropout(fc1,0.5)

    # 声明第六层全连接层的变量并实现前向传播过程，输入为一组512 的向量
    # 输出为一组10 的向量
    with tf.variable_scope("layer6-fc2"):
        fc2_weighs = tf.get_variable("weight", [FC_SIZE,NUM_LABELS], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weighs))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=
        tf.constant_initializer(0.1))
        # fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weighs) + fc1_biases)
        # if train: fc1 = tf.nn.dropout(fc1, 0.5)
        logit = tf.matmul(fc1,fc2_weighs)+fc2_biases


    #返回最后前向传播的结果
    return logit

