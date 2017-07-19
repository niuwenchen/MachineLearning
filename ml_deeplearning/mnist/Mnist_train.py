#encoding:utf-8
#@Time : 2017/7/18 15:43
#@Author : JackNiu
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
INPUT_NODE = 784    # 有多少个维度就有多少个输入节点
OUTPUT_NODE=10      #有多少个类别就有多少个输出节点

# 配置神经网络的参数
LAYER1_NODE=500     # 500个节点，一个隐藏层
BATCH_SIZE=100      # 一个训练batch中的训练数据个数，数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降

LEARNING_RATE_BASE = 0.8    #基础的学习率
LEARNING_RATE_DECAY =0.99   #学习率的衰减率
REGULARIZATION_RATE=0.0001  # 正则化 lambda系数
TRAINING_STEPS  =30000      # 训练轮数
MOVING_AVERAGE_DECAY=0.99        # 滑动平均衰减率

'''
一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。定义ReLU
激活函数的三层全连接神经网络，通过加入隐藏层实现了多层网络结构，通过ReLU激活函数实现了去线性化
也支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型
'''
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    # avg_class 滑动平均类
    if avg_class == None:
        # 隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

        #输出层的前向传播结果，
        return tf.matmul(layer1,weights2)+biases2
    else:
        # 使用avg_class 函数计算得出变量的滑动平均值
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

# 模型训练的过程

def inference1(input_tensor,reuse=False):
    with tf.variable_scope('layer1',reuse=reuse):
        weigths = tf.get_variable("weights",[INPUT_NODE,LAYER1_NODE],initializer=
                                  tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=
                                 tf.constant_initializer(0.0))
        layer1= tf.nn.relu(tf.matmul(input_tensor,weigths)+biases)

    # 定义输出层
    with tf.variable_scope('layer2', reuse=reuse):
        weigths = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=
        tf.constant_initializer(0.0))
        layer2 = tf.matmul(input_tensor, weigths) + biases

    return layer2

'''
调用方式
初次使用
y=inference1(x)

测试
new_x=...
测试的时候，获取或创建变量的部分就不再改变，将创建变量的部分放进了函数中
简化了方法的写法
new_y=inference1(new_y,True)

'''


def train(mnist):
    x=tf.placeholder(dtype=tf.float32,shape=[None,INPUT_NODE],name="x-input")
    y_=tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

    # 生成隐藏层的参数
    weights1= tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2= tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y=inference(x,None,weights1,biases1,weights2,biases2)
    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，也不需要优化，trainable=False
    global_steps= tf.Variable(0,trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_steps
    )
    # 在所有代表神经网络参数的变量上使用滑动平均，其他就不需要了
    # trainable_variables 返回的是GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有

    variables_averages_op = variable_averages.apply(tf.trainable_variables())


    # 计算使用了滑动平均之后的前向传播结果。
    average_y= inference(x,variable_averages,weights1,biases1,weights2,biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数，这里使用了tF中提供的sparse_softmax_cross_entropy_with_logits
    # 函数来计算交叉熵，当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的额计算，MNIST问题的推案中
    #包含了0-9数字中的一个，所以可以使用这个函数来计算交叉熵损失，第一个参数是神经网络不包括softmax层的前向传播结果
    # 第二个是训练数据的正确答案。tf.argmax() 得到正确答案的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean= tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 一般只计算权重的正则化损失
    regularization =regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean+regularization
    # 设置指数衰减的学习率,初始的global_steps
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_steps,mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数
    # 梯度下降求的是什么，weight，最终目的是什么， 假设现在weighs的值确定，那么就需要固定weight然后变换某一个变量来达到优化的目的
    # 这里通过学习率的变化来使误差最小。

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)

    # 那么最终的需要训练的就是: train_step来最小化误差，
    # 每一遍训练既需要通过反向传播更新设计网络中的参数，又需要更新每一个参数的滑动平均值
    # with tf.control_dependencies([train_step,variables_averages_op]):
    #     train_op = tf.no_op(name='train')
    train_op=tf.group(variables_averages_op,train_step)

    # 校验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_predication= tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    # 将一个布尔型的数值转换为实数型，然后计算平均值，平均值模型就是在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_predication,tf.float32))

    # 初始化训练并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed ={x:mnist.test.images,y_:mnist.test.labels}
        for k in tf.trainable_variables():
            print(k.name,k.eval())

        # 迭代训练
        for i in range(TRAINING_STEPS):
            if i %1000 ==0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validatoin accuracy using average model is %g"%(i,validate_acc))
                print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))

                # 训练数据
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
                # print("After %d training steps"%i)
                # for k in tf.trainable_variables():
                #     print(k.name, k.eval())
        # 训练结束后，在测试数据上检测神经网络模型的最终正确性
        print("here")
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g"%(TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("/MNIST_data/",one_hot=True)
    train(mnist)

# tf.app.run 会调用上面的main函数
if __name__=='__main__':
    tf.app.run()