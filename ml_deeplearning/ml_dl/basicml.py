#encoding:utf-8
#@Time : 2017/7/14 13:17
#@Author : JackNiu

import tensorflow as tf

#初始化变量和模型参数

def inference(X):
    # 计算推断模型在数据Ｘ上的输出，并将结果返回
    pass

def  loss(X,Y):
    # 计算损失
    pass

def inputs():
    # 读取或生成训练数据X及其期望输出Y
    pass

def train(total_loss):
    #
    pass

def evaluate(sess,X,Y):
    #　评估
    pass

# 在一个会话对象中启动数据流图，搭建流程
with tf.Sessin() as sess:
    tf.global_variables_initializer().run()
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)

    corrd = tf.train.Coordinator() # 线程
    # 启动所有的队列，(sess,coord,daemon,start,collection='')
    threads= tf.train.start_queue_runners(sess=sess,coord=corrd)

    # 实际的迭代参数
    training_steps=1000
    for step in range(training_steps):
        sess.run([train_op])

        if step % 10 ==0:
            print("loss: ",sess.run([total_loss]))
    evaluate(sess,X,Y)
    corrd.request_stop()
    # join操作，等待所有线程执行完成
    corrd.join(threads)
    sess.close()
