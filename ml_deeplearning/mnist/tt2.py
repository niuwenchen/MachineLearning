#encoding:utf-8
#@Time : 2017/11/3 14:33
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
def inference(input_tensor,regularizer):
    with tf.variable_scope("layer1"):
        weights =get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    #返回最后前向传播的结果
    return layer2



# # 训练过程
# import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#
# # 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "mnist.ckpt"
#
#
# def train(mnist):
#     x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name="x-input")
#     y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
#     regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#
#     # 前向传播过程
#     y = inference(x, regularizer)
#     global_step = tf.Variable(0, trainable=False)
#
#     variable_averages = tf.train.ExponentialMovingAverage(
#         MOVING_AVERAGE_DECAY, global_step
#     )
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
#     learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
#                                                global_step, mnist.train.num_examples / BATCH_SIZE,
#                                                LEARNING_RATE_DECAY)
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#     train_op = tf.group(variables_averages_op, train_step)
#
#     # 持久化TF类
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         tf.initialize_all_variables().run()
#         for i in range(TRAINING_STEPS):
#             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
#             # 需要保存的是loss，global_step
#             if i % 1000 == 0:
#                 print("After %d training step(s), lossing on trainng batch is %g" % (i, loss_value))
#                 # 文件会在后面加上gloabl_step作为标记
#                 saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
#         for k in tf.trainable_variables():
#             print(k.name, k.eval())
#
#
# def main(argv=None):
#     mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
#     train(mnist)
#
#
# # tf.app.run 会调用上面的main函数
# if __name__ == '__main__':
#     tf.app.run()




EVAL_INTERVAL_SECS=10
import time

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果，不关注正则损失化
        y=inference(x,None)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        # 滑动平均模型
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY
        )
        variabels_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variabels_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    for var in tf.trainable_variables():
                        print(var.name,var.eval())

                    # 获取迭代轮数
                    global_step= ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step(s), validation acuracy = %g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/MNIST_data/",one_hot=True)
    evaluate(mnist)

# tf.app.run 会调用上面的main函数
if __name__=='__main__':
    tf.app.run()