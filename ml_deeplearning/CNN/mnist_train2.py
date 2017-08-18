#encoding:utf-8
#@Time : 2017/7/20 10:00
#@Author : JackNiu
import os
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import  ml_deeplearning.CNN.mnist_inference as mnist_inference
import numpy as np


# 配置神经网络参数
BATCH_SIZE=100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY =0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="model/"
MODEL_NAME="cnn.ckpt"

def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                                mnist_inference.IMAGE_SIZE,
                                                mnist_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播过程
    y=mnist_inference.inference(x,True,regularizer)
    global_step= tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(variables_averages_op,train_step)

    # 持久化TF类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs=np.reshape(xs,(BATCH_SIZE,mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            # print(np.shape(reshape_xs))
            _,loss_value,step= sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs,y_:ys})
            # 需要保存的是loss，global_step
            if i%1000 ==0:
                print("After %d training step(s), lossing on trainng batch is %g" % (i, loss_value))
                # 文件会在后面加上gloabl_step作为标记
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
def main(argv=None):
    mnist = input_data.read_data_sets("/MNIST_data/",one_hot=True)
    train(mnist)

# tf.app.run 会调用上面的main函数
if __name__=='__main__':
    tf.app.run()





