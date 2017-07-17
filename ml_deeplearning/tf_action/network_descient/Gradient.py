#encoding:utf-8
#@Time : 2017/7/17 19:32
#@Author : JackNiu
import tensorflow as tf

batch_size=10
x=tf.placeholder(tf.float32,shape=[batch_size,2],name='x-input')
y_ = tf.placeholder(tf.float32,shape=[batch_size,1],name='y-input')

# 定义神经网络结构和优化算法
loss=...
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练神经网络
with tf.Session() as sess:
    # 参数初始化

    for i in range(STEPS):
        # 准备batch_size个训练数据
        current_X,cutrrent_Y=...
        sess.run(train_step,feed_dict={x:current_X,y_:cutrrent_Y})
