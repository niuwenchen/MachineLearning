#encoding:utf-8
#@Time : 2017/7/17 13:47
#@Author : JackNiu
import tensorflow as tf

w1=tf.Variable(tf.random_normal([2,3],stddev=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1))

# 定义placeholder 作为存放输入数据的地方
x=tf.placeholder(dtype=tf.float32,shape=[3,2],name="input")
a=tf.matmul(x,w1)
y= tf.matmul(a,w2)

sess = tf.Session()
init_op =tf.initialize_all_variables()
sess.run(init_op)

# placeHolder 的数据， 字典的形式， 用feed_dict 填充
feed_dict ={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}
y_=sess.run(y,feed_dict=feed_dict)

#定义损失函数来刻画和预测值与真实值之间的误差
# 交叉熵，昨晚看过的，就是熵

cross_entropy= -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
learning_rate=0.001
# 定义反向传播算法中优化神经网络的参数
# GradientDescentOptimizer, AdamOptimizer, MomentumOptimizer
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# run(train_step)就可以对TRAINABLE_VARIABLES集合中的变量进行优化，使得在当前batch下损失函数最小
