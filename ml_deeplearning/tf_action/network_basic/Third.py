#encoding:utf-8
#@Time : 2017/7/17 13:58
#@Author : JackNiu

# 完整的程序训练神经网络解决二分类问题
import tensorflow as tf
from  numpy.random import RandomState

# 定义训练数据batch的大小
batch_size=8
# 定义神经网络的参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y_input")

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

cross_entorpy= -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entorpy)

rdm = RandomState(1)
dataset_size=128

X=rdm.rand(dataset_size,2)

# 定义规则给出样本的标签
Y=[[int(X1+X2<1)] for (X1,X2)in X]

#创建一个会话运行TF程序
with tf.Session() as sess:
    init_op= tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    Steps= 10000
    for i in range(Steps):
        start=(i* batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        # 通过选取的样本训练神经网络,反复运行优化算法
        sess.run(train_step,feed_dict={x:X[start:end], y_:Y[start:end]})
        if i %1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵
            total_cross_entropy = sess.run(cross_entorpy,feed_dict={x:X,y_:Y})
            print("After %d training steps,corss entropy on all data is%s"%(i,total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

