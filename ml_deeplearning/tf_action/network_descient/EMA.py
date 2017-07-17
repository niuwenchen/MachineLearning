#encoding:utf-8
#@Time : 2017/7/17 22:55
#@Author : JackNiu
# 滑动平均模型

import tensorflow as tf

# 滑动平均
v1=tf.Variable(0,dtype=tf.float32)
step= tf.Variable(0,trainable=False)

# 定义滑动平均实体, 衰减率， num_updates,
ema = tf.train.ExponentialMovingAverage(0.99,step)

maintain_average_op = ema.apply([v1])
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # ema.average()获取滑动平均之后的取值
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(v1,5))
    # 更新v1的滑动平均值，更新衰减率
    # ema.average==> ema.apply(v1)
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    #
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

