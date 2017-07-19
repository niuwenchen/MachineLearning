#encoding:utf-8
#@Time : 2017/7/19 23:07
#@Author : JackNiu
import tensorflow as tf

v=  tf.Variable(0,dtype=tf.float32,name="v")
ema= tf.train.ExponentialMovingAverage(0.99)

print(ema.variables_to_restore())
# maintain_average_op=ema.apply(tf.all_variables())

for var in tf.all_variables():
    print(var.name)

saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#
#     sess.run(tf.assign(v,10))
#     sess.run(maintain_average_op)
#
#     saver.save(sess,'model/model.ckpt')
    # print(sess.run([v,ema.average(v)]))

with tf.Session() as sess:
    saver.restore(sess,'model/model.ckpt')
    print(sess.run(v))

