#encoding:utf-8
#@Time : 2017/7/20 10:39
#@Author : JackNiu
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import  input_data

import  ml_deeplearning.mnist.mnist_inference as mnist_inference
import ml_deeplearning.mnist.mnist_train2  as mnist_train

# 每10秒架子啊一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果，不关注正则损失化
        y=mnist_inference.inference(x,None)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        # 滑动平均模型
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variabels_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variabels_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
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
            # time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/MNIST_data/",one_hot=True)
    evaluate(mnist)

# tf.app.run 会调用上面的main函数
if __name__=='__main__':
    tf.app.run()
