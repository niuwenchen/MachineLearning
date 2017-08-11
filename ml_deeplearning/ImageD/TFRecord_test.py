#encoding:utf-8
#@Time : 2017/7/24 17:46
#@Author : JackNiu

import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data
import  numpy as np

# 生成整数型的属性
def __int64__feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串的属性
def __bytes__feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist  =input_data.read_data_sets("D:\PycharmProjects\MachineLearning\ml_deeplearning\CNN\MNIST_data",dtype=tf.uint8,one_hot=True)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples= mnist.train.num_examples

#输出 TF
filename="tfrecord/output.tfrecords"
writer= tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw= images[index].tostring()
    # 转换
    example=tf.train.Example(features = tf.train.Features(feature={
        'pixel':__int64__feature(pixels),
        'label':__int64__feature(np.argmax(labels[index])),
        'image_raw': __bytes__feature(image_raw)
    }))
    writer.write(example.SerializeToString())

writer.close()

