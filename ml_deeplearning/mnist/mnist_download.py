#encoding:utf-8
#@Time : 2017/7/18 15:36
#@Author : JackNiu
from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)

print("Training data size: ",mnist.train.num_examples)