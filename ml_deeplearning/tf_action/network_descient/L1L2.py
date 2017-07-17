#encoding:utf-8
#@Time : 2017/7/17 20:02
#@Author : JackNiu
import tensorflow as tf

# 获得一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称“losses”的集合中
def get_weight(shape,lambda_):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    # add_to_collection 函数将这个新生成变量L2正则化损失项加入集合
    # 这个函数的第一个参数losses是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lambda_)(var))
    # var 为 weights， 而且将这个weight对应的正则化损失加入集合

    return var

x=tf.placeholder(tf.float32,shape=(None,2))
y_= tf.placeholder(tf.float32,shape=(None,1))
batch_size=8
# 定义了每一层网络中节点的个数
layer_dimension =[2,10,10,10,1]

n_layers=len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer=x
in_dimension = layer_dimension[0]

# 每一层的权重应当是没有联系的，重新生成的

for i in range(1,n_layers):
    out_dimension=layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension=layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
# 按照前面所说的，误差由两部分构成，一是模型在训练上的表现，而是过度拟合的修正
# 模型训练可以通过最后的均方差来求出 ，L2修正则是每一层的L2修正
mes_loss = tf.reduce_mean(tf.square(y_ -cur_layer))



#将均方无擦加入损失函数
tf.add_to_collection("losses",mes_loss)

loss= tf.add_n(tf.get_collection('losses'))
