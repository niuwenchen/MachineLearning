## 图像识别与卷积神经网络
第5章通过MNIST数据集验证了第4章介绍的神经网络设计与优化的方法。从实验的结果可以看出，神经网络的
结构会对神经网络的准确率产生巨大的影响。卷积神经网络(Convolutional Neural Network,CNN)
卷积神经网络的应用非常广泛，在自然语言处理、医药发现、灾难气候发现甚至为其人工智能程序中都有应用。
通过卷积神经网络在图像识别上的应用来讲解卷积神经网络的基本原理以及如何使用TF实现卷积神经网络。

数据集 Cifar： http://www.cs.toronto.edu/~kriz/cifar.html

Cifar数据集最大的区别在于图片由黑白变成的彩色，且分类的难度也相对更高。 在Cifar-10数据集上，人工标注的正确率
大概为94%，这比MNIST数据集上的人工表现要低的多。

第一:现实生活中的图片分辨率远高于32*32，而且分辨率不固定，第二，物体类别很多，无论是10种还是100种远远不够，而且一张图片中不会只出现
一个种类的物体。

http://wordnet.princeton.edu

ImageNet是一个基于WordNet的大型图像数据库,历届LSVRC竞赛的题目和数据集

http://www.image-net.org/challenges/LSVRC


### 6.2 卷积神经网络简介
前面的神经网络每两层之间的所有节点都是有边相连的，所以本书称这种网络结构为全连接网络结构。

和全连接神经网络一样，卷积神经网络中的每一个节点都是一个神经元。在全连接神经网络中，没相邻两层之间的节点都有边
相连，于是一般会将每一层全连接层中的节点组织称一列，方便显示连接结构。而对于卷积神经网络，相邻两层之间
只有部分节点向量，为了展示每一层神经元的维度，一般会将每一层卷积层的节点组织成一个三维矩阵。

卷积神经网络和全连接神经网络的唯一区别就在于神经网络中相邻两层的连接方式。为什么全连接神经网络无法很好的处理图像数据
32*32*3  32*32表示图片的大小，有1024个像素点， *3 表示三种颜色，1024(red)+1024(green)+1024(blue)

    [[[r,g,b],[r,g,b],....[r,g,b]32],
    [[r,g,b],[r,g,b],....[r,g,b]32],
    [[r,g,b],[r,g,b],....[r,g,b]32],
    ...
    [[r,g,b],[r,g,b],....[r,g,b]32],
    ]

1 输入层

    图像中，一般代表图片的像素矩阵。三维矩阵的深度代表了图像的色彩通道，黑白图片是1，RGB是3.
    从输入层开始，卷积神经网络通过不同的神经网络结构将上一层的三维矩阵转化为下一层的三维矩阵，直到最后的全连阶层
    
2 卷积层
    
    卷积层的每一个节点的输入只是上一层神经网络的一小块，这个小块常用的大小有5*5 或 3*3。
    卷积层试图将神经网络中的每一小块进行更加深入的分析从而得到抽象程度更高的特征。一般来说，通过卷积层处理过的节点矩阵
    会变得更深，所以上图可以看到经过卷积层之后的节点矩阵的深度会增加。
    
3 池化层

    池化层神经网络不会改变三维矩阵的深度，但是可以缩小矩阵的大小。池化操作可以认为是一个将分辨率较高的图片
    转换为分辨率较低的图片。通过池化层，可以进一步缩小全连接层中节点的个数，从而达到减少整个神经网络中参数的目的。
    
4 全连阶层

    在经过多轮卷积层和池化层的处理之后，在卷积神经网络的最后一般会是一个1到2个连接层来给出最后的分类结果。
    经过几轮卷积层和池化层处理之后，可以认为图像中的信息已经被抽象成了信息含量更高的特征。可以将卷积层和池化层
    看成自动图像特征提取的过程。在特征提取玩之后，仍需要使用全连接层来完成分类任务。
    
5 Softmax层
    
    可以得到当前样例属于不同种类的概率分布情况。
    
### 6.3 卷积神经网络常用结构

    过滤器可以将当前层神经网络上的一个子节点矩阵转换为下一层神经网络上的一个单位节点矩阵。
    单位节点是一个长和宽都为1，但是深度不限的节点矩阵
    
    当前神经网络节点矩阵：过滤器处理的矩阵深度和当前层设计网络节点矩阵的深度是一致的
    单位节点矩阵的深度: 过滤器的深度
    过滤器的尺寸是一个过滤器输入节点矩阵的大小，而深度指的是输出单位节点矩阵的深度。
    输入矩阵的尺寸为过滤器的尺寸，单位矩阵的深度为过滤器的深度。
    
    2*2*3 --> 1*1*5 的单位节点矩阵
    过滤器的前向传播过程和全连接层类似，需要的参数是2*2*3*5+5= 65 个参数
    
说明

    使用过滤器计算节点g(i)的输出，卷积层结构的前向传播过程就是通过将一个过滤器从神经网络当前层的左上角移动到右下角，并且
    在移动中计算每一个对应的单位矩阵得到的。
    
卷积层的参数设定

    在卷积神经网络中，每一个卷积层中使用的过滤器中的参数都是一样的，这是卷积神经网络的一个重要性质。
    只管上理解，共享过滤器的参数可以使得图像上的内容不受位置的影响。以MNIST手写数字为例，无论数字"1"出现在左上角还是右下角，
    图片中的种类都是不变的。 因为在左上角和右下角使用的过滤器参数相同，所以通过卷积层之后无论数字在图像上的
    哪个位置，得到的结果都一样。
    
    以Cafar-10为例，假设第一层卷积层使用尺寸为5*5，深度为16的过滤器，那么这个卷积层的参数个数为5*5*3*16+16ge=1216个
    32*32*3 的输入数据
    5*5     卷积层尺寸
    
    卷积层的参数个数和图片的大小无关，只和过滤器的尺寸、深度以及当前层节点矩阵的深度有关
                                    5*5           16            3

池化层说明
    
    在卷积层之间往往会加上一个池化层(polling layer),池化层可以非常有效的缩小矩阵的尺寸，从而减少最后全连接层中的参数。
    使用池化层既可以加快计算速度也防止过拟合问题的作用。
    
    池化层前向传播的过程也是通过移动一个类似过滤器的结构完成的。不过池化层过滤器中的计算不是节点的加权和，
    而是采用更加简单的最大值或者平均值运算。使用最大值操作的池化层被称之为最大池化层(max pooling),这是被用的
    最多的池化层结构，使用平均值操作的池化层被称之为平均池化层.
    
    卷积层和池化层中过滤器移动的方式是相似的，唯一的区别在于卷积层使用的过滤器是横跨整个深度的，而池化层使用的
    过滤器只影响一个深度的节点。所以池化层的过滤器除了在长和宽两个维度移动之外，还需要在深度这个维度移动
    
    
## 6.4 经典卷积网络模型
### 6.4.1 LeNet-5模型
节点数目的计算

卷积层

    输入数据32*32*1 , 过滤器的尺寸是5*5,深度为6，步长为1，不用0填充， 则这一层的输出尺寸为28*28
    深度为6，共有5*5*1*6+6=156个参数，
    下一层的输入是28*28*6 [28,28,6] 6个维度
    每个维度和5*5=25 个当前节点相连，所以本层卷积层共有4704*(25+1) = 122304 个连接
    

池化层

    输入是28*28*6的节点矩阵，过滤器大小为2*2，长和宽步长均为2，所以输出矩阵14*14*6
    
第三层卷积层

    14*14*6--> 5*5 深度为16， 不实用全0填充，步长为1，输出矩阵10*10*16， 参数5*5*6*16+16=2416个参数
    10*10*16*(25+1 )=41600个连接
    
第四层池化层

    输入为10*10*16，过滤器为2*2，步长为2， 输出5*5*16
    
第五层 全连接层

    5*5*16的输入数据，并将这组数据拉成一个向量，就和神经网络对接了，输出节点为120个，
    参数个数 5*5*16*120 + 120 = 48120个参数
    
第六层 全连接层

    输入个数120个，输出84个
    
第七层 全连接层

    输入节点84个，输出节点10个，总共84*10+10=850个参数
    

具体过程:

	（1） 输入数据格式的转变
	格式要求 4维，
	reshape_xs=np.reshape(xs,(BATCH_SIZE,mnist_inference.IMAGE_SIZE,                                  mnist_inference.IMAGE_SIZE,
	mnist_inference.NUM_CHANNELS))
	Batch_size不用考虑是固定的，再就是行列，以及深度
	
	（2） 权重格式的改变
	4维， 行，列，输入数据通道数目，深度
	conv1_weights= tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
	initializer=tf.truncated_normal_initializer(stddev=0.1))

	（3）计算公式的改变
	前向计算:计算卷积
	conv1= tf.nn.conv2(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
	计算softamx输出
	relu1= tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

	（4） 池化层的计算
	pool1= tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

	（5） 继续卷积层、池化层
	（6） 全连接层
	需要的数据是一行数据，不再是多维，
	pool_shape = pool2.get_shape().as_list()
	nodes= pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

	全连接层weight
	fc1_weighs = tf.get_variable("weight",[nodes,FC_SIZE],
	initializer=tf.truncated_normal_initializer(stddev=0.1))
	
	只有全连接层的权重需要加入正则化
	if regularizer != None:
       tf.add_to_collection("losses",regularizer(fc1_weighs))
	
	比较有用的方法
	if train: fc1= tf.nn.dropout(fc1,0.5)
	
	（7） 输出层
	fc2_weighs =tf.get_variable("weight",[FC_SIZE,NUM_LABELS], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
	
	logit = tf.matmul(fc1,fc2_weighs)+fc2_biases
	得到结果

	（8） 训练过程详解
	训练模式和前面是一样的，主要是在定义前向训练过程，
	用python code模式，发现复杂的地方是在反向训练优化的过程中
	用Tensorflow发现是在前向训练过程比较复杂，至于反向传播
	（8.1） 定义滑动模型
	variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
	（8.2） 定义评判标准  交叉熵
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
	（8.3） 定义损失函数
	loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
	（8.4） 根据学习率优化损失函数即反向传播过程
	learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,                                               global_step,mnist.train.num_examples/BATCH_SIZE,
     LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	（8.5） 定义反向运行Op
	train_op = tf.group(variables_averages_op,train_step)
	（8.6） 训练过程，feed_dict填充
	xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs=np.reshape(xs,(BATCH_SIZE,mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            # print(np.shape(reshape_xs))
            _,loss_value,step= sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs,y_:ys})

结论： 子MNIST测试数据上，上面给出的卷积神经网络可以达到99.4%的正确率。相比第5章中最高的98.4%的正确率，卷积神经网络可以巨幅提高神经网络在MNIST数据集上的正确率。

LeNet-5模型就无法很好的处理类似ImageNet这样比较大的图像数据集。如何设计卷积神经网络的架构？

	输入层-->(卷积层+-->池化层?)+-->全连阶层+

在过滤器的深度上，大部分卷积神经网络都采用逐层递增的方式，每经过一次池化层后，卷积层过滤器的深度会乘以2，虽然不同的模型会使用不同的数字，但是逐层递增是比较普遍的模式。卷积层的步长一般为1，但是有的模型中也会使用2，或者3作为步长。池化层的配置相对简单，用的最多的就是最大池化层。池化层的过滤器变长一般为2或者3，步长也一般为2或者3.


## Inception-v3 模型
Inception是一种和LeNet-5完全不同的卷积神经网络结构。在Lenet-5模型中，不同卷积层通过串联的方式连接在一起，
而Inception-v3模型中的Inception结构是将不同的卷积层通过并联的方式结合在一起，

    Lenet-5中的卷积层用的边长 是1*1， 3*3, 5*5 的过滤器，Inception模块同时使用所有不同尺寸的
    过滤器，然后再将得到的矩阵拼接起来。
   
    Inception模块会首先使用不同尺寸的过滤器输入矩阵。分别是1*1，3*3，5*5，不同的矩阵代表了Inception模块中的
    一条计算路径。虽然过滤器的大小不同，但是如果所有过滤器都使用全0填充且步长为1，那么前向传播神经网络得到的
    结果矩阵的长和宽多余输入矩阵一致。
    
        全0填充:  输入矩阵/步长 都一样的尺寸
    深度为三个模块的和
    
    Inception-v3模型总共有46层，由11个Inception模块组成。总共有96个卷积层。
    
    # 直接使用Tf原始API实现卷积层

    import tensorflow as tf
    with tf.variable_scope(scope_name):
    weights = tf.get_variable("weight",....)
    biases = tf.get_variable('biases',...)
    conv = tf.nn.conv2d()
    relu = tf.nn.relu(tf.nn.bias_add(conv,biases))

    # 使用TF-Slim实现卷积层，通过TF-Slim可以在一行中实现一个卷积层的前向传播算法
    # slim.conv2d函数有3个参数是必填的，第一个参数为输入节点矩阵，第二个参数是当前卷积层过滤器的深度
    # 第三个参数是过滤器的尺寸，可选的参数有过滤器步长，是否全0填充，激活函数的选择等等
    net= tf.Slim.conv2d(input,32,[3,3])
    

## 卷积神经网络迁移学习
迁移学习就是将上一个问题上训练好的模型通过简单的调整使其适用于一个新的问题。介绍如何利用ImageNet数据集上训练好的
Inception-v3模型来解决一个新的图像分类问题，可以保留训练好的Inception-v3模型中所有卷积层的参数
，只是替换最后一层全连接层。在最后一层全连接层之前的网络层称之为瓶颈层

将新的图像通过训练好的卷积神经网络知道瓶颈层的过程可以看成是对图像进行特征提取的过程。。
在训练好的inception-v3模型中，因为将瓶颈层的输出再通过一个单层的全连接层神经网络可以很好的区分1000种类别的图像，所以有理由
认为瓶颈层的输出的节点向量被认为任何图像一个更加精简且表达能力更强的特征向量。于是，在新数据集上，可以直接利用这个训练好的神经网络对图像进行特征提取，然后再将提取的特征向量作为输出来训练一个单层的全连接神经网络处理新的分类问题。

### TF实现迁移学习



    
    
    
            
   