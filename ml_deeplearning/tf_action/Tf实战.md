http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/state_ops.html

## tf.ConfigProto

    tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    sess1= tf.InteractiveSession(config = config)
    sess2=tf.Session(config= config)
    
    allow_soft_plcaement=True时，GPU上的运算可以被放到CPU上运行，
        1. 当运算无法在gPU上执行
        2. 没有GPU资源（机器上只有一个gPU）
        3. 运算输出包含对CPU计算结果的引用
    
### 3.4.3 神经网络参数与Tensorflow变量
Tensorflow集合的概念，所有的变量都会被自动的加入GraphKeys.VARIABLES这个集合。通过tf.all_variables函数可以拿到
当前计算图上所有的变量。拿到计算图上所有的变量有助于持久化整个计算图的运行状态。
trainable参数来区分需要优化的参数(比如神经网络中的参数)和其他参数（比如迭代的轮数）。
Tensorflow中提供的神经网络优化算法会将GraphKeys.TRAINABLE_VARIABLES集合中的变量作为默认的优化对象。

变量的类型是不可以改变的，type变量的维度可以改变，需要设置参数 validate_shaoe=False

    w1=tf.Variable(tf.random_normal([2,3]),stddev=1),name="w1")
    s2=... ([2,2])
    tf.assign(w1,w2)   执行失败
    tf.assign(w1,w2,validate_shape=False)
    
TF 提供了placeholder机制用于提供输入数据，这个位置的数据在程序运行时指定。

    x=tf.placeholder(tf.float32,shape=[1,2],name="input")
    
    反复运行优化算法的效果是可以从下面看出来的
    After 0 training steps,corss entropy on all data is0.0674925
    After 1000 training steps,corss entropy on all data is0.0163385
    After 2000 training steps,corss entropy on all data is0.00907547
    After 3000 training steps,corss entropy on all data is0.00714436
    After 4000 training steps,corss entropy on all data is0.00578471
    After 5000 training steps,corss entropy on all data is0.00430222
    After 6000 training steps,corss entropy on all data is0.00280812
    After 7000 training steps,corss entropy on all data is0.00137464
    After 8000 training steps,corss entropy on all data is2.11566e-05
    
    1. 定义神经网络的结构和前向传播的输出结果
    2. 定义损失函数以及选择反向传播优化的算法
    3. 生成会话(tf.Session)并且在训练数据上反复运行反向传播优化算法
    

## 4 深层神经网络
虽然神经网络有两层，但是和单层的神经网络并没有区别，只通过线性变换，任意层的全连接神经网络和单层神经网络的表达能力
没有任何区别，而且它们都是线性模型。然而线性模型能够解决的问题是有限的。这就是线性模型最大的局限性，
也是为什么深度学习要强调非线性。

    sigmoid:
        f(x) =1/1+e(-x)
    tanh: f(x) = 1-e(-2x)/(1+e(-2x))
    激活函数不是线性的，则整个神经网路模型也就不再是线性的了。
    
    
### 多层神经网络解决异或运算
深度学习的一个重要特征---多层变换，解决异或

损失函数定义

通常神经网络解决多分类问题最常用的方法是设置n个输出节点，其中n为类别的个数。对于每一个样例，设计网络可以得到的一个n维数组作为输出结果
数组中的每一个维度对应一个类别。在理想情况下，如果一个样本属于类别k，那么这个类别所对应的输出节点的输出值应该为1，
而其他节点的输出都为0.交叉熵是常用的评判方法之一，刻画了两个概率分布之间的距离，是分类问题中比较常用的一种损失函数。。

    两个概率分布p,q, 通过q来表示p的交叉熵为  H(p,q) = - p(x)logq(x)
    交叉熵刻画的是两个概率分布之间的距离，但是神经网络的输出却不一定是概率分布
    
    如果将分类问题中的一个样例属于某一个类别看成一个概率时间，那么训练数据的正确答案就符合一个概率分布。
    softmax将神经网络的输出变成 一个概率分布，
    
    H(p,q) 表示用概率q==> p的困难程度，q代表预测值，p代表正确答案，交叉熵值越小，两个概率分布越接近。
    
    正确答案(1,0,0),某模型经过softmax回归之后的预测答案是(0.5,0.4,0.1) 则交叉熵
    H((1,0,0),(0.5,0.4,0.1)) = -(1*log(0.5)+0*log(0.4)+0*log(0.1))=0.3
    另一个是(0.8,0.1,0.1)
    H((1,0,0),(0.8,0.1,0.1)) = 0.1
    则第二个更好
    
    cross_entropy = -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-10,1.0)))
    tf.clip_by_value 将一个张量中的数值限制在一个范围之内，这样可以避免一些运算错误(比如log0是无效的)。
    
        v=tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
        print tf.clip_by_value(v,2.5,4.5).eval()
        [[2.5,2.5,3],[4,4.5,4.5]]
        
    * : 两个矩阵中的元素相乘
    
    tf.nn.softmax_cross_entropy_with_logits(y,y_) softmax之后的交叉熵。
    
    回归问题是一个任意实数，对于回归问题，最常用的损失函数是均方误差(MSE,mean squared error)
    
        MSE(u,y')= sum(yi-yi')^2/ n
        
    
    
### 4.3 神经网络优化算法
通过反向传播算法(backpropagation)和梯度下降算法(gradient decent)调整神经网络中参数的取值。

为什么梯度下降算法并不能保证被优化的函数达到全局最优解，因为梯度的求解是一个跳动的过程，并不是连续的

可以用随机梯度下降算法代替梯度下降算法来降低计算时间，随机优化某一条训练数据上的损失函数，这样每一轮参数更新的速达就大大加快了。

        for i in range(STEPS):
        # 准备batch_size个训练数据
        current_X,cutrrent_Y=...
        sess.run(train_step,feed_dict={x:current_X,y_:cutrrent_Y})


### 4.4 进一步优化
学习率的设置

    学习率控制参数更新的幅度，如果幅度过大，可能导致参数在极优值的两侧来回移动，
    J(x)=x^2, 学习率为1
    
        轮数  当前轮参数值      梯度*学习率      更新后参数值
        1       5               2*5*1=10        5-10=5
        2       -5              2*(-5)*1=-10    -5(--10)=5
        .....
    当学习率为0.001，迭代5次之后，x的值将为4.95，要将x训练到0.05需要大约2300轮；而当学习率为0.3， 只需要5轮
    TF提供学习率设置方法————指数衰减法。 tf.train.exponential_decay函数实现了指数衰减学习策略，
    这个函数，可以先使用较大的学习率快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定。
    
    exponential_decay如下实现:
    
    decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
    decayed_learning_rate:每一轮使用的学习率
    learning_rate:为事先设定的初始学习率
    decay_rate: 为衰减系数,衰减率
    decay_steps: 为衰减速度，通常代表了完整的使用一遍训练数据锁需要的迭代轮数，这个迭代轮数也就是总训练样本数目除以每一个batch
    中的训练样本数。这种设置的常用场景是每完整的训练过完一遍训练数据，学习率就减小一次。可以使得训练集中的所有数据对模型训练
    有相等的作用。
    
    learning_reat = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
    每训练100次后乘以学习率
    tf.train.exponential_decay(learning_rate=0.1,
                           global_step=1000,decay_steps=100,decay_rate=0.96,staircase=True)
    
### 过拟合问题
为了避免过拟合问题，一个常用的方法就是正则化(regularization).正则化的思想就是在损失函数中加入刻画模型复杂程序的指标。
假设损失函数是J(*), 在优化时是优化J(*)+lambda*R(w), R(w)刻画的是模型的复杂程度，而lambda表示模型复杂损失在
总孙志中的比例，模型复杂度只由w决定，

    L1正则化
        R(w)=||w||1= sum(wi)
    L2正则化
        R(w)=||w||2^2=sum(wi^2)
    基本的思想都是通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪音。
    L1会让参数更加稀疏，就是更多的参数w和b变成0，类似特征重要性选取，但L2不会。
    L2不会将很小值的参数调整为0，
    
        loss= tf.reduce_mean(tf.square(y_ - y))+
            tf.contrib.layers.l2_regularizer(lambda)(w)
        均方误差函数，刻画了模型在训练上的表现,
        第二个部分就是正则化，防止模型过度拟合训练数据中的随机噪音
        
    
    当网络结构复杂之后定义网络结构的部分和计算损失函数的部分可能不再同一个函数中，这样通过变量这种方式
    计算损失函数就更不方便了
    
    有一个注意的： 在TF中如何计算最优解？
        train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
        就这一句，使loss最小
        

### 4.4.3 滑动平均模型
采用随机梯度下降算法训练神经网络时，使用滑动平均模型在很多应用中都可以提高模型在测试数据中的表现

随机梯度下降法是采取其中的batch 数据，用滑动平均模型在数据层的选择上进行了更加健壮的分析

        
    






















