##MNIST数字识别问题
    
    1. 定义神经网络的结构和前向传播的输出结果
    2. 定义损失函数以及选择反向传播优化的算法
    3. 生成会话(tf.Session)并且在训练数据上反复运行反向传播优化算法
    
    通过损失函数进行计算差距，再通过反向传播算法对神经网络参数的取值进行优化更新，尽量使在batch这个数据上的计算结果与答案更接近。之后继续进行下一次迭代。
    minimize(loss,global_step)
    An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.
      
    反向传播优化算法：目前TensorFlow框架一共支持七种不同的优化器，最常用的三种是tf.train.GradientDescentOptimizer、tf.train.AdamOptimizer、tf.trainMomentumOptimizer。
    
    认知上的差异: 在做python实现神经网络Bp的时候，显式的更新权重去减小误差，但是TF这里就完成了对内部所有变量的训练以及更新，
    Add operations to minimize `loss` by updating `var_list
    有一个var_list 去被训练更新。
    
    训练过程
    xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            
### 5.2.2 使用验证数据集判断模型效果
虽然一个神经网络模型的效果最终是通过测试数据来评判的，但是我们不能直接通过模型过度拟合测试数据，从而失去对未知数据的预判能力。
因为一个神经网络模型的最终目标是对位置数据提供判断，所以为了估计模型在未知数据上的效果，需要保证测试数据在训练
过程中是不可见的。只有这样才能保证通过测试数据评估出来的效果和在真实应用场景下模型对未知数据玉盘的效果是结晶的。

除了使用验证数据集，还可以采用交叉验证cross validation的方式来验证模型效果，但因为神经网络训练时间本身就比较长，采用cross validation
会花费大量时间。所以在海量数据的情况下，一般会采用验证数据集的形式来评测模型。

### 5.3 变量管理
前面的ingerence函数包括了神经网络中的所有参数，然而，当神经网络的结构更加复杂，参数更多时，就需要一个更好的方式来传递和管理身网络中的
参数了。TF提供了通过变量名称来创建或获取一个变量的机制。通过该机制在不同的函数中可以直接通过变量的名字来使用变量，而不需要
将变量通过参数的形式到处传递。 TF中通过变量名称获取变量的机制通过tf.get_variable和tf.variable_scope函数实现的。

    v=tf.get_variable("V",shape=[1],initializer=tf.constant_initializer(1.0))
    v=tf.Variable(tf.constant(1.0,shape=[1],name="V")
    
    initializer 的初始化函数
    tf.constant_initializer
    tf.random_normal_initializer
    tf.truncated_normal_initializer
    tf.random_uniform_initializer
    tf.uniform_unit_scaling_initializer
    tf.zeros_initializer
    tf.ones_initializer
    
    如果需要通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生成一个上下文管理器，并明确指定在这个上下文管理器中，tf.get_variable将直接获取已经生成的变量。
    变量空间 variable_scope,name_scope
    
    with tf.variable_scope("foo"):
        v=tf.get_variable("V",shape=[1],initializer = tf.constant_initializer(1.0))
    
    再次在下面声明就会报错，如果设置reuse=True,就可以获取已经声明过的变量，仅且仅能获取声明过的变量
    with tf.variable_scope("foo",reuse=True):
        v=tf.get_variable("V",shape=[1],initializer = tf.constant_initializer(1.0))
    
    
    

### 5.4 模型持久化,保存的是这个会话
tf.train.Saver() 
    
    result=v1+v2
    saver.save(sess,"./model/test.ckpt")
    三个文件
    test.ckpt.meta  保存了TF计算图的结构,也可以说是神经网络的网络结构
    test.ckpt.data  每一个变量的取值
    test.ckpt.index 
    checkpoint  模型文件列表
    
    
    恢复:
    result =v1-v2
    saver.restore(sess,'./model/test.ckpt')
    print(sess.run(result))(注意 result还是和以前一样定义)
    恢复后并不是之前的3，而是-1，也就是说，这里保存的只是原始变量数据，而不是操作数据
    
    
    保存部分变量
    saver = tf.train.Saver([v1])
    只有变量v1被加载进来，并且被执行initializer方法，其他的都没有初始化，是不能用的
    
    saver = tf.train.Saver({"v1":v2})
    
    v= tf.Variable()....
    maintain_average_op=ema.apply(tf.all_variables())
    saver.save(sess,...) 这里保存了两个变量，v和影子变量，v:0   v/ExponentialMovingAverage:0
    生成变量： v/ExponentialMovingAverage:0
    
    ******实测******
        一般意义上认为，滑动变量必须得经过apply，再average之后才会生成影子变量，但是这里的保存方式
        是sess，并没有别的操作，但是还是保存了，那就认为是这样的，只要有保存sess，那么就会将影子变量默认执行
        说不清这是个什么原理，可能在生成影子变量的那一刻起就保存了
    
    加载的时候可以不用加载v，而是影子变量的值，在使用训练好的模型的时候就不再需要调用函数来获取变量的滑动平均值了。
    saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
    
  
  
### 持久化技术及数据格式

    message MetaInfoDef{
        string meta_graph_version =1
        OpList stripped_op_list =2;
        google.protobuf.Any any_info =3;
        repeated string tags=4;
    }
   
    graph_def属性
    主要记录了TF计算图上的节点信息，每一个节点对应了TF的一个运算，meta_info_def中已经包含了所有运算的具体信息，所以graph_def属性
    只关注运算的链接结构
    
    
### 最佳实践样例

本节将介绍一个TF训练神经网络模型的最佳实践。将训练和测试分成两个独立的程序，这可以使得每一个组件更加灵活。比如训练神经网络的程序可以持续输出训练好的模型，而测试程序可以每个一段时间检验最新模型的正确率，比如模型效果更好，
则将这个模型提供给产品使用。除了将不同功能模块分开，还将前向传播的过程抽象成一个单独的库函数。

前向模型没有什么疑惑的，就是一个神经网络前向计算的过程，主要是声明了两个层次，隐藏层和输出层； 主要一个是在这个里面增加了
参数weight和biase初始化的过程。那么这两个变量就在这里被初始化了，后面执行什么操作和这个前向网络无关，也就是说，再次执行
inference操作获取输出值和第一次初始化有关，但是后面这两个值都发生了改变，到底是谁在改变它？就是BP层。

训练层: 定义训练过程，并不定义准确率，准确率涉及到了验证，这里只关注在训练的过程中出现的参数，也就是损失函数loss

采用交叉熵

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

然后用步长变化lambda也就是学习率利用随机梯度下降法去优化这个loss，使loss最小

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
执行真正运行操作

    需要执行train_step 去更新参数，去执行滑动操作去平均参数，去执行loss去获取损失值
    保存这个会话，这个会话中有什么？ 知道的就是四个参数和对应的影子变量
    那么验证过程需要什么，也就是这四个参数，用这四个参数去实施计算过程。
    
验证过程

我们假定现在已经有四个变量weight，biases，如何验证，加载验证数据，执行验证操作，评价标准是accuracy

    y=mnist_inference.inference(x,None)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
这只是流程，还未执行，也就代表还未加载四个变量，现在从model中加载四个变量，用一个简单的方式加载滑动变量

    variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variabels_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variabels_to_restore)
    
    saver.restore(sess,ckpt.model_checkpoint_path) 恢复四个变量
    现在进行必要的accuracy计算，也就是run操作
    run-->accuracy--> inference--> 就是将现有的四个值应用到现有的模型，
    get_variable()方法可以获取已经存在的变量的值
    
验证流程也完成，一步异步都是很明确的，现在就是里面的一些小的计算过程需要详细一下。
























