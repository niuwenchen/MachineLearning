## Tensorflow
* 数值计算库
    Tensorflow提供了一个可使用户用数学方法从零开始的函数和类的广泛套件。这使得具有一定技术背景的用户可迅速而直观的创建自定义，具有高灵活性的模型。
    
* DataFlowGraphs(数据流图)
    计算模型是有向图(directed graph),其中每个节点代表了一些函数或计算，而边代表了数值，矩阵或张量
    通过将计算分解为一些小的、容易微分的环节，Tensorflow能够自动计算任意节点关于其他对第一个节点的输出产生影响的任意节点的导数
    (Tensorflow称为Operation)，计算任何节点的导数或梯度对于搭建机器学习模型非常重要。
    通过计算的分解，将计算分布在多个CPU，GPU以及其他计算设备上更加容易，即只需将完整的、较大的数据流图
    分解为一些较小的计算图，并让每台计算设备负责一个独立的计算子图
    
    张量：n维矩阵，2阶张量等价于标准矩阵，3阶张量就是立方数组
  
* 分布式功能
    Tensorflow与特定集群管理器(Kubermetes)的本地兼容性正在得到改善

* 软件套件
    TensorBoard是一个包含在任意标准Tensorflow安装中的图可视化软件。当用户在Tensorflow中引入某些TensorBoard的特定运算时，TensorBoard
    可读取由Tensorflow计算图导出的文件，并对分析模型的行为提供有价值的参考。他对概括统计量、分析训练过程以及调试Tensorflow代码都极有帮助。
    
    Tensorflow Serving是一个可为部署训练的Tensorflow模型带来便利的软件，利用内置的TF函数，用户可讲自己的模型导出到由TF Serving在本地读取
    的文件中，之后，会启动一个高性能服务器。该服务器可接收输入数据，并将其送入预训练的模型，然后将模型的输出结果返回。TF Serving还可以在就模型
    和新模型之间无缝切换，不会给最总用户带来任何停机时间。虽然Serving可能是TF生态系统中认可度最低的组成，却有可能使TF有别于其他竞争者的重要
    因素。

## 安装TF


## TF基础
依赖关系的构成，如果节点之间有依赖关系，那么当这个依赖节点之间的数据没有传递时，这个运算就会被中断
TF的数据类型:

    tf.float32，tf.float64,tf.int8,tf.int16,tf.int32,tf.int64,tf.uint8,tf.string,tf.bool,tf.complex64
    tf.qint8,tf.qint32,tf.quint8
    
    建议养成显式声明Tensor对象的数值属性的习惯，否则可能会导致TypeMismatchError，当然，如果给处理字符串时-创建字符串Tensor对象时，不要指定dtype属性
    
名称作用域组织数据流图(name scope)
名称作用域非常易于使用，且在用TensorBoard对Graph对象可视化时极有价值。名称作用域运行将Op划分到一些较大的，有名称的语句块中。当以后用
Tensorflow加载数据流图时，每个名称作用域对其自己的Op进行封装，从而获得更好的可视化效果。将Op添加到with tf.name_scope(<name>)中


### 3.2.5 TF的Graph对象
创建更多的数据流图，以及如何让多个数据流图协同工作

    g=tf.Graph()
    得到Graph对象，可以利用Graph.as_default()方法得到上下文管理器，添加OP结合with语句，可以利用上下文管理器通知TF
    需要将一些Op添加到某个特定的Graph对象中。
    with g.as_defualt():
    a=tf.mul(2,3)
    
当TF对象被加载时，会自动创建一个Graph对象，并将其作为默认的数据流图，因此，在Graph.as_dafault()上下文管理器之外定义
的任何OP，TS对象都会自动放置在默认的数据流图中:

    #放置在默认的数据流图中
    in_default_graph =tf.add(1,2)
    #放置在数据流图g中
    with g.as_default():
        in_graph_g = tf.mul(2,3)
    #由于不在with语句块中，下面的OP将放置在默认的数据流图中
    also_in_default_grpph= tf.sub(5,1)
    如果希望获得默认数据流图的句柄，可以使用tf.get_default_graph()函数
    
大多数时候，默认的数据流图就足够了，但是，如果需要定义多个互相之间不存在依赖关系的模型，则创建多个Graph对象十分有用。
当需要在单个文件中定义多个数据流图时，最佳实践是不实用默认数据流图，或为其立即分配句柄。

    获得句柄的方式
    (1) 忽略默认数据流图
        g1=tf.Graph()  定义新的数据流图
        g2=tf.Graph()
    (2) 获得默认句柄
        g1=tf.get_default_graph()
        g2=tf.Graph()
        
### TF Session
Session负责数据流图的执行，
.target 指定了要使用的执行引擎，对于大多数应用，该参数去默认的空字符串，在分布式设置中使用Session对象时，该参数用于联结不同的tf.train.Server实例
.graph  指定了将要在Session对象中加载的Graph对象，其默认值为None，表示将要使用当前默认数据流图，当使用多个数据流时。最好的方式是显式传入希望运行的Graph对象
-config 参数允许用户指定配置Session对象所需的选项，如限制CPU或GPU的使用数目，为数据流图设置优化参数以及日志选项等

    sess = tf.Session()
    sess = tf.Session(graph=tf.get_default_graph())  是等价的

说明：
    
    Tensorflow是基于图的计算系统，而图的节点则是由操作（Operations）来构成的，而图的各个节点之间则是由张量(Tensor)
    作为边来连接在一起的，所以Tensorflow的计算过程就是一个Tensor流图，Tensorflow的图则必须在一个Session中来计算。
    
    图的输出和在不在session 或 graph中没关系，但运行必须有session的参与
    
    http://www.cnblogs.com/lienhua34/p/5998853.html




