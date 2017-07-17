## Tensorflow api 教程

### Constants,Sequences,And Random Values

    构造一个维度为n*n的dim:
    x = tf.placeholder(dtype=tf.int32,shape=[3,3])   tf.shape(x)
    
    随机:
    var = tf.Variable(tf.random_uniform([2, 3]), name="xx") 随机生成一个[0,1]之内的实数
    这个和前面的不同,必须得将这个变量初始化 initialize_all_variables()
    
    tf.set_random_seed(100) 这个seed一样的话，就会生出一样的随机数
    
    constant 在使用的时候不需要initial_all_variables()
    variables 必须得实例化
    
    saving and restored 
        class tf.train.Saver
    
    
    Checkpoints 是二进制文件将map变量映射到tensor值，
    
    saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
    ...
    saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    