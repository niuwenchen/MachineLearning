#encoding:utf-8
#@Time : 2017/7/28 10:13
#@Author : JackNiu

'''
TF Inception_v3 处理
'''

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE=2048

#INCEPTION-v3模型中代表瓶颈层结果的张量名称，谷歌提供的模型中，这个张量名称就是
#'pool_3/_reshape:0'。 在训练的模型中，可以通过tensor.name来获取张量的名称
BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'

#图像输入张量对应的名称
JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

# 下载的谷歌训练好的模型文件目录
MODEL_DIR='incep_model/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE='classify_image_graph_def.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存
# 在文件中
CACHE_DIR='tmp/bottleneck'

INPUT_DATA='flower_data/flower_photos'

# 验证的数据百分比
VALIDATION_PERCENTGE=10

#测试的数据百分比
TEST_PERCENTAGE=10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS=4000
BATCH=100

# 读取多有图片列表并按照训练，验证，测试数据分开

def create_image_list(testing_percentage,validation_percentage):
    # 得到的所有图片都存在result中，这个字典的key为类别的名称，value存储了所有的图片名称
    result={}
    # 获取当前目录下的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # print(sub_dirs)
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        #获取当前目录下所有的有效图片文件
        extensions = ['jpg','jpeg','JPG',"JPEG"]
        file_list=[]
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,"*."+extension)

            file_list.extend(glob.glob(file_glob))

        if not file_list: continue
        # print("file_list",file_list)

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()
        # 初始化当前训练数据集，测试数据集和验证数据集
        training_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练数据集，测试数据集和验证数据集
            chance = np.random.randint(100)
            # print("base_name",base_name)
            if chance <validation_percentage:
                validation_images.append(base_name)
            elif chance <(testing_percentage+ validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 将当前类别的数据放入数据字典
        result[label_name]={
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images
        }
    return result
'''
这个函数通过类别名称、所属数据集和图片标号获取一张图片的弟子。
image_list 给出所有图片信息
image_dir 给出了根目录，存放图片数据的根目录和存放图片特征向量的根目录地址不同。
category: 给出获取的图片是在训练数据集、测试数据集还是验证数据集。

'''
def get_image_path(image_list,image_dir,label_name,index,category):
    label_lists = image_list[label_name]
    category_list =label_lists[category]
    mod_index = index %len(category_list)
    #获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path

'''
这个函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量
'''

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

'''
使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
'''
def run_bollleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    # 将当前图片作为输入计算瓶颈张量的值，这个瓶颈张量的值就是这张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})

    #经过卷积神经网络处理的结果是一个四维数组，将这个结果压缩成一个特征向量（一维）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


'''
获取一张图片经过Inception-v3处理之后的特征向量，这个函数会试图寻找已经计算且保存下来的特征向量
如果找不到则计算这个特征向量，然后保存到文件
'''
def get_or_create_bttleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path  = os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        image_data = gfile.FastGFile(image_path,'rb').read()
        bottleneck_values = run_bollleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        bottleneck_string = ','.join(str(x)  for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values =[float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

'''
随机获取一个batch的图片作为训练数据
'''
def get_random_cached_bottleneck(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bttleneck(
            sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor
        )
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

'''
获取全部的测试数据，在最终测试的时候需要者唉所有的测试数据集上计算正确率
'''

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index ,label_name in enumerate(label_name_list):
        category = 'testing'
        for index,unused_based_name in enumerate(image_lists[label_name][category]):
            bottleneck=get_or_create_bttleneck(
                sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor
            )
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

def main():
    image_lists = create_image_list(TEST_PERCENTAGE,VALIDATION_PERCENTGE)
    n_classes =  len(image_lists.keys())

    '''
        读取已经训练好的Inception-v3模型，谷歌训练好的模型保存在了GraphDefProtocol Buffer中
        里面保存了每一个节点取值的计算方法法以及变量的取值
    '''
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def =tf.GraphDef()
        graph_def.ParseFromString(f.read())

    '''
        加载读取的Inception-V3模型，并返回数据输入对应的张量以及计算瓶颈层结果对应的张量
    '''
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]
    )

    '''
    定义新的神经网络输入，这个输入就是新的图片经过Inception-V3模型前向传播达到瓶颈层时的节点取值
    '''
    bottleneck_input = tf.placeholder(
        tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name="BottleneckInputPlaceholder"
    )

    ground_truth_input = tf.placeholder(
        tf.float32,[None,n_classes],name='GroundTruthInput'
    )

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights)+biases
        final_tensor = tf.nn.softmax(logits)

    '''
    定义损失熵函数
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,ground_truth_input
    )
    cross_entropy_mean = tf.readce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    '''
    计算正确率
    '''
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),
                                      tf.argmax(ground_truth_input,1))
        evaluation_step = tf.readuce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(STEPS):
            '''
                每次获取一个batch的训练数据
            '''
            train_bottlenecks,train_ground_truths=\
                get_random_cached_bottleneck(
                    sess,n_classes,image_lists,BATCH,'training',
                    jpeg_data_tensor,bottleneck_tensor
                )
            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,
                                           ground_truth_input:train_ground_truths})

            if i %100 == 0 or i+1 ==STEPS:
                validation_bottlenecks, validation_ground_truths = \
                    get_random_cached_bottleneck(
                        sess, n_classes, image_lists, BATCH, 'validation',
                        jpeg_data_tensor, bottleneck_tensor
                    )

                validation_accury=sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                ground_truth_input: validation_ground_truths})

                print("Step %d :validation accuracy on random sampled %d examples=%.1f%%"%(i,BATCH,validation_accury*100))

        test_bottlenecks, test_ground_truths = \
            get_test_bottlenecks(
                sess,  image_lists, n_classes,
                jpeg_data_tensor, bottleneck_tensor
            )

        test_accury = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                 ground_truth_input: test_ground_truths})

        print("Final test accuracy = %.1f%%" % (test_accury * 100))

if __name__=="__main__":
    tf.app.run()




