# tensorflow学习记录  

[tensorflow学习介绍](https://www.jianshu.com/p/87581c7082ba)  

## 开始运行  

vscode不能自动补全tensorflow库，需要在launch.json文件中加入"pythonPath"填入python地址

```python
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入，np.random.rand(a, b)表示生成a行b列的（0-1）的随机数
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))  # tf.random_uniform([a, b], low, high)
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
```

## AlexNet  

&emsp;&emsp;AlexNet由5个卷积层和3个全连接层组成，主要对LeNet网络的改进为：  

* 采用RelU替代sigmoid， 提高了网络的非线性

* 引入Dropout训练方式，增强了网络的健壮性  

* 通过LRN(Local Responce Normalization)提高了网络的适应性（目前由BN替代）

* 通过Data Augmentation证明了大量数据对于模型的作用

```python
# coding=utf-8
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


# define print shape
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []

    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(
            tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1),
            name='weights')
        wx = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(0.0, shape=[64], dtype=tf.float32),
            trainable=True,
            name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv1 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv1)

    lrn1 = tf.nn.lrn(
        conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(
        lrn1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(
            tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
            name='weights')
        wx = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(0.0, shape=[192], dtype=tf.float32),
            trainable=True,
            name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv2 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    lrn2 = tf.nn.lrn(
        conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(
        lrn2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(
            tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32,
                                stddev=1e-1),
            name='weights')
        wx = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(0.0, shape=[384], dtype=tf.float32),
            trainable=True,
            name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv3 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(
            tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                stddev=1e-1),
            name='weights')
        wx = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(0.0, shape=[256], dtype=tf.float32),
            trainable=True,
            name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv4 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(
            tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                stddev=1e-1),
            name='weights')
        wx = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(0.0, shape=[256], dtype=tf.float32),
            trainable=True,
            name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv5 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv5)

    pool5 = tf.nn.max_pool(
        conv5,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool5')
    print_activations(pool5)
    return pool5, parameters


# define time run
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


# main fun
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(
            tf.random_normal([batch_size, image_size, image_size, 3],
                             dtype=tf.float32,
                             stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # time
        time_tensorflow_run(sess, pool5, "Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")


# run
run_benchmark()
```

## VGG16  


