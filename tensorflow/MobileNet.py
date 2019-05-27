# coding=utf-8
import tensorflow as tf
import math
import numpy as np

batch_size = 32
num_batches = 1000


# define conv
def conv(input, name, kh, kw, n_out, dh, dw, p):
    """
    定义平常卷积
    input:输入，维度为[batch_size, high, weight, channel]
    name:操作名称
    kh:卷积核的high
    kw:卷积核的weight
    n_out:输出通道数
    dh:high操作的步长
    dw:weight操作的步长
    p:包含所有参赛的列表
    """
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + "w",
            shape=[kh, kw, n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input, kernel, [1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(wx_add_b, name=scope)
        p += [kernel, biases]
        print(str(name) + ' ' + str(conv.shape))

        return relu


def dwconv(input, name, kh, kw, dh, dw, p):
    """
    深度卷积
    """
    # 获取输入通道数
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + "w",
            shape=[kh, kw, n_in, 1],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        dwconv = tf.nn.depthwise_conv2d(input, kernel, [1, dh, dw, 1], padding="SAME")
        bias_init_val = tf.constant(0.0, shape=[n_in], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(dwconv, biases)
        relu = tf.nn.relu(wx_add_b, name=scope)
        p += [kernel, biases]
        print(str(name) + ' ' + str(dwconv.shape))

        return relu


# max_pool
def max_pool(input, name, kh, kw, dh, dw):
    return tf.nn.max_pool(
        input,
        ksize=[1, kh, kw, 1],
        strides=[1, dh, dw, 1],
        padding='SAME',
        name=name
    )


# define mobel
def inference(input):
    p = []
    conv1 = conv(input, name="conv1", kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    print('pool1 ' + str(pool1.shape))

    dwconv1 = dwconv(pool1, name="dwconv1", kh=3, kw=3, dw=1, dh=1, p=p)
    conv2 = conv(dwconv1, name="conv2", kh=1, kw=1, n_out=64, dw=1, dh=1, p=p)
    dwconv2 = dwconv(conv2, name="dwconv2", kh=3, kw=3, dw=1, dh=1, p=p)
    conv3 = conv(dwconv2, name="conv3", kh=1, kw=1, n_out=128, dw=1, dh=1, p=p)
    pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    print('pool2 ' + str(pool2.shape))

    dwconv3 = dwconv(pool2, name="dwconv3", kh=3, kw=3, dw=1, dh=1, p=p)
    conv4 = conv(dwconv3, name="conv4", kh=1, kw=1, n_out=128, dw=1, dh=1, p=p)
    dwconv4 = dwconv(conv4, name="dwconv4", kh=3, kw=3, dw=1, dh=1, p=p)
    conv5 = conv(dwconv4, name="conv5", kh=1, kw=1, n_out=256, dw=1, dh=1, p=p)
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")
    print('pool3 ' + str(pool3.shape))

    dwconv5 = dwconv(pool3, name="dwconv5", kh=3, kw=3, dw=1, dh=1, p=p)
    conv6 = conv(dwconv5, name="conv6", kh=1, kw=1, n_out=256, dw=1, dh=1, p=p)
    dwconv6 = dwconv(conv6, name="dwconv6", kh=3, kw=3, dw=1, dh=1, p=p)
    conv7 = conv(dwconv6, name="conv7", kh=1, kw=1, n_out=512, dw=1, dh=1, p=p)
    pool4 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")
    print('pool4 ' + str(pool4.shape))
    # for i in range(5): 
    return pool4


# main fun
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(
            tf.random_normal([batch_size, image_size, image_size, 3],
                             dtype=tf.float32,
                             stddev=1e-1))
        # conv_relu = conv(images, "conv", 3, 3, 32, 1, 1, p_conv)
        # dwconv_relu = dwconv(conv_relu, "dwconv", 3, 3, 32, 1, 1, p_dwconv)

        predict = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        


run_benchmark()
