import numpy as np
from math import sqrt
from scipy.misc import imread, imsave, imresize
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# 快排
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# print(quicksort([4, 5, 21, 35, 2, 54, 2, 4, 6, 9, 38, 7]))


# 基本数据类型
def number_type(isPrint):
    if isPrint:
        x = 3
        print(type(x))
        print(x)
        print(x+1)
        print(x-1)
        print(x*2)
        print(x**2)
        x += 1
        print(x)
        x *= 2
        print(x)
        y = 2.5
        print(type(y))
        print(y, y+1, y-1, y*2, y**2)


# number_type(True)

# 布尔
def bool_test(isPrint):
    t = True
    f = False
    print(type(t))
    print(t and f)
    print(t or f)
    print(not t)
    print(t != f)

# bool_test(True)


# 字符串
def string_test(isPrint):
    hello = 'hello'
    world = 'world'
    print(hello)
    print(len(hello))
    hw = hello + ' ' + world
    print(hw)
    hw12 = '%s %s %d' % (hello, world, 12)
    print(hw12)
    s = hello
    print(s.capitalize())  # 首字母大写
    print(s.upper())  # 全部大写
    print(s.rjust(7))  # 右对齐，空格填充
    print(s.center(7))  # 居中对齐，空格填充
    print(s.replace('l', '(ell)'))  # 替换
    print('  world  '.strip())  # 去掉空格

# string_test(True)


# 容器
# 列表
def Lists_test(isPrint):
    xs = [3, 1, 2]
    print(xs, xs[2])
    print(xs[-1])
    xs[2] = 'foo'
    print(xs)
    xs.append('bar')
    print(xs)
    x = xs.pop()  # 将最后一个元素取出
    print(x, xs)


# Lists_test(True)

# 切片Slicing
def Slcing(nums):
    print(nums)
    print(nums[2:4])  # 不包含最后一个元素
    print(nums[2:])
    print(nums[:2])
    print(nums[:])
    print(nums[:-1])
    nums[2:4] = [8, 9]
    print(nums)


nums = [0, 1, 2, 3, 4, 5]
# Slcing(nums)
# 循环Loops


def Loops(arr):
    for a in arr:
        print(a)
    # 如果想要在循环体内访问每个元素的指针，可以使用内置的enumerate函数
    for idx, a in enumerate(arr):
        print('#%d:%s' % (idx+1, a))


animals = ['cat', 'dog', 'monkey']
# Loops(animals)
# 列表推导


def List_comprehensions(nums):
    square = []
    # for x in nums:
    #     square.append(x ** 2)
    square = [x ** 2 for x in nums]
    square = [x ** 2 for x in nums if x % 2 == 0]
    return square


# print(List_comprehensions(nums))
# 字典Dictionaries，用来存储键和值
# d = {'cat': 'cute', 'dog': 'furry'}


def Dictionaries(d):
    print(d['cat'])
    print('cat' in d)
    d['fish'] = 'wet'
    print(d['fish'])
    print(d.get('monkey', 'N/A'))
    print(d.get('fish', 'N/A'))
    del d['fish']
    print(d.get('fish', 'N/A'))


# Dictionaries(d)
# print(d)
# 循环Loops
d = {'person': 2, 'cat': 4, 'spider': 8}


def Dictionaries_loops(d):
    for animals in d:
        legs = d[animals]
        print('A %s has %d legs' % (animals, legs))
    # 如果要访问键和对应的值，使用items方法
    for animals, legs in d.items():
        print('A %s has %d legs' % (animals, legs))


# Dictionaries_loops(d)
# 字典推导
def Dictionaries_comprehensions(nums):
    square = {x: x ** 2 for x in nums if x % 2 == 0}
    print(square)


# Dictionaries_comprehensions(nums)
# 集合Sets，集合是独立不同个体的无序集合
animals = {'cat', 'dog'}


def sets_test(set):
    print('cat' in set)
    print('fish' in set)
    set.add('fish')
    print('fish' in set)
    print(len(set))
    set.add('cat')
    print(len(set))
    set.remove('cat')
    print(len(set))
    print(set)


# sets_test(animals)
# 循环loops


def set_loops(isPrint=True):
    animals = {'cat', 'dog', 'fish'}
    # 与列表操作相同
    for idx, animal in enumerate(animals):
        print('#%d: %s' % (idx+1, animal))


# set_loops()
# 集合推导set comprehensions


def set_comprehension(isPrint=True):
    nums = {int(sqrt(x)) for x in range(30)}
    print(nums)


# set_comprehension()
# 元组Tuples，元组是一个有序列表（不可改变），元组可以在字典中作为键，还可以作为集合的元素，列表不行。


def tuples_test(isPrint=True):
    d = {(x, x + 1): x for x in range(10)}
    print(d)
    t = (5, 6)
    print(type(t))
    print(d[t])
    print(d[(1, 2)])


# tuples_test()
# 函数Functions


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


# for i in [-1, 0, 1]:
#     print(sign(i))
# 类classes


class Greeter(object):

    # 构造函数constructor
    def __init__(self, name):
        self.name = name  # 建立实例变量

    # 实例方法
    def greet(self, loud=False):
        if loud:
            print("Hello, %s!" % self.name.upper())
        else:
            print('Hello, %s' % self.name)


# g = Greeter('Freed')
# g.greet()
# g.greet(loud=True)
# numpy,numpy是python中用于科学技算的核心库，它提供了高性能的多维数组对象，以及相关工具


def numpy_test(isPrint=True):
    a = np.array([1, 2, 3])
    print(type(a))
    print(a.shape)
    print(a[0], a[1], a[2])
    a[0] = 5
    print(a)

    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(b)
    print(b.shape)
    print(b[0, 0], b[0, 1], b[1, 0])


# numpy_test(True)
# 特殊数组创建


def numpy_array(isPrint=True):
    a = np.zeros((3, 3))
    print(a)
    b = np.ones((3, 3))
    print(b)
    c = np.full((2, 3), 'best')
    print(c)
    d = np.eye(3)  # 对角线为1的数组
    print(d)
    e = np.random.random((3, 3))
    print(e)


# numpy_array(True)
# 数组访问

def numpy_invers(isPrint=True):
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = a[:2, 1:3]
    print(b)
    b[0, 0] = 77  # 数组a的值同时被改变
    print(a[0, 1])

    row_r1 = a[1, :]  # a是一个二维数组，第一个参数确定hang，第二个参数确定列
    row_r2 = a[1:2, :]
    row_r3 = a[:, :]
    print(row_r1, row_r1.shape)  # 一维
    print(row_r2, row_r2.shape)  # 二维
    print(row_r3)

    col_c1 = a[:, 1]
    col_c2 = a[:, 1:2]
    print(col_c1, col_c1.shape)
    print(col_c2, col_c2.shape)


# numpy_invers(True)
# 整型数组访问
a = np.array([[1, 2], [3, 4], [5, 6]])


def numpy_int(arr):
    print(arr[[0, 1, 2], [0, 1, 0]])
    print(np.array([arr[0, 0], arr[1, 1], arr[2, 0]]))
    print(arr[[0, 0], [1, 1]])
    print(arr[0, 1], arr[0, 1])
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = np.array([0, 2, 0, 1])
    print(a[np.arange(4), b])
    a[np.arange(4), b] += 10
    print(a)


# numpy_int(a)
# 布尔类型访问


def numpy_bool():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    bool_idx = (a>2)

    print(bool_idx)

    print(a[bool_idx])

    print(a[a > 2])


# numpy_bool()
# 数据类型


def numpy_dtype():
    x = np.array([1, 2])
    print(x.dtype)

    y = np.array([1.0, 2.0])
    print(y.dtype)

    x = np.array([1, 2], dtype=np.int64)
    print(x.dtype)


# numpy_dtype()
# 数组计算


def numpy_compute():
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)

    print(x+y)
    print(np.add(x, y))

    print(x-y)
    print(np.subtract(x, y))

    print(x*y)
    print(np.multiply(x, y))

    print(x/y)
    print(np.divide(x, y))

    print(np.sqrt(x))


# numpy_compute()
# *是逐元素相乘，dot是矩阵乘法


def numpy_dot():
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)

    v = np.array([9, 10])
    w = np.array([11, 12])

    print(v.dot(w))
    print(np.dot(v, w))

    print(x.dot(v))
    print(np.dot(x, v))

    print(x.dot(y))
    print(np.dot(x, y))


# numpy_dot()
# sum函数，axis=0计算每一列的和加成一行，axis=1计算每一行的和加成一列


def numpy_sum():
    x = np.array([[1, 2], [3, 4]])

    print(np.sum(x))
    print(np.sum(x, axis=0))
    print(np.sum(x, axis=1))
    print(x.T)


# numpy_sum()
# 不同维度数组直接进行运算


def numpy_Broad():
    """
    1.如果数组的秩不同，使用1来将秩较小的数组进行扩展，直到两个数组的尺寸的长度都一样。
    2.如果两个数组在某个维度上的长度是一样的，或者其中一个数组在该维度上长度为1，那么我们就说这两个数组在该维度上是相容的。
    3.如果两个数组在所有维度上都是相容的，他们就能使用广播。
    4.如果两个输入数组的尺寸不同，那么注意其中较大的那个尺寸。因为广播之后，两个数组的尺寸将和那个较大的尺寸一样。
    5.在任何一个维度上，如果一个数组的长度为1，另一个数组长度大于1，那么在该维度上，就好像是对第一个数组进行了复制。  
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    v = np.array([1, 0, 1])

    print(x+v)

    v = np.array([1, 2, 3])
    w = np.array([4, 5])

    print(np.reshape(v, (3, 1))*w)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x+v)

    print((x.T + w).T)
    print(x + np.reshape(w, (2, 1)))
    print(np.reshape(w, (2, 1)))

    print(x*2)


# numpy_Broad()
# scipy


def numpy_scipy():
    img = imread('E:/GitHub/Interview-question-collection/picture/1.jpg')
    print(img.dtype, img.shape)
    img_tinted = img * [1, 0.95, 0.9]
    img_tinted = imresize(img_tinted, (300, 300))
    imsave('1_tinted.jpg', img_tinted)


# numpy_scipy()
# 计算集合中点的距离


def numpy_eu():
    x = np.array([[0, 1], [1, 0], [2, 0]])
    print(x)
    d = squareform(pdist(x, 'euclidean'))
    print(d)


# numpy_eu()
# 绘图


def plt_show():
    x = np.arange(0, 3*np.pi, 0.1)
    y = np.sin(x)

    print(np.pi)
    plt.plot(x, y)
    plt.show()


# plt_show()
# 一图多线


def plt_more():
    x = np.arange(-3*np.pi, 3*np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    y_tanh = np.tanh(x)
    
    plt.plot(x, y_cos)
    plt.plot(x, y_sin)
    plt.plot(x, y_tanh)
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.legend(['cos', 'sin', 'tanh'])
    plt.show()


# plt_more()
# 使用subplot函数在同一幅图中画不同的东西


def plt_subplot():
    x = np.arange(0, 3*np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.subplot(211)
    plt.plot(x, y_sin)
    plt.title('sin')

    plt.subplot(212)
    plt.plot(x, y_cos)
    plt.title('cos')

    plt.show()


# plt_subplot()


def imshow_pic():
    img = imread('1_tinted.jpg')
    img_tinted = img * [1, 0.50, 0.25]
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    print(img.dtype, img)
    print(img_tinted.dtype, img_tinted)
    plt.imshow(np.uint8(img_tinted))
    plt.show()


# imshow_pic()
