import tensorflow as tf
import numpy as np
print(tf.__version__)
print("0----")
#行为主义 基于控制论，构建感知动作控制系统
#符号注意 用 算术逻辑表达式，求解问题描述为表达式
#联结主义 仿生学
# 数据，搭建网络，优化参数，应用网络
#梯度；函数对各参数求偏导后的向量、
#梯度下降法: 沿损失函数梯度下降的方向，寻找损失函数的最小值，得到最优参数方法
# 张量 tf.constant( 张量内容, dtype= 数据类型)
w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.05
epoch = 40
a = tf.constant([1, 5], dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w+1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print("after %s epoch, w is %lf ,loss is %lf" % (epoch, w.numpy(), loss))

# 讲numpy 数据类型转换成 tensor 数据类型
#tf.convert_to_tensor(数据名, dtype= 数据类型)
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print(a)
print(b)
#创建 全为 0 的张量
# tf.zeros(维度)
# 创建全为 1 的张量
# tf.ones(维度)
# 创建指定值 的张量
# tf.fill(维度，张量值)
a = tf.zeros([2, 3])
b = tf.ones([2, 2])
c = tf.fill([2, 1], 5)
print(a)
print(b)
print(c)
# 生成正态分布的随机数，默认均值为0，标准差为1
# tf.random.normal(维度， mean = 均值， stddev = 标准差)
# 生成截断式正态分布随机数
# tf.random.truncated_normal(维度， mean = 均值， stddev = 标准差)
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print(d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print(e)
# 强制tensor转换为该数据类型
# tf.cast(张量名， dtype= 数据类型)
# 计算张良维度上元素最小值
# tf.reduce_min(张量名)
# tf.reduce_max(张量名)
# tf.reduce_sum(zhangliangming)

x1 = tf.constant([1.,2.,3.], dtype= tf.float64)
print(x1)
x2 = tf.cast(x1, dtype=tf.int32)
print(x2)
print(tf.reduce_sum(x2), tf.reduce_min(x2))
# axis 再一个二维张量或者数组中，可以通过调整axis = 0或 1 控制执行维度。
# axis =0  代表跨行 按列计算 axis = 1 代表跨列 按行计算
# 不指定则所有元素参与计算
# tf.reduce_mean(张量名 ，axis= 操作轴)
# 计算张量和
# tf.reduce_sum
x = tf.constant([[1, 2, 3],
                [2, 3, 4]])
print(x)
print(tf.reduce_mean(x))
print(tf.reduce_sum(x, axis=0))
# tf.Variable(初始值) 标记可训练
# w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
# tf.add(张量1，张良2)
# tf.subtract()
# tf.multiply()
# tf.divide() 维度相同
a1 = tf.ones([1, 3])
b1 = tf.fill([1, 3], 3.)
print(tf.add(a1, b1))
print(tf.subtract(a1, b1))
print(tf.multiply(a1, b1))
print(tf.divide(b1, a1))
# 张量开平方   n次方 开放
# tf.square(张量名) tf.pow(张量名，n次方)   tf.sqrt()
a = tf.fill([1, 2], 3.) #3. float32
b = tf.fill([1,2], 3) # 3 int32
print(a)
print(b)
print(tf.pow(a,3))
print(tf.sqrt(a))
# 矩阵乘法
# tf.matmul(矩阵1， 矩阵2)
a = tf.ones([2, 3]) #默认 float32
b = tf.fill([3,2], 3.)
print(tf.matmul(a, b))
# tf.data.Dataset.from_tensor_slices()
# 切分 传入张量的第一维度，生成输入特征、标签对，构建数据集
# data = tf.data.Dataset.from_tensor_slices(输入特征，标签)
# numpy 和 tensor 都可以用这个语句
features = tf.constant([12,23,10,17])
labels = tf.constant([0,1,1,0])
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
print(dataset)
for ele in dataset:
    print(ele)

# tf.GradientTape
# with结构记录计算过程， gradient求出张量梯度
# with tf.GradientTape() as tape:
#     若干计算过程
#     grad = tape.gradient(函数，对谁求导)

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w,2)
grad = tape.gradient(loss, w)
print(grad)   # w =3  w2 求导 就是 2w  = 6
# enumerate py 内建函数，它可便利每个元素
seq=['one','two','three']
for i,ele in enumerate(seq):
    print(i,ele)
# tf.one_hot  讲待转换数据 转换为 one_hot 形式输出读热码 做分类标记 1 是 0 非
# tf.one_hot(待转换数据，depth = 几个分类)
classes = 3
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print(output)
# tf.nn.softmax 使输出符合概率分布
# 当n分类n个输出 符合概率分布 和为1
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print("after softmax ypro is ", y_pro)
# assign_sub 赋值操作，更新参数的值并返回
# assign_sub前 先用tf.Variable 定义变量w 为可训练
# w.assign_sub(w要自减的内容)
w = tf.Variable(4)
w.assign_sub(1)
print(w)
# tf.argmax( 返回张量沿指定维度最大值索引)
test = np.array([[1, 2, 3],
                 [2, 3, 4],
                 [5, 4, 3],
                 [8, 7, 2]])
print(test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))
