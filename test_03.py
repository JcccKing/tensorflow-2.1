import tensorflow as tf
import numpy as np

# tf.where(条件语句 ，真 返回a , 假 返回 b) 条件语句
a = tf.constant([1, 2, 3, 0, 1])
b = tf.constant([0, 1, 2, 4, 5])
c = tf.where(tf.greater(a, b), a, b) # greater 进行元素比较
print(c)
#np.random.RandomState.rand(维度) 返回 0-1 之间随机数
rdm = np.random.RandomState(seed= 1)
a = rdm.rand() # 返回随机一个数字
b = rdm.rand(2,3) # 返回2行3 列 随机数字矩阵
print(a)
print(b)
#np.vstack(将两个 数组按垂直方向叠加)
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.vstack((a, b))
print('叠加后：', c) # [[1 2 3]  [2 3 4]]
# np.mgrid[起始值：结束值：步长，起始值：结束值：步长]
# x.ravel() 将x 变成一维数组， 把，前变量拉直
# np.c_[] 使返回的间隔数值点配对
x, y = np.mgrid[1:3:1, 2:4:0.5]
grid = np.c_[x.ravel(),y.ravel()]
print("x", x)
print('y',y)
print('grid\n', grid)
# 神经网络复杂度 NN多用层数和参数个数表示
# 空间复杂度 层数 = 隐藏层 + 1个输出层 ， 总参数 = 总w +总b
# 时间复杂度 运算次数
# 指数衰减学习率 先用较大学习率，找到最优解，再逐渐减小学习率，使模型在训练后稳定
# 指数衰减学习率 = 初试学习率 × 学习率衰减率

# epoch = 40
# lr_base = 0.2
# lr_decay = 0.99
# lr_step = 1
# for epoch in range(epoch):
#     lr = lr_base * lr_decay ** (epoch/lr_step)
#     with tf.GradientTape() as tape:
#         loss = tf.square(w + 1)
#     grads = tape.gradient(loss, w)
#     w.assign_sub(lr * grads)
#     print(epoch, w.numpy(), loss, lr)

# tf.nn.sigmoid(x) 激活函数
# 特点： 易造成梯度消失； 输出非0均值 收敛慢 ；幂运算复杂 训练时间长
# tf.math.tanh()
# 特点 输出是0 均值 易造成梯度消失 幂运算复杂 训练时间长
#
# tf.nn.relu() 某些神经元永远不被激活 参数不能更新，解决了梯度消失问题，只需判断输入是否大于0，计算速度快
# 收敛速度 远快于前两个
# tf.nn.leaky_relu()
# 初学者： 首选 relu,学习率 设置较小值，输入特征标准化，让输入特征满足以0为均值 1为标准差的正态分布
# 初始参数中心化，让随机生辰参数 满足以0为均值
#
# 损失函数 loss y 与y_已知答案 差距
# mse （mean squared error)  ce（cross entropy)
# loss_mse = tf.reduce_mean(tf.square(y_ - y))
seed = 23455
rdm = np.random.RandomState(seed=seed)
x = rdm.rand(32, 2)# 32组 0-1
y_ = [[x1 + x2 + (rdm.rand() / 10.0-0.05)] for (x1, x2) in x]
x = tf.cast(x, dtype= tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
epoch = 1000
lr = 0.002
cost = 99
formats = 1
#自定义 酸奶 成本 1 元 ，利润 99
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        #loss_mse = tf.reduce_mean(tf.square(y_ - y))
        loss_zdy = tf.reduce_sum(tf.where(tf.greater(y,y_), (y-y_)*cost, (y_ -y) *formats))
    grads = tape.gradient(loss_zdy, w1) #求偏导
    w1.assign_sub(lr * grads)
    if epoch % 500 == 0:
        print("agter %d training steps, w1 is " % (epoch) )
        print(w1.numpy())
print("final w1 is ", w1.numpy())
#自定义损失函数 loss = tf.reduce_sum(tf.where(tf.greater(y,y_),cost(y-y_), frofit(y_ - y)
# 交叉熵损失函数 cross entropy
# 二分类已知答案 y_ =(1,0) 预测 y1 =(0.6,0.4) y2 = (0.8, 0.2) 哪个更接近
# tf.losses.categorical_crossentropy(y_, y)

loss_ce1 = tf.losses.categorical_crossentropy([1,0],[0.6,0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1,0],[0.8,0.2])
print(loss_ce1)
print(loss_ce2)
# softmax 与交叉函数结合
# 输出 先过softmax函数，再计算y与y_交叉熵损失函数。
# tf.nn.softmax_cross_entropy_with_logits(y_, y)
y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)
print(loss_ce1)
print(loss_ce2)