import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# 欠拟合 和 过拟合
# 欠拟合 ：增加输入特征项，增加网络参数，减少正则化参数
# 过拟合： 数据清洗 增大训练集 采用正则化 增大正则化参数
# 正则化缓解过拟合  利用给w增加权值，弱化了训练数据的噪声
# loss = loss(y与 y_) +REGULARIZER *LOSS(W)
# 模型中所有参数的损失函数，如交叉熵、均方误差  用超参数 给出参数w 在总loss中的比例
# 即正则化权重， w 需要正则化的参数
# 正则化选择
# l1 正则化大概率使很多参数变为零，因此可通过稀疏参数，减少参数的数量降低复杂度
# l2 正则化会使参数很接近零 可通过减少参数值大小降低复杂度
# with tf.GradientTape() as tape:
#     h1 = tf.matmul(x_train, w1) + b1
#     h1 = tf.nn.relu(h1)
#     y = tf.matmul(h1, w2) + b2
#     loss_mse = tf.reduce_mean(tf.square(y_train - y))
#
#     loss_regularization =[]
#     loss_regularization.append((tf.nn.l2_loss(w1)))
#     loss_regularization = tf.reduce_sum(loss_regularization)
#     loss = loss_mse +0.03 * loss_regularization
# variables =[w1, b1, w2, b2]
# grads = tape.gradient(loss, variables)

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])
x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

y_c = [['red' if y else 'blue'] for y in y_train]

#转换 x 的数据类型 ，否则后面矩阵相乘 数据类型报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

#from _tensor_slices 函数切分传入的丈量第一个维度，生成数据集
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
#生成神经网络参数，输入层为 2个神经元，隐藏层为 11 个神经元， 1 层隐藏层 ，输出层我诶 1个神经元
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 =tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.005
epoch = 800
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            #前向传播
            h1 = tf.matmul(x_train, w1) +b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) +b2
            #采用均方误差 mse  = mean（sum （y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            #添加 l2正则化
            loss_regularization =[]
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))

            #求和
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization

        #计算 loss 对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        #梯度更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

        #每20个，打印loss
    if epoch %20 ==0:
        print('epoch',epoch,'loss:', float(loss))

#预测部分
print('---------predict-----------')
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
#将网络坐标点喂入神经网络，进行预测
probs =[]
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

#取第0列给x1，取第 1 列 给x2
x1 = x_data[:,0]
x2 = x_data[:,1]

probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color= np.squeeze(y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()