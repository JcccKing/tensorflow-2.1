import tensorflow as tf
from sklearn import datasets
import numpy as np
from pandas import DataFrame
import pandas as pd
from matplotlib import pylab as plt
import time
x_data = datasets.load_iris().data #返回数据集所有输入特征
y_data = datasets.load_iris().target #返回数据集 标签
# print(x_data)
# print(y_data)
# x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# pd.set_option('display.unicode.east_asian_width', True)
# print('x_data add index: \n', x_data)
#
# x_data['类别'] = y_data
# print('x_data add a column: \n', x_data)

# 准备数据 数据集乱序 生成训练集和测试机 配成
# 搭建网络 参数优化 测试结果 acc /loss 查看效果 可视化

# 数据集乱序
np.random.seed(116) #使用相同的seed 使输入特征标签 一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
# 数据集分出永不相见的训练集 和 测试机
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
#转换x的数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

#配成 输入特征，标签 对，每次 喂入一小搓 batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#定义 神经网络中所有可训练参数

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
#嵌套循环 迭代 更新参数 显示 当前loss

lr = 0.1
epoch = 300
train_loss_results =[] #记录 损失
test_acc =[] #测试准确度 记录
loss_all = 0 #每轮分成四个 step 记录四个和
#训练部分
for epoch in range(epoch):
    for step,(x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape: #with 结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1 #神经网络乘加计算
            y = tf.nn.softmax(y) #符合概率分布
            y_ = tf.one_hot(y_train, depth=3) #独热码方式
            loss = tf.reduce_mean(tf.square(y_ - y)) #采用均方误差方式 求损失函数
            loss_all +=loss.numpy() # 将每个step 计算出的loss 累加，为后续loss 提供数据
            # 计算总loss
        grads = tape.gradient(loss, [w1, b1])
        #实现梯度更新 w1 = w1 -lr * w1_grad b = b - lr* b_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
    print('epoch:{}, loss:{}'.format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0

# 计算当前参数 传播后的准确度，显示当前acc
    total_correct = 0 #预测对的样本个数
    total_number = 0 #总样本数
    for x_test, y_test in test_db:
        #更新后的参数预测
        y = tf.matmul(x_test, w1) + b1 # y为 预测结果
        y = tf.nn.softmax(y) #符合概率分布
        pred = tf.argmax(y, axis=1) #返回 y 中最大索引值， 预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype) #调整 数据类型和 标签一直

        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct) #讲每个 batch corrent 加起来
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct/total_number
    test_acc.append(acc)
    print('test_acc', acc)
#绘制 loss 曲线
plt.title('loss function curve')
plt.xlabel('epoch')
plt.ylabel('loss')
# 啄点画出
plt.plot(train_loss_results, label= '$Loss$')
plt.legend()
plt.show()

#acc/loss
plt.title('acc curve') # 图片标签
plt.xlabel('epoch') #x 轴名称
plt.ylabel('acc') #y 轴名称
plt.plot(test_acc, label="$Accuracy$") #桌垫 画出
plt.legend()
plt.show()