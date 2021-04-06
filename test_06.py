import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
# tf.keras 搭建神经八股
# import； train ， test ; model = tf.keras.models.Sequential;
# model.compile()  , model.fit(), model.summary()

#tf.keras.models.Sequential 视为容器
# tf.keras.layers.Flatten() 拉直层 吧输入层 拉直 变成一位数组
#
# 全连接层
# tf.keras.layers.Dense(神经元个数,activation="激活函数",kernel_regularizer=哪种正策划)
# activation = relu,softmax,sigmod,tanh,
# kernel_regularizer = tf.keras.regularizers.l1(), tf.keras.regularizers.l2()
# 卷基层： tf.keras.layers.Conv2D(filters= 卷积核个数, kernel_size = 卷积核尺寸, strides= 卷积步长,
#                             padding=valid /  same)
# LSTM层 tf.keras.layers.LSTM()
# model.compile(optimizer =优化器, loss = 损失函数 , metrics = ['准确率'])
# optimizer 可选：
# 'sgd' tf.keras.optimizers.SGD(lr= , momentum= 动量参数)
# 'adagrad' tf.keras.optimizers.Adagrad(lr=)
# 'adadelta' tf.keras.optimizers.Adadelta(lr=)
# 'adam' tf.keras.optimizers.Adam(lr= , beta_1=0.9 , beta_2= 0.999)
# loss 可选
# 'mse' tf.keras.losses.MeanSquaredError()
# 'spare_categorical_crossentropy' tf.keras.losses.SparseCategoricalCrossentropy(bool=, reduction=,)
# metrics 可选
# 'accuracy' y_ - y
# 'categorical_accuracy' 读热码 概率分布
# 'sparse_categorical_accuracy' ：
# model.fit(训练集的输入特征， 训练集的标签， batch_size = ,epochs =
# validation_data =(测试集的输入特征， 测试集的标签)
# validation_split =从训练集划分多比例测试集
# validation_freq =多少次epoch测试一次
# model.summary() 打印

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer= tf.keras.regularizers.l2())
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=500,
          validation_split=0.2, validation_freq=20)
model.summary()

