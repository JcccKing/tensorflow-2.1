import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
# mnist 数据集 28 * 28 个元素值 0 是黑色 255是白色
from matplotlib import pyplot as plt
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 0-1 之间的值
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), #输入数据集 拉直 784个 元素
    tf.keras.layers.Dense(128, activation='relu'), # 第一层网络 128个神经元， relu 激活函数
    tf.keras.layers.Dense(10, activation='softmax') #第二层输出曾 softmax 使 输出成概率分布
])
# class MnistModel(Model):
#     def __init__(self):
#         super(MnistModel, self).__init__()
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(10, activation='softmax')
#     def call(self, x):
#         x = self.flatten(x)
#         x = self.d1(x)
#         y = self.d2(x)
#         return y
#
# model = MnistModel()
#配置训练方法
model.compile(optimizer='adam', #
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # 不满足组概率分布 写 true
              metrics=['sparse_categorical_accuracy']) #输出标签 使 数值

checkpoin_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoin_save_path +'.index'):
    print('-------------load the model ----------')
    model.load_weights(checkpoin_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoin_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

# model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5,
#           validation_data=(x_test, y_test), validation_freq=1)
history = model.fit(x_train, y_train, batch_size=32, epochs=5,
          validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])
model.summary()

# #可视化训练集输入特征的第一个元素
# plt.imshow(x_train[0], cmap='gray') #绘制灰度图
# plt.show()
#
# #p 打印出 训练姐第一个元素
# print('x_train[0] \n', x_train[0]) # 28 *28
# print('y_train[0]  \n', y_train[0])
#
# # 打印形状
# print('x_train,shape \n', x_train.shape)
# print('y_train.shape \n', y_train.shape)
# print('x_test,shape \n', x_test.shape)
# print('y_test.shape \n', y_test.shape)