# 复现 resnet
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPool2D
import os
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
dataset = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
# LeNet  卷积网络开篇之作 共享卷积核，减少网络参数
# AlexNet 使用激活函数Relu 提升训练速度 使用Dropout 缓解 过拟合
# VggNet 小尺寸卷积核减少参数， 网络结构规整，适合并行加速
# Inception一层内使用不同尺寸卷积核，提升感知力，使用批标准化，环节梯度消失
# ResNet 层间残差挑连，引入前方信息，环节模型退化，使卷积网络层数加深

class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        # 为 true 对输入进行采样， 就是 1*1 卷积核做卷积操作，保证x 和fx 维度相同 顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                                  padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs # residual 等于输入值本身
        # 讲 输入通过卷积 bn层 激活层 计算f(x)

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)
        # 拼接
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        out = self.a2(y + residual)
        return out

class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, kernel_size=(3, 3), strides=1, padding='same',
                         use_bias=False,)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet 网络结构
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]): # 第几个卷积层
                if block_id != 0 and layer_id == 0: # 对除去第一个块  外的block 输入进行采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)

                self.blocks.add(block)
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = ResNet18([2, 2, 2, 2])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path ='./checkpoint/ResNet18.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model -----------')
    model.load_weights(checkpoint_save_path)

cpcallpacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cpcallpacks])
model.summary()