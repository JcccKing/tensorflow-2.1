# 复现 精简版本 inception
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

class ConvBnRelu(Model):
    def __init__(self,ch, kernelsz=3, strides=1, padding ='same'):
        super(ConvBnRelu, self).__init__()
        self.model = tf.keras.Sequential([
            Conv2D(ch,kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])
    def call(self, x):
        x = self.model(x)
        return x

class Inceptionblock(Model):
    def __init__(self, ch, strides = 1):
        super(Inceptionblock, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBnRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBnRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBnRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBnRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBnRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3,strides=1,padding='same')
        self.c4_2 = ConvBnRelu(ch, kernelsz=1, strides=strides)
    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        x = tf.concat([x1,x2_2, x3_2, x4_2], axis=3)
        return x

class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16):
        super(Inception10, self).__init__()
        self.in_channels =init_ch
        self.out_channels =init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBnRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()

        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id ==0:
                    block = Inceptionblock(self.out_channels, strides=2)
                else:
                    block = Inceptionblock(self.out_channels, strides=1)
                self.blocks.add(block)

            self.out_channels *=2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = Inception10(num_blocks=2,num_classes=10)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path ='./checkpoint/Inception.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model -----------')
    model.load_weights(checkpoint_save_path)

cpcallpacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cpcallpacks])
model.summary()