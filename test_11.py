import tensorflow as tf
from tensorflow.keras import Model
import os
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
# 搭建神经网络 5*5的卷积核 过 2*2 池化和 步长2
# 过 128 全连接 一层 10 全连接
# 加载数据集
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
# 卷积模型
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.f2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

model = Baseline()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 打断点 判断有没有 缓存模型
checkpoint_save_path = './checkpoint/cifar10.ckpt'
if os.path.exists(checkpoint_save_path +'.index'):
    print('-----------load the model-------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
          validation_freq=1, callbacks=[cp_callback])
model.summary()
# 写入 模型的 参数
# file = open('./weights_cifar10.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
# 显示训练曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(1, 2 ,2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='validation loss')
plt.legend()
plt.show()
