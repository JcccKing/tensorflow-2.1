# 复现 AlexNet
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from matplotlib import pyplot as plt
import os
np.set_printoptions(threshold=np.inf)

dataset = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
# 卷积 96 *3 *3 步长 1 valid 标准 relu 最大池化 2* 2 步长 2
# 卷积 256 *3 *3 步长 1 valid relu 池化 2*2 步长 2
# 卷积 383 *3 *3 步长 1 same relu  *2
# 卷积 256 *3 *3 步长1 same relu
# 拉直
# 全连接 2048 relu dropout 0.5 *2
# 全连接 10 softmax

class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid')
        self.b2 = tf.keras.layers.BatchNormalization()
        self.a2 = tf.keras.layers.Activation('relu')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c3 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.c4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.p3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d1 = tf.keras.layers.Dropout(0.5)
        self.f2 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.5)
        self.f3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)

        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y

model = AlexNet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/AlexNet.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model---------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                             save_weights_only=True,
                                            save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])
model.summary()

file = open('./weights_AlexNet.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='valid accuracy')
plt.title('training and validation accuracy ')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='valid loss')
plt.title('training and validation loss ')
plt.legend()
plt.show()