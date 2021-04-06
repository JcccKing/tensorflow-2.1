# 复现 LeNet
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
from tensorflow.keras import Model
dataset = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0
# 设计卷积
#     c ( 核 6 *5 * 5 , 步长1，填充valid)
#     b（NONE）
#     A（sigmod）
#     P（max 核 2*2 步长 2，填充valid）
#     d（None)
#     c ( 核 16 *5 * 5 , 步长1，填充valid)
#     b（NONE）
#     A（sigmod）
#     P（max 核 2*2 步长 2，填充valid）
#     d（None)
#     Flatten
#     Dense(神经元 120，激活sigmod)
#     dense（84 ，激活 sigmod）
#     dense(10, softmax)
#
class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid')
        #self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('sigmoid')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid')
        #self.d1 = tf.keras.layers.Dropout(0.2)

        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid')
        #self.b2 = tf.keras.layers.BatchNormalization()
        self.a2 = tf.keras.layers.Activation('sigmoid')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid')
        #self.d2 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(120, activation='sigmoid')
        self.f2 = tf.keras.layers.Dense(84, activation='sigmoid')
        self.f3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

model = LeNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/LeNet.ckpt'
if os.path.exists(checkpoint_save_path +'.index'):
    print('---------load the model----------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                   save_weights_only=True,
                                   save_best_only=True)
history = model.fit(x_train, y_train, batch_size=16, epochs=1, validation_data=(x_test, y_test),
          validation_freq=1, callbacks=[cp_callback])

model.summary()
file = open('./weights_LeNet.txt', 'w')
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
plt.plot(val_acc, label='testing accuracy')
plt.title('training and validing accuracy ')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='testing loss')
plt.title('training and validing loss ')
plt.legend()
plt.show()