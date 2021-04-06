#fashion 数据集 6w 28*28 像素点衣裤图片和标签 ，用于训练
# 1w 28*28 标签 用于测试
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import os
np.set_printoptions(threshold=np.inf)
Mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = Mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #给数据增加一个维度

# image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 1, 所有数据乘以该数值
#     rotation_range=15, 角度旋转
#     width_shift_range=.15,  宽度偏移量
#     height_shift_range=.15, 高度偏移量
#     horizontal_flip=True, 水平翻转
#     zoom_range=0.5) 随即缩放范围
#
# image_gen_train.fit(x_train)

# class FashionModel(Model):
#     def __init__(self):
#         super(FashionModel, self).__init__()
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(10, activation='softmax')
#
#     def call(self, x):
#         x = self.flatten(x)
#         x = self.d1(x)
#         y = self.d2(x)
#         return y
#
# model = FashionModel()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
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
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label= 'Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='validing loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

# 自制数据集
# from PIL import Image
# def generateds(path, txt):
#     f = open(txt, 'r')
#     contents = f.readlines()
#     f.close()
#     x, y_ =[], []
#     for content in contents:
#         value = content.split()
#         img_path = path + value[0]
#         img = Image.open(img_path)
#         img = np.array(img.convert('L'))
#         img =img / 255.
#         x.append(img)
#         y_.append(value[1])
#         print('loading :' + content)
#     x = np.array()
#     y_ = np.array(y_)
#     y_ = y_.astype(np.int64)
#     return x, y_

#数据增强 增大数据量
# image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale= 所有数据集乘以该数值,
#     rotation_range= 随机旋转角度范围,
#     width_shift_range= 随机宽度偏移量,
#     height_shift_range= 随机高度偏移量,
#     horizontal_flip= 是狗随机水平翻转,
#     zoom_range= 随机缩放范围 [1-n,1+n] )
# image_gen_train.fit(x_train)
# 例如：
# image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale= 1. /1. ,
#     rotation_range= 45 ,
#     width_shift_range= .15,
#     height_shift_range= .15,
#     horizontal_flip= False,
#     zoom_range=0.5
# )
# image_gen_train.fit(x_train)

# 读取保存模型
# load_weight(路径文件名)
# checkpoint_save_path = "./checkpoint/mnist.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model ----------')
#     model.load_weights(checkpoint_save_path)
# 保存模型：
# tf.keras.callbacks.ModelCheckpoint(
#     filepath=路径文件名,
#     save_best_only= True/False,
#     save_best_only= True/False
# )
# history = model.fit( callbacks=[cp_callback])
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)
# history = model.fit(x_train,y_train, batch_size=32,epochs=5,
#                     validation_data=(x_test,y_test),validation_freq=1,
#                     callbacks=[cp_callback])

# 提取可训练参数
# model.trainable_variables 返回模型可训练参数
# 设置输出格式
# np.set_printoptions(threshold=超过多少省略显示)
# print(model.trainable_variables)
# file = open('./weights.txt','w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) +'\n')
# file.close()
# acc曲线 和 loss 曲线
# history = model.fit(训练集数据,训练集标签, batch_size=,epochs=,validation_data=(测试数据),
#                     validation_split=用作测试的比例,validation_freq=测试频率)
# history 训练集 loss loss
# 测试机loss val_loss
# 训练集准确率 sparse_categorical_accuracy
# 测试集准确率 val_sparse_categorical_accuracy
# acc = history.history['sparse_categorical_accuracy']
# val_acc =history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']