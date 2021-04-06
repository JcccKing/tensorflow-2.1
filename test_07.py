import tensorflow as tf
# 非连续性网络 可以用class
# class Mymodel(Model):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         定义网络结构快
#     def __call__(self, *args, **kwargs):
#         调用网络结构快，实现前向传播
#         return y
#
# model = Mymodel()
# class IrisModel(Mymodel):
#     def __init__(self):
#         super(IrisModel, self).__init__()
#         self.dl = Dense(3)
#     def __call__(self, *args, **kwargs):
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='sigmoid',
                         kernel_regularizer=tf.keras.regularizers.l2())
    def call(self, x):
        y = self.d1(x)
        return y
model = IrisModel()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=500,
          validation_split=0.2, validation_freq=20)
model.summary()
