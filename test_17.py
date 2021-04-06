import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
# 循环核： 参数时间共享，循环层提取时间信息
# 循环神经网络， 借助循环核提取时间特征后，送入全连接神经网络
# tf.keras.layers.SimpleRNN(记忆体个数，activation=激活函数, 不写默认tanh
# return_sequences=是否每个时刻输出ht到下一层)
# RNN 送入时，维度【送入样本数， 循环核时间展开步数，每个事件步输入特征个数】
#记忆体个数  = 隐藏层神经元个数 = 隐状态的维度
#编码 abcde 读热码
input_word ='abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_to_onehot = {0: [1., 0., 0., 0., 0.],
                1: [0., 1., 0., 0., 0.],
                2: [0., 0., 1., 0., 0.],
                3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}
x_train = [  id_to_onehot[w_to_id['a']],
             id_to_onehot[w_to_id['b']],
             id_to_onehot[w_to_id['c']],
             id_to_onehot[w_to_id['d']],
             id_to_onehot[w_to_id['e']] ]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
#使 x——train 符合SimpleRNN 输入要求，[送入样本数， 循环核事件展开步数，每个时间布输入特征个数】
# 此处整个数据集送入
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax') # 全连接 相当于 输出层
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path ='./checkpoint/one_hot_abcde.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------load the model-------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss') # 由于fit没有输出测试集，不计算测试集准确度
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()


file = open('./weights_onehot_abcde.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='training accuracy')
plt.title('training accuracy ')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.title('training  loss ')
plt.legend()
plt.show()

Prenum = int(input('输出测试的字母个数：'))
for i in range(Prenum):
    a = input('输入的字母是：')
    alphabet = [id_to_onehot[w_to_id[a]]]
    alphabet = np.reshape(alphabet, (1, 1, 5))
    res = model.predict([alphabet])
    pred = tf.argmax(res, axis=1)
    pred = int(pred)
    tf.print(a + '->' + input_word[pred])