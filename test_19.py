import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# Embedding 一种编码方法 用于测试 英文 用低维向量
# tf.keras.layers.Embedding(词汇大小， 编码维度)
# 输入Embedding x_train维度 【送入样本数， 循环核维度】

input_word ='abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

x_train = [w_to_id['a'], w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e']]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5, 2),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/Embedding_abced.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model-------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()

file = open('./weights_Embedding.txt', 'w')
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
    alphabet = [w_to_id[a]]
    alphabet = np.reshape(alphabet, (1, 1))
    res = model.predict([alphabet])
    pred = tf.argmax(res, axis=1)
    pred = int(pred)
    tf.print(a + '->' + input_word[pred])