import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
input_word ='abcdefghijklmnopqrstuvwxyz'
w_to_id ={
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16,
    'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
training_set_scaled =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
x_train =[]
y_train =[]
for i in range(4, 26):
    x_train.append(training_set_scaled[i-4:i])
    y_train.append(training_set_scaled[i])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(26, 2),
    tf.keras.layers.SimpleRNN(10),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path ='./checkpoint/rnn_embedding_4.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------load the model------')
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
    alphabet1 = input('输入的字母是：')
    alphabet = [w_to_id[a]  for a in alphabet1]
    # 验证效果送入一个样本，送入样本数1，输出4个字母出结果，循环核时间展开步数为4
    alphabet = np.reshape(alphabet, (1, 4))
    res = model.predict([alphabet])
    pred = tf.argmax(res, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])