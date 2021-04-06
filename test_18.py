import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

input_word='abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_to_onehot = {0: [1., 0., 0., 0., 0.],
                1: [0., 1., 0., 0., 0.],
                2: [0., 0., 1., 0., 0.],
                3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}
x_train =[ [id_to_onehot[w_to_id['a']],id_to_onehot[w_to_id['b']],id_to_onehot[w_to_id['c']],id_to_onehot[w_to_id['d']]],
           [id_to_onehot[w_to_id['b']],id_to_onehot[w_to_id['c']],id_to_onehot[w_to_id['d']],id_to_onehot[w_to_id['e']]],
           [id_to_onehot[w_to_id['c']],id_to_onehot[w_to_id['d']],id_to_onehot[w_to_id['e']],id_to_onehot[w_to_id['a']]],
           [id_to_onehot[w_to_id['d']],id_to_onehot[w_to_id['e']],id_to_onehot[w_to_id['a']],id_to_onehot[w_to_id['b']]],
           [id_to_onehot[w_to_id['e']],id_to_onehot[w_to_id['a']],id_to_onehot[w_to_id['b']],id_to_onehot[w_to_id['c']]]
           ]
y_train = [w_to_id['e'],w_to_id['a'], w_to_id['b'], w_to_id['c'], w_to_id['d']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 4, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/one_hot_abcde2.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------load the model ----')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 monitor='loss') #没有测试集 根据 最优loss 保存模型
history = model.fit(x_train, y_train, batch_size=32, epochs=10,callbacks=[cp_callback])
model.summary()

file = open('./weights_onehot_abcde2.txt', 'w')
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
    alphabet = [id_to_onehot[w_to_id[a]] for a in alphabet1]
    alphabet = np.reshape(alphabet, (1, 4, 5))
    res = model.predict([alphabet])
    pred = tf.argmax(res, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])