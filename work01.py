import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random as random
import numpy as np
import csv
# 假设x_data和y_data都有10笔，分别代表宝可梦进化前后的cp值
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
# 这里采用最简单的linear model：y_data=b+w*x_data
# 我们要用gradient descent把b和w找出来

def getGrad(b,w):
    b_grad =0
    w_grad =0
    for i in range(10):
        b_grad +=(-2.0) *(y_data[i] - (b + w * x_data[i]))
        w_grad +=(-2.0 * x_data[i]) * (y_data[i] - (b +w * x_data[i]))
    return (b_grad, w_grad)



x = np.arange(-200, -100, 1)  # bias
y = np.arange(-5, 5, 0.1)  # weight
Z = np.zeros((len(x), len(y)))  # color
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]

        # Z[j][i]存储的是loss
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - (b + w * x_data[n])) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# 由于我们是规定了iteration次数的，因此原因应该是learning
# rate不够大，这里把它放大10倍

# y_data = b + w * x_data
b = -120  # initial b
w = -4  # initial w
lr = 1 # learning rate 放大10倍
iteration = 100000  # 这里直接规定了迭代次数，而不是一直运行到b_grad和w_grad都为0(事实证明这样做不太可行)

# store initial values for plotting，我们想要最终把数据描绘在图上，因此存储过程数据
b_history = [b]
w_history = [w]
lr_b = 0
lr_w = 0
# iterations
for i in range(iteration):
    # get new b_grad and w_grad
    b_grad, w_grad = getGrad(b, w)

    lr_b = lr_b + b_grad ** 2
    # 平方
    lr_w = lr_w + w_grad ** 2

    b -= lr/np.sqrt(lr_b) * b_grad
    w -= lr/np.sqrt(lr_w) * w_grad

    # # update b and w
    # b -= lr * b_grad
    # w -= lr * w_grad
    #
    # store parameters for plotting
    b_history.append(b)
    w_history.append(w)

print("the function will be y_data="+str(b)+"+"+str(w)+"*x_data")
error=0.0
for i in range(10):
    print("error "+str(i)+" is: "+str(np.abs(y_data[i]-(b+w*x_data[i])))+" ")
    error+=np.abs(y_data[i]-(b+w*x_data[i]))
average_error=error/10
print("the average error is "+str(average_error))

# plot the figure
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()





# data = pd.read_csv('train.csv', encoding='big5')
# data = data.iloc[:, 3:]
# data[data == 'NR'] = 0
# raw_data = data.to_numpy()
# month_data ={}
# for month in range(12):
#     sample = np.empty([18, 480])
#     for day in range(20):
#         sample[:, day* 24 :( day +1) * 24] = raw_data[18 *(20 * month +day ) :18 * (20 * month +day +1), :]
#     month_data[month] = sample
#
# x = np.empty([12 * 471, 18 * 9], dtype = float)
# y = np.empty([12 * 471,1], dtype= float)
# for month in range(12):
#     for day in range(20):
#         for hour in range(24):
#             if day == 19 and hour > 14:
#                 continue
#             x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day *24 + hour+9].reshape(1,-1)
#             y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day *24 + hour + 9]
# print('x----')
# print(x)
# print('y----')
# print(y)
# print('-----------------')
#
# mean_x = np.mean(x, axis =0)
# std_x = np.std(x, axis= 0)
# for i in range(len(x)):
#     for j in range(len(x[0])):
#         if std_x[j] !=0:
#             x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# print(x)
# print('----------------------')
# x_train_set = x[:math.floor(len(x) * 0.8), :]
# y_train_set = y[:math.floor(len(y) * 0.8), :]
# x_validation = x[math.floor(len(x) * 0.8):, :]
# y_validation = y[math.floor(len(x) * 0.8):, :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))
#
# dim = 18 * 9 +1
# w = np.zeros([dim, 1])
# x = np.concatenate((np.ones([12 * 471 ,1]), x), axis=1).astype(float)
# learning_rate = 100
# iter_time = 1000
# adagrad = np.zeros([dim, 1])
# eps = 0.000000001
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x,w) - y, 2)) /471 /12)
#     if t % 100 == 0 :
#         print( str(t) +"loss:" + str(loss))
#     gradient = 2 * np.dot(x.transpose(), np.dot(x,w) - y )
#     adagrad += gradient ** 2
#     w = w - learning_rate * gradient / np.sqrt(adagrad +eps)
# np.save('weight.py',w)
