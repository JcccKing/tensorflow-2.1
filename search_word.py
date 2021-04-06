# 1.若干层特征提取
# 卷积 计算 认为是一种有效提取特征方法
# 1.卷积计算过程 2.感受野 3.全零填充 4.TF 描述卷积计算层 5.批便准化 6.池化 7.舍弃 8.卷积神经网络
# cifar10数据集 卷积神经网络搭建示例 实现leNet alexNet vggnet inceptionnet resnet
# 感受野 ：卷积神经网络各输出特征图中的每个像素点，在原始输入图片上映射区域大小。
# 全0填充 padding same 输入特征图边长/卷积核步长 / \
# valid 输入特征图边长 - 卷积核长 +1 / 步长
# 卷积是 特征提取器， CBAPD 卷积 批便准化 激活 池化 舍弃
# 两个3 * 3 卷积核 在 输入特征宽度、高为x 卷积计算步长为1
# 参数量： 9 + 9 = 18 计算量 18 x^2 - 108x +180
# 9 *(x-2)*(x-2) + 9*((x-2)-2)*((x-2) -2)
# import tensorflow as tf
# model = tf.keras.models.Sequential([
#     Conv2D(6,5,padding='valid', activation ='sigmoid'),
#     MaxPool2D(2, 2)
#     Conv2D(6,(5,5), padding ="valid", activation ="sigmoid")
#     MaxPool2D(2,(2,2)),
#     Conv2D(filter=6, kernel_size=(5,5),padding='valid', activation  ='sigmoid'),
#     MalPool2D(pool_size=(2,2), strides = 2),
#     Flatten(),
#     Dense(10, activation ='softmax')
# ]
# )
#
# # tf.keras.layers.Conv2D(
# #     filters = 卷积核个数,
# #     kernel_size= ()卷积核尺寸, 写核高h，核宽w
# #     strides = 滑动步长, 横纵向相同写 步长整数，（纵向步长h，横向步长w）默认 1
# #     padding ='same' or'valid', 使用全零填充是 same 不使用valid
# #     activation ='relu' or 'sigmoid' or 'tanh' or 'softmax' # 如有BN 此处可以不写
# #     input_shape = (高，宽，通道数) # 输入特征图维度，可省略
# # )
# 批标准化 使数据 0 -1 之间 的 标准差分布
# 批标准化 ： 对一小批数据 做标准化处理
# 批标准化后，第k个卷积核 的输出特征图，中第i个像素点
# Bn 再卷积后 再激活前
# tf.keras.models.Sequential([
#     Convn2D(filters = 6, kernel_size=(5,5), padding ='same'),
#     BatchNormalization(),#BN层
#     Activation('relu'),
#     MaxPool2D(Pool_size=(2,2), strides =2, padding='same'),
#     Dropout(0.2),
# ])
# 池化 用于减少特征数据量，最大值池化可提取图片纹理，均值池化可保留背景特征
# tf.keras.layers.MaxPool2D(
#     pool_size= 池化核尺寸，#正方形写核长整数，或（核高h，核宽w）
#     strides = 池化步长，#步长整数，或（纵向步长h， 横向步长w） 默认pool——size
#     padding ='valid or same' #使用全0填充 same 不使用valid
# )
# tf.keras.layers.AveragePooling2D(
#     pool_size= 池化核尺寸， 正方形写核长整数，或（核高h，核宽w）
#     stride是= 池化步长， 步长整数，或（纵向步长h，横向步长w），默认为pool——size
#     padding = valid or same 使用全0填充是same 不使用是valid
#
# )
# model = not tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5 ) ,padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2,padding='same'),
#     tf.keras.layers.Dropout(0.2)
# ])