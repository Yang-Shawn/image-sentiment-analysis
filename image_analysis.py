
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
import scipy

from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

img_rows, img_cols = 100, 100
img_channels = 1

path0 = "images/neg"
path1 = "images/pos"
path2 = "images_resized"

listing0 = os.listdir(path0) 
listing1 = os.listdir(path1) 
listing =listing0 + listing1
num_samples=size(listing)
print("图片数据集大小：",num_samples)


import math

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

number=500
list=[[],[]] * number #困难之处！！！！！！！！创建一个以二维数组为元素的一维数组来存储矩阵
k=0
#图像数据预处理
for file in listing0:
    imgl = [[0 for row in range(img_rows)] for col in range(img_cols)]
    im = Image.open(path0 + '//' + file)   
    img = im.resize((img_rows,img_cols))#将每张图片调整为300*300尺寸
    img_array=img.load()
    for i in range(img_rows):
      for j in range (img_cols):
          r, g, b = img_array[i,j]
          h,s,v=rgb2hsv(r, g, b)
          if h>300 and h<=360 or h>0 and h<=25:
            h=0
          elif h>25 and h<=41:
            h=1
          elif h>41 and h<=75:
            h=2
          elif h>75 and h<=156:
            h=3
          elif h>156 and h<=201:
            h=4
          elif h>201 and h<=272:
            h=5
          elif h>272 and h<=285:
            h=6
          elif h>285 and h<=330:
            h=7
          if s>0.1 and s<0.65:
            s=0
          elif s>=0.65 and s<=1:
            s=1
          v=0
          imgl[i][j] = 2*h+s+v
    list[k]=imgl
    k +=1
    print("k:",k)
print(list[1])

for file in listing1:
    imgl = [[0 for row in range(img_rows)] for col in range(img_cols)]
    im = Image.open(path1 + '//' + file)   
    img = im.resize((img_rows,img_cols))#将每张图片调整为300*300尺寸
    img_array=img.load()
    for i in range(img_rows):
      for j in range (img_cols):
          r, g, b = img_array[i,j]
          h,s,v=rgb2hsv(r, g, b)
          if h>300 and h<=360 or h>0 and h<=25:
            h=0
          elif h>25 and h<=41:
            h=1
          elif h>41 and h<=75:
            h=2
          elif h>75 and h<=156:
            h=3
          elif h>156 and h<=201:
            h=4
          elif h>201 and h<=272:
            h=5
          elif h>272 and h<=285:
            h=6
          elif h>285 and h<=330:
            h=7
          if s>0.1 and s<0.65:
            s=0
          elif s>=0.65 and s<=1:
            s=1
          v=0
          imgl[i][j] = 2 * h + s + v
    list[k]=imgl
    #print("list:::",list[k][0])
    k +=1
    print("k:",k)

imnbr = len(list)
print("len(list)：",imnbr)

immatrix = array([array(list[i]).flatten() for i in range(500)],'f')
                
label=np.ones((num_samples,),dtype = int)#初始化全部标注为1
label[0:250]=0
label[250:]=1
data,Label = shuffle(immatrix,label, random_state=2)#随机排序
train_data = [data,Label]


print (train_data[0].shape)
print (train_data[1].shape)


batch_size = 4# 每次训练和梯度更新块的大小。
nb_classes = 2 # 共有两种类型，积极，消极
nb_epoch = 20 # 迭代次数。

nb_filters = 4
nb_pool = 2
nb_conv = 3

(X, y) = (train_data[0],train_data[1])

#划分90%为训练集和10%为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

#类型转换
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#对数据进行归一化到0-1 因为图像数据最大是15
X_train /= 15
X_test /= 15


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Y_train共nb_classes=2个类别，keras要求格式为binary class matrices,
#转化一下，直接调用keras提供的这个函数
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#i = 256
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])

model = Sequential()#建立模型

#第一个卷积层，2个卷积核，每个卷积核大小3*3。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)

#第二个卷积层，32个卷积核，每个卷积核大小3*3。
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)

#第三个卷积层，32个卷积核，每个卷积核大小3*3。
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout3 = Activation('relu')
model.add(convout3)

#采用maxpooling，poolsize为(2,2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

#按概率来将x中的一些元素值置零，并将其他的值放大。
#用于进行dropout操作，一定程度上可以防止过拟合 
#x是一个张量，而keep_prob是一个[0,1]之间的值。
#x中的各个元素清零的概率互相独立，为1-keep_prob,
#而没有清零的元素，则会统一乘以1/keep_prob, 
#目的是为了保持x的整体期望值不变。
model.add(Dropout(0.5))

#全连接层，先将前一层输出的二维特征图flatten为一维的,压扁平准备全连接。
model.add(Flatten())
model.add(Dense(512))#添加512节点的全连接
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))#添加输出3个节点
model.add(Activation('softmax'))#采用softmax激活
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#用于训练一个固定迭代次数的模型
#返回：记录字典，包括每一次迭代的训练误差率和验证误差率；
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_test, Y_test))

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_split=0.2)


#展示模型在验证数据上的效果
#返回：误差率或者是(误差率，准确率)元组（if show_accuracy=True）
score = model.evaluate(X_test, Y_test,  verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
  

target_names = ['class 0(Negative)', 'class 1(Positive)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

