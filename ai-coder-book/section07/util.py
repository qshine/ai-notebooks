import numpy as np
import os

from skimage import transform

from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

def preprocess_image(image, size):
    '''
    对图片进行缩放
    '''
    img = transform.resize(image, (size, size))
    return img


def create_model(num_classes, img_size):
    model = Sequential()

    # 每个卷积层都用relu函数, 然后用dropout层选取一些节点不进行处理来提高速度
    model.add(Conv2D(32, (3, 3), padding='same', Activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', Activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    # 通过flattern层来变为一维向量输入
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # 第二层使用softmax来获取每个类别的概率
    model.add(Dense(num_classes, activation='softmax'))

    return model

