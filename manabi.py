import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob


TRAIN_NUM = 75


def main():
    names = ['human', 'murai']
    image_size = 50

    X_train = []
    y_train = []
    for index, name in enumerate(names):
        dir = "./data/divide/" + name + "/a"
        files = glob.glob(dir + "/*.jpg")
        for i, file in enumerate(files[:TRAIN_NUM]):
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X_train.append(data)
            y_train.append(index)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for index, name in enumerate(names):
        dir = "./data/divide/" + name + "/b"
        files = glob.glob(dir + "/*.jpg")
        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X_test.append(data)
            y_test.append(index)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32')
    X_train = X_train / 255.0

    X_test = X_test.astype('float32')
    X_test = X_test / 255.0

    # 正解ラベルの形式を変換
    y_train = np_utils.to_categorical(y_train, 2)
    # 正解ラベルの形式を変換
    y_test = np_utils.to_categorical(y_test, 2)

    # CNNを構築
    #model = Sequential()

    # model.add(Conv2D(32, (3, 3), padding='same',
    #          activation='relu'))
    # model.add(Activation('relu'))
    #model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    #model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    #model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2))
    # model.add(Activation('softmax'))

    # コンパイル
    # model.compile(loss='categorical_crossentropy',
    #              optimizer='SGD', metrics=['accuracy'])
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    # 分類したい人数を入れる
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # コンパイル
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100)
    #print(model.evaluate(X_test, y_test))
    model.save("./data/model/model.h5")


if __name__ == "__main__":
    main()
