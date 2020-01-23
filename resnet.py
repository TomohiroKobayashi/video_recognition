import sys
import argparse
from PIL import Image
"""
#yoloを呼び出すためにパスを追加
sys.path.append("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/研究用コード/keras-yolo3")
from yolo import YOLO, detect_video


def detect_img(yolo,img):
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        #continueはwhile文で無限ループしてる時だけ
        #continue
    else:
        r_image = yolo.detect_image(image)
        #画像を保存
        r_image.save('test1_done.png')
        #r_image.show()
    yolo.close_session()
"""

#以下はデータオーグメンテーションの関数
def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image
def vertical_flip(image):
    image = image[::-1, :, :]
    return image

from scipy.ndimage.interpolation import rotate
"""
from scipy.misc import imresize
def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = imresize(image, (h, w))
    return image
"""

def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = Image.fromarray(image)
    image = np.asarray(image.resize((h,w)))
    return image

#独自データで学習
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from livelossplot import PlotLossesKeras
import random
#順に立位・歩き、立位・停止、立位・採食。立位・その場足踏み、伏臥位、横臥位
#folder = ["11","12","14","17","21","31"]
folder = ["11","21","31"]
image_size = 128

X = []
Y = []
for index, name in enumerate(folder):
    print(name+ ":" + str(index))
    dir = "train_data_crop/" + name
    files = glob.glob(dir + "/*.jpg")

    #ランダムに950個取得する場合
    #l = list(np.arange(len(files)))
    #rnd_list =  random.sample(l, 950)
    for i in range(len(files)):
        image = Image.open(files[i]).convert("RGB")
        #image = Image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        if name == "11" or name=="12" or name=="14"or name=="17":
            Y.append(0)
        elif name == "21":
            Y.append(1)
        elif name=="31":
            Y.append(2)
        #Y.append(index)

#タークオイズも学習データに追加
for index, name in enumerate(folder):
    print(name+ ":" + str(index))
    dir = "tarcro_train/" + name
    files = glob.glob(dir + "/*.jpg")

    for i in range(len(files)):
        image = Image.open(files[i]).convert("RGB")
        #image = Image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        if name == "11" or name=="12" or name=="14"or name=="17":
            Y.append(0)
        elif name == "21":
            Y.append(1)
        elif name=="31":
            Y.append(2)

import collections
c = collections.Counter(Y)
print(c.most_common())

X = np.array(X)
Y = np.array(Y)

print("学習データ数は:"+str(len(Y)))
print("X_shape:"+str(X.shape))

X = X.astype('float32')
X = X / 255.0

# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, 3)

# 学習用データと検証用データ
X_train = X
y_train = Y

print("before:"+str(y_train[:10]))

#学習データのシャッフル
for l in [X_train, y_train]:
    np.random.seed(1)
    np.random.shuffle(l)
print("after:"+str(y_train[:10]))

#テストデータも同様に作成(タムロブライト)
X_test = []
Y_test = []
for index, name in enumerate(folder):
    print("test")
    print(name+ ":" + str(index))
    dir = "test_data/" + name
    files = glob.glob(dir + "/*.jpg")
    #アンダーサンプリングとして横臥位に合わせて950枚ずつとする
    count = 0
    for i, file in enumerate(files):
        image = Image.open(file).convert("RGB")
        #image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_test.append(data)
        #Y_test.append(index)
        if name == "11" or name=="12" or name=="14"or name=="17":
            Y_test.append(0)
        elif name == "21":
            Y_test.append(1)
        elif name=="31":
            Y_test.append(2)
        count += 1

import collections
c = collections.Counter(Y_test)
print(c.most_common())

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test.astype('float32')
X_test = X_test / 255.0

# 正解ラベルの形式を変換
Y_test = np_utils.to_categorical(Y_test, 3)

#X_test, X_val, y_test, y_val = train_test_split(X_test, Y_test, test_size=0.20)

#データジェネレータの作成
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=180,  # randomly rotate images in the range
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=True  # randomly flip images vertically
    )
datagen.fit(X_train)



#Resnet50の定義
#モデル作成・学習
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import tensorflow as tf
import keras
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D,Input
from keras.layers import Dense, Dropout, Flatten, Activation,GlobalAveragePooling2D,Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import optimizers


# ResNet50のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(image_size, image_size, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dropout(0.5))
#最終層の出力はクラス数によって変える
top_model.add(Dense(3, activation='softmax'))

# ResNet50とFC層を結合してモデルを作成
resnet50_model = Model(input=resnet50.input, output=top_model(resnet50.output))

"""
#ResNet50の一部の重みを固定
for layer in resnet50_model.layers[:100]:
    layer.trainable = False
"""

# 多クラス分類を指定
resnet50_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])
resnet50_model.summary()

epochs = 50

#学習の実行
"""
hist = resnet50_model.fit_generator(datagen.flow(X_train, y_train, batch_size=75),
                        epochs=35, validation_data=(X_test, y_test),
                        steps_per_epoch=X_train.shape[0])
"""

#history = resnet50_model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test,Y_test))
history = resnet50_model.fit_generator(datagen.flow(X_train,y_train,batch_size=128),epochs=epochs, validation_data=(X_test,Y_test))


json_string = resnet50_model.to_json()
open('test.json', 'w').write(json_string)
resnet50_model.save_weights('test.hdf5')

# evaluate
score = resnet50_model.evaluate(X_test, Y_test)
print("test loss", score[0])
print("test acc",  score[1])

import matplotlib.pyplot as plt
epochs = range(1, len(history.history['acc']) + 1)

fig1 = plt.figure()
plt.plot(epochs, history.history['loss'], label='Training loss', ls='-')
plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig1.savefig("loss.png")

fig2 = plt.figure()
plt.plot(epochs, history.history['acc'],  label='Training acc')
plt.plot(epochs, history.history['val_acc'], label="Validation acc")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
fig2.savefig("acc.png")
