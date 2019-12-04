import sys
import argparse
from PIL import Image

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
folder = ["11","12","14","17","21","31"]
image_size = 128

X = []
Y = []
for index, name in enumerate(folder):
    print(name+ ":" + str(index))
    dir = "annotation_snowflower/" + name
    files = glob.glob(dir + "/*.jpg")
    """
    #アンダーサンプリングとして横臥位に合わせて950枚ずつとする
    count = 0
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
        count += 1
        if count >= 950:
            break
    """
    #ランダムに950個取得する場合
    rnd_list =  random.sample(np.arange(len(files)), 950)
    for i in rnd_list:
        image = Image.open(files[i])
        image = Image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X = X / 255.0

# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, 6)

# 学習用データと検証用データ
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)

#テストデータも同様に作成(タムロブライト)
X_test = []
Y_test = []
for index, name in enumerate(folder):
    print("test")
    print(name+ ":" + str(index))
    dir = "annotation_tamrobraito/" + name
    files = glob.glob(dir + "/*.jpg")
    #アンダーサンプリングとして横臥位に合わせて950枚ずつとする
    count = 0
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_test.append(data)
        Y_test.append(index)
        count += 1
        if count >= 50:
            break

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test.astype('float32')
X_test = X_test / 255.0

# 正解ラベルの形式を変換
Y_test = np_utils.to_categorical(Y_test, 6)


# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))

epochs = 50

# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

#訓練
history = model.fit(X_train, y_train, batch_size=128, epochs=epochs,callbacks=[PlotLossesKeras()], validation_data=(X_val,y_val))

json_string = model.to_json()
open('test.json', 'w').write(json_string)
model.save_weights('test.hdf5')

# evaluate
score = model.evaluate(X_test, Y_test)
print("test loss", score[0])
print("test acc",  score[1])

"""
import matplotlib.pyplot as plt
x = range(epochs)
plt.plot(x, history.history['val_acc'], label="acc")
plt.title("accuracy")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.plot(x, history.history['loss'], label="loss")
plt.title("loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
"""

"""
FLAGS = None
if __name__ == '__main__':
    img = "1819502.jpg"
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--image', default="False", action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    FLAGS = parser.parse_args()
    detect_img(YOLO(**vars(FLAGS)),img)
    #detect_img(YOLO({"image":img}))
"""
