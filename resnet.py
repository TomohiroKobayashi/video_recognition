import sys
import argparse
from PIL import Image


#以下はデータオーグメンテーションの関数
def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image
def vertical_flip(image):
    image = image[::-1, :, :]
    return image

from scipy.ndimage.interpolation import rotate

def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = Image.fromarray(image)
    image = np.asarray(image.resize((h,w)))
    return image
"""
# モデルの生成
def generate_model(input_shape, block_f, blocks, block_sets, block_layers=2, first_filters=32, kernel_size=(3,3)):
  inputs = Input(shape=input_shape)

  # 入力層
  x = Conv2D(filters=first_filters, kernel_size=kernel_size, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPool2D((2, 2))(x)
  x = Dropout(0.2)(x)

  # 畳み込み層
  for s in range(block_sets):
    filters =  first_filters * (2**s)

    for b in range(blocks):
      x = block_f(x, kernel_size, filters, block_layers)

    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)

  # 出力層
  x = Flatten()(x)
  x = Dropout(0.4)(x)
  x = Dense(100)(x)
  x = ReLU()(x)
  outputs = Dense(6, activation='softmax')(x)

  model = Model(input=inputs, output=outputs)

  return model

# shortcut connection無しのブロック
def plain_block(x, kernel_size, filters, layers):
  for l in range(layers):
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

  return x

# shortcut path有りのブロック (residual block)
def residual_block(x, kernel_size, filters, layers=2):
  shortcut_x = x

  for l in range(layers):
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    if l == layers-1:
      if K.int_shape(x) != K.int_shape(shortcut_x):
        shortcut_x = Conv2D(filters, (1, 1), padding='same')(shortcut_x)  # 1x1フィルタ

      x = Add()([x, shortcut_x])

    x = ReLU()(x)

  return x
"""
from keras.layers import Conv2D, Activation, BatchNormalization, Add, Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
import time
import pickle

# 経過時間用のコールバック
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)


class ResNet:
    def __init__(self, n, framework, channels_first=False, initial_lr=0.01, nb_epochs=100):
        self.n = n
        self.framework = framework
        # 論文通りの初期学習率=0.1だと発散するので0.01にする
        self.initial_lr = initial_lr
        self.nb_epochs = nb_epochs
        self.weight_decay = 0.0005
        # MX-Netではchannels_firstなのでその対応をする
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"
        self.bn_axis = 1 if channels_first else -1
        # Make model
        self.model = self.make_model()

    # オリジナルの論文に従って、サブサンプリングにPoolingではなくstride=2のConvを使う
    def subsumpling(self, output_channels, input_tensor):
        return Conv2D(output_channels, kernel_size=1, strides=(2,2), data_format=self.data_format, kernel_regularizer=l2(self.weight_decay))(input_tensor)

    # BN->ReLU->Conv->BN->ReLU->Conv をショートカットさせる(Kaimingらの研究による)
    # https://www.slideshare.net/KotaNagasato/resnet-82940994
    def block(self, channles, input_tensor):
        # ショートカット元
        shortcut = input_tensor
        # メイン側
        x = BatchNormalization(axis=self.bn_axis)(input_tensor)
        x = Activation("relu")(x)
        x = Conv2D(channles, kernel_size=3, padding="same", data_format=self.data_format, kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation("relu")(x)
        x = Conv2D(channles, kernel_size=3, padding="same", data_format=self.data_format, kernel_regularizer=l2(self.weight_decay))(x)
        # 結合
        return Add()([x, shortcut])

    def make_model(self):
        input = Input(shape=(3, 128, 128)) if self.channels_first else Input(shape=(128, 128, 3))
        # 3->16にチャンネル数を増やす
        x = Conv2D(16, kernel_size=3, padding="same", data_format=self.data_format, kernel_regularizer=l2(self.weight_decay))(input)
        # 32x32x16のブロックをn回
        for i in range(self.n):
            x = self.block(16, x)
        # 16x16x32
        x = self.subsumpling(32, x)
        for i in range(self.n):
            x = self.block(32, x)
        # 8x8x64
        x = self.subsumpling(64, x)
        for i in range(self.n):
            x = self.block(64, x)
        # Global Average Pooling
        x = GlobalAveragePooling2D(data_format=self.data_format)(x)
        x = Dense(6, activation="softmax")(x)
        # model
        model = Model(input, x)
        return model

    def lr_schduler(self, epoch):
        x = self.initial_lr
        if epoch >= self.nb_epochs * 0.5: x /= 10.0
        if epoch >= self.nb_epochs * 0.75: x /= 10.0
        return x

    def train(self, X_train, y_train, X_val, y_val):
        # コンパイル
        self.model.compile(optimizer=SGD(lr=self.initial_lr, momentum=0.9), loss="categorical_crossentropy", metrics=["acc"])
        # Data Augmentation
        traingen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=4./32,
            height_shift_range=4./32,
            horizontal_flip=True)
        valgen = ImageDataGenerator(
            rescale=1./255)
        # Callback
        time_cb = TimeHistory()
        lr_cb = LearningRateScheduler(self.lr_schduler)
        # Train
        history = self.model.fit_generator(traingen.flow(X_train, y_train, batch_size=128), epochs=self.nb_epochs,
                                           steps_per_epoch=len(X_train)/128, validation_data=valgen.flow(X_val, y_val),
                                           callbacks=[time_cb, lr_cb]).history
        history["time"] = time_cb.times
        # Save history
        file_name = f"{self.framework}_n{self.n}.dat"
        with open(file_name, "wb") as fp:
            pickle.dump(history, fp)


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
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, ReLU, Flatten, Dense, Add, Dropout

#順に立位・歩き、立位・停止、立位・採食。立位・その場足踏み、伏臥位、横臥位
folder = ["11","12","14","17","21","31"]
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
        Y.append(index)
        #data Augumentation
        if random.random() < 0.4:
            data = horizontal_flip(data)
            X.append(data)
            Y.append(index)
        if random.random() < 0.4:
            data = vertical_flip(data)
            X.append(data)
            Y.append(index)
        if random.random() < 0.4:
            data = random_rotation(data)
            X.append(data)
            Y.append(index)

X = np.array(X)
Y = np.array(Y)
print("X_shape:"+str(X.shape))

X = X.astype('float32')
X = X / 255.0

# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, 6)

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
        Y_test.append(index)
        count += 1

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test.astype('float32')
X_test = X_test / 255.0

# 正解ラベルの形式を変換
Y_test = np_utils.to_categorical(Y_test, 6)

#X_test, X_val, y_test, y_val = train_test_split(X_test, Y_test, test_size=0.20)
epochs = 50
#input_shape = X_train.shape[1:]
input_shape=(128,128,3)
"""
# residualモデル
residual_model  = generate_model(input_shape, residual_block, blocks=6, block_sets=2, first_filters=32)
residual_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

#訓練
history = residual_model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test,Y_test))
"""
batch_size=128

net = ResNet(3, "keras_tf", nb_epochs=epochs)
    # train
net.train(X_train, y_train, X_test, Y_test)

json_string = model.to_json()
open('test.json', 'w').write(json_string)
model.save_weights('test.hdf5')

# evaluate
score = model.evaluate(X_test, Y_test)
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
