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
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

'''
ResNet18: nb_blocks = [2,2,2,2], wide = 2, nottleneck = False
ResNet34: nb_blocks = [3,4,6,3], wide = 2, nottleneck = False
ResNet50: nb_blocks = [3,4,6,3], wide = 2, nottleneck = True
ResNet101: nb_blocks = [3,4,23,3], wide = 2, nottleneck = True
ResNet152: nb_blocks = [3,8,36,3], wide = 2, nottleneck = True
WideResNet: nb_blocks = [3,3,3], wide >= 3, nottleneck = False
'''

def resnet(nb_blocks = [3,4,6,3], wide = 2, bottleneck = False):
  input = Input(shape=(128, 128, 3), dtype=tf.float32)
  X = input
  n_filter = 64
  X = Conv2D(n_filter, (3,3),  padding="same",kernel_initializer='he_normal')(X)

  if bottleneck == False:

    for i, repete in enumerate(nb_blocks):
      for j in range(repete):
        ''' BN - Conv - BN - ReLu - Conv - BN
      Han et al. arXiv:1610.02915'''

        shortcut = X
        if i>0 and j == 0:
          shortcut =  Conv2D(n_filter, (1, 1), strides=(2, 2),
                            kernel_initializer='he_normal')(shortcut)
          X = BatchNormalization()(X)
          X = Conv2D(n_filter, (3,3), strides= (2,2), padding="same", kernel_initializer='he_normal')(X)
          X = BatchNormalization()(X)

        else:
          X = BatchNormalization()(X)
          X = Conv2D(n_filter, (3,3), padding="same", kernel_initializer='he_normal')(X)
          X = BatchNormalization()(X)

        X = Activation("relu")(X)
        X = Conv2D(n_filter, (3,3), padding="same",kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)

        # ショートカットとマージ
        X = Add()([X, shortcut])
      n_filter *= wide

  if bottleneck == True:
    for i, repete in enumerate(nb_blocks):
      for j in range(repete):
        ''' BN - Conv(1,1) - BN - ReLu - Conv(3,3) - BN - ReLu - Conv(1,1) - BN
        Han et al. arXiv:1610.02915'''

        shortcut = X
        if i==0 and j ==0:
          shortcut =  Conv2D(n_filter * 4, (1, 1), kernel_initializer='he_normal')(shortcut)

        if i>0 and j == 0:
          shortcut =  Conv2D(n_filter * 4, (1, 1), strides=(2, 2),
                            kernel_initializer='he_normal')(shortcut)
          X = BatchNormalization()(X)
          X = Conv2D(n_filter, (1,1), strides= (2,2), padding="same", kernel_initializer='he_normal')(X)
          X = BatchNormalization()(X)
        else:
          X = BatchNormalization()(X)
          X = Conv2D(n_filter, (1,1), padding="same", kernel_initializer='he_normal')(X)
          X = BatchNormalization()(X)


        X = Activation("relu")(X)
        X = Conv2D(n_filter, (3,3), padding="same",kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        X = Conv2D(n_filter * 4, (1,1), padding="same",kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)

        # ショートカットとマージ
        X = Add()([X, shortcut])
      n_filter *= wide


  # 全結合
  X = Activation("relu")(X)
  X = GlobalAveragePooling2D()(X)
  X = Dropout(0.5)(X)
  y = Dense(6, activation="softmax")(X)
  # モデル
  model = Model(inputs=[input], outputs=[y])
  return model

#wide resudual network
model = resnet(nb_blocks = [3,3,3], wide = 4)

# モデルをコンパイル　学習係数は 0.003
model.compile( tf.train.AdamOptimizer(learning_rate=3e-3), loss="categorical_crossentropy",
              metrics=["acc"])


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
history = model.fit(X_train, y_train, batch_size=128,
                               epochs=epochs, validation_data=(X_test,Y_test))
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
