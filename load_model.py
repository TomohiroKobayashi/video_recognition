import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from keras.models import model_from_json
import json
import scipy, os

keras_model = "test.json"
keras_param = "test.hdf5"

imsize = (128,128)

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

if __name__ == "__main__":
    pic = "106202.jpg"

    model = model_from_json(open(keras_model).read())
    model.load_weights(keras_param)
    model.summary()

    img = load_image(pic)

    prd = model.predict(np.array([img]))
    print(prd)
    prelabel = np.argmax(prd, axis=1)
