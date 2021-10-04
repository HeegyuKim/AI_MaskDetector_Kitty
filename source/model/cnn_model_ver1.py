from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


from tensorflow import keras
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import os
import shutil
from keras.preprocessing import image
import cv2

print(tf.__version__)

img_withmask_dir = './train/with_mask'
img_witouthmask_dir = './train/without_mask'

def_target_size = 64
 

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(def_target_size, def_target_size))

    #region test
    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.show()    
    #print(img_path)
    #endregion

    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    rgb_tensor /= 255.
    return rgb_tensor
    


categories = ['mask','nomask']
nb_class=len(categories)

x = []
y = []

for i in os.listdir(img_withmask_dir):
    img_path = os.path.join(img_withmask_dir, i)
    img_tensor = preprocess_img(img_path)
    x.append(img_tensor)
    y.append(0)
    
for i in os.listdir(img_witouthmask_dir):
    img_path = os.path.join(img_witouthmask_dir, i)
    img_tensor = preprocess_img(img_path)
    x.append(img_tensor)
    y.append(1)

x = np.array(x)
print(x.shape) 
y = np.array(y)
print(y.shape) 


X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size=0.1)

Y_train = keras.utils.to_categorical(Y_train, 2)

print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)

print('X_test shape : ', X_test.shape)
print('Y_test shape : ', Y_test.shape)

#region 내사진 테스트
# img_test_me_dir = './AI_Mask_Detector/train/test_me'
# test_x = []
# for i in os.listdir(img_test_me_dir):
#     img_path = os.path.join(img_test_me_dir, i)
#     img_tensor = preprocess_img(img_path)
#     test_x.append(img_tensor)
#
# X_test = np.array(test_x)
# print('X_test2 shape : ', X_test.shape)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(def_target_size, def_target_size, 3)),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(2, activation='softmax')
# ])
#endregion


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(def_target_size, def_target_size, 3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_split=0.1)              

model.save('./AI_MASK_DETECTOR/model.h5')

# 예측
predictions = model.predict(X_test)

#test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
#print('\n테스트 정확도:', test_acc)


# 이미지 시각화 에러처리
roofCnt = 8*10
if len(X_test) < roofCnt:
    roofCnt = len(X_test)

# #이미지 시각화
plt.figure(figsize=(15,15))
for i in range(roofCnt):
    plt.subplot(8,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)

    if predictions[i][0] > predictions[i][1]:
        label = 'Mask ' + str(int(predictions[i][0] * 100))
    else:
        label = 'No Mask ' + str(int(predictions[i][1] * 100))

    plt.xlabel(label) 
    # plt.xlabel(categories[Y_train[i]])
plt.show()
