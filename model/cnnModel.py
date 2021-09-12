import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.datasets.fashion_mnist import load_data
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras import models
#from tensorflow.keras.layers import Dense, Input
#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.utils import plot_model

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

def_target_size = 150

print(len(os.listdir(img_withmask_dir)))
print(os.listdir(img_withmask_dir)[:10])

def preprocess_img(img_path, target_size):
    img = image.load_img(img_path, target_size=(target_size, target_size))
    #img = image.load_img(img_path, grayscale=True, target_size=(target_size, target_size))

    img_tensor = image.img_to_array(img)

    #print(img_path)

    # scaling into [0, 1]
    img_tensor /= 255.
    return img_tensor


categories = ['mask','nomask']
nb_class=len(categories)

x = []
y = []

for i in os.listdir(img_withmask_dir):
    img_path = os.path.join(img_withmask_dir, i)
    img_tensor = preprocess_img(img_path, def_target_size)
    x.append(img_tensor)
    y.append(0)
    
for i in os.listdir(img_witouthmask_dir):
    img_path = os.path.join(img_witouthmask_dir, i)
    img_tensor = preprocess_img(img_path, def_target_size)
    x.append(img_tensor)
    y.append(1)

x = np.array(x)
print(x.shape) 
y = np.array(y)
print(y.shape) 


X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size=0.2)

print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)

print('X_test shape : ', X_test.shape)
print('Y_test shape : ', Y_test.shape)

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(def_target_size, def_target_size, 3)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')
# ])

# inputs = keras.Input(shape=(def_target_size, def_target_size, 3))
# h = keras.layers.Flatten(inputs)
# h = keras.layers.Dense(128)(h)
# h = keras.layers.Activation('relu')(h)
# h = keras.layers.Dense(128)(h)
# h = keras.layers.Activation('relu')(h)
# outputs = keras.layers.Dense(2)(h)
# outputs = keras.layers.Activation('softmax')(outputs)

# model = keras.Model(inputs=inputs, outputs=outputs)

# inputs = keras.Input(shape=(def_target_size, def_target_size, 3))
# h = keras.layers.Conv2D(filters=4, kernel_size=4, padding='same', activation='relu')(inputs)
# h = keras.layers.MaxPooling2D(pool_size=(2, 2))(h)
# h = keras.layers.Conv2D(filters=8, kernel_size=4, padding='same', activation='relu')(h)
# h = keras.layers.MaxPooling2D(pool_size=(2, 2))(h)
# h - keras.layers.Flatten()(h)
# h = keras.layers.Dense(32, activation='relu')(h)
# outputs = keras.layers.Dense(2, activation='softmax')(h)
# model = keras.Model(inputs=inputs, outputs=outputs)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(def_target_size, def_target_size, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=5, validation_split=0.1)              

#model.save('20210912.h5')

# 예측
predictions = model.predict(X_test)

#test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
#print('\n테스트 정확도:', test_acc)

# test
# img_name = '1-with-mask.jpg'
# img_path = os.path.join(img_dir, img_name)
# print(img_path)

# img = image.load_img(img_path, target_size=(512, 512))
# img_tensor = image.img_to_array(img)
# print(img_tensor.shape)

# # expand a dimension (3D -> 4D)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# print(img_tensor.shape)

# # scaling into [0, 1]
# img_tensor /= 255.

# print(img_tensor[0])

# #이미지 시각화
# plt.rcParams['figure.figsize'] = (10, 10) # set figure size
# plt.imshow(image.array_to_img(x[0]))
# plt.show()

# #이미지 시각화
plt.figure(figsize=(10,10))
for i in range(80):
    plt.subplot(8,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)

    if predictions[i][0] > predictions[i][1]:
        label = 'Mask'
    else:
        label = 'No Mask'

    plt.xlabel(label) 
    # plt.xlabel(categories[Y_train[i]])
plt.show()