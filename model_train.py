
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow import keras
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
import numpy as np

import wandb
from wandb.keras import WandbCallback

import os

from model.hkmodel import SimpleCNNModel, preprocess_img

wandb.init(project='mask-detector', entity='heegyukim')

img_withmask_dir = './train/with_mask'
img_witouthmask_dir = './train/without_mask'

def_target_size = 128


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
y = np.array(y)


X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size=0.1, random_state=123)

model = SimpleCNNModel()
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, validation_split=0.1, callbacks=[WandbCallback()])              
model.save('./simple.h5')