
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow import keras
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

from model.hkmodel import SimpleCNNModel, preprocess_img

img_test_me_dir = './train/test_me'
def_target_size = 128

test_x = []
for i in sorted(os.listdir(img_test_me_dir)):
    img_path = os.path.join(img_test_me_dir, i)
    img_tensor = preprocess_img(img_path, def_target_size)
    test_x.append(img_tensor)

X_test = np.array(test_x)

model = SimpleCNNModel()
model.load_weights("simple.h5")

predictions = model.predict(X_test)

# #이미지 시각화
plt.figure(figsize=(10,10))
for i in range(X_test.shape[0]):
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
plt.savefig("test.png")
