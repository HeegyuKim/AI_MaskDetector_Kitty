import tensorflow as tf
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

faceModel = './AI_Mask_Detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
faceConfig = './AI_Mask_Detector/deploy.prototxt'
img_withmask_dir = './AI_Mask_Detector/train/with_mask'
img_witouthmask_dir = './AI_Mask_Detector/train/without_mask'
def_target_size = 300

net = cv2.dnn.readNet(faceModel, faceConfig)
 
x = []
y = []
test_x = []

print(tf.__version__)

def preprocess_img(img_path, y_Data, isTestMe = False):
    #img = image.load_img(img_path, target_size=(def_target_size, def_target_size))
    #img = image.load_img(img_path, grayscale=True, target_size=(def_target_size, def_target_size))

    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.show()    

    #print(img_path)

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = img.shape[:2]

    face =[]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

        face = img[y1:y2, x1:x2]
        #width, height, channel = face.shape

        face = cv2.resize(face, dsize=(def_target_size, def_target_size))

        #face = image.load_img(face, target_size=(def_target_size, def_target_size))

        #print(x1, y1, x2, y2, width, height)
        #img[0:width, 0:height] = face    

        #print(face.shape)
        #plt.imshow(face, cmap=plt.cm.binary)
        #plt.show()  


        rgb_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
        rgb_tensor /= 255.

        if isTestMe == False:
            x.append(rgb_tensor)
            y.append(y_Data)
        else:
            test_x.append(rgb_tensor)
    


categories = ['mask','nomask']
nb_class=len(categories)


for i in os.listdir(img_withmask_dir):
    img_path = os.path.join(img_withmask_dir, i)
    img_tensor = preprocess_img(img_path, 0)
    #x.append(img_tensor)
    #y.append(0)
    
for i in os.listdir(img_witouthmask_dir):
    img_path = os.path.join(img_witouthmask_dir, i)
    img_tensor = preprocess_img(img_path, 1)
    #x.append(img_tensor)
    #y.append(1)


x = np.array(x)
print(x.shape) 
y = np.array(y)
print(y.shape)

X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size=0.1)


#내사진 테스트
# test_x.clear()
# img_test_me_dir = './AI_Mask_Detector/train/test_me'
# for i in os.listdir(img_test_me_dir):
#     img_path = os.path.join(img_test_me_dir, i)
#     img_tensor = preprocess_img(img_path, 0, True)

# X_test = np.array(test_x)
# print('X_test2 shape : ', X_test.shape)


Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)

print('X_test shape : ', X_test.shape)
print('Y_test shape : ', Y_test.shape)
 


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

model.fit(X_train, Y_train, epochs=4, validation_split=0.1)              

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
plt.figure(figsize=(10,10))
for i in range(roofCnt):
    plt.subplot(8,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)

    if predictions[i][0] > predictions[i][1]:
        label = 'Mask' + str(round(predictions[i][0],2))
    else:
        label = 'No' + str(round(predictions[i][1],2))

    plt.xlabel(label) 
    # plt.xlabel(categories[Y_train[i]])
plt.show()
