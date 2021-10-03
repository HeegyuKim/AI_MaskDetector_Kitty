import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


image_path = "./train/my_photo"
def_target_size = 128

def preprocess_img(img_path, target_size=def_target_size):
    img = image.load_img(img_path, target_size=(target_size, target_size))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    return img_tensor
    
x = []

for i in os.listdir(image_path):
    img_path = os.path.join(image_path, i)
    img_tensor = preprocess_img(img_path, def_target_size)
    x.append(img_tensor)
    
x = np.array(x)

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

model.load_weights('20210912.h5')


predictions = model.predict(x)
print(predictions)

# #이미지 시각화
plt.figure(figsize=(10,10))
for i in range(2):
    plt.subplot(8,10,i*2+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[i], cmap=plt.cm.binary)

    if predictions[i][0] > predictions[i][1]:
        label = f'Mask! {int(predictions[i][0] * 100)}'
    else:
        label = f'No Mask! {int(100 * predictions[i][1])}'

    plt.xlabel(label) 
    # plt.xlabel(categories[Y_train[i]])

plt.savefig('test_output.png')