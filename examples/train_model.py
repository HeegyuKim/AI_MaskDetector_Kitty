import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import numpy as np

from mask_detector.model import get_default_model

img_width, img_height = 64, 64
batch_size = 32
train_data_dir = "./dataset/train/"


# ImageDataGenerator를 이용해서 학습 데이터를 불러오고 데이터 전처리 및 증강을 수행합니다.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    subset="training",
)  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    subset="validation",
)  # set as validation data

# 간단한 모델을 만든 뒤 학습합니다.
model = get_default_model(img_width, img_height)
print(model.summary())

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 가장 적은 validation loss를 기록한 모델을 저장하고
# validation loss가 감소하지 않을 경우 조기 종료합니다.
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3),
    ModelCheckpoint(
        filepath="./resource/model/model_best.h5",
        monitor="val_loss",
        save_best_only=True,
    ),
]

model.fit_generator(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=callbacks,
)
