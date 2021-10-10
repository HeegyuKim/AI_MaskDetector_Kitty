from tensorflow import keras

from .pretrained import *


def get_default_model(input_width, input_height):

    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(input_width, input_height, 3),
        )
    )
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model
