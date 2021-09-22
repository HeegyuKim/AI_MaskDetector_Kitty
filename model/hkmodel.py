import tensorflow as tf

from keras.preprocessing import image

def preprocess_img(img_path, target_size=128):
    img = image.load_img(img_path, target_size=(target_size, target_size))
    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    rgb_tensor /= 255.
    return rgb_tensor

class SimpleCNNModel(tf.keras.models.Sequential):
    
    def __init__(self, def_target_size=128):
        super(SimpleCNNModel, self).__init__([
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
        