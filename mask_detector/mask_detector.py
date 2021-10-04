import tensorflow as tf


default_mask_detector_model_path = "./resource/model/model.h5"

def to_tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=tf.float32) if not isinstance(x, tf.Tensor) else x
    
class MaskDetector:
    def __init__(self, weight_path=default_mask_detector_model_path):
        self.model = tf.keras.Sequential([
            tf.keras.models.load_model(weight_path)
            ])
        self.input_image_size = (64, 64)
    
    def _preprocess(self, image):
        rgb_tensor = to_tensor(image, dtype=tf.float32)
        rgb_tensor /= 255.
        return rgb_tensor
    
    def predict_one(self, image):
        image = self._preprocess(image)
        image = tf.expand_dims(image, 0)
        preds = self.model.predict(image)
        return preds[0][0]
        
    def predict(self, images):
        images = self._preprocess(images)
        preds = self.model.predict(images)
        return preds[:, 0]