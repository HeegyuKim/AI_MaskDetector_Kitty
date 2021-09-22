import os
import numpy as np


def get_images_from_dir(dir, preprocess_func):
    x = []
    
    for i in sorted(os.listdir(dir)):
        img_path = os.path.join(dir, i)
        img_tensor = preprocess_func(img_path)
        x.append(img_tensor)
    
    return np.array(x)