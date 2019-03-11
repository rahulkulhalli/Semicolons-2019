import numpy as np

def preprocessing_function(x):
    return x / 255.

input_config = {
    'target_size': (224, 224),
    'color_mode': 'RGB',
    'im_framework': 'opencv',
    'reference_image': 247. * np.ones((1, 224, 224, 3))
}

custom_objects = None

class_map = {0: 'Normal', 1: 'Pneumonia'}