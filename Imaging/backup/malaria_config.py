import numpy as np


def preprocessing_function(x):
    return x

input_config = {
    'target_size': (224, 224),
    'color_mode': 'RGB',
    'im_framework': 'opencv',
    'reference_image': np.zeros((1, 224, 224, 3))
}

custom_objects = None

class_map = {0: 'Uninfected', 1: 'Parasitized'}

reference_image = np.zeros(input_config['target_size']+(3,))