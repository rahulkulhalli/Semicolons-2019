# Imports
import os, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import models

import model_utils
import visualizers
import image_utils

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_name', type=str, help='Filename of the input image', default='')
    parser.add_argument('--save_path', type=str, help='Directory to save output images in.', default='.')

    parser.add_argument('--dataset', type=str, help='Dataset to visualize e.g. Malaria, Pneumonia', default='pneumonia')

    parser.add_argument('--model_filename', type=str, help='Model Filename. Defaults to Malaria.',
                        default='/home/ubuntu/image_tests/image/Models/pneumonia.h5')

    parser.add_argument('--deployment', type=str, help='Deployment server: One of "local" or "remote". '
                                                       'Defaults to local. Local denotes any instance'
                                                       'without GPU or < 64GB memory',
                        default='remote')

    args = parser.parse_args()

    return args

args = get_args()

assert args.image_name, 'Please provide path to input image.'
image_name = args.image_name.split(os.path.sep)[-1]
print (image_name)
args.dataset = args.dataset.lower()

# Assert malaria data for local compute as local compute hangs for Pneumonia
if args.deployment == 'local':
    assert args.dataset == 'malaria', 'LOCAL COMPUTE WILL HANG FOR PNEUMONIA'

if args.dataset == 'malaria':
    import malaria_config as config
elif args.dataset == 'pneumonia':
    import pneumonia_config as config

# Model params
input_config = config.input_config
class_map = config.class_map
last_conv_layer = config.last_conv_layer

model = models.load_model(args.model_filename)
# Keep a copy of the model since modifications to model affect the original
# model due to shallow copy
# TODO: Optionally load on different GPU
model_mod = models.load_model(args.model_filename)

# Create wrapper model
# TODO: Find last conv layer automatically
print('Creating Wrapper model.')
application_model = model_utils.ApplicationModel(model, model_mod,
                                     make_linear=True, preprocessing_function=config.preprocessing_function, last_conv_layer=last_conv_layer,
                                     custom_objects=config.custom_objects)
print('Complete.')
# Initialize visualizers
print('Initializing Integrated Gradients.')
integrated_vis = visualizers.IntegratedGradientsVisualizer(application_model)
print('Initializing GradCAM.')
grad_cam = visualizers.GradCAMVisualizer(application_model)

'''
Entry point here
'''
x, img = image_utils.read_image(
    args.image_name,
    im_framework=input_config['im_framework'],
    target_size=input_config['target_size'],
    mode=input_config['color_mode']
)

#TODO: Dynamic colormap
# For now, hardcode colormap
cmap = plt.cm.copper

print('Running Inference on {}'.format(args.image_name))
class_scores = application_model.predict(x, output='linear')
class_probabilities, predicted_class = model_utils.softmax(class_scores)

# Generate bar graph of top 5 (or less) classes
savename = image_name.rpartition('.')[0] + '_prob_graph.png'
image_utils.generate_bargraph(class_probabilities, class_map, os.path.join(args.save_path, savename))

# Print probabilities
print_str = 'Prediction:\n'
for idx, _class in class_map.items():
    print_str += '{}: {}\n'.format(_class, class_probabilities[idx])
print(print_str)

print('Generating Explanations.')
'''
integrated_image = integrated_vis.calculate(x, class_scores, predicted_class,
                                            reference_image=input_config['reference_image'],
                                            cmap=cmap, percentile=99)
'''
print('1 / 2')
actmap_image = grad_cam.calculate(x, class_scores, predicted_class)
print('2 / 2')

# Save to disk
# Integrated Gradients
'''
print (args.save_path)
savename = image_name.rpartition('.')[0] + '_exp2.png'
mask = integrated_image[..., -1] > 0
beta = np.ones(x.shape[1:3])
beta[mask] = 0.5
beta = np.expand_dims(beta, axis=-1)
super_image = (beta * x[0, :, :, ::-1] + (1 - beta) * integrated_image[..., :-1]).astype(np.uint8)
cv2.imwrite(os.path.join(args.save_path, savename), super_image)
'''
savename = image_name.rpartition('.')[0] + '_exp3.png'
mask = actmap_image[..., -1] > 0
beta = np.ones(x.shape[1:3])
beta[mask] = 0.8
beta = np.expand_dims(beta, axis=-1)
super_image = (beta * x[0, :, :, ::-1] + (1 - beta) * actmap_image[..., :-1]).astype(np.uint8)
cv2.imwrite(os.path.join(args.save_path, savename), super_image)

savename = image_name.rpartition('.')[0] + '_exp2.png'
cv2.imwrite(os.path.join(args.save_path, savename), super_image)

print('Complete.')
