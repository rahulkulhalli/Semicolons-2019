import os, warnings

import cv2
from PIL import Image

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt = matplotlib.pyplot
import numpy as np

def VisualizeImageGrayscale(image_3d, percentile=99, cmap=None):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    normalized_image = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    if cmap:
        # Convert grayscale to colormap images and remove alpha channel
        cmapped_image = cmap(normalized_image)
        image = cmapped_image[:, :, :-1][:, :, ::-1]

        alpha_channel = np.expand_dims(normalized_image, axis=-1) # This works fine
        image = np.concatenate([image, alpha_channel], axis=-1)
        image = (255. * image).astype(np.uint8)
        return image
        # image = (cmapped_image[:, :, :3] * 255).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # # Add alpha channel
        # alpha_channel = (255. * cmapped_image[:, :, -1]).astype(np.uint8)
        # return np.concatenate([image, np.expand_dims(alpha_channel, axis=-1)], axis=-1)
    else:
        return normalized_image


def ShowGrayscaleImage(im, title='', ax=None, alpha=0.5, color_map='copper', save_path=''):
    if ax is None:
        plt.figure()
    plt.axis('off')
    if not isinstance(im, (list, tuple)):
        plt.imshow(im, cmap=color_map, vmin=0, vmax=1)
    else:
        if len(im) > 2:
            warnings.warn(
                '{} images passed, visualizing first two. Second image will be visualized with alpha={}'.format(len(im),
                                                                                                                alpha))
        plt.imshow(im[0])
        plt.imshow(im[1], cmap=color_map, vmin=0, vmax=1, alpha=alpha)
    # if save_path:
    #     plt.savefig(save_path + '.png', dpi=300, )
    # else:
    #     plt.title(title)
    plt.title(title)


def ShowHeatmap(im, title='', ax=None, alpha=0.5):
    if ax is None:
        plt.figure()
    plt.axis('off')
    if not isinstance(im, (list, tuple)):
        plt.imshow(im, cmap='jet')
    else:
        plt.imshow(im[0])
        plt.imshow(im[1], cmap='jet', alpha=alpha)
    plt.title(title)


def ShowRGB(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im)
    plt.title(title)


def read_image(im_path, im_framework='PIL', target_size=None, mode='RGB'):
    if im_framework.lower() == 'pil':
        img = Image.open(im_path)
        if target_size:
            img = img.resize(target_size)

        # Convert of image RGB -> for display purpose
        if img.mode != 'RGB':
            img_to_display = img.convert('RGB')

            # Convert for input x if conversion specified for model input
            if mode == 'RGB':
                img = img_to_display.copy()

        x = np.array(img)
        img.close()

    elif im_framework.lower() == 'opencv':
        img = cv2.imread(im_path)
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        if mode == 'RGB':
            if len(img.shape) == 3:
                x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_to_display = x.copy()

            # Check grayscale image
            if len(img.shape) < 3:
                img_to_display = img.copy()
                img = np.expand_dims(img, axis=-1)
                x = np.concatenate([img, img, img], axis=-1)
        else:
            # Dummy channel for keras conv layer
            x = np.expand_dims(img, axis=-1)
            img_to_display = img
    return np.expand_dims(x, axis=0), img_to_display

def stack_explanation_images(im_dir, im_framework='opencv', target_size=(224, 224), mode='RGB'):
    X = []
    classes = os.listdir(im_dir)
    for c in classes:
        path = os.path.join(im_dir, c)
        images = os.listdir(path)
        for im_name in images:
            im_path = os.path.join(path, im_name)
            x, img = read_image(im_path, im_framework=im_framework, target_size=target_size, mode=mode)
            X.append(x[0])

    return np.array(X)

def generate_bargraph(class_probabilities, class_map, save_path):
    sorted_indices = np.argsort(class_probabilities)
    # Select Top 5 or less
    to_display = sorted_indices[:5]
    classes = np.vectorize(lambda x: class_map[x])(to_display)
    probabilities = class_probabilities[to_display]
    n_classes = len(classes)
    fig, ax = plt.subplots(dpi=300, figsize=(4, 0.5 * n_classes))
    ax.set_xlim([0., 1.])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('Class Probability')
    # White, with completely transparent background
    ax.set_facecolor((1, 1, 1, 0))
    plt.barh(classes, probabilities, color='blue', alpha=0.5, height=0.8)
    for i, c in enumerate(classes):
        plt.text(0.01, i, c, fontsize=12, alpha=1., verticalalignment='center')
    plt.savefig(save_path, transparent=True, bbox_inches='tight')

def get_colormap(x, mode='RGB'):
    if mode=='RGB':
        flag = cv2.COLOR_RGB2HSV
    elif mode=='BGR':
        flag = cv2.COLOR_RGB2HSV
    else:
        flag = cv2.COLOR_RGB2HSV
    x_hsv = cv2.cvtColor(x, flag)

    if np.mean(x_hsv[..., 0]) == 0 and np.mean(x_hsv[..., 1]) == 0:
        #Grayscale image if hue is 0 and saturation is 0
        colormap = 'copper'
    else:
        # Mean Hue (Pure Color)
        colormap = 'gray'
