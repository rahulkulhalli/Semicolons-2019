import os, sys

import numpy as np
import keras.backend as K
import cv2
from PIL import Image
from PIL.ImageFilter import GaussianBlur

from model_utils import ApplicationModel
import image_utils
sys.path.insert(0, os.path.abspath('./libs/IntegratedGradients'))
import IntegratedGradients

class GuidedBackpropVisualizer():

    def __init__(self, model):
        assert isinstance(model, ApplicationModel)
        self.model = model

    def gradient_function(self, preds, output_neuron):

        # For now, assert multi-class
        assert preds.shape[0] > 1, 'Provide multi-class output for now.'

        one_hots = np.zeros((1, preds.shape[0]))
        one_hots[:, output_neuron] = 1.
        loss_out = one_hots * self.model.guided_model.output
        input_grads = K.gradients(loss_out, self.model.guided_model.input)

        outputs = [self.model.guided_model.output]
        if type(input_grads) in {list, tuple}:
            outputs += input_grads
        else:
            outputs.append(input_grads)

        f_outputs = K.function([self.model.guided_model.input], outputs)

        return f_outputs

    def calculate(self, x, preds, output_neuron, cmap=None, percentile=99):
        f = self.gradient_function(preds, output_neuron)
        function_preds, guided_gradients = f([self.model.preprocessing_function(x)])

        return self.postprocess(guided_gradients[0], cmap, percentile), function_preds

    def postprocess(self, x, cmap, percentile):
        return image_utils.VisualizeImageGrayscale(x, percentile=percentile, cmap=cmap)


class IntegratedGradientsVisualizer():

    def __init__(self, model):
        assert isinstance(model, ApplicationModel)
        self.model = model
        self.ig = IntegratedGradients.integrated_gradients(model.model)

    def calculate(self, x, preds, output_neuron, reference_image=None,
                  cmap=None, percentile=99):
        assert len(x.shape) == 4
        if not reference_image is None:
            assert len(reference_image.shape) == 4

        if reference_image is None:
            reference_image = self.model.preprocessing_function(np.zeros(x.shape))

        explanation = self.ig.explain(
            self.model.preprocessing_function(x)[0],
            reference=reference_image[0],
            outc=output_neuron
        )
        return self.postprocess(explanation, cmap, percentile)

    def postprocess(self, x, cmap, percentile):
        preprocessed_image =  image_utils.VisualizeImageGrayscale(x, percentile=percentile, cmap=cmap)
        mask = preprocessed_image[..., -1] > 0
        preprocessed_image = np.expand_dims(mask, axis=-1) * preprocessed_image
        return preprocessed_image


class GradCAMVisualizer():

    def __init__(self, model, filter_radius=7):
        assert isinstance(model, ApplicationModel)
        self.model = model
        self.im_filter = GaussianBlur(radius=filter_radius)

    def gradient_function(self, preds, output_neuron):
        # For now, assert multi-class
        assert preds.shape[0] > 1, 'Provide multi-class output for now.'

        one_hots = np.zeros((1, preds.shape[0]))
        one_hots[:, output_neuron] = 1.
        loss_out = one_hots * self.model.model_mod.output
        cam_grads = K.gradients(
            # Gradient of output layer
            loss_out,
            # wrt output of final conv layer
            self.model.model_mod.get_layer(self.model.last_conv_layer).output)
        alpha_tensor = K.mean(cam_grads[0], axis=(0, 1, 2))

        cam = self.model.model_mod.get_layer(self.model.last_conv_layer).output
        scaled_map = cam[0] * alpha_tensor
        grad_cam = K.relu(K.sum(scaled_map, axis=-1))

        outputs = [grad_cam]

        cam_func = K.function([self.model.model_mod.input], outputs)

        return cam_func

    def calculate(self, x, preds, output_neuron):
        cam_f = self.gradient_function(preds, output_neuron)
        grad_cam = cam_f([self.model.preprocessing_function(x)])

        # Width by Height
        target_size = (x.shape[2], x.shape[1])
        processed_grad_cam = self.post_process(grad_cam[0], target_size)

        return processed_grad_cam

    def post_process(self, activation_map, target_size):
        # Normalize 0 - 1
        vmax = np.max(activation_map)
        vmin = np.min(activation_map)

        scaled_map = np.clip((activation_map - vmin) / (vmax - vmin), 0, 1)

        # Normalize 0 - 255, Resize, Smoothen
        scaled_map = scaled_map * 255.
        scaled_map = scaled_map.astype(np.uint8)
        act_image = Image.fromarray(scaled_map)
        act_image = act_image.resize(target_size)
        act_image = act_image.filter(self.im_filter)

        # Renormalize
        act_image = np.array(act_image)
        vmax = np.max(act_image)
        vmin = np.min(act_image)

        act_image = np.clip((act_image - vmin) / (vmax - vmin), 0, 1)

        # Apply colourmap and return
        heatmap = cv2.applyColorMap((act_image * 255).astype(np.uint8), cv2.COLORMAP_JET)

        '''
        Experimental, remove if not working
        '''
        # Go on varying intensity
        # alpha_channel = np.copy(act_image)
        # alpha_channel = (255. * alpha_channel).astype(np.uint8)

        # Should take care of Red, Yellow
        alpha_channel = heatmap[:, :, -1] > 0
        # Green
        alpha_channel |= heatmap[:, :, 1] > 0
        alpha_channel = (255 * alpha_channel).astype(np.uint8)
	
        heatmap *= np.expand_dims(alpha_channel > 0, axis=-1)
        heatmap = np.concatenate(
            [heatmap, np.expand_dims(alpha_channel, axis=-1)],
            axis=-1
        )

        return heatmap
