import os, tempfile

import numpy as np
# import cv2
import tensorflow as tf
from tensorflow.python.framework import ops

from keras import activations, models

class ApplicationModel():

    def __init__(self, model, model_mod, make_linear, preprocessing_function, last_conv_layer=12, custom_objects=None):

        self.custom_objects = custom_objects
        self.preprocessing_function = preprocessing_function

        self.model = model
        self.make_linear = make_linear
        if make_linear:
            self.model_mod = self.linear_output(model_mod)
        else:
            self.model_mod = model_mod

        model_modifier = GuidedModelModifier()
        self.guided_model = model_modifier.modify_model_backprop(
            self.model_mod, 'guided', custom_objects=self.custom_objects
        )

        # TODO: Find last conv layer automatically
        self.last_conv_layer = model_mod.layers[last_conv_layer].name

    def predict(self, x, output='logits', verbose=0):

        if output == 'logits':
            preds = self.model.predict(self.preprocessing_function(x))[0]
        elif output == 'linear':
            preds = self.model_mod.predict(self.preprocessing_function(x))[0]

        if verbose == 1:
            print(preds)

        return preds

    def linear_output(self, model_non_l):

        model_non_l.layers[-1].activation = activations.linear
        model_non_l.save('temp_inc.h5')
        model_l = models.load_model('temp_inc.h5', custom_objects=self.custom_objects)
        os.remove('temp_inc.h5')
        return model_l

    # def get_last_conv(self):
    #


class GuidedModelModifier():

    def __init__(self):
        self._MODIFIED_MODEL_CACHE = dict()
        self._BACKPROP_MODIFIERS = {
            'guided': self._register_guided_gradient,
            'rectified': self._register_rectified_gradient
        }

    def _register_guided_gradient(self, name):
        if name not in ops._gradient_registry._registry:
            @tf.RegisterGradient(name)
            def _guided_backprop(op, grad):
                dtype = op.outputs[0].dtype
                gate_g = tf.cast(grad > 0., dtype)
                gate_y = tf.cast(op.outputs[0] > 0., dtype)
                return gate_y * gate_g * grad

    def _register_rectified_gradient(self, name):
        if name not in ops._gradient_registry._registry:
            @tf.RegisterGradient(name)
            def _relu_backprop(op, grad):
                dtype = op.outputs[0].dtype
                gate_g = tf.cast(grad > 0., dtype)
                return gate_g * grad

    def modify_model_backprop(self, model, backprop_modifier, custom_objects=None):
        modified_model = self._MODIFIED_MODEL_CACHE.get((model, backprop_modifier))
        if modified_model is not None:
            return modified_model

        model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        try:
            # 1. Save original model
            model.save(model_path)

            # 2. Register modifier and load modified model under custom context.
            modifier_fn = self._BACKPROP_MODIFIERS.get(backprop_modifier)
            if modifier_fn is None:
                raise ValueError("'{}' modifier is not supported".format(backprop_modifier))
            modifier_fn(backprop_modifier)

            # 3. Create graph under custom context manager.
            with tf.get_default_graph().gradient_override_map({'Relu': backprop_modifier}):
                #  This should rebuild graph with modifications.
                modified_model = models.load_model(model_path, custom_objects=custom_objects)

                # Cache to improve subsequent call performance.
                self._MODIFIED_MODEL_CACHE[(model, backprop_modifier)] = modified_model
                return modified_model
        finally:
            os.remove(model_path)

def softmax(y):
    max_score_idx = np.argmax(y)
    max_score = y[max_score_idx]
    stable_y = y - max_score
    exponentials = np.exp(stable_y)
    return (exponentials / np.sum(exponentials), max_score_idx)
