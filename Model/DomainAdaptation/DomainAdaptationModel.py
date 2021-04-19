import math

from tensorflow_addons.layers import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model, models

from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd


class DomainAdaptationModel(Model):

    def __init__(self, feature_extractor, prediction_layer):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.prediction_layer = prediction_layer
        self.freeze_after_epochs = None
        self.epoch_count = 0


    def build(self, input_shape):
        self.feature_extractor.build(input_shape)
        super(DomainAdaptationModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if self.prediction_layer:
            x = self.prediction_layer(self.feature_extractor(inputs))

        else:
            x = self.feature_extractor(inputs)

        return x


    def freeze_feature_extractor(self):
        self.feature_extractor.trainable = False
