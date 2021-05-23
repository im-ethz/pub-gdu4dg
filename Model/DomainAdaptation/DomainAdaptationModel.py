import math

from tensorflow_addons.layers import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model, models


from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd

from Model.DomainAdaptation.domain_adaptation_layer import DGLayer


class DomainAdaptationModel(Model):
    """ """

    def __init__(self, feature_extractor, prediction_layer):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.prediction_layer = prediction_layer
        self.freeze_after_epochs = None
        self.epoch_count = 0
        self.dg_layer = None

    def build(self, input_shape):
        """

        Parameters
        ----------
        input_shape :


        Returns
        -------

        """
        self.feature_extractor.build(input_shape)
        super(DomainAdaptationModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """

        Parameters
        ----------
        inputs :

        training :
             (Default value = None)
        mask :
             (Default value = None)

        Returns
        -------

        """
        if self.prediction_layer:
            x = self.prediction_layer(self.feature_extractor(inputs))

        else:
            x = self.feature_extractor(inputs)

        return x


    def freeze_feature_extractor(self):
        """ """
        self.feature_extractor.trainable = False


    def get_dg_layer(self):
        if self.dg_layer is None:
            for l in range(len(self.prediction_layer.layers)):
                if isinstance(self.prediction_layer.layers[l], DGLayer):
                    self.dg_layer_index = l
                    self.dg_layer = self.prediction_layer.layers[l]

        return self.dg_layer


