import numpy as np
import tensorflow as tf

from SimulationExperiments.experiment_ecgs.ECG_utils.datareader import DataReader
from SimulationExperiments.experiment_ecgs.ECG_utils.utils import load_weights


class WeightedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, pos_weight, name="weighted_ce", **kwargs):
        super(WeightedCrossEntropyLoss, self).__init__(name=name, **kwargs)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

        Returns:
          Loss values with the shape `[batch_size, d0, .. dN-1]`.
        """

        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight, name=self.name)


class ChallengeLoss(tf.keras.losses.Loss):
    def __init__(self, name="challenge_loss", **kwargs):
        super(ChallengeLoss, self).__init__(name=name, **kwargs)
        self.class_weights = tf.convert_to_tensor(
            load_weights('weights.csv', list(DataReader.get_label_maps(path="tables/")[0].keys())).astype(np.float32))

    def _get_individual_score(self, y_true, y_pred):
        normalizer = tf.reduce_sum(y_true + y_pred - tf.multiply(y_true, y_pred), axis=1)
        normalizer = tf.clip_by_value(normalizer, clip_value_min=1, clip_value_max=tf.float32.max)

        confusion_matrix = tf.matmul(tf.transpose(tf.cast(y_true, tf.float32)),
                                     tf.truediv(tf.cast(y_pred, tf.float32), tf.expand_dims(normalizer, axis=1)))

        return -tf.reduce_sum(tf.math.multiply(self.class_weights, confusion_matrix))

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

        Returns:
          Loss values with the shape `[batch_size, d0, .. dN-1]`.
        """
        y_pred = tf.sigmoid(y_pred)
        return tf.vectorized_map(lambda y: self._get_individual_score(y[0], y[1]), elems=[y_true, y_pred])
