import numpy as np
import tensorflow as tf

from SimulationExperiments.experiment_ecgs.ecg_config import Config


class MultilableAccuracy(tf.keras.metrics.MeanIoU):
    def __init__(self, name="multilable_acc", **kwargs):
        super(MultilableAccuracy, self).__init__(num_classes=2, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, tf.cast(tf.greater(y_pred, tf.constant(0.0)), dtype=tf.float32),
                             sample_weight=sample_weight)


class WeightedCrossEntropy(tf.keras.metrics.Metric):
    def __init__(self, pos_weights, name="weighted_crossentropy", **kwargs):
        super(WeightedCrossEntropy, self).__init__(name=name, **kwargs)
        self.pos_weights = pos_weights
        self.weighted_crossentropy = self.add_weight(name="wce", initializer="zeros")
        self.cnt = self.add_weight(name="cnt", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=self.pos_weights,
                                                          name=self.name)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.weighted_crossentropy.assign_add(tf.reduce_sum(tf.reduce_mean(values, axis=1)))
        self.cnt.assign_add(tf.shape(y_true)[0])

    def result(self):
        return self.weighted_crossentropy / tf.cast(self.cnt, dtype=tf.float32)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.weighted_crossentropy.assign(0.0)
        self.cnt.assign(0)


class ChallengeMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_challenge_metric', **kwargs):
        super(ChallengeMetric, self).__init__(name=name, **kwargs)
        self.observed_score = self.add_weight(name='observed', initializer='zeros')
        self.correct_score = self.add_weight(name='correct', initializer='zeros')
        self.inactive_score = self.add_weight(name='inactive', initializer='zeros')

        self.normal_class = '426783006'
        self.normal_index = Config.HASH_TABLE[0][self.normal_class]
        self.class_weights = tf.convert_to_tensor(Config.loaded_weigths, dtype=tf.float32)

    @staticmethod
    def _get_confusion(y_true, y_pred):
        normalizer = tf.reduce_sum(tf.cast(tf.logical_or(y_true, y_pred), tf.float32), axis=1)
        normalizer = tf.clip_by_value(normalizer, clip_value_min=1, clip_value_max=tf.float64.max)
        return tf.matmul(tf.transpose(tf.cast(y_true, tf.float32)),
                         tf.truediv(tf.cast(y_pred, tf.float32), tf.expand_dims(normalizer, axis=1)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.greater(y_true, tf.constant(0.0))
        y_pred = tf.greater(y_pred, tf.constant(0.0))
        observed_score = tf.reduce_sum(tf.math.multiply(self.class_weights, self._get_confusion(y_true, y_pred)))
        correct_score = tf.reduce_sum(tf.math.multiply(self.class_weights, self._get_confusion(y_true, y_true)))
        tf.assert_greater(correct_score, tf.subtract(observed_score, 1e-5),
                          message="Correct score should be greater than observed score")

        self.observed_score.assign_add(observed_score)
        self.correct_score.assign_add(correct_score)

        y_inactive = tf.pad(tf.ones_like(tf.expand_dims(y_pred[:, self.normal_index], axis=1)),
                            paddings=tf.constant([[0, 0], [self.normal_index, 24 - self.normal_index - 1]]),
                            mode="CONSTANT")
        inactive_score = tf.reduce_sum(tf.math.multiply(self.class_weights, self._get_confusion(y_true, y_inactive)))
        tf.assert_greater(correct_score, tf.subtract(inactive_score, 1e-5),
                          message="Correct score should be greater than inactive score")
        self.inactive_score.assign_add(inactive_score)

    def result(self):
        return tf.truediv(tf.math.subtract(self.observed_score, self.inactive_score),
                          tf.math.subtract(self.correct_score, self.inactive_score))


def compute_challenge_metric_custom(res, lbls, normalize=True):
    normal_class = '426783006'

    normal_index = Config.HASH_TABLE[0][normal_class]

    lbls = lbls > 0
    res = res > 0

    weights = Config.loaded_weigths

    observed_score = np.sum(weights * get_confusion(lbls, res))

    if not normalize:
        return observed_score

    correct_score = np.sum(weights * get_confusion(lbls, lbls))

    inactive_outputs = np.zeros_like(lbls)
    inactive_outputs[:, normal_index] = 1
    inactive_score = np.sum(weights * get_confusion(lbls, inactive_outputs))

    normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)

    return normalized_score


def get_confusion(lbls, res):
    normalizer = np.sum(lbls | res, axis=1)
    normalizer[normalizer < 1] = 1

    A = lbls.astype(np.float32).T @ (res.astype(np.float32) / normalizer.reshape(normalizer.shape[0], 1))

    return A
