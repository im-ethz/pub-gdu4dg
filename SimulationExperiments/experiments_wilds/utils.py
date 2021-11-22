import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.keras.initializers import Initializer


class GSM(Initializer):
    """Initializer that generates tensors with a normal distribution.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed`
        for behavior.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_init = tf.initializers.random_normal(mean=mean, stddev=stddev, seed=seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported.
        Raises:
          ValueError: If the dtype is not floating point
          :param **kwargs:
        """
        W_0 = self._random_init(shape // 2, dtype)
        return tf.concat([tf.concat([W_0, tf.negative(W_0)], axis=0), tf.concat([tf.negative(W_0), W_0], axis=0)],
                         axis=1)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class Ortho(Initializer):
    """Initializer that generates tensors with a normal distribution.
    Args:
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed`
        for behavior.
    """

    def __init__(self, seed=None):
        self.seed = seed
        self._random_init = tf.initializers.orthogonal(seed=seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported.
        Raises:
          ValueError: If the dtype is not floating point
          :param **kwargs:
        """
        shape_0 = [shape_dim // 2 for shape_dim in shape]
        W_0 = self._random_init(shape_0, dtype)
        return tf.concat([tf.concat([W_0, tf.negative(W_0)], axis=0), tf.concat([tf.negative(W_0), W_0], axis=0)],
                         axis=1)

    def get_config(self):
        return {"seed": self.seed}


def get_lr_callback(decay_starts=100, decay=-0.05):
    def scheduler(epoch, lr):
        if epoch < decay_starts:
            return lr
        else:
            return lr * tf.math.exp(decay)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return lr_callback