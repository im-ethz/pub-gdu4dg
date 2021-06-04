import random

import numpy as np
import tensorflow as tf
from scipy.signal import firwin, filtfilt

__all__ = ["Compose", "ZScore", "RandomStretch", "RandomAmplifier", "Resample", "SnomedToOneHot"]


class Compose(object):
    """Composes several transforms together.
    Example:
        transforms.Compose([
            transforms.HardClip(10),
            transforms.ToTensor(),
            ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_sample, **kwargs):
        if self.transforms:
            for t in self.transforms:
                data_sample = t(data_sample, **kwargs)
        return data_sample


class ZScore:
    """Returns Z-score normalized data"""

    def __init__(self, mean=0, std=1000):
        self.mean = mean
        self.std = std

    def __call__(self, sample, **kwargs):
        sample = sample - np.array(self.mean).reshape(-1, 1)
        sample = sample / self.std

        return sample


class RandomStretch:
    """
    Class randomly stretches temporal dimension of signal
    """

    def __init__(self, p=0, max_stretch=0.1):
        self.probability = p
        self.max_stretch = max_stretch

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            relative_change = 1 + tf.experimental.numpy.random.rand(1)[0] * 2 * self.max_stretch - self.max_stretch
            if relative_change < 1:
                relative_change = 1 / (1 - relative_change + 1)

            new_len = int(relative_change * self.sample_length)

            stretched_sample = np.zeros((self.sample_channels, new_len))
            for channel_idx in range(self.sample_channels):
                stretched_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample_length - 1, new_len),
                                                             np.linspace(0, self.sample_length - 1, self.sample_length),
                                                             sample[channel_idx, :])

            sample = stretched_sample
        return sample


class RandomAmplifier:
    """
    Class randomly amplifies signal
    """

    def __init__(self, p=0, max_multiplier=0.2):
        self.probability = p
        self.max_multiplier = max_multiplier

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            for channel_idx in range(sample.shape[0]):
                multiplier = 1 + random.random() * 2 * self.max_multiplier - self.max_multiplier

                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if multiplier < 1:
                    multiplier = 1 / (1 - multiplier + 1)

                sample[channel_idx, :] = sample[channel_idx, :] * multiplier

        return sample


class Resample:
    def __init__(self, output_sampling=500):
        self.output_sampling = int(output_sampling)

    def __call__(self, sample, input_sampling, gain):

        sample = sample.astype(np.float32)
        for k in range(sample.shape[0]):
            sample[k, :] = sample[k, :] * gain[k]

        # Rescale data
        self.sample = sample
        self.input_sampling = int(input_sampling)

        factor = self.output_sampling / self.input_sampling

        len_old = self.sample.shape[1]
        num_of_leads = self.sample.shape[0]

        new_length = int(factor * len_old)
        resampled_sample = np.zeros((num_of_leads, new_length))

        for channel_idx in range(num_of_leads):
            tmp = self.sample[channel_idx, :]

            ### antialias
            if factor < 1:
                q = 1 / factor

                half_len = 10 * q
                n = 2 * half_len
                b, a = firwin(int(n) + 1, 1. / q, window='hamming'), 1.
                tmp = filtfilt(b, a, tmp)

            l1 = np.linspace(0, len_old - 1, new_length)
            l2 = np.linspace(0, len_old - 1, len_old)
            tmp = np.interp(l1, l2, tmp)
            resampled_sample[channel_idx, :] = tmp

        return resampled_sample


class SnomedToOneHot(object):
    """Returns one hot encoded labels"""

    def __init__(self):
        pass

    def __call__(self, snomed_codes, mapping):
        encoded_labels = np.zeros(len(mapping)).astype(np.float32)
        for code in snomed_codes:
            if code not in mapping:
                continue
            else:
                encoded_labels[mapping[code]] = 1.0

        return encoded_labels
