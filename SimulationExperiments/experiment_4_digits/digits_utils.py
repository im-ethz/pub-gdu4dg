import os
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import _SumKernel
import tensorflow_probability as tfp

img_shape = (32, 32, 3)


def MMD(x1, x2, kernel):
    return np.mean(kernel.matrix(x1, x1)) - 2 * np.mean(kernel.matrix(x1, x2)) + np.mean(kernel.matrix(x2, x2))


def get_mmd_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1
    mmd_matrix = np.zeros((num_ds, num_ds))
    for i in range(num_ds):
        x_i = x_data[i]
        for j in range(i, num_ds):
            x_j = x_data[j]
            mmd_matrix[i, j] = mmd_matrix[j, i] = MMD(x_i, x_j, kernel=kernel)
    return mmd_matrix


def sigma_median(x_data, sample_size=5000):
    x_data = x_data[:sample_size]
    sigma_median = np.median(euclidean_distances(x_data, x_data))
    return sigma_median


def get_domainnet_feature_extractor(dropout=0.5):
    feature_exctractor = tf.keras.Sequential([
        Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same", input_shape=img_shape)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same")
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Conv2D(128, strides=(1, 1), kernel_size=(5, 5), padding="same")
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Flatten()
        , Dense(3072)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , Dropout(dropout)

        , Dense(2048)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
    ], name='feature_extractor_domainnet_digits')

    return feature_exctractor


def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Conv2D(64, kernel_size=(2, 2), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Flatten()
        , Dense(100, activation="relu")
        , Dense(100, activation="relu")
    ], name='feature_extractor')
    return feature_exctractor


def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.05)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))


def get_kernel_sum(sigma_list):
    amplitude_list = [1]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in
               sigma_list for amplitude in amplitude_list] + \
              [tfp.math.psd_kernels.MaternFiveHalves(length_scale=sigma, amplitude=amplitude) for sigma in sigma_list
               for amplitude in amplitude_list] + \
              [tfp.math.psd_kernels.RationalQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in sigma_list
               for amplitude in amplitude_list]
    kernel_sum = _SumKernel(kernels=kernels)
    return kernel_sum


