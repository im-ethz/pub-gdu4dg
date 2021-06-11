from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import tensorflow_probability as tfp

from sklearn.metrics.pairwise import euclidean_distances


def sigma_median(x_data):
    sigma_median = np.median(euclidean_distances(x_data, x_data))
    return sigma_median


def MMD(x1, x2, kernel):
    return np.mean(kernel.matrix(x1, x1)) - 2 * np.mean(kernel.matrix(x1, x2)) + np.mean(kernel.matrix(x2, x2))


def get_kme_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1

    kme_matrix = np.zeros((num_ds, num_ds))

    for i in range(num_ds):
        x_i = x_data[i]

        for j in range(i, num_ds):
            x_j = x_data[j]
            kme_matrix[i, j] = kme_matrix[j, i] = np.mean(kernel.matrix(x_i, x_j))

    return kme_matrix


def get_mmd_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1
    mmd_matrix = np.zeros((num_ds, num_ds))

    kme_mat = get_kme_matrix(x_data, kernel)
    for i in range(num_ds):
        for j in range(num_ds):
            mmd_matrix[i, j] = mmd_matrix[j, i] = kme_mat[i, i] - 2*kme_mat[i, j] + kme_mat[j, j]

    return mmd_matrix





