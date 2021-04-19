import io
import random


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def transform_one_hot(labels):
    "transforms labels into one hot representation"

    if type(labels) != list:
        encoder = OneHotEncoder().fit(labels)
        one_hot_labels = encoder.transform(labels).toarray()

    else:
        encoder = OneHotEncoder().fit((np.concatenate(labels, axis=0)))
        one_hot_labels = [encoder.transform(label).toarray() for label in labels]

    return one_hot_labels


def decode_one_hot_vector(labels):
    return np.expand_dims(np.argmax(labels, axis=1), axis=-1)


def convert_seq_to_func_model(model):
    "converts a sequential keras model into a functional keras model"
    from tensorflow.python.keras import layers, models

    input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
    prev_layer = input_layer

    for layer in model.layers:
        if isinstance(layer, models.Sequential):
            for seq_layer in layer.layers:
                prev_layer = seq_layer(prev_layer)

        else:
            prev_layer = layer(prev_layer)


    funcmodel = models.Model([input_layer], [prev_layer])

    return funcmodel


if __name__ == "__main__":
    pass