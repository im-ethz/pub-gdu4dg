import os
import pandas as pd
from tqdm import tqdm
import random
from os.path import expanduser, isfile
import pickle
from utils.constants import DATA_DIR
# from utils.data.custom_datasets import ClassificationDataset, NocturnalDataset, ShortTermDataset


def load_data(directory="/wrangled_concatenated_data/", shuffle=True, filenames=False):
    """
    loads the wrangled data

    :arg
        data_dir: (string) path to the data
    :return
        df_list: list of pandas dataframes containing the data series from the wrangled data directory
        if filenames:
            files: list of loaded files
    """
    path = DATA_DIR + directory
    files = os.listdir(path)
    if shuffle:
        random.shuffle(files)

    df_list = []

    print("Loading files")
    for f in tqdm(files):
        df = pd.read_csv(path + f)
        df.dropna(inplace=True)
        df_list.append(df)

    if filenames:
        return df_list, files
    else:
        return df_list


def load_training_data(directory):
    path = DATA_DIR + directory

    df = pd.read_pickle(path + "dataset.csv")

    with open(path + "train_idx.txt", "rb") as fp:
        train_idx = pickle.load(fp)

    with open(path + "val_idx.txt", "rb") as fp:
        val_idx = pickle.load(fp)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[val_idx]

    return train_df, test_df


def load_classification_data():

    data_dir = expanduser("~/bg-forecasting/data/classification/")

    df = pd.read_pickle(data_dir + "dataset.csv")

    # load train_idx and val_idx
    with open(data_dir + "train_idx.txt", "rb") as fp:
        train_idx = pickle.load(fp)

    with open(data_dir + "val_idx.txt", "rb") as fp:
        val_idx = pickle.load(fp)

    # split the dataframe into train and test data
    train_data = ClassificationDataset(df.loc[train_idx])
    val_data = ClassificationDataset(df.loc[val_idx])

    return train_data, val_data


def load_long_term_data(args):

    df = pd.read_pickle(expanduser("~/bg-forecasting/data/long_term/") + "nocturnal_prenocturnal_12h.csv")

    # load train_idx and val_idx
    with open(expanduser("~/bg-forecasting/data/long_term/") + "train_idx_n_pn_12h.txt", "rb") as fp:
        train_idx = pickle.load(fp)

    with open(expanduser("~/bg-forecasting/data/long_term/") + "test_idx_n_pn_12h.txt", "rb") as fp:
        val_idx = pickle.load(fp)

    # split the dataframe into train and test data
    train_data = NocturnalDataset(df.loc[train_idx], window=args.window, sampled_labels=args.sampled_labels)
    val_data = NocturnalDataset(df.loc[val_idx], window=args.window, sampled_labels=args.sampled_labels)

    return train_data, val_data


def load_short_term_data(filename="short_term_i30_h15.csv"):

    data_dir = expanduser("~/bg-forecasting/data/short_term/")
    if not isfile(data_dir + filename):
        raise FileNotFoundError(f"The file {filename} does not exist in the directory {data_dir}.")

    df = pd.read_pickle(data_dir + filename)

    # load train_idx and val_idx
    with open(data_dir + "train_idx_i30_h15.txt", "rb") as fp:
        train_idx = pickle.load(fp)

    with open(data_dir + "test_idx_i30_h15.txt", "rb") as fp:
        val_idx = pickle.load(fp)

    # split the dataframe into train and test data
    train_data = ShortTermDataset(df.loc[train_idx])
    val_data = ShortTermDataset(df.loc[val_idx])

    return train_data, val_data
