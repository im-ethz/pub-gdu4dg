import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from itertools import compress
from utils.data import load_data, split_data
import matplotlib.pyplot as plt
from more_itertools import consecutive_groups


def sample_period_distribution(dataset):
    """returns a histogram of the sampling periods in the data set"""

    def int_difference(a, b):
        return int(a - b)

    v_difference = np.vectorize(int_difference)

    sample_periods = {}
    print("Computing frequency of sampling periods")
    for d in tqdm(dataset):
        d_list = split_data(d)
        for ds in d_list:
            periods = v_difference(ds.relative_time.values[1:], ds.relative_time.values[:-1])
            for idx in range(periods.shape[0]):
                try:
                    sample_periods[periods[idx]] += 1
                except KeyError:
                    sample_periods[periods[idx]] = 1

    sample_periods = {k: np.log(v) for k, v in sample_periods.items()}
    sample_periods = OrderedDict(sorted(sample_periods.items()))

    print(f"Most frequent sampling period: {max(sample_periods, key=sample_periods.get)}")

    plt.bar(list(sample_periods.keys()), sample_periods.values(), align='center')
    plt.xticks(range(0, 181, 15), list(sample_periods.keys())[::15])
    plt.xlabel("Sample periods (min)")
    plt.ylabel("log-frequency")
    # plt.title("Distribution of sample periods")
    plt.savefig(os.path.expanduser("~") + "/plots/sample_period_distribution.pdf", bbox_inches="tight")
    plt.show()


def series_length_distribution(dataset, filenames):
    series_lengths = {}
    print("Computing the length of each data series")

    for i, df_patient in tqdm(enumerate(dataset)):
        d_list = split_data(df_patient)
        for ds in d_list:
            time_span = int((ds.relative_time.iloc[-1] - ds.relative_time.iloc[0]) / 60)
            if time_span >= 4320:
                print(filenames[i])
            try:
                series_lengths[time_span] += 1
            except KeyError:
                series_lengths[time_span] = 1

    # series_lengths = {k: np.log(v) for k, v in series_lengths.items()}
    series_lengths = OrderedDict(sorted(series_lengths.items()))

    weighted_sum = 0
    for k, v in zip(list(series_lengths.keys()), series_lengths.values()):
        weighted_sum += k * v

    print(f"The average data series length is {weighted_sum / len(list(series_lengths.keys()))}")

    plt.bar(list(series_lengths.keys()), series_lengths.values(), align='center')
    # plt.xticks(range(0, 181, 15), list(sample_periods.keys())[::15])
    plt.xlabel("Length of a data series (h)")
    plt.ylabel("Frequency")
    # plt.title("Distribution of sample periods")
    # plt.savefig(os.path.expanduser("~") + "/plots/series_length_distribution.pdf", bbox_inches="tight")
    plt.show()


def file_lengths_distribution(data_list):
    lengths = [len(i) for i in data_list]
    lengths = np.asarray(lengths)
    print(f"Minimum number of data points in a single file: {np.min(lengths)}")
    print(f"Maximum number of data points in a single file: {np.max(lengths)}")
    print(f"Median number of data points in a single file: {np.median(lengths)}")
    print(f"Average number of data points in a single file: {np.mean(lengths)}")
    print(f"Number of files with more than 15,000 data points: {(lengths > 15000).sum()}")
    _ = plt.hist(lengths, bins=100)
    plt.xlabel("Data points per file")
    plt.ylabel("Frequency")
    # plt.title("Distribution of amount of data points per file")
    plt.savefig(os.path.expanduser("~") + "/plots/amount_data_points_distribution.pdf", bbox_inches="tight")
    plt.show()


def quantification_inf_missingness():
    """
    quantifies the frequency and fraction of 'Hoch'/'Niedrig' values; only happens with the dexcom sensor
    """
    path = os.path.expanduser("~") + "/data/raw_data/"
    files = os.listdir(path)
    mask = ["dex" in f for f in files]
    print(f"There are {sum(mask)} patients out of {len(files)} which use the dexcom sensor.")
    files = list(compress(files, mask))

    filenames_inf_missingness = []
    df_inf_missingness = []

    for f in tqdm(files):
        df = pd.read_excel(path + f, engine='openpyxl')
        df.columns = df.iloc[0]
        df = df[['GlucoseValue']]
        df = df.iloc[1:]
        if "Hoch" in df.values or "Niedrig" in df.values:
            filenames_inf_missingness.append(f)
            df_inf_missingness.append(df)

    print(f"{len(filenames_inf_missingness)} patients have records in which informative missingness occurs.")
    fraction = np.zeros((len(df_inf_missingness)))

    for idx, df in enumerate(df_inf_missingness):
        n = len(df)
        n_missingness = len(df.loc[np.logical_or(df.values == "Hoch", df.values == "Niedrig")])
        fraction[idx] = n_missingness / n

    print(f"The minimum fraction of of informative missingness is {np.min(fraction)}")
    print(f"The median fraction of informative missingness is {np.median(fraction)}")
    print(f"The average fraction of informative missingness is {np.mean(fraction)}")
    _ = plt.hist(fraction, bins=50)
    plt.xlabel("Fraction of informative missingness")
    plt.ylabel("Frequency")
    plt.savefig(os.path.expanduser("~") + "/plots/inf_missingness_distribution.pdf", bbox_inches="tight")
    plt.show()


def quantify_noc_hypoglycemia(dataset, threshold=3.9, min_length=15, histogram=False):
    """
    quantitative evaluation of occurrence of hypoglycemia

    :param
        data_list: list of dataframes of the wrangled and concatenated cgm data
    """
    n_patients = len(dataset)
    hypoglycemic_episodes = []

    print("Extracting nocturnal data")
    n_nights = 0
    n_hypo_nights = 0
    n_hypo_episodes = 0
    counter = 0

    for df in tqdm(dataset):
        df.loc[:, 'datetime'] = pd.to_datetime(df.datetime)
        df = df.set_index(df.datetime)
        night_df = df.between_time("22:00", "06:00")
        pd.set_option('mode.chained_assignment', None)
        night_df.loc[:, 'hypoglycemic'] = night_df.glucose_value <= threshold
        nights = split_data(night_df, max_gap=8)
        n_nights += len(nights)

        # check if the blood glucose value has ever been below the threshold
        if sum(night_df.hypoglycemic) > 0:
            counter += 1
            for n in nights:
                if sum(n.hypoglycemic) > 0:
                    n_hypo_nights += 1
                    groups = [list(g) for g in consecutive_groups(list(np.where(n.hypoglycemic)[0]))]
                    n_hypo_episodes += len(groups)
                    for g in groups:
                        potential_hypo_episode = n.iloc[g]
                        if potential_hypo_episode.relative_time.iloc[-1] > min_length:
                            hypoglycemic_episodes.append(potential_hypo_episode)

    episode_lengths = [d.relative_time.iloc[-1] for d in hypoglycemic_episodes]

    print(f"There are {n_patients} patients in the data set.")
    print(f"{counter} patients had had critically low blood sugar levels at night.")
    print(f"There were {len(hypoglycemic_episodes)} hypoglycemic episodes.")

    print(f"In total, {n_nights} nights are recorded in the data set.")
    print(f"{n_hypo_nights} nights out of these ({round(n_hypo_nights / n_nights * 100, 2)}%) contain a hypoglycemic episode.")

    print(f"The longest episode is {np.max(episode_lengths)} min long.")
    print(f"The shortest episode is {np.min(episode_lengths)} min long.")
    print(f"The average episode is {round(np.mean(episode_lengths), 2)} min long.")
    print(f"The median episode is {np.median(episode_lengths)} min long.")

    if histogram:
        _ = plt.hist(episode_lengths, bins=100)
        plt.xlabel("Length of the hypoglycemic episode (min)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.expanduser("~") + "/plots/n_hypoglycemia_length_distribution.pdf", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    data = load_data(data_dir="/data/wrangled_concatenated_data/")
    quantify_noc_hypoglycemia(data, threshold=3.9, min_length=15, histogram=True)
