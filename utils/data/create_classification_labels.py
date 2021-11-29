import numpy as np
from tqdm import tqdm


def _find_labels(sequence, hypo_threshold=3.9, hyper_threshold=13.9, min_time=15, sampling_period=5):
    sequence = np.array(sequence)
    hypo_points = np.array(sequence <= hypo_threshold)
    hyper_points = np.array(sequence >= hyper_threshold)

    min_points = int(min_time / sampling_period)

    hypo = _find_label(sequence, hypo_points, min_time=min_points)
    hyper = _find_label(sequence, hyper_points, min_time=min_points)

    return hypo, hyper


def _find_label(sequence, boolean_arr, min_time):
    """
    finds the label for the passed sequence according to the boolean array and the minimum time

    :param sequence: (np.array) sequence for which to find the label
    :param boolean_arr: (np.array) array of booleans which is used to cut the sequence into chunks
    :param min_time: (int) the minimum time length of an episode so that the sequence is labeled as true
    :return: label: (bool)
    """

    label = False

    # finding the points where critical episodes start/end
    indices = np.flatnonzero(boolean_arr[1:] != boolean_arr[:-1]) + 1
    episodes = np.split(sequence, indices)
    episodes = episodes[0::2] if boolean_arr[0] else episodes[1::2]

    for episode in episodes:
        if len(episode) >= min_time:
            label = True

    return label


def create_labels(df, sampling_period=5):

    print("Generating labels...")

    hypo_labels = []
    hyper_labels = []

    for idx in tqdm(range(len(df))):

        labels = _find_labels(df.loc[idx, 'nocturnal_resampled'], sampling_period=sampling_period)
        hypo_labels.append(labels[0])
        hyper_labels.append(labels[1])

    df['hypoglycemia'] = hypo_labels
    df['hyperglycemia'] = hyper_labels

    return df
