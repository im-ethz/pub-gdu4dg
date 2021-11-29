import os
from utils.constants import DATA_DIR
import pandas as pd
import numpy as np


def filter_patients(return_list=False, df=None, **kwargs):
    """filters the patients by the passed criteria

    :return: filtered dataframe
    """
    patients = pd.read_csv(DATA_DIR + "patients_characteristics.csv")

    if 'diabetes_type' in kwargs:
        patients.drop(patients.index[np.array(patients['diabetes_type'] != kwargs['diabetes_type'])], inplace=True)
    if 'therapy' in kwargs:
        patients.drop(patients.index[np.array(patients['therapy'] != kwargs['therapy'])], inplace=True)
    if 'gender' in kwargs:
        patients.drop(patients.index[np.array(patients['gender'] != kwargs['gender'])], inplace=True)
    if 'sensor' in kwargs:
        patients.drop(patients.index[np.array(patients['code'].apply(lambda s: s.split('_')[1]) != kwargs['sensor'])],
                      inplace=True)
        
    relevant_patients = patients['code'].tolist()

    if return_list:
        return relevant_patients
    else:
        assert df is not None, "You must pass a dataframe if return_list=False"

    # find the column which contains patient codes
    code_column = ''
    for col in list(df.columns):
        try:
            if sum(df[col].str.contains(relevant_patients[0])) > 0:
                code_column = col
                break
        except AttributeError:
            pass

    filtered_df = df.drop(df.index[df[code_column].apply(lambda s: "_".join(s.split("_")[:3]) not in relevant_patients)])

    return filtered_df


def generate_subset(file, **kwargs):
    """
    creates a subset of the passed file. Each patient in the file fulfills the criteria passed as kwargs

    :param file:
    :param kwargs:
    :return:
    """

    df = pd.read_pickle(DATA_DIR + file)
    print(f"The original file contains {len(df)} samples.")

    df = filter_patients(df=df, **kwargs)
    print(f"The filtered file contains {len(df)} samples.")

    train_df, test_df = custom_train_test_split(df)

    new_directory = '_'.join(v for _, v in kwargs.items())
    save_to = file.split('/')[0] + '/' + new_directory + '/'

    if not os.path.exists(DATA_DIR + save_to):
        os.makedirs(DATA_DIR + save_to)

    train_df.to_pickle(DATA_DIR + save_to + "train_set.csv")
    test_df.to_pickle(DATA_DIR + save_to + "test_set.csv")

    print(f"Subset successfully saved to {DATA_DIR + save_to}")


def custom_train_test_split(df, arr=None, test_size=0.2, random_seed=0):
    """splits dataframe into train and test set. A custom function is necessary to ensure that a patient
    only occurs in either of the two."""

    # fix the random seed for reproducibility
    np.random.seed(random_seed)

    n_samples = len(df)
    n_test_samples = int(test_size * n_samples)

    train_df = df.copy().reset_index(drop=True)
    test_df = pd.DataFrame()

    if arr is not None:
        assert len(arr) == len(df), "The passed array must have the same length as the passed dataframe."
        train_arr = arr.copy()
        test_arr = np.array([])

    while len(test_df) < n_test_samples:

        random_patient = np.random.choice(train_df['patient_code'])
        indices, = np.where(train_df['patient_code'] == random_patient)

        test_df = pd.concat([test_df, train_df.loc[indices]])
        train_df = train_df.drop(indices).reset_index(drop=True)

        if arr is not None:
            test_arr = np.append(test_arr, train_arr[indices])
            train_arr = np.delete(train_arr, indices)

    print("Fraction of train samples:", round(len(train_df) / n_samples, 3) * 100, "%")
    print("Fraction of test samples:", round(len(test_df) / n_samples, 3) * 100, "%")
    print("Patients in the train set:", len(np.unique(train_df['patient_code'])))
    print("Patients in the test set:", len(np.unique(test_df['patient_code'])))

    if arr is not None:
        assert len(train_arr) == len(train_df), "There is not a domain index for every training sample."
        return train_df, test_df, train_arr, test_arr
    else:
        return train_df, test_df
