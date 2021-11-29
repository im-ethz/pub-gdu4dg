from os.path import expanduser, isfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import data
from utils.constants import DATA_DIR
from GP_regression import gp_sampling
import torch


def short_term_forecast_dataset(pretrained_model, prenocturnal_length=1, beginning_of_night=22,
                                end_of_night=6, sample_period=1):

    model_path = expanduser("~/pretrained_models/gps/") + pretrained_model + ".pth"
    assert isfile(model_path), f"There is no model like {pretrained_model}"

    df_list = data.load_data(data_dir="/data/wrangled_concatenated_data/", shuffle=True)

    length_of_windows = prenocturnal_length + end_of_night + 24 - beginning_of_night

    windows = []

    for df in tqdm(df_list):
        ls_df = data.split_data(df)
        windows.extend(data.generate_windows(ls_df, window_length=length_of_windows, tolerance=1,
                                             at_time=beginning_of_night-prenocturnal_length))

    pretrained_model = torch.load(model_path)

    df = pd.DataFrame(columns=['datetime', 'glucose_values', 'relative_time', 'sampled_values'])

    for w in tqdm(windows):

        # shift relative time s.t. time is relative to beginning_of_night - prenocturnal_length
        night_start = pd.to_datetime(pd.to_datetime(w.datetime.iloc[0].date())
                                     + pd.to_timedelta(str(beginning_of_night - prenocturnal_length) + "h"))
        time_diff = np.timedelta64(w['datetime'].iloc[0] - night_start).astype('timedelta64[m]').astype(float)
        w['relative_time'] = w.loc[:, 'relative_time'] + time_diff

        sampled_values = gp_sampling(w, pretrained_model, time_to_sample=length_of_windows, period_length=sample_period)

        new_row = {'datetime': w['datetime'].values, 'glucose_values': w['glucose_value'].values,
                   'relative_time': w['relative_time'].values, 'sampled_values': sampled_values}
        df = df.append(new_row, ignore_index=True)

    df.to_pickle(expanduser("~/bg-forecasting/data/") + f"short_term_"
                                                        f"{beginning_of_night - prenocturnal_length}_"
                                                        f"{end_of_night}.csv")


def prepare_short_term_forecast(filename, input_time_length, horizon, beginning_of_night=23,
                                end_of_night=6):
    """
    Splits the data from the file into chunks for short-term prediction. The labels are always part of the
    nocturnal data

    :param filename: (string) the file which contains continuously sampled labels and which is meant to be split
    :param input_time_length: (int) the time length of the input for short term predictions
    :param horizon: (int) the prediction horizon of the model
    :param beginning_of_night: (int) the beginning of the night
    :param end_of_night: (int) the end of the night
    """
    data_dir = expanduser("~/bg-forecasting/data/")
    assert isfile(data_dir + filename), f"The file {filename} does not exist in the data directory {data_dir}"

    df = pd.read_pickle(data_dir + filename)
    print(f"There are {len(df)} nights in the data set.")

    time_beginning_dataset = int(filename.split('_')[-2])
    assert (beginning_of_night-time_beginning_dataset)*60 >= input_time_length

    start_idx = (beginning_of_night - time_beginning_dataset) * 60 - input_time_length
    sample_length = df.iloc[0]['sampled_values'].shape[0]
    chunks_per_sample = (sample_length - start_idx) // (input_time_length + horizon)
    print(f"Each sample will be split into {chunks_per_sample} chunks.")

    st_dataset = pd.DataFrame(columns=['inputs', 'labels'])

    print("Splitting the data samples into chunks.")
    for sample_number in tqdm(range(len(df))):
        for i in range(chunks_per_sample):
            # for the sake of readability
            beginning_of_chunk = start_idx + i * (input_time_length + horizon)
            inputs = np.array(df.iloc[sample_number]['sampled_values']
                              [beginning_of_chunk:beginning_of_chunk + input_time_length])
            labels = np.array(df.iloc[sample_number]['sampled_values']
                              [beginning_of_chunk + input_time_length:
                               beginning_of_chunk + input_time_length + horizon])

            new_row = {'inputs': inputs, 'labels': labels}
            st_dataset = st_dataset.append(new_row, ignore_index=True)

    print(f"There are {len(st_dataset)} training samples in the resulting file.")

    st_dataset.to_pickle(data_dir + f"short_term_i{input_time_length}_h{horizon}.csv")


def create_short_term_dataset(window_length=4, sample_period=5):

    ls_df, filenames = data.load_data(filenames=True)

    windows = []
    patients = []

    print("Splitting the data series into windows.")
    for i in tqdm(range(len(ls_df))):
        continuous_dfs = data.split_data(ls_df[i])
        new_windows = data.generate_windows(continuous_dfs, window_length=window_length)
        windows.extend(new_windows)
        patients.extend([filenames[i].split('.')[0] for _ in range(len(new_windows))])

    pretrained_model = torch.load(expanduser("~/pretrained_models/gps/") + "gp_w12_b8_lr0.01" + ".pth")
    df = pd.DataFrame()

    print("Resampling the data")
    for idx in tqdm(range(len(windows))):
        p_code = patients[idx]
        resampled = np.array(gp_sampling(windows[idx], pretrained_model, time_to_sample=window_length, period_length=sample_period))
        glucose_values = np.array(windows[idx].glucose_value)
        times = np.array(windows[idx].relative_time)
        datetime = np.array(windows[idx].datetime)

        new_row = {'patient_code': p_code, 'resampled_values': resampled, 'glucose_values': glucose_values,
                   'relative_times': times, 'datetime': datetime}

        df = df.append(new_row, ignore_index=True)

    df.to_pickle(DATA_DIR + f"short_term_dataset_{window_length}h.csv")
    print(f"Data set saved to {DATA_DIR}short_term_dataset_{window_length}h.csv")
