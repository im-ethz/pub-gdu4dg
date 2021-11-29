import os
parallel_procs = "2"
os.environ["OMP_NUM_THREADS"] = parallel_procs
os.environ["MKL_NUM_THREADS"] = parallel_procs
os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs

from os.path import expanduser, isfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import data
from utils.constants import DATA_DIR
from GP_regression import gp_sampling
import torch


def entire_night_forecast_dataset(pretrained_model, prenocturnal_length=12, beginning_of_night=22,
                                  end_of_night=6, sample_period=5):

    model_path = expanduser("~/pretrained_models/gps/") + pretrained_model + ".pth"
    assert isfile(model_path), f"There is no model like {pretrained_model}"

    df_list, filenames = data.load_data("wrangled_concatenated_data/", shuffle=True, filenames=True)

    patients = []
    prenocturnal = []
    nocturnal = []
    print("Splitting files into nocturnal and prenocturnal data")
    for i, df in enumerate(tqdm(df_list)):
        data_day, data_night = data.fetch_nocturnal_data(df, hours_before=prenocturnal_length,
                                                         beginning_of_night=beginning_of_night,
                                                         end_of_night=end_of_night)
        patients.extend([filenames[i].split('.')[0] for _ in range(len(data_day))])
        prenocturnal.extend(data_day)
        nocturnal.extend(data_night)

    pretrained_model = torch.load(model_path)

    # compute mask and length of nocturnal labels so that it does not need to be computed when loading the data
    night_length = (24 - beginning_of_night) + end_of_night
    assert night_length >= 0
    df = pd.DataFrame(columns=['patient_code', 'prenocturnal', 'nocturnal_resampled', 'nocturnal', 'nocturnal_times'])

    # generating data set
    for i in tqdm(range(len(prenocturnal))):
        p_code = patients[i]
        pn = np.array(gp_sampling(prenocturnal[i], pretrained_model, time_to_sample=prenocturnal_length, period_length=sample_period))
        n_resampled = np.array(gp_sampling(nocturnal[i], pretrained_model, time_to_sample=night_length, period_length=sample_period))
        n = np.array(nocturnal[i].glucose_value)
        n_t = np.array(nocturnal[i].relative_time, dtype=int)

        new_row = {'patient_code': p_code, 'nocturnal_resampled': n_resampled, 'prenocturnal': pn, 'nocturnal': n,
                   'nocturnal_times': n_t}
        df = df.append(new_row, ignore_index=True)

    df.to_pickle(DATA_DIR + f"nocturnal_prenocturnal_{prenocturnal_length}h.csv")
    print(f"Saved data set to  {DATA_DIR}nocturnal_prenocturnal_{prenocturnal_length}h.csv")


if __name__ == '__main__':
    entire_night_forecast_dataset("gp_w12_b8_lr0.01", prenocturnal_length=12, sample_period=5)
