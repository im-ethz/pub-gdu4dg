import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.data import load_data


def compute_relative_time(df):
    """
    :arg
        df: (pd.DataFrame) the dataframe for which the relative times should be computed
    :return
        df: (pd.DataFrame) a dataframe with a column of minutes passed since the first timestamp
    """
    first_timestamp = df['datetime'].iloc[0]
    df.loc[:, 'relative_time'] = (pd.to_timedelta(df.datetime - first_timestamp).values / 6e10).astype(float)

    return df


def create_dir_if_necessary(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory {directory}")


def generate_windows(ls_df, window_length=24, tolerance=1, at_time=None):
    """

    :param
        ls_df: (list) the dataframes to cut into windows
        window_length: (int) the time length (h) of a data series, much longer series will be split, much shorter series will be ignored
        tolerance: (int) windows which are shorter than window_length - tolerance will be neglected
        at_time: (int, optional) if set the dataframes will be cut into windows which always start at this time
    :return
        windows_list: (list) list of pd.DataFrames which are windows of desired length
    """

    if not isinstance(ls_df, list):
        ls_df = [ls_df]

    windows_list = []

    for ds in ls_df:

        ds['datetime'] = pd.to_datetime(ds.loc[:, 'datetime'])
        # check into how many the dataframe needs to be cut
        if at_time is not None:
            assert isinstance(at_time, int), "at_time must be of type int"

            if pd.to_datetime(ds.datetime.iloc[0]).hour < at_time:
                first_timestamp = pd.to_datetime(
                    pd.to_datetime(ds.datetime.iloc[0].date()) + pd.to_timedelta(str(at_time) + "h"))
            else:
                first_timestamp = pd.to_datetime(
                    pd.to_datetime(ds.datetime.iloc[0].date() + pd.to_timedelta("1d")) + pd.to_timedelta(
                        str(at_time) + "h"))

            ds.set_index(pd.to_datetime(ds.datetime.values), inplace=True)
            try:
                idx = ds.index.get_loc(first_timestamp, method="backfill")
                n_windows = (ds.relative_time.iloc[-1] - ds.relative_time.iloc[idx]) / (60 * 24)
                ds = compute_relative_time(ds.iloc[idx:])
            except KeyError:
                n_windows = 0

        else:
            n_windows = ds.relative_time.iloc[-1] / (60 * window_length)
            first_timestamp = ds.datetime.iloc[0]

        if n_windows > 1:
            splits = int(n_windows - 1e-6)
            if at_time is not None:
                # as the windows always start at the same time,
                # the time difference between the starting points must be a whole day
                splits_dt = np.asarray(
                    pd.Series(first_timestamp + (i + 1) * pd.to_timedelta("24h") for i in range(splits)))
            else:
                splits_dt = np.asarray(pd.Series(
                    first_timestamp + (i + 1) * pd.to_timedelta(str(window_length) + "h") for i in range(splits)))

            # split the data at the computed times (split_dts)
            # and make sure that the series are not shorter than window_length - tolerance
            # 24 is subtracted from window length, so that it is computed from the starting point,
            # i.e. a day before the first split
            if at_time is not None:
                window_ends_at = pd.to_datetime(pd.to_datetime(splits_dt[0])
                                                + pd.to_timedelta(str(window_length - 24) + "h"))
                w = ds.loc[np.logical_and(ds.datetime <= splits_dt[0], ds.datetime <= window_ends_at)]
            else:
                w = ds.loc[ds.datetime <= splits_dt[0]]

            w = compute_relative_time(w)
            if w.relative_time.iloc[-1] >= (window_length - tolerance) * 60:
                windows_list.append(w)

            for idx in range(len(splits_dt) - 2):
                # splitting at the computed times and reducing the dataframes to the required window length
                window_ends_at = pd.to_datetime(pd.to_datetime(splits_dt[idx])
                                                + pd.to_timedelta(str(window_length) + "h"))
                w = ds.loc[np.logical_and(np.logical_and(ds.datetime > splits_dt[idx],
                                                         ds.datetime <= splits_dt[idx + 1]),
                                          ds.datetime <= window_ends_at)]

                if not w.empty:
                    w = compute_relative_time(w)
                    if w.relative_time.iloc[-1] >= (window_length - tolerance) * 60:
                        windows_list.append(w)

            # the final iteration cannot be done in the loop due to key errors
            window_ends_at = pd.to_datetime(pd.to_datetime(splits_dt[-1])
                                            + pd.to_timedelta(str(window_length) + "h"))
            w = ds.loc[np.logical_and(ds.datetime > splits_dt[-1], ds.datetime <= window_ends_at)]

            w = compute_relative_time(w)
            if w.relative_time.iloc[-1] >= (window_length - tolerance) * 60:
                windows_list.append(w)

        else:
            # else means that the data series is shorter than a whole day
            if ds.relative_time.iloc[-1] >= (window_length - tolerance) * 60:
                windows_list.append(compute_relative_time(ds))

    return windows_list


def fetch_nocturnal_data(df, hours_before=16, beginning_of_night=22, end_of_night=6):
    """
    only for quick testing and debugging for now
    """

    ls_df = split_data(df)
    night_starts_at = beginning_of_night
    night_ends_at = end_of_night
    day_starts_at = night_starts_at - hours_before
    sample_length = hours_before + night_ends_at + 24 - night_starts_at

    day_start_str = str(day_starts_at) + ":00"
    night_start_str = str(night_starts_at) + ":00"
    night_end_str = str(night_ends_at) + ":00"

    # cut data series into separate days
    list_of_days = []
    if hours_before == 16:
        for ds in ls_df:
            list_of_days.extend(generate_windows(ds, window_length=24, tolerance=1, at_time=night_ends_at))
    else:
        for ds in ls_df:
            ds.loc[:, 'datetime'] = pd.to_datetime(ds.datetime)
            ds = ds.set_index(ds.datetime)
            ds = ds.between_time(day_start_str, night_end_str)
            if len(ds) > 1:
                pd.set_option('mode.chained_assignment', None)
                sub_list = split_data(ds, max_gap=day_starts_at - night_ends_at)
                for sl in sub_list:
                    list_of_days.extend(generate_windows(sl, window_length=sample_length, at_time=day_starts_at))

    days = []
    nights = []

    for day in list_of_days:

        day.set_index(pd.to_datetime(day.datetime.values), inplace=True)
        date = day.datetime.iloc[0].date()
        night_start = pd.to_datetime(pd.to_datetime(date) + pd.to_timedelta(str(night_starts_at) + "h"))

        prenocturnal_df = day.between_time(str(night_starts_at - hours_before) + ":00", str(night_starts_at) + ":00",
                                           include_end=False)
        nocturnal_df = day.between_time(str(night_starts_at) + ":00", str(night_ends_at) + ":00", include_end=False)

        if not prenocturnal_df.empty:
            time_diff = np.timedelta64(nocturnal_df['datetime'].iloc[0] - night_start).astype('timedelta64[m]').astype(
                float)
            nocturnal_df = compute_relative_time(nocturnal_df)
            nocturnal_df['relative_time'] = nocturnal_df.loc[:, 'relative_time'] + time_diff

            days.append(compute_relative_time(prenocturnal_df))
            nights.append(nocturnal_df)

    return days, nights


def _series_of_relevant_length(series, length, tolerance):
    length_of_series = (series.relative_time.iloc[-1] - series.relative_time.iloc[0]) / 60
    relevant = length_of_series >= (length - tolerance)
    return relevant


def split_data(df, max_gap=3):
    """
    splits data series into multiple shorter but continuous series if there are gaps that are bigger than max_gap
    """

    def int_difference(a, b):
        return int(a - b)

    v_diff = np.vectorize(int_difference)

    pd.set_option('mode.chained_assignment', None)
    periods = v_diff(df.relative_time[1:], df.relative_time[:-1])
    needs_split = (True if np.max(periods) > max_gap * 60 else False)
    ls_df = []
    df.loc[:, "datetime"] = pd.to_datetime(df.datetime)

    # split df where the periods are bigger than max_gap
    if not needs_split:
        ls_df.append(df)
    else:
        where_to_split = np.where(periods > max_gap * 60)[0] + 1
        ls_df.append(df.iloc[:where_to_split[0]])
        for idx in range(len(where_to_split) - 1):
            ls_df.append((df.iloc[where_to_split[idx]:where_to_split[idx + 1]]).reset_index(drop=True))
        ls_df.append(df.iloc[where_to_split[-1]:])

    ls_df = [compute_relative_time(d) for d in ls_df if not len(d) <= 1]

    return ls_df


def wrangle_data(characteristics, cgm_data):
    """
    - standardises the cgm data and translates patient's characteristics
    - the format of the dataframe is different for each type of cgm sensor
    """
    patient = {'age': int(characteristics['age_at visit (y)'].values),
               'HbA1c': float(characteristics['HbA1c_at visit (%)'].values)}
    gender_code = int(characteristics['Geschlecht (1=m, 2=f)'].values)
    device_code = int(characteristics['device (1=medtronic, 2=dexcom, 3=libre)'].values)

    if gender_code == 1:
        patient['gender'] = 'male'
    elif gender_code == 2:
        patient['gender'] = 'female'
    else:
        patient['gender'] = 'unknown'

    if device_code == 1:
        patient['device'] = 'medtronic'
        cgm_data[['date_only', 'rubbish']] = cgm_data['Date'].astype(str).str.split(' ').tolist()
        cgm_data['Time'] = cgm_data['Time'].astype(str)
        mask = (cgm_data['Time'].str.len() == 8)
        cgm_data = cgm_data.loc[mask]
        cgm_data.loc[:, 'datetime'] = pd.to_datetime(cgm_data['date_only'] + " " + cgm_data['Time'], dayfirst=True)
        cgm_data = cgm_data[['datetime', 'Sensor Glucose (mmol/L)']]
        cgm_data.columns = ['datetime', 'glucose_value']
        cgm_data['glucose_value'] = cgm_data['glucose_value'].astype('float64')

    elif device_code == 2:
        patient['device'] = 'dexcom'
        cgm_data = cgm_data[['GlucoseDisplayTime', 'GlucoseValue']]
        cgm_data.loc[:, 'GlucoseDisplayTime'] = pd.to_datetime(cgm_data['GlucoseDisplayTime'], dayfirst=True)
        cgm_data = cgm_data.replace({'Hoch': None, 'Niedrig': None})
        cgm_data.loc[:, 'GlucoseValue'] = cgm_data['GlucoseValue'].astype('float64')
        cgm_data.columns = ['datetime', 'glucose_value']

    elif device_code == 3:
        patient['device'] = 'libre'
        cgm_data[['date_only', 'rubbish']] = cgm_data['Date'].astype(str).str.split(' ').tolist()
        cgm_data['Time'] = cgm_data['Time'].astype(str)
        mask = (cgm_data['Time'].str.len() == 8)
        cgm_data = cgm_data.loc[mask]
        cgm_data.loc[:, 'datetime'] = pd.to_datetime(cgm_data['date_only'] + " " + cgm_data['Time'], dayfirst=True)
        cgm_data.loc[:, 'glucose_value'] = cgm_data['Value'] / 18
        cgm_data = cgm_data[['datetime', 'glucose_value']]

    else:
        patient['device'] = 'unknown'

    cgm_data = cgm_data.sort_values(by='datetime').reset_index(drop=True)

    return patient, cgm_data


def clean_and_concatenate():
    """generates a single data series for each patient"""
    path_from = os.path.expanduser("~") + "/data/wrangled_data/"
    path_to = os.path.expanduser("~") + "/data/wrangled_concatenated_data/"
    patient_codes = [(i.split(".")[0]).split("_")[:-1] for i in os.listdir(path_from)]
    patient_codes = ["_".join(j) for j in patient_codes]
    unique_patient_codes = []
    for p in patient_codes:
        if p not in unique_patient_codes:
            unique_patient_codes.append(p)

    for patient in tqdm(unique_patient_codes):
        files = glob(path_from + patient + "*")
        ls_df = [pd.read_csv(f, usecols=["datetime", "glucose_value"]) for f in files]
        df = pd.concat(ls_df, ignore_index=True)
        df.loc[:, "datetime"] = pd.to_datetime(df.datetime)
        df = df.sort_values(by='datetime').reset_index(drop=True)
        first_timestamp = pd.to_datetime(df['datetime'].iloc[0])
        df.loc[:, 'relative_time'] = (pd.to_timedelta(df.datetime - first_timestamp).values / 6e10).astype(float)
        df.to_csv(path_to + patient + ".csv", index=False)


def clean_only():
    data_dir = os.path.expanduser("~/bg-forecasting/data/")

    load_from = data_dir + "raw_data/"
    save_to = data_dir + "wrangled_data/"
    create_dir_if_necessary(save_to)

    dirty_files = list(map(lambda s: s.split('.')[0], os.listdir(load_from)))
    clean_files = list(map(lambda s: s.split('.')[0], os.listdir(save_to)))
    remaining_files = [item for item in dirty_files if item not in clean_files]
    remaining_files = list(map(lambda s: s + ".xlsx", remaining_files))
    patients = pd.read_excel(data_dir + '/patients.xlsx', engine='openpyxl')

    pd.set_option('mode.chained_assignment', None)
    print("Cleaning files...")
    for file in tqdm(remaining_files):
        patient_code = file.split('.')[0]
        characteristics = patients.loc[patients['Code (med=medtronic; dex=dexcom; fs=freestyle libre)'] == patient_code]
        df = pd.read_excel(load_from + file, engine='openpyxl')
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        patient, df = wrangle_data(characteristics, df)
        df = compute_relative_time(df)
        df.to_csv(save_to + patient_code + ".csv", index=False)


def make_readable(data):
    readable_dict = {'code': str(data['Code (med=medtronic; dex=dexcom; fs=freestyle libre)'].values)[2:-2],
                     'date': pd.to_datetime(data['date visit'].values[0]),
                     'age': int(data['age_at visit (y)'].values),
                     'HbA1c (%)': float(data['HbA1c_at visit (%)'].values)}

    gender = int(data['Geschlecht (1=m, 2=f)'].values)
    if gender == 1:
        readable_dict['gender'] = 'm'
    elif gender == 2:
        readable_dict['gender'] = 'f'
    else:
        readable_dict['gender'] = 'd'

    diabetes_type = int(data['Type Diabetes (1=DM1, 2=DM2, 3=GDM, 4=MODY, 5=other)'].values)
    if diabetes_type == 1:
        readable_dict['diabetes_type'] = 'DM1'
    elif diabetes_type == 2:
        readable_dict['diabetes_type'] = 'DM2'
    elif diabetes_type == 3:
        readable_dict['diabetes_type'] = 'GDM'
    elif diabetes_type == 4:
        readable_dict['diabetes_type'] = 'MODY'
    else:
        readable_dict['diabetes_type'] = 'other'

    therapy = int(data[
                      'Therapy (1=MDI, 2=CSII, 3=no insulin antidiabetics [nia], 4=basal insulin, 5=insulin & nia, 6=other)'].values)
    if therapy == 1:
        readable_dict['therapy'] = 'MDI'
    elif therapy == 2:
        readable_dict['therapy'] = 'CSII'
    elif therapy == 3:
        readable_dict['therapy'] = 'no insulin antidiabetics [nia]'
    elif therapy == 4:
        readable_dict['therapy'] = 'basal insulin'
    elif therapy == 5:
        readable_dict['therapy'] = 'insulin & nia'
    else:
        readable_dict['therapy'] = 'other'

    return readable_dict


def remove_duplicate_timestamps():
    """DEPRECATED"""
    raise DeprecationWarning

    path = "/data/wrangled_concatenated_data/"
    ls_data, filenames = load_data(data_dir=path, shuffle=False, filenames=True)
    for i in tqdm(range(len(ls_data))):
        df = ls_data[i]
        df = df.groupby('datetime', as_index=False).mean()
        assert df.datetime.is_unique, f"A problem with {filenames[i]} occurred."
        df = df.sort_values(by='datetime').reset_index(drop=True)
        df.to_csv(os.path.expanduser("~") + path + filenames[i], index=False)


def reduce_df(df, samples_per_patient=10, random_seed=None):

    if samples_per_patient > 10:
        print("There are only 10 samples for each patient in the data set. \n"
              "Therefore, sampling from the same row more than once is allowed "
              "for patients with fewer samples.")

    reduced_df = pd.DataFrame()
    patients = np.unique(df['patient_code'].values)

    for p in tqdm(patients):
        sub_df = df.loc[df['patient_code'] == p]
        try:
            reduced_df = pd.concat([reduced_df, sub_df.sample(n=samples_per_patient, random_state=random_seed)])
        except ValueError:
            reduced_df = pd.concat(
                [reduced_df, sub_df.sample(n=samples_per_patient, replace=True, random_state=random_seed)])

    return reduced_df.reset_index(drop=True)
