import os
import pickle

import numpy as np
from ECG_utils.datareader import DataReader
from ecg_config import Config
from sklearn.model_selection import train_test_split
from tqdm import tqdm

snomed_table = DataReader.read_table(path="tables/")
idx_mapping, label_mapping = DataReader.get_label_maps(path="tables/")


class ECGData(object):
    DOMAINS = ("cpsc", "cpsc-extra", "incart", "ptb", "ptb-xl", "g12ec")
    BASE_PATH = "training_data"
    NUM_CLASSES = 24
    CLASSES = Config.SNOMED_24_ORDERD_LIST

    def __init__(self):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_ecgs()


def load_ecg_data(input_directory):
    # Load data.
    print('Loading data...')

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    num_files = len(header_files)

    print('Extracting features...')

    # Extract features
    features = list()
    labels = list()

    for i in tqdm(range(num_files)):
        sample, y, sample_length, age, sex = load_ecg_file(header_files[i], transform=Config.TRANSFORM_DATA_TRAIN,
                                                           encoder=Config.TRANSFORM_LBL, remap=Config.REMAP)
        features.append(sample)
        labels.append(y)

    return features, labels


def load_ecgs(domains=ECGData.DOMAINS):
    x_train_dict = {}
    y_train_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    for domain in domains:
        print(domain)

        features, labels = load_ecg_data(os.path.join(ECGData.BASE_PATH, domain))
        x_train_dict[domain], x_test_dict[domain], y_train_dict[domain], y_test_dict[domain] = train_test_split(
            features, labels)

        def save(obj, filename):
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)

        save(x_train_dict[domain], os.path.join(ECGData.BASE_PATH, domain + "_train_features_ECG.pkl"))
        save(y_train_dict[domain], os.path.join(ECGData.BASE_PATH, domain + "_train_labels_ECG.pkl"))
        save(x_test_dict[domain], os.path.join(ECGData.BASE_PATH, domain + "_test_features_ECG.pkl"))
        save(y_test_dict[domain], os.path.join(ECGData.BASE_PATH, domain + "_test_labels_ECG.pkl"))

        def load(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)

        x_train_dict[domain] = load(os.path.join(ECGData.BASE_PATH, domain + "_train_features_ECG.pkl"))
        y_train_dict[domain] = load(os.path.join(ECGData.BASE_PATH, domain + "_train_labels_ECG.pkl"))
        x_test_dict[domain] = load(os.path.join(ECGData.BASE_PATH, domain + "_test_features_ECG.pkl"))
        y_test_dict[domain] = load(os.path.join(ECGData.BASE_PATH, domain + "_test_labels_ECG.pkl"))

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


def load_ecg_file(header_file_name, transform=None, encoder=None, remap=None):
    # Read data
    sample_file_name = header_file_name.replace('.hea', '.mat')
    sample = DataReader.read_sample(sample_file_name)
    header = DataReader.read_header(header_file_name, snomed_table, remap=remap)

    sampling_frequency, resolution, age, sex, snomed_codes = header

    # Transform sample
    if transform:
        sample = transform(sample, input_sampling=sampling_frequency, gain=1 / np.array(resolution))

    sample_length = sample.shape[1]

    # Transform label
    if encoder is not None:
        y = encoder(snomed_codes, idx_mapping)

    return sample, y, sample_length, age, sex
