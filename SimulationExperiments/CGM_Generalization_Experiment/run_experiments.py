import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import os
import sys
import keras
import logging
import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd

DATA_DIR = "/wave/bg-prediction/data/"
import random

import tensorflow as tf

tf.random.set_seed(1234)

import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import _SumKernel

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

from Model.utils import decode_one_hot_vector
from Visualization.evaluation_plots import plot_TSNE

from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from Model.DomainAdaptation.domain_adaptation_callback import DomainCallback



def init_gpu(gpu, memory):
    used_gpu = gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[used_gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[used_gpu], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)


init_gpu(gpu=0, memory=6000)

# file path to the location where the results are stored

res_file_dir = None
SOURCE_SAMPLE_SIZE = None
TARGET_SAMPLE_SIZE = None


def digits_classification(method, TARGET_DOMAIN=None, single_best=False, single_source_domain=None,
                          task='hypo-classification',
                          batch_norm=False,
                          lr=0.001,
                          save_file=False, save_plot=False, save_feature=False,
                          activation="tanh",
                          bias=False,
                          fine_tune=False,
                          kernel=None,
                          run=None):
    domain_adaptation_spec_dict = {
        "num_domains": 5,
        "domain_dim": 10,
        "sigma": 7.5,
        'softness_param': 2,
        "similarity_measure": method,  # MMD, IPS
        "lambda_orth": 1e-8,
        "bias": bias,
        "source_sample_size": SOURCE_SAMPLE_SIZE,
        "target_sample_size": TARGET_SAMPLE_SIZE
    }

    # architecture used as feature extractor
    architecture = domain_adaptation_spec_dict["architecture"] = "lenet"  # "DomainNet"# "LeNet"

    domain_adaptation_spec_dict["kernel"] = "custom" if kernel is not None else "single"

    # used in case of "projected"
    domain_adaptation_spec_dict["orth_reg"] = reg = "SRIP"
    domain_adaptation_spec_dict['reg_method'] = orth_reg_method = reg if method == 'projected' else 'none'

    # training specification
    use_optim = domain_adaptation_spec_dict['use_optim'] = 'adam'  # "SGD"
    optimizer = tf.keras.optimizers.SGD(lr) if use_optim.lower() == "sgd" else tf.keras.optimizers.Adam(lr)

    batch_size = domain_adaptation_spec_dict['batch_size'] = 512
    domain_adaptation_spec_dict['epochs'] = num_epochs = 2
    domain_adaptation_spec_dict['epochs_FT'] = num_epochs_FT = 250
    domain_adaptation_spec_dict['lr'] = lr
    domain_adaptation_spec_dict['dropout'] = dropout = 0.5
    domain_adaptation_spec_dict['patience'] = patience = 10

    # network specification
    domain_adaptation_spec_dict['batch_normalization'] = batch_norm
    from_logits = False if activation == "softmax" else True

    ##########################################
    ###     PREPARE DATA
    ##########################################

    # define domains based on sensor, type of diabetes, and therapy
    # load data with domain as dict key
    # train and validation set with training domains
    # test data with test domains

    # options for first argument are regression, hypo-classification, and hyper-classification
    X_dict, Y_dict = load_data(task=task,
                               reduce_data=True,
                               samples_per_patient=10,
                               input_length=60,
                               horizon=60,
                               random_seed=None)

    train_domains = [k for k in X_dict.keys()]
    test_domain = random.choice(train_domains)
    train_domains.remove(test_domain)
    print(f"{test_domain} is the test domain.")

    X_train = np.concatenate([np.stack(X_dict[d]) for d in train_domains], axis=0)
    Y_train = np.concatenate([np.stack(Y_dict[d]) for d in train_domains], axis=0)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    X_test = np.stack(X_dict[test_domain])
    Y_test = np.stack(Y_dict[test_domain])

    ##########################################
    ###     FEATURE EXTRACTOR
    ##########################################
    if architecture.lower() == "lenet":
        feature_extractor = get_lenet_feature_extractor()

    else:
        architecture = "lstm"
        feature_extractor = get_lstm_feature_extractor()



    ##########################################
    ###     PREDICTION LAYER
    ##########################################
    prediction_layer = tf.keras.Sequential([], name='prediction_layer')
    domain_adaptation = True if method is not None else False
    if domain_adaptation:

        if task == 'regression':
            # prediction of an hour with sampling period length with 5min
            units = 12
        else:
            # binary label prediction
            units = 1

        num_domains = domain_adaptation_spec_dict['num_domains']
        sigma = domain_adaptation_spec_dict['sigma']
        domain_dim = domain_adaptation_spec_dict['domain_dim']
        similarity_measure = domain_adaptation_spec_dict["similarity_measure"]
        softness_param = domain_adaptation_spec_dict["softness_param"]
        prediction_layer.add(DGLayer(domain_units=num_domains,
                                     N=domain_dim,
                                     softness_param=softness_param,
                                     units=units,
                                     kernel=kernel,
                                     sigma=sigma,
                                     activation=activation,
                                     bias=bias,
                                     similarity_measure=similarity_measure,
                                     orth_reg_method=orth_reg_method,
                                     ))

    else:
        method = "SOURCE_ONLY"
        if task == 'regression':
            # prediction of an hour with sampling period length with 5min
            prediction_layer.add(Dense(12))
        else:
            # binary label prediction
            prediction_layer.add(Dense(1))

    callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
    domain_callback = DomainCallback(test_data=X_test, train_data=X_train, print_res=True,
                                     max_sample_size=5000)
    ##########################################
    ###     INITIALIZE MODEL
    ##########################################
    # DomainAdaptationModel has one feature_extractor (that may be used in the fine tune stage) and one prediction layer
    model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

    # currently there is an error due to incompatible tf, numpy, and python versions with the 'tf' environment
    model.build(input_shape=(X_train.shape[0], X_train.shape[1], 1))
    model.feature_extractor.summary()
    model.prediction_layer.summary()

    if task == 'regression':
        metrics = [tf.keras.metrics.MeanAbsoluteError()]
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=metrics,
        )
    else:
        metrics = [tf.keras.metrics.BinaryAccuracy(),
                   tf.keras.metrics.BinaryCrossentropy(from_logits=True)]
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=metrics,
        )

    run_start = datetime.now()

    hist = model.fit(x=X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y=Y_train, epochs=num_epochs, verbose=2,
                     batch_size=batch_size, shuffle=False,
                     validation_data=(X_val.reshape(X_val.shape[0], X_val.shape[1], 1), Y_val),
                     callbacks=[callback, domain_callback] if domain_adaptation else callback,
                     )
    run_end = datetime.now()

    # model evaluation
    model_res = model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), Y_test, verbose=0)
    metric_names = model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names
    print(eval_df)

    if save_plot or save_file:
        run_id = np.random.randint(0, 10000, 1)[0]
        save_dir_path = os.path.join(res_file_dir, "run_" + str(run))
        create_dir_if_not_exists(save_dir_path)
        save_dir_path = os.path.join(save_dir_path, "SINGLE_BEST") if single_best else os.path.join(save_dir_path,
                                                                                                    "SOURCE_COMBINED")

        create_dir_if_not_exists(save_dir_path)

        save_dir_path = os.path.join(save_dir_path, TARGET_DOMAIN[0])
        create_dir_if_not_exists(save_dir_path)

        if single_best:
            save_dir_name = method.upper() + "_" + train_domains[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(run_id)
        else:
            save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

        save_dir_path = os.path.join(save_dir_path, save_dir_name)
        create_dir_if_not_exists(save_dir_path)

    if save_plot or save_feature:
        X_DATA = model.predict(X_test)
        Y_DATA = decode_one_hot_vector(Y_test)

        if save_feature:
            df_file_path = os.path.join(save_dir_path, method.upper() + "_feature_data.csv")
            pred_df = pd.DataFrame(X_DATA, columns=["x_{}".format(i) for i in range(10)])
            pred_df['label'] = Y_DATA
            pred_df.to_csv(df_file_path)

        if save_plot:
            file_name = "TSNE_PLOT_" + method.upper() + ".png"
            tsne_file_path = os.path.join(save_dir_path, file_name)
            plot_TSNE(X_DATA, Y_DATA, plot_kde=False, file_path=tsne_file_path, show_plot=False)

    if save_file:
        hist_df = pd.DataFrame(hist.history)
        duration = run_end - run_start

        file_name_hist = 'history_' + method.upper() + '.csv'
        hist_file_path = os.path.join(save_dir_path, file_name_hist)
        hist_df.to_csv(hist_file_path)

        model_res = model.evaluate(X_test, Y_test, verbose=2)
        metric_names = model.metrics_names
        eval_df = pd.DataFrame(model_res).transpose()
        eval_df.columns = metric_names

        test_sources = ",".join(test_domain)
        train_sources = ",".join(train_domains)

        eval_df['source_domain'] = train_sources
        eval_df['target_domain'] = test_sources

        # rund specifications
        domain_adaptation_parameter_names = list(domain_adaptation_spec_dict.keys())
        domain_adaptation_parameters_df = pd.DataFrame(domain_adaptation_spec_dict.values()).transpose()
        domain_adaptation_parameters_df.columns = domain_adaptation_parameter_names

        eval_df = pd.concat([eval_df, domain_adaptation_parameters_df], axis=1)
        eval_df['duration'] = duration
        eval_df['run_id'] = run_id
        eval_df['trained_epochs'] = len(hist_df)

        file_name_eval = 'spec_' + method.upper() + '.csv'
        eval_file_path = os.path.join(save_dir_path, file_name_eval)
        eval_df.to_csv(eval_file_path)

    ##########################################
    #               FINE TUNE                #
    ##########################################

    if domain_adaptation is False and fine_tune:

        feature_extractor_filepath = os.path.join(save_dir_path, 'feature_extractor.h5.tmp')
        feature_extractor.save(feature_extractor_filepath)

        for similarity_measure in ['cosine_similarity', 'mmd', 'projected']:

            prediction_layer = tf.keras.Sequential([], name='prediction_layer')

            num_domains = domain_adaptation_spec_dict['num_domains']

            feature_extractor = keras.models.load_model(feature_extractor_filepath)
            feature_extractor.trainable = False

            # sigma is estimated based on the median heuristic, with sample size of 5000 features
            # sigma = domain_adaptation_spec_dict['sigma']
            sigma = domain_adaptation_spec_dict['sigma'] = sigma_median(feature_extractor.predict(X_train))
            print("\n\n\n ESTIMATED SIGMA: {sigma} ".format(sigma=str(np.round(sigma, 3))))

            #######################################
            ###     PREDICTION LAYER
            #######################################
            domain_dim = domain_adaptation_spec_dict['domain_dim']
            domain_adaptation_spec_dict["similarity_measure"] = similarity_measure
            softness_param = domain_adaptation_spec_dict["softness_param"]
            domain_adaptation_spec_dict['reg_method'] = orth_reg_method = reg if method == 'projected' else 'none'

            prediction_layer.add(DGLayer(domain_units=num_domains,
                                         N=domain_dim,
                                         softness_param=softness_param,
                                         units=units,
                                         kernel=kernel,
                                         sigma=sigma,
                                         activation=activation,
                                         bias=bias,
                                         similarity_measure=similarity_measure,
                                         orth_reg_method=orth_reg_method,
                                         ))

            model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

            model.build(input_shape=X_train.shape)
            model.feature_extractor.summary()
            model.prediction_layer.summary()

            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                          metrics=metrics)

            callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
            domain_callback = DomainCallback(test_data=X_val, train_data=X_train, print_res=True,
                                             max_sample_size=5000)

            hist = model.fit(x=X_train, y=Y_train.astype(np.float32), epochs=num_epochs_FT, verbose=2,
                             batch_size=batch_size, shuffle=False, validation_data=(X_val, Y_val),
                             callbacks=[callback, domain_callback]
                             )
            model.evaluate(X_test, Y_test, verbose=2)

            if save_plot or save_file:
                run_id = np.random.randint(0, 10000, 1)[0]
                save_dir_path = os.path.join(res_file_dir, "run_" + str(run))
                create_dir_if_not_exists(save_dir_path)
                save_dir_path = os.path.join(save_dir_path, "SINGLE_BEST") if single_best else os.path.join(
                    save_dir_path,
                    "SOURCE_COMBINED")
                create_dir_if_not_exists(save_dir_path)

                save_dir_path = os.path.join(save_dir_path, TARGET_DOMAIN[0])
                create_dir_if_not_exists(save_dir_path)

                if single_best:
                    save_dir_name = method.upper() + "_" + train_domains[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(
                        run_id)
                else:
                    save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

                save_dir_path = os.path.join(save_dir_path, save_dir_name)
                create_dir_if_not_exists(save_dir_path)

            if save_plot or save_file:
                X_DATA = model.predict(X_test)
                Y_DATA = decode_one_hot_vector(Y_test)

                if save_feature:
                    df_file_path = os.path.join(save_dir_path, method.upper() + "_FT_feature_data.csv")
                    pred_df = pd.DataFrame(X_DATA, columns=["x_{}".format(i) for i in range(10)])
                    pred_df['label'] = Y_DATA
                    pred_df.to_csv(df_file_path)

                if save_plot:
                    file_name = "TSNE_PLOT_" + method.upper() + "_FT" + ".png"
                    tsne_file_path = os.path.join(save_dir_path, file_name)
                    plot_TSNE(X_DATA, Y_DATA, plot_kde=False, file_path=tsne_file_path, show_plot=False)

            if save_file:
                hist_df = pd.DataFrame(hist.history)
                duration = run_end - run_start

                file_name_hist = 'history_' + method.upper() + "_FT" + '.csv'
                hist_file_path = os.path.join(save_dir_path, file_name_hist)
                hist_df.to_csv(hist_file_path)

                # prepare results
                model_res = model.evaluate(X_test, Y_test, verbose=2)
                metric_names = model.metrics_names
                eval_df = pd.DataFrame(model_res).transpose()
                eval_df.columns = metric_names

                test_sources = ",".join(TARGET_DOMAIN)
                train_sources = ",".join(train_domains)

                eval_df['source_domain'] = train_sources
                eval_df['target_domain'] = test_sources

                # run specifications
                domain_adaptation_parameter_names = list(domain_adaptation_spec_dict.keys())
                domain_adaptation_parameters_df = pd.DataFrame(domain_adaptation_spec_dict.values()).transpose()
                domain_adaptation_parameters_df.columns = domain_adaptation_parameter_names

                eval_df = pd.concat([eval_df, domain_adaptation_parameters_df], axis=1)
                eval_df['duration'] = duration
                eval_df['run_id'] = run_id
                eval_df['trained_epochs'] = len(hist_df)

                file_name_eval = 'spec_' + method.upper() + "_FT" + '.csv'
                eval_file_path = os.path.join(save_dir_path, file_name_eval)
                eval_df.to_csv(eval_file_path)

        os.remove(feature_extractor_filepath)

    tf.keras.backend.clear_session()
    return None


def MMD(x1, x2, kernel):
    return np.mean(kernel.matrix(x1, x1)) - 2 * np.mean(kernel.matrix(x1, x2)) + np.mean(kernel.matrix(x2, x2))


def get_mmd_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1
    mmd_matrix = np.zeros((num_ds, num_ds))
    for i in range(num_ds):
        x_i = x_data[i]
        for j in range(i, num_ds):
            x_j = x_data[j]
            mmd_matrix[i, j] = mmd_matrix[j, i] = MMD(x_i, x_j, kernel=kernel)
    return mmd_matrix


def sigma_median(x_data, sample_size=5000):
    x_data = x_data[:sample_size]
    sigma_median = np.median(euclidean_distances(x_data, x_data))
    return sigma_median


def get_lstm_feature_extractor():
    feature_extractor = tf.keras.Sequential(
        [LSTM(512), Dense(512, activation='relu'), Dense(256, activation='relu')],
        name='feature_extractor')
    return feature_extractor


def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential(
        [Conv1D(32, kernel_size=3, activation='relu'), BatchNormalization(),
         MaxPool1D(pool_size=2, strides=2), Conv1D(64, kernel_size=2, activation='relu'), BatchNormalization(),
         GlobalAveragePooling1D(), Dense(100, activation="relu"), Dense(100, activation="relu")],
        name='feature_extractor')
    return feature_exctractor


def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.05)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))


def assign_patients_to_domains(df, domains):
    print("Assigning patients to domains...")
    X_dict = {}
    Y_dict = {}
    domains_patients = []

    for d in domains:
        criteria = d.split('_')
        domains_patients.append(set(filter_patients(
            return_list=True, therapy=criteria[0], diabetes_type=criteria[1], sensor=criteria[2])))

    for idx, row in df.iterrows():
        for idy, patients_in_domain in enumerate(domains_patients):
            if row['patient_code'] in patients_in_domain:
                try:
                    cur_X = X_dict[domains[idy]]
                    cur_Y = Y_dict[domains[idy]]
                    X_dict[domains[idy]] = np.concatenate([cur_X, row['inputs'].reshape(1, -1)], axis=0)
                    Y_dict[domains[idy]] = np.concatenate([cur_Y, np.array(row['labels']).reshape(1, -1)], axis=0)
                except KeyError:
                    X_dict[domains[idy]] = row['inputs'].reshape(1, -1)
                    Y_dict[domains[idy]] = np.array(row['labels']).reshape(1, -1)

                break

    return X_dict, Y_dict


def load_data(task, reduce_data=False, samples_per_patient=None, input_length=60, horizon=60, random_seed=None):
    domains = ['CSII_DM1_DIA', 'CSII_DM1_dex', 'CSII_DM1_med', 'MDI_DM1_DIA',
               'MDI_DM1_dex', 'MDI_DM2_DIA', 'MDI_other_DIA',
               'insulin & nia_DM2_DIA']

    if reduce_data:
        assert samples_per_patient is not None, "You must pass samples_per_patient if reduce_data is true!"

    if task == 'regression':
        if input_length is None or horizon is None:
            raise ValueError("You must pass both an input_length and a horizon (in minutes)"
                             "when performing regression!")
        if input_length + horizon > 4 * 60:
            raise ValueError("The sum of input_length and horizon must not be greater than 240 min.")
        df = pd.read_pickle(DATA_DIR + "short_term_dataset_4h.csv")

        if reduce_data:
            print(f"Reducing the data set to {samples_per_patient} samples per patient...")
            df = reduce_df(df, samples_per_patient=samples_per_patient, random_seed=random_seed)
        # train_df, test_df = custom_train_test_split(df, test_size=test_size, random_seed=None)

        points_input = int(input_length / 5)  # 5 is the sampling period length
        points_prediction = int(horizon / 5)

        df = pd.DataFrame(data={'patient_code': df['patient_code'].values,
                                'inputs': pd.Series(df['resampled_values'].values).apply(
                                    lambda arr: arr[-(points_input + points_prediction):-points_prediction]),
                                'labels': pd.Series(df['resampled_values'].values).apply(
                                    lambda arr: arr[-points_prediction:])})

        X_dict, Y_dict = assign_patients_to_domains(df, domains)

    elif task == 'hypo-classification' or task == 'hyper-classification':
        df = pd.read_pickle(DATA_DIR + 'nocturnal_prenocturnal_12h.csv')

        if reduce_data:
            print(f"Reducing the data set to {samples_per_patient} samples per patient...")
            df = reduce_df(df, samples_per_patient=samples_per_patient, random_seed=random_seed)
        # train_df, test_df = custom_train_test_split(df, test_size=test_size, random_seed=None)

        if task == 'hypo-classification':
            labels = df['hypoglycemia'].values
        else:
            labels = df['hyperglycemia'].values

        df = pd.DataFrame(data={'patient_code': df['patient_code'].values,
                                'inputs': pd.Series(df['prenocturnal'].values),
                                'labels': labels})

        X_dict, Y_dict = assign_patients_to_domains(df, domains)

    else:
        raise NotImplementedError("The available tasks are: regression, hypo-classification, and hyper-classification!")

    return X_dict, Y_dict


def get_kernel_sum(sigma_list):
    amplitude_list = [1]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in
               sigma_list for amplitude in amplitude_list] + \
              [tfp.math.psd_kernels.MaternFiveHalves(length_scale=sigma, amplitude=amplitude) for sigma in sigma_list
               for amplitude in amplitude_list] + \
              [tfp.math.psd_kernels.RationalQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in sigma_list
               for amplitude in amplitude_list]
    kernel_sum = _SumKernel(kernels=kernels)
    return kernel_sum


def run_experiment(experiment, gpu=None):
    if gpu is not None:
        init_gpu(gpu)

    #try:
    digits_classification(**experiment)
    # except Exception:
    #  import traceback
     #   traceback.print_exc()
      #  pass

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

    filtered_df = df.drop(
        df.index[df[code_column].apply(lambda s: "_".join(s.split("_")[:3]) not in relevant_patients)])

    return filtered_df


GPUS = [2]
# GPUS = [2]

if __name__ == "__main__":

    # for i in range(5):
    for i in [0]:
        experiments = []
        for method in ['cosine_similarity', 'MMD', 'projected']:
            for kernel in [None]:
                for batch_norm in [True]:  # , False]:
                    for bias in [False]:  # , True]:
                        # for single_source in [['mnistm'], ['mnist'], ['svhn'], ['syn'], ['usps']]:
                        experiments.append({
                            'method': method,
                            'kernel': kernel,
                            'batch_norm': batch_norm,
                            'bias': bias,
                            # 'single_source_domain': single_source,
                            'run': i
                        })

        print(f'Running {len(experiments)} experiments on {len(GPUS)} GPUs')

        if len(GPUS) > 1:
            init_gpu(GPUS)
            pool = multiprocessing.pool.Pool(processes=min(len(experiments), len(GPUS)))
            for i, experiment in enumerate(experiments):
                pool.apply_async(run_experiment, (experiment, None))
            pool.close()
            pool.join()
        else:
            for experiment in experiments:
                run_experiment(experiment)
