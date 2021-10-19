import argparse
import logging
import os
import sys
import warnings
import shutil
from datetime import datetime

# import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
# from silence_tensorflow import silence_tensorflow
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import *
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import _SumKernel

from DomainAdaptationModel import DomainAdaptationModel
from domain_adaptation_callback import DomainCallback
from domain_adaptation_layer import DGLayer
from utils import decode_one_hot_vector
from d5_dataloader import load_digits

# from evaluation_plots import plot_TSNE

warnings.filterwarnings("ignore", category=DeprecationWarning)
# silence_tensorflow()

tf.random.set_seed(1234)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

'''
nohup /local/home/sfoell/anaconda3/envs/gdu4dg/bin/python3.8 -u /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiment_4_digits/digits_5_classification.py > /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiment_4_digits/orth_000.log 2>&1 &
'''

# file path to the location where the results are stored

res_file_dir = "/cluster/home/alinadu/GDU/pub-gdu4dg/results/ensemble/"
SOURCE_SAMPLE_SIZE = 25000
TARGET_SAMPLE_SIZE = 9000
img_shape = (32, 32, 3)


# load data once in the program and keep in class
class DigitsData(object):
    DOMAINS = ('mnistm', 'mnist', 'syn', 'svhn', 'usps')
    BASE_PATH = "/cluster/home/alinadu/GDU/pub-gdu4dg/datasets/"
    NUM_CLASSES = 10

    # CLASSES = Config.SNOMED_24_ORDERD_LIST

    def __init__(self, test_size=SOURCE_SAMPLE_SIZE):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_digits(test_size=test_size,
                                                                                               base_path=DigitsData.BASE_PATH)


def digits_classification(method, TARGET_DOMAIN, single_best=False, single_source_domain=None, batch_norm=False,
                          lr=0.001, save_file=True, save_plot=False, save_feature=False, activation="tanh",
                          lambda_sparse=1e-3,  # 1e-1,
                          lambda_OLS=1e-3,  # 1e-1,
                          lambda_orth=1e-3,  # 1e-1,
                          early_stopping=True, bias=False, fine_tune=True, kernel=None, data: DigitsData = None,
                          run=None):
    domain_adaptation_spec_dict = {"num_domains": 5, "domain_dim": 10, "sigma": 7.5, 'softness_param': 2,
                                   "similarity_measure": method,  # MMD, IPS
                                   "img_shape": img_shape, "bias": bias, "source_sample_size": SOURCE_SAMPLE_SIZE,
                                   "target_sample_size": TARGET_SAMPLE_SIZE}

    # architecture used as feature extractor
    architecture = domain_adaptation_spec_dict["architecture"] = "DomainNet"  # "DomainNet"# "LeNet"

    domain_adaptation_spec_dict["kernel"] = "custom" if kernel is not None else "single"

    # specification of regularization
    domain_adaptation_spec_dict["lambda_sparse"] = lambda_sparse
    domain_adaptation_spec_dict["lambda_OLS"] = lambda_OLS
    domain_adaptation_spec_dict["lambda_orth"] = lambda_orth

    # used in case of "projected"
    domain_adaptation_spec_dict["orth_reg"] = reg = "SRIP"
    domain_adaptation_spec_dict['reg_method'] = orth_reg_method = reg if method == 'projected' else 'none'

    # training specification
    use_optim = domain_adaptation_spec_dict['use_optim'] = 'adam'  # "SGD"
    optimizer = tf.keras.optimizers.SGD(lr) if use_optim.lower() == "sgd" else tf.keras.optimizers.Adam(lr)

    batch_size = domain_adaptation_spec_dict['batch_size'] = 512
    domain_adaptation_spec_dict['epochs'] = num_epochs = 100 if early_stopping else 250
    domain_adaptation_spec_dict['epochs_FT'] = num_epochs_FT = 100 if early_stopping else 250
    domain_adaptation_spec_dict['lr'] = lr
    domain_adaptation_spec_dict['dropout'] = dropout = 0.5
    domain_adaptation_spec_dict['patience'] = patience = 10

    # network spacification
    domain_adaptation_spec_dict['batch_normalization'] = batch_norm
    from_logits = activation != "softmax"

    ##########################################
    ###     PREPARE DATA
    ##########################################
    SOURCE_DOMAINS = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    # print(single_source_domain)
    # dataset used in K3DA

    x_source_tr = np.concatenate([data.x_train_dict[source.lower()] for source in SOURCE_DOMAINS if
                                  source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
    y_source_tr = np.concatenate([data.y_train_dict[source.lower()] for source in SOURCE_DOMAINS if
                                  source.lower() != TARGET_DOMAIN[0].lower()], axis=0)

    # tf.data.Dataset.from_tensor_slices((x_source_tr, y_source_tr)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    x_source_tr, y_source_tr = shuffle(x_source_tr, y_source_tr, random_state=1234)

    x_source_te = np.concatenate([data.x_test_dict[source.lower()] for source in SOURCE_DOMAINS if
                                  source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
    y_source_te = np.concatenate([data.y_test_dict[source.lower()] for source in SOURCE_DOMAINS if
                                  source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
    x_source_te, y_source_te = shuffle(x_source_te, y_source_te, random_state=1234)
    x_val, y_val = x_source_te, y_source_te

    x_target_te = np.concatenate([data.x_test_dict[source] for source in TARGET_DOMAIN], axis=0)
    y_target_te = np.concatenate([data.y_test_dict[source] for source in TARGET_DOMAIN], axis=0)
    x_target_te, y_target_te = shuffle(x_target_te, y_target_te, random_state=1234)

    x_val, y_val = shuffle(x_target_te, y_target_te, random_state=1234)

    print("\n FINISHED LOADING DIGITS")

    ##########################################
    ###     FEATURE EXTRACTOR
    ##########################################
    if architecture.lower() == "lenet":
        feature_extractor = get_lenet_feature_extractor()
    else:
        feature_extractor = get_domainnet_feature_extractor(dropout=dropout)


    ##########################################
    ###     PREDICTION LAYER
    ##########################################
    num_domains = domain_adaptation_spec_dict['num_domains']
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x_tilda = feature_extractor(inputs)
    # AD: Append `num_domains` classification heads Dense(10) and average 
    # their predictions to get an ensemble. This way we match the complexity
    # of the ensemble model with the GDU level complexity and make the
    # comparison more fair.
    preds = [Dense(10)(x_tilda) for _ in range(num_domains)]
    outputs = tf.keras.layers.average(preds)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    callback = [EarlyStopping(patience=patience, restore_best_weights=True)]

    if early_stopping:
        callbacks = [callback]
    else:
        callbacks = None

    ##########################################
    ###     INITIALIZE MODEL
    ##########################################

    model.build(input_shape=x_source_tr.shape)
    model.summary()

    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.CategoricalCrossentropy(from_logits=from_logits)]

    print('\n\n\n BEGIN TRAIN:\t METHOD:' + "Ensemble" + "\t\t\t TARGET_DOMAIN:" + TARGET_DOMAIN[0] + "\n\n\n")

    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                  metrics=metrics, )

    run_start = datetime.now()

    hist = model.fit(x=x_source_tr, y=y_source_tr, epochs=num_epochs, verbose=2, batch_size=batch_size, shuffle=False,
                     validation_data=(x_val, y_val), callbacks=callbacks, )
    run_end = datetime.now()

    # model evaluation
    model_res = model.evaluate(x_target_te, y_target_te, verbose=0)
    metric_names = model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names
    print(eval_df)

    save_dir_path = os.path.join(res_file_dir, "run_" + str(run))
    create_dir_if_not_exists(save_dir_path)
    save_dir_path = os.path.join(save_dir_path, "SINGLE_BEST") if single_best else os.path.join(save_dir_path,
                                                                                                "SOURCE_COMBINED")
    create_dir_if_not_exists(save_dir_path)

    save_dir_path = os.path.join(save_dir_path, TARGET_DOMAIN[0])
    create_dir_if_not_exists(save_dir_path)

    run_id = np.random.randint(0, 10000, 1)[0]
    save_dir_name = "ENSEMBLE" + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

    save_dir_path = os.path.join(save_dir_path, save_dir_name)
    create_dir_if_not_exists(save_dir_path)

    hist_df = pd.DataFrame(hist.history)
    duration = run_end - run_start

    # file_name_hist = 'history_' + method.upper() + '.csv'
    file_name_hist = 'history_' + filename + '.csv'

    hist_file_path = os.path.join(save_dir_path, file_name_hist)
    hist_df.to_csv(hist_file_path)

    model_res = model.evaluate(x_target_te, y_target_te, verbose=2)
    metric_names = model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names

    test_sources = ",".join(TARGET_DOMAIN)
    train_sources = ",".join(SOURCE_DOMAINS)

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

    file_name_eval = 'spec_' + filename + '.csv'
    eval_file_path = os.path.join(save_dir_path, file_name_eval)
    eval_df.to_csv(eval_file_path)


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


def get_domainnet_feature_extractor(dropout=0.5):
    feature_exctractor = tf.keras.Sequential(
        [Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same", input_shape=img_shape), BatchNormalization(),
         tf.keras.layers.ReLU(), MaxPool2D(pool_size=(2, 2), strides=(2, 2))

            , Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same"), BatchNormalization(),
         tf.keras.layers.ReLU(), MaxPool2D(pool_size=(2, 2), strides=(2, 2))

            , Conv2D(128, strides=(1, 1), kernel_size=(5, 5), padding="same"), BatchNormalization(),
         tf.keras.layers.ReLU(), MaxPool2D(pool_size=(2, 2), strides=(2, 2))

            , Flatten(), Dense(3072), BatchNormalization(), tf.keras.layers.ReLU(), Dropout(dropout)

            , Dense(2048), BatchNormalization(), tf.keras.layers.ReLU()], name='feature_extractor_domainnet_digits')

    return feature_exctractor


def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential([Conv2D(32, kernel_size=(3, 3), activation='relu'), BatchNormalization(),
                                              MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                                              Conv2D(64, kernel_size=(2, 2), activation='relu'), BatchNormalization(),
                                              MaxPool2D(pool_size=(2, 2), strides=(2, 2)), Flatten(),
                                              Dense(100, activation="relu"), Dense(100, activation="relu")],
                                             name='feature_extractor')
    return feature_exctractor


def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.05)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))


def get_kernel_sum(sigma_list):
    amplitude_list = [1]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in
               sigma_list for amplitude in amplitude_list] + [
                  tfp.math.psd_kernels.MaternFiveHalves(length_scale=sigma, amplitude=amplitude) for sigma in sigma_list
                  for amplitude in amplitude_list] + [
                  tfp.math.psd_kernels.RationalQuadratic(length_scale=sigma, amplitude=amplitude) for sigma in
                  sigma_list for amplitude in amplitude_list]
    kernel_sum = _SumKernel(kernels=kernels)
    return kernel_sum


def run_experiment(experiment):
    try:
        digits_classification(**experiment)
    except Exception:
        import traceback
        traceback.print_exc()
        pass


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    #tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    parser = argparse.ArgumentParser(description='Run a single GDU experiment for ECG data (PhysionetChallenge2020)')
    parser.add_argument('--target', metavar='target_domain', type=str,  # nargs=1,
                        choices=DigitsData.DOMAINS, help='held out target domain for testing', required=True)

    parser.add_argument('--timestamp', metavar='exp_datetime', type=str, nargs=1,
                        help='experiment datetime for storing the results')

    parser.add_argument('--filename', metavar='file name', type=str, nargs=1, help='name of the result CSV file')
    parser.add_argument('--run_id', metavar='run_id', type=str, nargs=1, help='run id')

    args = parser.parse_args()

    target_domain = args.target
    # single_source_domain = args.single_source
    # print("target: ", target_domain, "source: ", single_source_domain)
    print("target: ", target_domain)
    i = args.run_id[0]
    # kernel = args.kernel
    # batch_norm = args.batch_norm
    # print(batch_norm)
    # bias = args.bias
    # fine_tune = args.fine_tune
    timestamp = args.timestamp[0]
    filename = args.filename[0]

    res_file_dir += timestamp + "/"
    # load data once
    digits_data = DigitsData()

    EARLY_STOPPING = True
    LAMBDA_SPARSE = 1e-3
    LAMBDA_ORTH = 1e-3

    run_experiment({'data': digits_data, 'method': None, 'kernel': None, 'TARGET_DOMAIN': [target_domain],
                    'lambda_sparse': LAMBDA_SPARSE,
                    'early_stopping': EARLY_STOPPING, 'run': i})
