'''
nohup /local/home/sfoell/anaconda3/envs/gdu_old/bin/python3.8 -u /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/digits5/digits_5_classification_random_ensemble.py > /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/digits5/digits_5_classification_random_ensemble.log 2>&1 &
'''
import os
import sys
# Find code directory relative to our directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from silence_tensorflow import silence_tensorflow

silence_tensorflow()


import itertools
import logging
from SimulationExperiments.digits5.d5_argparser import parser_args
import pandas as pd
import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)

from datetime import datetime
from sklearn.utils import shuffle

from tensorflow.python.keras.callbacks import EarlyStopping  # TODO: tensorflow.python.keras.callbacks
from SimulationExperiments.digits5.digits_utils import *

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
from SimulationExperiments.digits5.d5_dataloader import load_digits


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


init_gpu(gpu=0, memory=8000)

# File path to the location where the results are stored
res_file_dir = "/local/home/sfoell/NeurIPS/results/2022_Response/RandomEnsemble"
SOURCE_SAMPLE_SIZE = 25000
TARGET_SAMPLE_SIZE = 9000
img_shape = (32, 32, 3)


# load data once in the program and keep in class
class DigitsData(object):
    def __init__(self, test_size=SOURCE_SAMPLE_SIZE):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_digits(test_size=test_size)


def digits_classification(method = "ERM random ensemble", TARGET_DOMAIN = ['mnistm'], single_best=False, single_source_domain=None, batch_norm=False,
                          lr=0.001, save_file=True, save_plot=False, save_feature=False, activation="tanh",
                          lambda_sparse=0,  # 1e-1,
                          lambda_OLS=0,  # 1e-1,
                          lambda_orth=0,  # 1e-1,
                          early_stopping=True, bias=False, fine_tune=True, kernel=None, data: DigitsData = None,
                          run=None, num_domains=5):

    domain_adaptation_spec_dict = {"num_domains": num_domains, "domain_dim": 10, "sigma": 7.5, 'softness_param': 2,
        "similarity_measure": method,
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
    domain_adaptation_spec_dict['epochs'] = num_epochs = 250 if early_stopping else 100
    domain_adaptation_spec_dict['epochs_FT'] = num_epochs_FT = 250 if early_stopping else 100
    domain_adaptation_spec_dict['lr'] = lr
    domain_adaptation_spec_dict['dropout'] = dropout = 0.5
    domain_adaptation_spec_dict['patience'] = patience = 10

    # network spacification
    domain_adaptation_spec_dict['batch_normalization'] = batch_norm
    from_logits = False if activation == "softmax" else True

    ##########################################
    ###     PREPARE DATA
    ##########################################
    if single_best:
        # in case where only one single source domain is chosen
        SOURCE_DOMAINS = single_source_domain
    else:
        SOURCE_DOMAINS = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    # print(single_source_domain)
    # dataset used in K3DA

    if single_best and (SOURCE_DOMAINS[0] == TARGET_DOMAIN[0].lower()):
        print('Source and target domain are the same! Skip!')
        return None
    else:
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

        print("\n FINISHED LOADING DIGITS")

    ##########################################
    ###     FEATURE EXTRACTOR
    ##########################################
    if architecture.lower() == "lenet":
        feature_extractor = get_lenet_feature_extractor()
    else:
        feature_extractor = get_domainnet_feature_extractor(dropout=dropout)

    # if batch_norm:
    #    feature_extractor.add(BatchNormalization())

    ##########################################
    ###     PREDICTION LAYER
    ##########################################
    num_domains = domain_adaptation_spec_dict['num_domains']
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x_tilda = feature_extractor(inputs)
    # Append `num_domains` classification heads Dense(10) and average
    # their predictions to get an ensemble. This way we match the complexity
    # of the ensemble model with the GDU level complexity and make the
    # comparison more fair.
    preds = [Dense(10)(x_tilda) for _ in range(5)]
    train_outputs = tf.keras.layers.average(preds)
    train_model = tf.keras.Model(inputs=inputs, outputs=train_outputs)

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(patience=patience, restore_best_weights=True))

    ##########################################
    ###     INITIALIZE MODEL
    ##########################################

    train_model.build(input_shape=x_source_tr.shape)
    #train_model.feature_extractor.summary()
    #train_model.prediction_layer.summary()

    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.CategoricalCrossentropy(from_logits=from_logits)]

    print('\n\n\n BEGIN TRAIN:\t METHOD:' + method.upper() + "\t\t\t TARGET_DOMAIN:" + TARGET_DOMAIN[0] + "\n\n\n")

    train_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
        metrics=metrics, )

    run_start = datetime.now()
    hist = train_model.fit(x=x_source_tr, y=y_source_tr, epochs=num_epochs, verbose=2, batch_size=batch_size, shuffle=False,
                     validation_data=(x_val, y_val), callbacks=callbacks, )

    model_res = train_model.evaluate(x_target_te, y_target_te, verbose=0)
    metric_names = train_model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names
    print(eval_df)

    run_end = datetime.now()

    # model evaluation
    
    # Sample ensemble weights aka betas from a probability simplex
    n = 5
    betas = np.random.exponential(scale=1.0, size=n)
    betas /= sum(betas)
    betas = tf.convert_to_tensor(betas.astype(dtype="float32"))
    outputs = tf.tensordot(preds, betas, axes=[0, 0])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.build(input_shape=x_source_tr.shape)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits), metrics=metrics, )

    model_res = model.evaluate(x_target_te, y_target_te, verbose=0)
    metric_names = model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names
    print(eval_df)

    tf.keras.backend.clear_session()
    return None


def run_experiment(experiment):
    try:
        digits_classification(**experiment)
    except Exception:
        import traceback
        traceback.print_exc()
        pass


def run_all_experiments(digits_data):
    for i in range(10):
        experiments = []
        for TARGET_DOMAIN in [['mnist'], ['mnistm'], ['svhn'], ['syn'], ['usps']]:
            experiments.append({'data': digits_data, 'TARGET_DOMAIN': TARGET_DOMAIN, 'num_domains': 5})

        print(f'Running {len(experiments)} experiments')

        for experiment in experiments:
            run_experiment(experiment)


if __name__ == "__main__":

    res_file_dir = res_file_dir
    digits_data = DigitsData()
    run_all_experiments(digits_data)

