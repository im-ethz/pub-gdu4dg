import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import os

from tqdm import tqdm

import pandas as pd

logging.disable(logging.WARNING)
os.environ["​TF_CPP_MIN_LOG_LEVEL"] = "3"
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

'''
nohup /local/home/euernst/anaconda3/envs/euernst_MT_gpu/bin/python3.8 -u /local/home/euernst/mt-eugen-ernst/SimulationExperiments/experiment_6_office31_modified/office_playground.py > /local/home/euernst/mt-eugen-ernst/SimulationExperiments/simulation_results/experiment_4/OFFICE_nohup.log 2>&1 &
'''

from Model.DomainAdaptation.domain_adaptation_layer import DomainAdaptationLayer
from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import *

from datetime import datetime

from dataloader_office31 import load_office31

from Model.utils import transform_one_hot, decode_one_hot_vector
from Visualization.evaluation_plots import plot_TSNE


def init_gpu(gpu, memory):
    used_gpu = gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[used_gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[used_gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)

init_gpu(gpu=1, memory=18000)


TRAIN_SAMPLE_SIZE = 20000
TEST_SAMPLE_SIZE = 9000
res_file_dir = "/headwind/misc/domain-adaptation/office31"


from office_data import office31

def transform_office_data(dataset, source_ds=True, target_ds=True, one_hot=False):
    x_tr_s = []
    x_tr_t = []
    y_tr_s = []
    y_tr_t = []
    for sample in dataset:
        if source_ds:
            x_tr_s.append(sample[0].numpy())
            y_tr_s.append(sample[2].numpy())

        if target_ds:
            x_tr_t.append(sample[1].numpy())
            y_tr_t.append(sample[3].numpy())

    x_tr_s = np.array(x_tr_s)
    y_tr_s = np.array(y_tr_s).astype(np.str)
    y_tr_s = np.expand_dims(y_tr_s, axis=-1)

    x_tr_t = np.array(x_tr_t)
    y_tr_t = np.array(y_tr_t).astype(np.str)
    y_tr_t = np.expand_dims(y_tr_t, axis=-1)

    if one_hot:
        y_tr_s, y_tr_t = transform_one_hot([y_tr_s, y_tr_t])

    source_data = [x_tr_s, y_tr_s]
    target_data = [x_tr_t, y_tr_t]

    return source_data, target_data

def process_office31(dataset='train', source_names=["dslr"], target_name="webcam", image_resize=(224, 224), one_hot=True):
    train_dict, val_dict, test_dict = office31(
                                source_names=source_names,
                                target_name=target_name,
                                seed=1,
                                image_resize=image_resize,
                                group_in_out=False,
                                framework_conversion="tensorflow",
                                office_path="/local/home/euernst/mt-eugen-ernst/SimulationExperiments/experiment_4_digits/office"
                            )

    for source_name in source_names:
        try:
            source_train.concatenate(train_dict[source_name + "_train"])
            source_val.concatenate(val_dict[source_name + "_val"])
            source_test.concatenate(test_dict[source_name + "_test"])

        except:
            source_train = train_dict[source_name + "_train"]
            source_val = val_dict[source_name + "_val"]
            source_test = test_dict[source_name + "_test"]

        if dataset == "train":
            source_data = source_train
            target_data = train_dict[target_name + "_train"]
        elif dataset == "val":
            source_data = source_val
            target_data = val_dict[target_name + "_val"]
        elif dataset == "test":
            source_data = source_test
            target_data = test_dict[target_name + "_test"]

    type(source_data)
    source_data.batch(512)#.map(lambda x, y: (x, y))

    if False:
        x_source = []
        y_source = []
        for smpl in tqdm(source_data):
            x_source.append(smpl[0].numpy())
            y_source.append(smpl[1].numpy())

        x_source = np.array(x_source)
        y_source = np.expand_dims(np.array(y_source).astype(np.str), axis=-1)

        x_target = []
        y_target = []
        for smpl in tqdm(target_data):
            x_target.append(smpl[0].numpy())
            y_target.append(smpl[1].numpy())

        x_target = np.array(x_target)
        y_target = np.expand_dims(np.array(y_target).astype(np.str), axis=-1)

        if one_hot:
            [y_source, y_target] = transform_one_hot([y_source, y_target])

        source_data = [x_source, y_source]
        target_data = [x_target, y_target]

    return source_data, target_data




x_train_dict, y_train_dict, x_test_dict, y_test_dict = load_office31(domains=['amazon', 'dslr', 'webcam'])

def office_classification(method, TARGET_DOMAIN, single_best=False, single_source_domain=None,
                          save_file=False, save_plot=False,
                          fine_tune=True,
                          kernel=None):
    run_start = datetime.now()

    domain_adaptation_spec_dict = {
        "num_domains": 5,
        "domain_dim": 100,
        "sigma": 10.5,
        'softness_param': 2,
        "similarity_measure": method,# MMD, IPS
        "domain_reg_param": 1e-1,
        "activation": "tanh",
        "img_shape": (224, 224, 3),
        "bias": False
    }
    run_id = np.random.randint(0, 10000, 1)[0]

    reg = 'SRIP' if method == 'normed' else None

    #architecture used as feature extractor
    architecture = domain_adaptation_spec_dict["architecture"] = "DomainNet"# "LeNet", "DomainNet"

    domain_adaptation_spec_dict["kernel"] = "custom" if kernel is not None else "single"

    bias = domain_adaptation_spec_dict["bias"]
    activation = domain_adaptation_spec_dict["activation"]

    # used in case of "normed"
    domain_adaptation_spec_dict["orth_reg"] = reg = "SRIP"
    domain_adaptation_spec_dict['reg_method'] = reg_method = reg if method == 'normed' else 'nOtHinG<3'

    # training specification
    batch_size = domain_adaptation_spec_dict['batch_size'] = 32
    domain_adaptation_spec_dict['epochs'] = num_epochs = 200
    lr = domain_adaptation_spec_dict['lr'] = 1e-4

    # domains that will be loaded
    SOURCE_DOMAINS = ['amazon', 'dslr', 'webcam'] if single_source_domain is None else single_source_domain + TARGET_DOMAIN

    x_source_tr = np.concatenate([x_train_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()])
    y_source_tr = np.concatenate([y_train_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()])

    x_source_te = np.concatenate([x_test_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()])
    y_source_te = np.concatenate([y_test_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()])

    x_target_te = np.concatenate([x_test_dict[source] for source in TARGET_DOMAIN], axis=0)
    y_target_te = np.concatenate([y_test_dict[source] for source in TARGET_DOMAIN], axis=0)

    print("\n FINISHED LOADING OFFICE31")
    print(x_source_te.shape)
    print(x_source_tr.shape)


    ###################################################
    ###     RESNET FEATURE-EXTRACTOR
    ###################################################
    feature_exctractor = tf.keras.applications.ResNet101(input_shape=(224, 224, 3))

    domain_adaptation = True if method is not None else False
    #domain_adaptation = False
    prediction_layer = models.Sequential([])
    prediction_layer.add(BatchNormalization())


    if domain_adaptation:
        num_domains = domain_adaptation_spec_dict['num_domains']
        sigma = domain_adaptation_spec_dict['sigma']
        domain_dim = domain_adaptation_spec_dict['domain_dim']
        similarity_measure = domain_adaptation_spec_dict["similarity_measure"]
        domain_reg_param = domain_adaptation_spec_dict["domain_reg_param"]

        prediction_layer.add(DomainAdaptationLayer(num_domains=num_domains,
                                        domain_dimension=domain_dim,
                                        softness_param=2,
                                        units=31,
                                        activation="tanh",
                                        sigma=sigma,
                                        similarity_measure=similarity_measure,
                                        domain_reg_method=reg_method,
                                        domain_reg_param=domain_reg_param))


    else:

        prediction_layer.add(Dense(31))


    model = DomainAdaptationModel(feature_extractor=feature_exctractor, prediction_layer=prediction_layer)
    model.build(input_shape=x_source_tr.shape)

    x_val = x_target_te
    y_val = y_target_te

    callback = None

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.CategoricalCrossentropy(from_logits=True)])

    print('\n BEGIN TRAIN:\t' + TARGET_DOMAIN[0] + "\n")

    hist = model.fit(x=x_source_tr, y=y_source_tr, epochs=num_epochs, verbose=2,
                   batch_size=batch_size, shuffle=True, validation_data=(x_val, y_val),
                   #callbacks=[cb]
                 )

    run_end = datetime.now()

    metrics = ['accuracy', tf.keras.metrics.CategoricalCrossentropy(from_logits=True)]

    if save_plot or save_file:
        save_dir_path = os.path.join(res_file_dir, "SINGLE_BEST") if single_best else os.path.join(res_file_dir, "SOURCE_COMBINED")
        create_dir_if_not_exists(save_dir_path)

        save_dir_path = os.path.join(save_dir_path, TARGET_DOMAIN[0])
        create_dir_if_not_exists(save_dir_path)

        if single_best:
            save_dir_name = method.upper() + "_" + SOURCE_DOMAINS[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(run_id)
        else:
            save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

        save_dir_path = os.path.join(save_dir_path, save_dir_name)
        create_dir_if_not_exists(save_dir_path)

    if save_plot:
        X_DATA = model.predict(x_target_te)
        Y_DATA = decode_one_hot_vector(y_target_te)

        df_file_path = os.path.join(save_dir_path, method.upper() + "_feature_data.csv")
        pred_df = pd.DataFrame(X_DATA, columns=["x_{}".format(i) for i in range(31)])
        pred_df['label'] = Y_DATA
        pred_df.to_csv(df_file_path)

        file_name = "TSNE_PLOT_" + method.upper() + ".png"
        tsne_file_path = os.path.join(save_dir_path, file_name)
        plot_TSNE(X_DATA, Y_DATA, plot_kde=False, file_path=tsne_file_path, show_plot=False)

    if save_file:
        hist_df = pd.DataFrame(hist.history)
        duration = run_end - run_start

        file_name_hist = 'history_' + method.upper() + '.csv'
        hist_file_path = os.path.join(save_dir_path, file_name_hist)
        hist_df.to_csv(hist_file_path)

        # perpare results
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

        file_name_eval = 'spec_' + method.upper() + '.csv'
        eval_file_path = os.path.join(save_dir_path, file_name_eval)
        eval_df.to_csv(eval_file_path)

    ##########################################
    #               FINE TUNE                #
    ##########################################
    if domain_adaptation is False and fine_tune:
        feature_exctractor.trainable = False

        for method in ['mmd', 'ips', 'normed']:
            num_domains = domain_adaptation_spec_dict['num_domains']
            sigma = domain_adaptation_spec_dict['sigma']
            domain_dim = domain_adaptation_spec_dict['domain_dim']
            domain_adaptation_spec_dict["similarity_measure"] = method
            domain_reg_param = domain_adaptation_spec_dict["domain_reg_param"]
            softness_param = domain_adaptation_spec_dict["softness_param"]

            prediction_layer.add(DomainAdaptationLayer(num_domains=num_domains,
                                                       domain_dimension=domain_dim,
                                                       softness_param=softness_param,
                                                       units=31,
                                                       kernel=kernel,
                                                       activation=activation,
                                                       sigma=sigma,
                                                       bias=bias,
                                                       similarity_measure=method,
                                                       domain_reg_method=reg_method,
                                                       domain_reg_param=domain_reg_param))


            model = DomainAdaptationModel(feature_extractor=feature_exctractor, prediction_layer=prediction_layer)
            model.build(input_shape=x_source_tr.shape)
            model.summary()

            model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=metrics)

            print('\n BEGIN FINE TUNING:\t' + TARGET_DOMAIN[0] + "\n")
            hist = model.fit(x=x_source_tr, y=y_source_tr.astype(np.float32), epochs=num_epochs, verbose=2,
                             batch_size=batch_size, shuffle=True, validation_data=(x_val, y_val),
                             callbacks=callback
                             )

            if save_plot:
                X_DATA = model.predict(x_target_te)
                Y_DATA = decode_one_hot_vector(y_target_te)

                df_file_path = os.path.join(save_dir_path, method.upper() + "_FT_feature_data.csv")
                pred_df = pd.DataFrame(X_DATA, columns=["x_{}".format(i) for i in range(31)])
                pred_df['label'] = Y_DATA
                pred_df.to_csv(df_file_path)

                file_name = "TSNE_PLOT_" + method.upper() + "_FT" + ".png"
                tsne_file_path = os.path.join(save_dir_path, file_name)
                plot_TSNE(X_DATA, Y_DATA, plot_kde=False, file_path=tsne_file_path, show_plot=False)

            if save_file:
                hist_df = pd.DataFrame(hist.history)
                duration = run_end - run_start

                file_name_hist = 'history_' + method.upper() + "_FT" + '.csv'
                hist_file_path = os.path.join(save_dir_path, file_name_hist)
                hist_df.to_csv(hist_file_path)

                # perpare results
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

                file_name_eval = 'spec_' + method.upper() + "_FT" + '.csv'
                eval_file_path = os.path.join(save_dir_path, file_name_eval)
                eval_df.to_csv(eval_file_path)

    tf.keras.backend.clear_session()
    return None

def img_normalization(x):
    return x / 255.0

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))


if __name__ == "__main__":
    for method in ['ips']:
        for TEST_SOURCES in [['amazon'], ['dslr'], ['webcam']]:
                _ = office_classification(method=method, TARGET_DOMAIN=TEST_SOURCES)
