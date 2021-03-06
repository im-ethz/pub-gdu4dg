import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


import os
import sys
import keras
import skimage
import logging
import multiprocessing


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import tensorflow as tf
tf.random.set_seed(1234)

import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from datetime import datetime
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

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../..')))
THIS_FILE = os.path.abspath(__file__)

from Model.utils import decode_one_hot_vector
from Visualization.evaluation_plots import plot_TSNE
from SimulationExperiments.experiment_4_digits.d5_dataloader import load_digits

from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from Model.DomainAdaptation.domain_adaptation_callback import DomainCallback

def init_gpu(gpu, memory):
    used_gpu = gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[used_gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[used_gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)

init_gpu(gpu=0, memory=6000)

# file path to the location where the results are stored
#res_file_dir = "/headwind/misc/domain-adaptation/digits/eugen"
#res_file_dir = "/local/home/sfoell/NeurIPS/results/source_combined_methods_domainb_20210601"
res_file_dir = "/local/home/sfoell/NeurIPS/results/test"
SOURCE_SAMPLE_SIZE = 25000
TARGET_SAMPLE_SIZE = 9000
img_shape = (32, 32, 3)

'''
nohup /local/home/sfoell/anaconda3/envs/gdu4dg/bin/python3.8 -u /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiment_4_digits/digits_5_classification_SF.py > /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiment_4_digits/visualizations.log 2>&1 &
'''

# load data once in the program and keep in class
class DigitsData(object):
    def __init__(self, test_size=SOURCE_SAMPLE_SIZE):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_digits(test_size=test_size)


def digits_classification(method, TARGET_DOMAIN, single_best=False, single_source_domain=None,
                          batch_norm=False,
                          lr=0.001,
                          save_file=True, save_plot=False, save_feature=True,
                          activation="tanh",
                          bias=False,
                          fine_tune=True,
                          kernel=None,
                          data: DigitsData=None,
                          run=None):

    domain_adaptation_spec_dict = {
        "num_domains": 5,
        "domain_dim": 10,
        "sigma": 7.5,
        'softness_param': 2,
        "similarity_measure": method,# MMD, IPS
        "lambda_orth": 1e-8,
        "img_shape": img_shape,
        "bias": bias,
        "source_sample_size": SOURCE_SAMPLE_SIZE,
        "target_sample_size": TARGET_SAMPLE_SIZE
    }

    #architecture used as feature extractor
    architecture = domain_adaptation_spec_dict["architecture"] ="DomainNet" # "DomainNet"# "LeNet"

    domain_adaptation_spec_dict["kernel"] = "custom" if kernel is not None else "single"

    # used in case of "projected"
    domain_adaptation_spec_dict["orth_reg"] = reg = "SRIP"
    domain_adaptation_spec_dict['reg_method'] = orth_reg_method = reg if method == 'projected' else 'none'

    # training specification
    use_optim = domain_adaptation_spec_dict['use_optim'] = 'adam' #"SGD"
    optimizer = tf.keras.optimizers.SGD(lr) if use_optim.lower() =="sgd" else tf.keras.optimizers.Adam(lr)

    batch_size = domain_adaptation_spec_dict['batch_size'] = 512
    domain_adaptation_spec_dict['epochs'] = num_epochs = 250
    domain_adaptation_spec_dict['epochs_FT'] = num_epochs_FT = 250
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
    #print(single_source_domain)
    # dataset used in K3DA

    if (single_best == True) & (SOURCE_DOMAINS[0] == TARGET_DOMAIN[0].lower()):
        print('Source and target domain are the same! Skip!')
        return None
    else:
        x_source_tr = np.concatenate([data.x_train_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
        y_source_tr = np.concatenate([data.y_train_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()], axis=0)

        #tf.data.Dataset.from_tensor_slices((x_source_tr, y_source_tr)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        x_source_tr, y_source_tr = shuffle(x_source_tr, y_source_tr, random_state=1234)

        x_source_te = np.concatenate([data.x_test_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
        y_source_te = np.concatenate([data.y_test_dict[source.lower()] for source in SOURCE_DOMAINS if source.lower() != TARGET_DOMAIN[0].lower()], axis=0)
        x_source_te, y_source_te = shuffle(x_source_te, y_source_te, random_state=1234)
        x_val, y_val = x_source_te, y_source_te

        x_target_te = np.concatenate([data.x_test_dict[source] for source in TARGET_DOMAIN], axis=0)
        y_target_te = np.concatenate([data.y_test_dict[source] for source in TARGET_DOMAIN], axis=0)
        x_target_te, y_target_te = shuffle(x_target_te, y_target_te, random_state=1234)

        print("\n FINISHED LOADING DIGITS")


    ##########################################
    ###     FEATURE EXTRACTOR
    ##########################################
    if architecture.lower() =="lenet":
        feature_extractor = get_lenet_feature_extractor()

    else:
        feature_extractor = get_domainnet_feature_extractor(dropout=dropout)

    #if batch_norm:
    #    feature_extractor.add(BatchNormalization())

    ##########################################
    ###     PREDICTION LAYER
    ##########################################
    prediction_layer = tf.keras.Sequential([], name='prediction_layer')
    domain_adaptation = True if method is not None else False
    if domain_adaptation:
        num_domains = domain_adaptation_spec_dict['num_domains']
        sigma = domain_adaptation_spec_dict['sigma']
        domain_dim = domain_adaptation_spec_dict['domain_dim']
        similarity_measure = domain_adaptation_spec_dict["similarity_measure"]
        softness_param = domain_adaptation_spec_dict["softness_param"]
        prediction_layer.add(DGLayer(domain_units=num_domains,
                                     N=domain_dim,
                                     softness_param=softness_param,
                                     units=10,
                                     kernel=kernel,
                                     sigma=sigma,
                                     activation=activation,
                                     bias=bias,
                                     similarity_measure=similarity_measure,
                                     orth_reg_method=orth_reg_method,
                                     ))

    else:
        method = "SOURCE_ONLY"
        prediction_layer.add(Dense(10, activation=activation))

    callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
    domain_callback = DomainCallback(test_data=x_source_te, train_data=x_source_tr, print_res=True, max_sample_size=5000)
    ##########################################
    ###     INITIALIZE MODEL
    ##########################################
    # DomainAdaptationModel has one feature_extractor (that may be used in the fine tune stage) and one prediction layer
    model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

    model.build(input_shape=x_source_tr.shape)
    model.feature_extractor.summary()
    model.prediction_layer.summary()

    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.CategoricalCrossentropy(from_logits=from_logits)]

    print('\n\n\n BEGIN TRAIN:\t METHOD:' + method.upper() + "\t\t\t TARGET_DOMAIN:" + TARGET_DOMAIN[0] + "\n\n\n")

    model.compile(
                  optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                  metrics=metrics,
                  )

    run_start = datetime.now()

    hist = model.fit(x=x_source_tr, y=y_source_tr, epochs=num_epochs, verbose=2,
                     batch_size=batch_size, shuffle=False,
                     validation_data=(x_val, y_val),
                     callbacks=[callback,domain_callback] if domain_adaptation else callback,
                     )
    run_end = datetime.now()

    # model evaluation
    model_res = model.evaluate(x_target_te, y_target_te, verbose=0)
    metric_names = model.metrics_names
    eval_df = pd.DataFrame(model_res).transpose()
    eval_df.columns = metric_names
    print(eval_df)

    if save_plot or save_file:
        run_id = np.random.randint(0, 10000, 1)[0]
        save_dir_path = os.path.join(res_file_dir, "run_" + str(run))
        create_dir_if_not_exists(save_dir_path)
        save_dir_path = os.path.join(save_dir_path, "SINGLE_BEST") if single_best else os.path.join(save_dir_path, "SOURCE_COMBINED")

        create_dir_if_not_exists(save_dir_path)

        save_dir_path = os.path.join(save_dir_path, TARGET_DOMAIN[0])
        create_dir_if_not_exists(save_dir_path)

        if single_best:
            save_dir_name = method.upper() + "_" + SOURCE_DOMAINS[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(run_id)
        else:
            save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

        save_dir_path = os.path.join(save_dir_path, save_dir_name)
        create_dir_if_not_exists(save_dir_path)

    if save_plot or save_feature:
        X_DATA = model.predict(x_target_te)
        Y_DATA = decode_one_hot_vector(y_target_te)

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

        file_name_eval = 'spec_' + method.upper() + '.csv'
        eval_file_path = os.path.join(save_dir_path, file_name_eval)
        eval_df.to_csv(eval_file_path)

        #x_digits_data_list = list(digits_data.x_train_dict.values()) + list(digits_data.x_test_dict.values())
        #x_digits_data_list = [np.array(digits_data.x_train_dict[key]) for key in digits_data.x_train_dict.keys()]
        #y_data_label = [decode_one_hot_vector(np.array(digits_data.y_train_dict[key])) for key in digits_data.y_train_dict.keys()]
        #x_data_label = [np.array(digits_data.x_train_dict[key]) for key in digits_data.x_train_dict.keys()]

        x_data = {key: model.feature_extractor.predict(np.array(digits_data.x_train_dict[key][0:100])) for key in digits_data.x_train_dict.keys()}


        first = True
        for key in list(x_data.keys()):
            if first:
                x_data_array = x_data[key]
                x_data_index = [key] * x_data[key].shape[0]
                first = False
            else:
                x_data_array = np.concatenate((x_data_array, x_data[key]), axis=0)
                x_data_index = np.concatenate((x_data_index, [key] * x_data[key].shape[0]),axis=0).flatten()



    ##########################################
    #               FINE TUNE                #
    ##########################################

    if domain_adaptation is False and fine_tune:

        feature_extractor_filepath = os.path.join(save_dir_path, 'feature_extractor.h5.tmp')
        feature_extractor.save(feature_extractor_filepath)

        for similarity_measure in ['MMD', 'cosine_similarity' ,'projected']:

            prediction_layer = tf.keras.Sequential([], name='prediction_layer')

            num_domains = domain_adaptation_spec_dict['num_domains']

            feature_extractor = keras.models.load_model(feature_extractor_filepath)
            feature_extractor.trainable = False

            # sigma is estimated based on the median heuristic, with sample size of 5000 features
            # sigma = domain_adaptation_spec_dict['sigma']
            sigma = domain_adaptation_spec_dict['sigma'] = sigma_median(feature_extractor.predict(x_source_tr))
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
                                         units=10,
                                         kernel=kernel,
                                         sigma=sigma,
                                         activation=activation,
                                         bias=bias,
                                         similarity_measure=similarity_measure,
                                         orth_reg_method=orth_reg_method,
                                         ))

            model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

            model.build(input_shape=x_source_tr.shape)
            model.feature_extractor.summary()
            model.prediction_layer.summary()

            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits), metrics=metrics)

            callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
            domain_callback = DomainCallback(test_data=x_source_te, train_data=x_source_tr, print_res=True, max_sample_size=5000)


            print('\n BEGIN FINE TUNING:\t' + method.upper() + "\t" + TARGET_DOMAIN[0] + "\n")
            hist = model.fit(x=x_source_tr, y=y_source_tr.astype(np.float32), epochs=num_epochs_FT, verbose=2,
                                   batch_size=batch_size, shuffle=False, validation_data=(x_val, y_val),
                                   callbacks=[callback, domain_callback]
                                   )
            model.evaluate(x_target_te, y_target_te, verbose=2)

            dg_layer = model.get_dg_layer()
            domain_basis = dg_layer.get_domain_basis()

            # Clustering "samples" of elementary domain

            first = True
            for key in list(domain_basis.keys()):
                if first:
                    domain_basis_array = domain_basis[key]
                    domain_basis_index = [key] * domain_basis[key].shape[0]
                    first = False
                else:
                    domain_basis_array = np.concatenate((domain_basis_array, domain_basis[key]), axis=0)
                    domain_basis_index = np.concatenate((domain_basis_index, [key] * domain_basis[key].shape[0]),
                                                        axis=0).flatten()

            file_name_db = 'domain_basis_df_' + similarity_measure.upper() + '.csv'
            file_name_db = os.path.join(save_dir_path, file_name_db)
            domain_basis_df = pd.DataFrame(domain_basis_array)
            domain_basis_df['index'] = domain_basis_index
            domain_basis_df.to_csv(file_name_db)

            file_name_x = 'x_data_df_' + similarity_measure.upper() + '.csv'
            file_name_x = os.path.join(save_dir_path, file_name_x)
            x_data_df_ = pd.DataFrame(x_data_array)
            x_data_df_['index'] = x_data_index
            x_data_df_.to_csv(file_name_x)

            from sklearn.manifold import TSNE
            embedded_domain_basis = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=1234).fit_transform(domain_basis_array)
            embedded_data = TSNE(n_components=2, perplexity=100, n_iter=5000, random_state=1234).fit_transform(x_data_array)
            embedding = np.concatenate((embedded_domain_basis, embedded_data))
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=np.concatenate((domain_basis_index, x_data_index)).flatten(), palette="bright")
            plt.show()

            # Heatmaps to visualize the MMD
            domain_basis.update({key.upper(): x_data[key] for key in x_data.keys()})

            dg_kernel = dg_layer.kernel

            domains_values = list(domain_basis.values())
            doamain_keys = list(domain_basis.keys())

            mmd_matrix = get_mmd_matrix(domains_values, kernel=dg_kernel)
            mmd_matrix_df = pd.DataFrame(mmd_matrix, columns=doamain_keys, index=doamain_keys)

            file_name_mmd = 'mmd_matrix_' + similarity_measure.upper() + '.csv'
            file_name_mmd = os.path.join(save_dir_path, file_name_mmd)

            mmd_matrix_df.to_csv(file_name_mmd)

            #ax = plt.axes()
            #sns.heatmap(mmd_matrix_df, annot=True, ax=ax, linewidths=.8, cmap="viridis",
            #            cbar_kws={'label': 'Maximum Mean Discrepancy'})
            #plot_label = {'cosine_similarity': 'Cosine Similarity', 'MMD': 'Maximum Mean Discrepancy',
            #              'projected': 'Projection'}
            #ax.set_title("Method: %s \n Target domin: %s" % (plot_label[method], TARGET_DOMAIN[0].upper()))
            #plt.show()

            # D_KME = {key: np.mean(dg_kernel.matrix(domain_basis[key.upper()], domain_basis[key.upper()]).numpy(), axis=0) for key in  x_data.keys()}
            # D_KME = pd.DataFrame.from_dict(D_KME).transpose()
            # D_KME_em = TSNE(n_components=2, random_state=1234).fit_transform(D_KME)

            # V_KME = {key: np.mean(dg_kernel.matrix(domain_basis[key], domain_basis[key]).numpy(), axis=0) for key in dg_layer.get_domain_basis().keys()}
            # V_KME  = pd.DataFrame.from_dict(V_KME).transpose()
            # V_KME_em = TSNE(n_components=2, random_state=1234).fit_transform(V_KME)

            # df = pd.DataFrame(D_KME_em, index=list(x_data.keys()), columns = ['Dim 1', 'Dim 2']).append(pd.DataFrame(V_KME_em, columns = ['Dim 1', 'Dim 2'], index = list(dg_layer.get_domain_basis().keys())))

            # sns.scatterplot(data = df, x='Dim 1', y='Dim 2', hue = df.index)
            # plt.show()

            #for key in domain_basis.keys():
            #    file_name_eval = 'domain_basis' + method.upper() + '_' + str(key) + '.csv'
            #    eval_file_path = os.path.join(save_dir_path, file_name_eval)
            #    pd.DataFrame(domain_basis[key]).to_csv(eval_file_path)

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
                    save_dir_name = method.upper() + "_" + SOURCE_DOMAINS[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(
                        run_id)
                else:
                    save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

                save_dir_path = os.path.join(save_dir_path, save_dir_name)
                create_dir_if_not_exists(save_dir_path)

            if save_plot or save_file:
                X_DATA = model.predict(x_target_te)
                Y_DATA = decode_one_hot_vector(y_target_te)

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
                model_res = model.evaluate(x_target_te, y_target_te, verbose=2)
                metric_names = model.metrics_names
                eval_df = pd.DataFrame(model_res).transpose()
                eval_df.columns = metric_names

                test_sources = ",".join(TARGET_DOMAIN)
                train_sources = ",".join(SOURCE_DOMAINS)

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
    for i in range (num_ds):
        x_i = x_data[i]
        for j in range (i, num_ds):
            x_j = x_data[j]
            mmd_matrix[i, j] = mmd_matrix[j, i] = MMD(x_i, x_j, kernel = kernel)
    return mmd_matrix

def sigma_median(x_data, sample_size=5000):
    x_data = x_data[:sample_size]
    sigma_median = np.median(euclidean_distances(x_data, x_data))
    return sigma_median


def get_domainnet_feature_extractor(dropout=0.5):
    feature_exctractor = tf.keras.Sequential([
        Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same", input_shape=img_shape)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Conv2D(64, strides=(1, 1), kernel_size=(5, 5), padding="same")
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Conv2D(128, strides=(1, 1), kernel_size=(5, 5), padding="same")
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        , Flatten()
        , Dense(3072)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
        , Dropout(dropout)

        , Dense(2048)
        , BatchNormalization()
        , tf.keras.layers.ReLU()
    ], name='feature_extractor_domainnet_digits')

    return feature_exctractor

def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Conv2D(64, kernel_size=(2, 2), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Flatten()
        , Dense(100, activation="relu")
        , Dense(100, activation="relu")
    ], name='feature_extractor')
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

    try:
        digits_classification(**experiment)
    except Exception:
        import traceback
        traceback.print_exc()
        pass


GPUS = [2]
# GPUS = [2]

if __name__ == "__main__":
    #from office31.dataloader_office31 import load_office31
    #x_train_dict, y_train_dict, x_test_dict, y_test_dict = load_office31(domains=['amazon', 'dslr', 'webcam'])

    # load data once
    digits_data = DigitsData()
    #for i in range(5):
    for i in [0]:
        experiments = []
        for method in [None]:
            for kernel in [None]:
                for TEST_SOURCES in [['mnistm'], ['mnist'], ['syn'], ['svhn'], ['usps']]:
                    for batch_norm in [True]:  # , False]:
                        for bias in [False]:  # , True]:
                            #for single_source in [['mnistm'], ['mnist'], ['svhn'], ['syn'], ['usps']]:
                                experiments.append({
                                    'data': digits_data,
                                    'method': method,
                                    'kernel': kernel,
                                    'batch_norm': batch_norm,
                                    'bias': bias,
                                    'TARGET_DOMAIN': TEST_SOURCES,
                                    #'single_source_domain': single_source,
                                    'run' : i
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
