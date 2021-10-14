import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import sys
import keras
import itertools
import logging
from d5_argparser import parser_args
import pandas as pd
import tensorflow as tf

tf.random.set_seed(1234)


from datetime import datetime
from sklearn.utils import shuffle

from tensorflow.python.keras.callbacks import EarlyStopping # TODO: tensorflow.python.keras.callbacks
from digits_utils import *

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

'''
nohup /local/home/pokanovic/miniconda3/envs/gdu4dg/bin/python3.8 -u /local/home/pokanovic/project2/SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --lambda_orth 0 --fine_tune False --run 0 > /local/home/pokanovic/project2/SimulationExperiments/experiment_4_digits/srip_orth_000.log 2>&1 &

'''

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
            tf.config.experimental.set_virtual_device_configuration(gpus[used_gpu], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)


init_gpu(gpu=0, memory=6000)

# File path to the location where the results are stored
res_file_dir = "/local/home/pokanovic/project2/results/frozen"
SOURCE_SAMPLE_SIZE = 25000
TARGET_SAMPLE_SIZE = 9000
img_shape = (32, 32, 3)


# load data once in the program and keep in class
class DigitsData(object):
    def __init__(self, test_size=SOURCE_SAMPLE_SIZE):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_digits(test_size=test_size)


def digits_classification(method, TARGET_DOMAIN, single_best=False, single_source_domain=None,
                          batch_norm=False,
                          lr=0.001,
                          save_file=True, save_plot=False, save_feature=False,
                          activation="tanh",
                          lambda_sparse=0,  # 1e-1,
                          lambda_OLS=0,  # 1e-1,
                          lambda_orth=0,  # 1e-1,
                          early_stopping=True,
                          bias=False,
                          fine_tune=True,
                          kernel=None,
                          data: DigitsData = None,
                          run=None):
    domain_adaptation_spec_dict = {
        "num_domains": 5,
        "domain_dim": 10,
        "sigma": 7.5,
        'softness_param': 2,
        "similarity_measure": method,  # MMD, IPS
        "img_shape": img_shape,
        "bias": bias,
        "source_sample_size": SOURCE_SAMPLE_SIZE,
        "target_sample_size": TARGET_SAMPLE_SIZE
    }

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

    batch_size = domain_adaptation_spec_dict['batch_size'] = 128
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

    if (single_best == True) & (SOURCE_DOMAINS[0] == TARGET_DOMAIN[0].lower()):
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
                                     lambda_sparse=lambda_sparse,
                                     lambda_OLS=lambda_OLS,
                                     lambda_orth=lambda_orth,
                                     ))

    else:
        method = "SOURCE_ONLY"
        prediction_layer.add(Dense(10, activation=activation))

    callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
    domain_callback = DomainCallback(test_data=x_source_te, train_data=x_source_tr, print_res=True,
                                     max_sample_size=5000)

    if early_stopping and domain_adaptation:
        callbacks = [callback, domain_callback]

    elif early_stopping and domain_adaptation == False:
        callbacks = [callback]

    elif early_stopping == False:
        callbacks = domain_callback if domain_adaptation else None

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
                     callbacks=callbacks,
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
        save_dir_path = os.path.join(save_dir_path, "SINGLE_BEST") if single_best else os.path.join(save_dir_path,
                                                                                                    "SOURCE_COMBINED")

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

    ##########################################
    #               FINE TUNE                #
    ##########################################

    if domain_adaptation is False and fine_tune:

        feature_extractor_filepath = os.path.join(save_dir_path, 'feature_extractor.h5.tmp')
        feature_extractor.save(feature_extractor_filepath)

        for similarity_measure in ['cosine_similarity', 'MMD', 'projected']:

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
                                         lambda_sparse=lambda_sparse,
                                         lambda_OLS=lambda_OLS,
                                         lambda_orth=lambda_orth,
                                         ))

            model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

            model.build(input_shape=x_source_tr.shape)
            model.feature_extractor.summary()
            model.prediction_layer.summary()

            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                          metrics=metrics)

            callback = [EarlyStopping(patience=patience, restore_best_weights=True)]
            domain_callback = DomainCallback(test_data=x_source_te, train_data=x_source_tr, print_res=True,
                                             max_sample_size=5000)

            if early_stopping:
                callbacks = [callback, domain_callback]

            else:
                callbacks = domain_callback

            print('\n BEGIN FINE TUNING:\t' + method.upper() + "\t" + TARGET_DOMAIN[0] + "\n")
            hist = model.fit(x=x_source_tr, y=y_source_tr.astype(np.float32), epochs=num_epochs_FT, verbose=2,
                             batch_size=batch_size, shuffle=False, validation_data=(x_val, y_val),
                             callbacks=callbacks
                             )
            model.evaluate(x_target_te, y_target_te, verbose=2)

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
                    save_dir_name = method.upper() + "_" + SOURCE_DOMAINS[0] + "_to_" + TARGET_DOMAIN[0] + "_" + str(
                        run_id)
                else:
                    save_dir_name = method.upper() + "_" + TARGET_DOMAIN[0] + "_" + str(run_id)

                save_dir_path = os.path.join(save_dir_path, save_dir_name)
                create_dir_if_not_exists(save_dir_path)

            if save_plot or save_feature:
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
                # print("\n\nSPEC_FILE \n", eval_df)

        #os.remove(feature_extractor_filepath)

    tf.keras.backend.clear_session()
    return None


def run_experiment(experiment):
    try:
        digits_classification(**experiment)
    except Exception:
        import traceback
        traceback.print_exc()
        pass


def run_all_experiments(digits_data, args):
    for i in [4]:
        experiments = []
        for experiment in itertools.product([args.method],
                                            [[args.TARGET_DOMAIN]],
                                            [True], [0, 1e-3, 1e-2, 1e-1], [0, 1e-3, 1e-2, 1e-1], [args.fine_tune]):
            experiments.append({
                'data': digits_data,
                'method': experiment[0],
                'kernel': None,
                'TARGET_DOMAIN': experiment[1],
                'lambda_sparse': experiment[3],
                'lambda_OLS': experiment[4],
                'lambda_orth': 0,
                'early_stopping': experiment[2],
                'run': i,
                'fine_tune': experiment[5]
            })

        print(f'Running {len(experiments)} experiments')

        for experiment in experiments:
            run_experiment(experiment)


if __name__ == "__main__":
    args = parser_args()
    res_file_dir = args.res_file_dir + args.TARGET_DOMAIN + '_' + str(args.method) + '_' + 'ft' if args.fine_tune else 'e2e'
    # load data once
    digits_data = DigitsData()
    if args.run_all:
        run_all_experiments(digits_data, args)
    else:
        experiment = {
            'data': digits_data,
            'method': args.method,
            'kernel': None,
            'TARGET_DOMAIN': [args.TARGET_DOMAIN],
            'lambda_sparse': args.lambda_sparse,
            'lambda_OLS': args.lambda_OLS,
            'lambda_orth': args.lambda_orth,
            'early_stopping': args.early_stopping,
            'run': args.running
        }
        run_experiment(experiment)

