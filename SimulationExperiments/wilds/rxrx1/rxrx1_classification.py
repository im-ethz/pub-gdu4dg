import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

import logging
import os
import pathlib
import sys
import warnings
import tempfile
import math


from datetime import datetime
from tensorflow import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import *

from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
import utils


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, training_steps):
        super(CustomSchedule, self).__init__()

        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.first_step = True

    def __call__(self, step):
        s = tf.cast(step, 'float32')
        progress = (s - self.warmup_steps) / tf.maximum(1.0, self.training_steps - self.warmup_steps)
            #lr = self.initial_lr * tf.maximum(0.0, 0.5 * (1.0 + tf.math.cos(tf.constant(math.pi) * progress)))
        f1 = lambda: self.initial_lr * s / tf.maximum(1.0, self.warmup_steps)
        f2 = lambda: self.initial_lr * tf.maximum(0.0, 0.5 * (1.0 + tf.math.cos(tf.constant(math.pi) * progress)))
        lr = tf.case([(tf.less(s, self.warmup_steps), f1)], default=f2)
        
        tf.summary.scalar('learning rate', data=lr, step=tf.cast(step, 'int8'))

        return lr




# n_training_steps = math.ceil(len(train_loader)/config.gradient_accumulation_steps) * config.n_epochs
n_epoch = 120
gradient_accumulation_steps = 1
n_training_steps = math.ceil(542 / gradient_accumulation_steps) * n_epoch #542

# probably, you can just hardcode this constant :)

# Also, I believe they actually used intial lr 1e-3 and not 1e-4 as reported in the paper. And a batch size of 72. At least, that's what their config says.


warnings.filterwarnings("ignore", category=DeprecationWarning)
# silence_tensorflow()
tf.random.set_seed(2341)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

# file path to the location where the results are stored
res_file_dir = "output"

# IMPORTANT: check this before running
# TODO: add to argument parser?
width, height = 256, 256
img_shape = (width, height, 3)
units = 1139


def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Conv2D(64, kernel_size=(2, 2), activation='relu')
        , BatchNormalization()
        , MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        , Flatten()
        , Dense(100, activation="relu")  # , kernel_initializer=utils.Ortho())
        , Dense(100, activation="relu")  # , kernel_initializer=utils.Ortho())
    ], name='feature_extractor')
    return feature_exctractor


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


def get_resnet(input_shape):
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                     pooling="avg", input_shape=input_shape)
    feature_extractor = tf.keras.Sequential([resnet], name='feature_extractor_resnet')
    return feature_extractor


def get_dense_net():
    return tf.keras.applications.densenet.DenseNet121(
        include_top=False, weights='imagenet',
        input_shape=img_shape, pooling='max')


def add_regularization(model, regularizer=tf.keras.regularizers.l2(1e-5)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


class RXRX1Classification():
    def __init__(self, method, timestamp, target_domain, train_generator, valid_generator, test_generator,
                 kernel=None, batch_norm=False, bias=False,
                 lambda_OLS = 1e-3, lambda_orth = 1e-3, lambda_sparse= 1e-3,
                 num_domains = 6, domain_dim = 10, sigma = 10,
                 save_file=True, save_plot=False,
                 save_feature=True, batch_size=64, fine_tune=False, lr=1e-3, activation=None,
                 feature_extractor='LeNet', run=0, only_fine_tune=False, feature_extractor_saved_path=None):
        """"
        Params:
        ----------------------
        only_fine_tune: 'bool'
            if this parameter is set to True, the feature extractor will not be fine tuned
        """
        super()
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator
        self.feature_extractor = feature_extractor
        self.method = "SOURCE_ONLY" if method is None else method
        self.target_domain = target_domain

        self.batch_norm = batch_norm
        self.lr = lr
        self.save_file = save_file
        self.save_plot = save_plot
        self.save_feature = save_feature
        self.activation = activation
        self.bias = bias
        self.fine_tune = fine_tune
        self.kernel = kernel
        self.batch_size = batch_size
        self.run = run
        self.only_fine_tune = only_fine_tune
        self.feature_extractor_saved_path = feature_extractor_saved_path
        self.lambda_OLS = lambda_OLS
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
        self.num_domains = num_domains
        self.domain_dim = domain_dim
        self.sigma = sigma
        self.n_epoch = 90
        self.gradient_accumulation_steps = 1
        self.n_training_steps = math.ceil(542 / self.gradient_accumulation_steps) * self.n_epoch  # 542

        self.run_id = np.random.randint(0, 10000, 1)[0]
        print(self.run_id)
        self.save_dir_path = 'pathSaving'
        self.da_spec = self.create_da_spec()
        self.optimizer = tfa.optimizers.AdamW(learning_rate=CustomSchedule(initial_lr=1e-3, warmup_steps= 5420, training_steps=self.n_training_steps), weight_decay=1e-5) #5420
        from_logits = self.activation != "softmax"

        if units == 1:
            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            self.metrics = [tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.BinaryCrossentropy(from_logits=from_logits),
                            tfa.metrics.F1Score(num_classes=units, average='macro')
                            ]
        else:
            self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
            self.metrics = [tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.CategoricalCrossentropy(from_logits=from_logits),
                            tfa.metrics.F1Score(num_classes=units, average='macro')
                            ]

        #logdir = "logging_rxrx1_{}_{}".format("ERM" if method == None else method.upper(), self.run_id)
        #logdir = os.path.join(self.save_dir_path, logdir)
        #self.callback = [keras.callbacks.TensorBoard(log_dir=logdir)]

        print("\n FINISHED LOADING WILDS")

    def save_evaluation_files(self, model, fine_tune=False):
        method = self.da_spec["similarity_measure"]
        if fine_tune:
            num_epochs = self.da_spec["epochs_FT"]
        elif fine_tune== False and self.method != "SOURCE_ONLY":
            num_epochs = self.da_spec["epochs_E2E"]
        else:
            num_epochs = self.n_epoch

        print(num_epochs)

        file_suffix = "_FT" if fine_tune else "E2E"
        run_start = datetime.now()

        hist = model.fit(x=self.train_generator,
                         epochs=num_epochs,
                         verbose=1,
                         validation_data=self.valid_generator,
                         #callbacks=self.callback,
                         )
        run_end = datetime.now()
        predictions = model.predict(self.test_generator)
        file_name_pred = "pred_rxrx1_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run)
        pred_file_path = os.path.join(self.save_dir_path, file_name_pred)
        # TODO np.save(pred_file_path, predictions), don't need it?

        if self.save_file:
            hist_df = pd.DataFrame(hist.history)
            duration = run_end - run_start

            file_name_hist = "history_rxrx1_{}_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run,
                                                                       self.run_id)
            hist_file_path = os.path.join(self.save_dir_path, file_name_hist)
            hist_df.to_csv(hist_file_path)

            # prepare results
            model_res = model.evaluate(self.test_generator, verbose=1)
            metric_names = model.metrics_names
            eval_df = pd.DataFrame(model_res).transpose()
            eval_df.columns = metric_names
            eval_df = pd.concat([eval_df, pd.DataFrame.from_dict([self.da_spec])], axis=1)
            eval_df['duration'] = duration
            eval_df['run_id'] = self.run_id
            eval_df['trained_epochs'] = len(hist_df)
            print('RUN ID: ', self.run_id, '\n\n')

            file_name_eval = "spec_rxrx1_{}_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run, self.run_id)
            eval_file_path = os.path.join(self.save_dir_path, file_name_eval)
            print('EVAL_DF\n\n', eval_df)
            eval_df.to_csv(eval_file_path)

            if self.save_feature:
                df_file_path = os.path.join(self.save_dir_path,
                                            "{}_{}_{}_{}_feature_data_rxrx1.csv".format(method.upper(), file_suffix,
                                                                                           self.run, self.run_id))
                pred_df = pd.DataFrame(predictions, columns=["x_{}".format(i) for i in range(1)])
                pred_df.to_csv(df_file_path)

    def create_da_spec(self):
        da_spec_dict = {"num_domains": self.num_domains,
                        "domain_dim": self.domain_dim,
                        "sigma": self.sigma,
                        'softness_param': 2,
                        "batch_size": self.batch_size,
                        "epochs_ERM": self.n_epoch,
                        "epochs_E2E": 120,
                        "epochs_FT": 90,
                        "orth_reg": "SRIP",
                        "architecture": self.feature_extractor,
                        "bias": self.bias,
                        "similarity_measure": self.method,
                        'lambda_OLS': self.lambda_OLS,
                        'lambda_orth': self.lambda_orth,
                        'lambda_sparse': self.lambda_sparse,
                        'lr': self.lr,
                        'batch_normalization': self.batch_norm,
                        "kernel": "custom" if self.kernel is not None else "single"}

        # used in case of "projected"
        da_spec_dict['reg_method'] = da_spec_dict["orth_reg"] if self.method == 'projected' else 'none'

        return da_spec_dict

    def add_da_layer(self, prediction_layer):
        num_domains = self.da_spec['num_domains']
        sigma = self.da_spec['sigma']
        domain_dim = self.da_spec['domain_dim']
        similarity_measure = self.da_spec["similarity_measure"]
        softness_param = self.da_spec["softness_param"]
        reg_method = self.da_spec['reg_method']
        lambda_OLS = self.da_spec['lambda_OLS']
        lambda_orth = self.da_spec['lambda_orth']
        lambda_sparse = self.da_spec['lambda_sparse']

        print(units)
        prediction_layer.add(tf.keras.layers.BatchNormalization())
        prediction_layer.add(
            DGLayer(domain_units=num_domains, N=domain_dim, softness_param=softness_param, units=units,
                    kernel=self.kernel, sigma=sigma, activation=self.activation, bias=self.bias,
                    similarity_measure=similarity_measure, orth_reg_method=reg_method,
                    lambda_OLS = lambda_OLS, lambda_orth = lambda_orth, lambda_sparse = lambda_sparse))

    def build_model(self, feature_extractor, prediction_layer, ):
        model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

        model.build(input_shape=(None, width, height, 3))
        model.feature_extractor.summary()
        model.prediction_layer.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model


    def build_dg_model(self, feature_extractor, prediction_layer, ):
        model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

        model.build(input_shape=(None, width, height, 3))
        model.feature_extractor.summary()
        model.prediction_layer.summary()

        n_epoch = self.da_spec["epochs_FT"] if self.method == "SOURCE_ONLY" and self.fine_tune else self.da_spec["epochs_E2E"]
        gradient_accumulation_steps = 1
        n_training_steps = math.ceil(542 / gradient_accumulation_steps) * n_epoch #1270

        lr_decayed_fn = CustomSchedule(initial_lr=1e-3, warmup_steps= 5420, training_steps=n_training_steps)
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_decayed_fn, weight_decay=1e-5)


        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def run_experiment(self):
        # Create output folder
        pathlib.Path(self.save_dir_path).mkdir(parents=True, exist_ok=True)

        # Define the feature extractor
        if self.feature_extractor.lower() == 'lenet':
            print("LeNet")
            feature_extractor = get_lenet_feature_extractor()
        elif self.feature_extractor.lower() == 'resnet':
            print("ResNet")
            feature_extractor = get_resnet((width, height, 3))
            # feature_extractor = add_regularization(feature_extractor)
        elif self.feature_extractor.lower() == 'domainnet':
            print('domainNet')
            feature_extractor = get_domainnet_feature_extractor()
        elif self.feature_extractor.lower() == 'densenet':
            feature_extractor = get_dense_net()
        else:
            raise ValueError('Feature extractor not possible')

        # Define prediction layer
        prediction_layer = tf.keras.Sequential([], name='prediction_layer')
        if self.method == "SOURCE_ONLY":
            prediction_layer.add(Dense(units, activation=self.activation))
            model = self.build_model(feature_extractor, prediction_layer)
            # , use_bias=self.bias)) #, kernel_initializer=utils.Ortho()))  # , activation=activation, use_bias=bias))
        else:
            self.add_da_layer(prediction_layer)
            model = self.build_dg_model(feature_extractor, prediction_layer)

        # Initialize model
        # DomainAdaptationModel has one feature_extractor (that may be used in the fine tune stage)
        # and one prediction layer
        #model = self.build_model(feature_extractor, prediction_layer)
        print("\n\n\n BEGIN TRAIN:\t METHOD:{}\t\t\t target_domain: {}\n\n\n".format(self.method.upper(),
                                                                                     'rxrx1'))
        if not self.only_fine_tune:
            self.save_evaluation_files(model)

        # Fine tuning, used only if no DA layer is used in previous stage
        if self.method == "SOURCE_ONLY" and self.fine_tune:

            feature_extractor_filepath = os.path.join(self.save_dir_path + str(self.run_id), 'feature_extractor_best')
            pathlib.Path(feature_extractor_filepath).mkdir(parents=True, exist_ok=True)
            feature_extractor.save(feature_extractor_filepath)
            if self.feature_extractor_saved_path is not None:
                feature_extractor = tf.keras.models.load_model(self.feature_extractor_saved_path)
            feature_extractor.trainable = False
            for method in ['cs', 'mmd', 'projected']:
                # feature_extractor = tf.keras.models.load_model(feature_extractor_filepath)
                # feature_extractor.trainable = False

                self.da_spec["similarity_measure"] = method
                prediction_layer = tf.keras.Sequential([], name='prediction_layer')  # TODO: not sure about this
                self.add_da_layer(prediction_layer)

                model = self.build_dg_model(feature_extractor, prediction_layer)

                print('\n BEGIN FINE TUNING:\t' + method.upper() + "\t\n")
                self.save_evaluation_files(model, fine_tune=True)

        tf.keras.backend.clear_session()
