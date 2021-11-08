import logging
import os
import pathlib
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import *

from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer

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

# file path to the location where the results are stored
res_file_dir = "output"

SOURCE_SAMPLE_SIZE = 25000
TARGET_SAMPLE_SIZE = 9000
width, height = 96, 96


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


def get_resnet(input_shape):
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape)


class CamelyonClassification():
    def __init__(self, method, timestamp, target_domain, train_generator, valid_generator, test_generator,
                 kernel=None, batch_norm=False, bias=False,
                 save_file=True, save_plot=False,
                 save_feature=True, batch_size=64, fine_tune=False, lr=0.001, activation=None,
                 feature_extractor='LeNet', run=0):
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

        self.run_id = np.random.randint(0, 10000, 1)[0]
        self.save_dir_path = 'pathSaving'
        self.da_spec = self.create_da_spec()
        self.optimizer = tf.keras.optimizers.SGD(lr) \
            if self.da_spec['use_optim'].lower() == "sgd" else tf.keras.optimizers.Adam(lr)

        from_logits = self.activation != "softmax"

        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.metrics = [tf.keras.metrics.BinaryAccuracy(),
                        # ChallengeMetric(),
                        # WeightedCrossEntropy(pos_weight),
                        # MultilableAccuracy(),
                        tf.keras.metrics.BinaryCrossentropy(from_logits=from_logits)]

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0001)
        file_path = os.path.join(self.save_dir_path, 'best_model.hdf5')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callback = [EarlyStopping(patience=self.da_spec["patience"], restore_best_weights=True), reduce_lr,
                         model_checkpoint]

        print("\n FINISHED LOADING CAMELYON")

    def save_evaluation_files(self, model, fine_tune=False):
        method = self.da_spec["similarity_measure"]
        num_epochs = self.da_spec["epochs_FT"] if fine_tune else self.da_spec["epochs"]
        file_suffix = "_FT" if fine_tune else "E2E"
        run_start = datetime.now()

        hist = model.fit(x=self.train_generator,
                         epochs=num_epochs,
                         verbose=1,
                         validation_data=self.valid_generator,
                         callbacks=self.callback, )
        run_end = datetime.now()
        predictions = model.predict(self.test_generator)
        file_name_pred = "pred_camelyon_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run)
        pred_file_path = os.path.join(self.save_dir_path, file_name_pred)
        # TODO np.save(pred_file_path, predictions), don't need it?

        if self.save_file:
            hist_df = pd.DataFrame(hist.history)
            duration = run_end - run_start

            file_name_hist = "history_camelyon_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run)
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

            file_name_eval = "spec_camelyon_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run)
            eval_file_path = os.path.join(self.save_dir_path, file_name_eval)
            eval_df.to_csv(eval_file_path)

            if self.save_feature:
                df_file_path = os.path.join(self.save_dir_path,
                                            "{}_{}_{}_feature_data_camelyon.csv".format(method.upper(), file_suffix, self.run))
                pred_df = pd.DataFrame(predictions, columns=["x_{}".format(i) for i in range(1)])
                pred_df.to_csv(df_file_path)

    def create_da_spec(self):
        da_spec_dict = {"num_domains": 5, "domain_dim": 10, "sigma": 5.5, 'softness_param': 2,
                        "domain_reg_param": 1e-3, "batch_size": self.batch_size, "epochs": 250, "epochs_FT": 250,
                        "dropout": 0.5, "patience": 10, "use_optim": "adam", "orth_reg": "SRIP",
                        "source_sample_size": SOURCE_SAMPLE_SIZE, "target_sample_size": TARGET_SAMPLE_SIZE,
                        "architecture": "LeNet", "bias": self.bias, "similarity_measure": self.method, 'lr': self.lr,
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
        prediction_layer.add(BatchNormalization())
        prediction_layer.add(
            DGLayer(domain_units=num_domains, N=domain_dim, softness_param=softness_param, units=1,
                    kernel=self.kernel, sigma=sigma, activation=self.activation, bias=self.bias,
                    similarity_measure=similarity_measure, orth_reg_method=reg_method))  # TODO: check orth_pen_method

    def build_model(self, feature_extractor, prediction_layer, ):
        model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

        model.build(input_shape=(None, width, height, 3))
        model.feature_extractor.summary()
        model.prediction_layer.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, )
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
            feature_extractor = get_resnet((width, height, 3))  # TODO: remove hard-coded references
        else:
            raise ValueError('Feature extractor not possible')

        # Define prediction layer
        prediction_layer = tf.keras.Sequential([], name='prediction_layer')
        if self.method == "SOURCE_ONLY":
            prediction_layer.add(Dense(1))  # , activation=activation, use_bias=bias))
        else:
            self.add_da_layer(prediction_layer)

        # Initialize model
        # DomainAdaptationModel has one feature_extractor (that may be used in the fine tune stage)
        # and one prediction layer
        model = self.build_model(feature_extractor, prediction_layer)
        print("\n\n\n BEGIN TRAIN:\t METHOD:{}\t\t\t target_domain: {}\n\n\n".format(self.method.upper(),
                                                                                     'camelyon'))
        self.save_evaluation_files(model)

        # Fine tuning, used only if no DA layer is used in previous stage
        if self.method == "SOURCE_ONLY" and self.fine_tune:

            feature_extractor_filepath = os.path.join(self.save_dir_path, 'feature_extractor.h5.tmp')
            feature_extractor.save(feature_extractor_filepath)

            for method in ['cs', 'mmd', 'projected']:
                feature_extractor = tf.keras.models.load_model(feature_extractor_filepath)
                feature_extractor.trainable = False

                self.da_spec["similarity_measure"] = method
                self.add_da_layer(prediction_layer)

                model = self.build_model(feature_extractor, prediction_layer)

                print('\n BEGIN FINE TUNING:\t' + method.upper() + "\t" + self.target_domain[0] + "\n")
                self.save_evaluation_files(model, fine_tune=True)

        tf.keras.backend.clear_session()
