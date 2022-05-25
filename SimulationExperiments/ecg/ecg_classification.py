import itertools
import logging
import os
import pathlib
# import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
# from silence_tensorflow import silence_tensorflow
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import *

from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
from ECG_utils.losses import WeightedCrossEntropyLoss
from ECG_utils.metrics import compute_challenge_metric_custom, ChallengeMetric, \
    WeightedCrossEntropy, MultilableAccuracy
from ecg_dataset import ECGData

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


def get_lenet_feature_extractor():
    feature_exctractor = tf.keras.Sequential(
        [Conv1D(32, kernel_size=3, activation='relu', input_shape=(None, 12)), BatchNormalization(),
         MaxPool1D(pool_size=2, strides=2), Conv1D(64, kernel_size=2, activation='relu'), BatchNormalization(),
         GlobalAveragePooling1D(), Dense(100, activation="relu"), Dense(100, activation="relu")],
        name='feature_extractor_lenet_ecg')
    return feature_exctractor


class ECGClassification:
    def __init__(self, method, target_domain, single_source_domain=None, batch_norm=False, lr=0.001, batch_size=64,
                 save_file=False, save_plot=False, save_feature=False, activation=None, bias=False, fine_tune=True,
                 timestamp="", kernel=None, data: ECGData = None):
        self.method = "SOURCE_ONLY" if method is None else method
        self.target_domain = target_domain
        if single_source_domain is not None:
            # in case where only one single source domain is chosen
            self.source_domains = [single_source_domain]
        else:
            self.source_domains = ECGData.DOMAINS

        self.batch_norm = batch_norm
        self.lr = lr
        self.save_file = save_file
        self.save_plot = save_plot
        self.save_feature = save_feature
        self.activation = activation
        self.bias = bias
        self.fine_tune = fine_tune
        self.kernel = kernel
        self.data = data
        self.batch_size = batch_size

        self.run_id = np.random.randint(0, 10000, 1)[0]
        if single_source_domain is not None:
            save_dir_name = self.method.upper() + "_" + single_source_domain + "_to_" + self.target_domain[
                0] + "_" + str(self.run_id)
        else:
            save_dir_name = self.method.upper() + "_" + self.target_domain[0] + "_" + str(self.run_id)
        self.save_dir_path = os.path.join(res_file_dir, timestamp,
                                          "SINGLE_BEST" if single_source_domain is not None else "SOURCE_COMBINED",
                                          self.target_domain[0], save_dir_name)

        self.da_spec = self.create_da_spec()
        self.optimizer = tf.keras.optimizers.SGD(lr) if self.da_spec[
                                                            'use_optim'].lower() == "sgd" else tf.keras.optimizers.Adam(
            lr)

        # Prepare data
        x_map_fn = {"train": lambda source: data.x_train_dict[source], "test": lambda source: data.x_test_dict[source]}
        y_map_fn = {"train": lambda source: data.y_train_dict[source], "test": lambda source: data.y_test_dict[source]}

        def get_domain_data(domain_list, mode="test"):
            print(domain_list, mode)
            x = list(itertools.chain(*list(map(x_map_fn[mode], domain_list))))
            x = [np.array(rec).transpose() for rec in x]
            # cut a recording if it's too long
            x = [rec if rec.shape[0] < 15000 else rec[:15000, :] for rec in x]
            y = list(itertools.chain(*list(map(y_map_fn[mode], domain_list))))
            return x, y

        def create_dataset(x, y, mode="test"):
            if mode == "train":
                x, y = shuffle(x, y, random_state=1234)
            return tf.data.Dataset.from_generator(lambda: zip(x, y),
                                                  output_shapes=(tf.TensorShape([None, 12]), tf.TensorShape([24])),
                                                  output_types=(tf.float32, tf.float32)).padded_batch(self.batch_size)

        no_target_domains = [source.lower() for source in self.source_domains if
                             source.lower() != self.target_domain[0].lower()]

        x_source_tr, y_source_tr = get_domain_data(no_target_domains, mode="train")
        x_source_te, y_source_te = get_domain_data(no_target_domains, mode="test")
        x_target_te, y_target_te = get_domain_data(self.target_domain, mode="test")

        print("Source data size:", np.array(x_source_tr).shape)
        print("Target data size:", np.array(x_target_te).shape)

        self.x_source_tr = create_dataset(x_source_tr, y_source_tr, mode="train")
        self.x_source_te = create_dataset(x_source_te, y_source_te, mode="test")
        self.x_target_te = create_dataset(x_target_te, y_target_te, mode="test")
        self.y_target_te = y_target_te

        from_logits = self.activation != "softmax"

        n_samples = np.array(y_source_tr).shape[0]
        pos_count = np.sum(y_source_tr, axis=0)
        neg_count = n_samples - pos_count
        pos_weight = neg_count / pos_count
        pos_weight[pos_weight == np.inf] = 1.0
        self.loss = WeightedCrossEntropyLoss(pos_weight=pos_weight)
        self.metrics = [tf.keras.metrics.BinaryAccuracy(), ChallengeMetric(), WeightedCrossEntropy(pos_weight),
                        MultilableAccuracy(), tf.keras.metrics.BinaryCrossentropy(from_logits=from_logits)]

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0001)
        file_path = os.path.join(self.save_dir_path, 'best_model.hdf5')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callback = [EarlyStopping(patience=self.da_spec["patience"], restore_best_weights=True), reduce_lr,
                         model_checkpoint]

        print("\n FINISHED LOADING ECGS")

    def save_evaluation_files(self, model, fine_tune=False):
        method = self.da_spec["similarity_measure"]
        num_epochs = self.da_spec["epochs_FT"] if fine_tune else self.da_spec["epochs"]
        file_suffix = "_FT" if fine_tune else ""
        run_start = datetime.now()

        hist = model.fit(x=self.x_source_tr, epochs=num_epochs, verbose=1, shuffle=False,
                         validation_data=self.x_source_te, callbacks=self.callback, )

        run_end = datetime.now()

        predictions = model.predict(self.x_target_te)
        file_name_pred = "pred_{}{}.csv".format(method.upper(), file_suffix)
        pred_file_path = os.path.join(self.save_dir_path, file_name_pred)
        np.save(pred_file_path, predictions)

        if self.save_file:
            hist_df = pd.DataFrame(hist.history)
            duration = run_end - run_start

            file_name_hist = "history_{}{}.csv".format(method.upper(), file_suffix)
            hist_file_path = os.path.join(self.save_dir_path, file_name_hist)
            hist_df.to_csv(hist_file_path)

            # prepare results
            model_res = model.evaluate(self.x_target_te, verbose=1)
            metric_names = model.metrics_names
            eval_df = pd.DataFrame(model_res).transpose()
            eval_df.columns = metric_names

            print(compute_challenge_metric_custom(predictions, np.array(self.y_target_te)))
            eval_df["challenge_metric"] = compute_challenge_metric_custom(predictions, np.array(self.y_target_te))

            eval_df['source_domain'] = ",".join(self.target_domain)
            eval_df['target_domain'] = ",".join(self.source_domains)

            eval_df = pd.concat([eval_df, pd.DataFrame.from_dict([self.da_spec])], axis=1)
            eval_df['duration'] = duration
            eval_df['run_id'] = self.run_id
            eval_df['trained_epochs'] = len(hist_df)

            file_name_eval = "spec_{}{}.csv".format(method.upper(), file_suffix)
            eval_file_path = os.path.join(self.save_dir_path, file_name_eval)
            eval_df.to_csv(eval_file_path)

            if self.save_feature:
                df_file_path = os.path.join(self.save_dir_path,
                                            "{}{}_feature_data.csv".format(method.upper(), file_suffix))
                pred_df = pd.DataFrame(predictions, columns=["x_{}".format(i) for i in range(ECGData.NUM_CLASSES)])
                pred_df.to_csv(df_file_path)

    def create_da_spec(self):
        da_spec_dict = {"num_domains": 10, "domain_dim": 10, "sigma": 5.5, 'softness_param': 2,
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
            DGLayer(domain_units=num_domains, N=domain_dim, softness_param=softness_param, units=ECGData.NUM_CLASSES,
                    kernel=self.kernel, sigma=sigma, activation=self.activation, bias=self.bias,
                    similarity_measure=similarity_measure, orth_reg_method=reg_method, ))

    def build_model(self, feature_extractor, prediction_layer,):
        model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

        model.build(input_shape=(None, None, 12))
        model.feature_extractor.summary()
        model.prediction_layer.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, )
        return model

    def run_experiment(self):
        # Create output folder
        pathlib.Path(self.save_dir_path).mkdir(parents=True)

        # Define the feature extractor
        print("LeNet")
        feature_extractor = get_lenet_feature_extractor()

        # Define prediction layer
        prediction_layer = tf.keras.Sequential([], name='prediction_layer')
        if self.method == "SOURCE_ONLY":
            prediction_layer.add(Dense(ECGData.NUM_CLASSES))  # , activation=activation, use_bias=bias))
        else:
            self.add_da_layer(prediction_layer)

        # Initialize model
        # DomainAdaptationModel has one feature_extractor (that may be used in the fine tune stage)
        # and one prediction layer
        model = self.build_model(feature_extractor, prediction_layer)
        print("\n\n\n BEGIN TRAIN:\t METHOD:{}\t\t\t target_domain: {}\n\n\n".format(self.method.upper(),
                                                                                     self.target_domain[0]))
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
