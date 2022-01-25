import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../..')))
THIS_FILE = os.path.abspath(__file__)


import argparse
import logging
import os
import pathlib
import sys
import warnings
from datetime import datetime




import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import *
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader




warnings.filterwarnings("ignore", category=DeprecationWarning)
# tf.random.set_seed(1234)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

batch_size = 16
warnings.filterwarnings("ignore", category=DeprecationWarning)
# silence_tensorflow()
# tf.random.set_seed(1234)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# file path to the location where the results are stored
res_file_dir = "output"

width, height = 448, 448
img_shape = (width, height, 3)
units = 182


import copy
import pdb
from typing import Dict, List, Union

import torch
from wilds.common.utils import get_counts
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
import warnings

class Grouper:
    """
    Groupers group data points together based on their metadata.
    They are used for training and evaluation,
    e.g., to measure the accuracies of different groups of data.
    """
    def __init__(self):
        raise NotImplementedError

    @property
    def n_groups(self):
        """
        The number of groups defined by this Grouper.
        """
        return self._n_groups

    def metadata_to_group(self, metadata, return_counts=False):
        """
        Args:
            - metadata (Tensor): An n x d matrix containing d metadata fields
                                 for n different points.
            - return_counts (bool): If True, return group counts as well.
        Output:
            - group (Tensor): An n-length vector of groups.
            - group_counts (Tensor): Optional, depending on return_counts.
                                     An n_group-length vector of integers containing the
                                     numbers of data points in each group in the metadata.
        """
        raise NotImplementedError

    def group_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the pretty name of that group.
        """
        raise NotImplementedError

    def group_field_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the name of that group.
        """
        raise NotImplementedError

class CombinatorialGrouper(Grouper):
    def __init__(self, dataset, groupby_fields):
        """
        CombinatorialGroupers form groups by taking all possible combinations of the metadata
        fields specified in groupby_fields, in lexicographical order.
        For example, if:
            dataset.metadata_fields = ['country', 'time', 'y']
            groupby_fields = ['country', 'time']
        and if in dataset.metadata, country is in {0, 1} and time is in {0, 1, 2},
        then the grouper will assign groups in the following way:
            country = 0, time = 0 -> group 0
            country = 1, time = 0 -> group 1
            country = 0, time = 1 -> group 2
            country = 1, time = 1 -> group 3
            country = 0, time = 2 -> group 4
            country = 1, time = 2 -> group 5

        If groupby_fields is None, then all data points are assigned to group 0.

        Args:
            - dataset (WILDSDataset or list of WILDSDataset)
            - groupby_fields (list of str)
        """
        if isinstance(dataset, list):
            if len(dataset) == 0:
                raise ValueError("At least one dataset must be defined for Grouper.")
            datasets: List[WILDSDataset] = dataset
        else:
            datasets: List[WILDSDataset] = [dataset]

        metadata_fields: List[str] = datasets[0].metadata_fields
        # Build the largest metadata_map to see to check if all the metadata_maps are subsets of each other
        largest_metadata_map: Dict[str, Union[List, np.ndarray]] = copy.deepcopy(datasets[0].metadata_map)
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, WILDSSubset):
                raise ValueError("Grouper should be defined with full dataset(s) and not subset(s).")

            # The first dataset was used to get the metadata_fields and initial metadata_map
            if i == 0:
                continue

            if dataset.metadata_fields != metadata_fields:
                raise ValueError(
                    f"The datasets passed in have different metadata_fields: {dataset.metadata_fields}. "
                    f"Expected: {metadata_fields}"
                )

            if dataset.metadata_map is None: continue
            for field, values in dataset.metadata_map.items():
                n_overlap = min(len(values), len(largest_metadata_map[field]))
                if not (np.asarray(values[:n_overlap]) == np.asarray(largest_metadata_map[field][:n_overlap])).all():
                    raise ValueError("The metadata_maps of the datasets need to be ordered subsets of each other.")

                if len(values) > len(largest_metadata_map[field]):
                    largest_metadata_map[field] = values

        self.groupby_fields = groupby_fields
        if groupby_fields is None:
            self._n_groups = 1
        else:
            self.groupby_field_indices = [i for (i, field) in enumerate(metadata_fields) if field in groupby_fields]
            if len(self.groupby_field_indices) != len(self.groupby_fields):
                raise ValueError('At least one group field not found in dataset.metadata_fields')

            metadata_array = torch.cat([dataset.metadata_array for dataset in datasets])
            grouped_metadata = metadata_array[:, self.groupby_field_indices]
            if not isinstance(grouped_metadata, torch.LongTensor):
                grouped_metadata_long = grouped_metadata.long()
                if not torch.all(grouped_metadata == grouped_metadata_long):
                    warnings.warn(f'CombinatorialGrouper: converting metadata with fields [{", ".join(groupby_fields)}] into long')
                grouped_metadata = grouped_metadata_long

            for idx, field in enumerate(self.groupby_fields):
                min_value = grouped_metadata[:,idx].min()
                if min_value < 0:
                    raise ValueError(f"Metadata for CombinatorialGrouper cannot have values less than 0: {field}, {min_value}")
                if min_value > 0:
                    warnings.warn(f"Minimum metadata value for CombinatorialGrouper is not 0 ({field}, {min_value}). This will result in empty groups")

            # We assume that the metadata fields are integers,
            # so we can measure the cardinality of each field by taking its max + 1.
            # Note that this might result in some empty groups.
            assert grouped_metadata.min() >= 0, "Group numbers cannot be negative."
            self.cardinality = 1 + torch.max(grouped_metadata, dim=0)[0]
            cumprod = torch.cumprod(self.cardinality, dim=0)
            self._n_groups = cumprod[-1].item()
            self.factors_np = np.concatenate(([1], cumprod[:-1]))
            self.factors = torch.from_numpy(self.factors_np)
            self.metadata_map = largest_metadata_map

    def metadata_to_group(self, metadata, return_counts=False):
        if self.groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = metadata[:, self.groupby_field_indices].long() @ self.factors

        if return_counts:
            group_counts = get_counts(groups, self._n_groups)
            return groups, group_counts
        else:
            return groups

    def group_str(self, group):
        if self.groupby_fields is None:
            return 'all'

        # group is just an integer, not a Tensor
        n = len(self.factors_np)
        metadata = np.zeros(n)
        for i in range(n-1):
            metadata[i] = (group % self.factors_np[i+1]) // self.factors_np[i]
        metadata[n-1] = group // self.factors_np[n-1]
        group_name = ''
        for i in reversed(range(n)):
            meta_val = int(metadata[i])
            if self.metadata_map is not None:
                if self.groupby_fields[i] in self.metadata_map:
                    meta_val = self.metadata_map[self.groupby_fields[i]][meta_val]
            group_name += f'{self.groupby_fields[i]} = {meta_val}, '
        group_name = group_name[:-2]
        return group_name

        # a_n = S / x_n
        # a_{n-1} = (S % x_n) / x_{n-1}
        # a_{n-2} = (S % x_{n-1}) / x_{n-2}
        # ...
        #
        # g =
        # a_1 * x_1 +
        # a_2 * x_2 + ...
        # a_n * x_n

    def group_field_str(self, group):
        return self.group_str(group).replace('=', ':').replace(',','_').replace(' ','')

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_loader, x_path=None, y_path=None, batch_size=32, save_file=True, load_files=True,
                 one_hot=False, return_weights=False, weights_path=None, leave_torch_shape=False):
        """
        :param data_loader: 'torch.DataLoader'
        :param x_path: 'string'
            specifies the location of the full numpy matrix for x values
        :param y_path: 'string'
            specifies the location of the full numpy matrix for x values
        :param batch_size: 'int'
        :param save_file: 'bool'
            if set to True, it will try to save the entire numpy matrix
        :param load_files: 'bool'
            if set to True loads the data entirely into a numpy array
        :param one_hot: 'bool'
            if set to True, y is returned as one_hot vector, necessary when there are more than 2 classes in the output
        :param return_weights: 'bool'
            if set to True weights are calculated to deal with class imbalance, inverse of the class frequency
        :param weights_path: 'string'
            location of weights array for imbalanced datasets
        :param leave_torch_shape: 'bool'
            if True returns torch shape as in the wilds dataloader
        """
        super(DataGenerator, self).__init__()
        self.data_loader = data_loader
        self.iterator = iter(data_loader)
        self.x_path = x_path
        self.y_path = y_path
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.x_full = None
        self.y_full = None
        self.weights = None
        self.save_file = save_file
        self.load_files = load_files
        self.one_hot = one_hot
        self.return_weights = return_weights
        self.leave_torch_shape = leave_torch_shape
        if self.return_weights:
            if self.weights_path is not None:
                self.weights = np.load(self.weights_path)
            else:
                self.weights = np.zeros(units)

        if self.load_files:
            self.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.y_full) / self.batch_size)) if self.load_files else len(self.data_loader)

    def load(self):
        """Load the entire dataset into the memory"""
        if self.x_path is not None and self.y_path is not None:
            self.x_full = np.load(self.x_path)
            self.y_full = np.load(self.y_path)
            print('Loaded ', self.x_path, self.y_path)
        else:
            for x, y, metadata in tqdm(self.data_loader):
                y = y.numpy()
                if self.return_weights:
                    unique, counts = np.unique(y, return_counts=True)
                    self.weights[unique] += counts
                if self.one_hot:
                    y = one_hot(y, units)
                x = x.permute(0, 2, 3, 1).numpy()
                #B, C, W, H = x.shape
                #x = x.reshape(B, W, H, C)
                if self.x_full is None:
                    self.x_full = x
                    self.y_full = y
                else:
                    self.x_full = np.concatenate([self.x_full, x])
                    self.y_full = np.concatenate([self.y_full, y])
            x_file_name = 'x_full' + str(np.random.randint(1000, size=1)[0]) + '.npy'
            y_file_name = 'y_full' + str(np.random.randint(1000, size=1)[0]) + '.npy'
            if self.save_file:
                try:
                    np.save(x_file_name, self.x_full)
                    np.save(y_file_name, self.y_full)
                    print('Saved as ', x_file_name, y_file_name)
                except Exception:
                    print('Not enough space to save ', x_file_name, y_file_name)
            else:
                print('Not saving files')

            if self.return_weights:
                self.weights = self.weights.sum() / self.weights

    def on_epoch_end(self):
        """Shuffle data at the end of every epoch"""
        if self.load_files:
            self.x_full, self.y_full = shuffle(self.x_full, self.y_full)
        else:
            pass

    def __getitem__(self, index):
        if self.load_files:
            x = self.x_full[index * self.batch_size:(index + 1) * self.batch_size]
            y = self.y_full[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            try:
                x, y, metadata = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                x, y, metadata = next(self.iterator)
            y = y.numpy()
            if self.one_hot:
                y = one_hot(y, units)
            if not self.leave_torch_shape:
                x = x.permute(0, 2, 3, 1).numpy()
                #B, C, W, H = x.shape
                #x = x.reshape(B, W, H, C)
        if self.return_weights:
            w = self.get_weights(y)
            return x, y, w
        else:
            return x, y

    def get_weights(self, y):
        """y has to be one_hot encoding"""
        return self.weights[np.argmax(y, axis=1)]


def one_hot(x, depth):
    return_arr = np.zeros((x.size, depth))
    return_arr[np.arange(x.size), x] = 1.0
    return return_arr


def get_resnet(input_shape):
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    feature_extractor = tf.keras.Sequential([resnet, tf.keras.layers.GlobalAveragePooling2D()],
                                            name='feature_extractor_resnet')
    return feature_extractor


class WildcamClassification():
    def __init__(self, method, timestamp, target_domain, train_generator, valid_generator, test_generator,
                 metadata, y_true, dataset,
                 kernel=None, batch_norm=False, bias=False,
                 save_file=True, save_plot=False,
                 save_feature=True, batch_size=64, fine_tune=False, lr=3e-5, activation=None,
                 feature_extractor='LeNet', run=0, only_fine_tune=False, feature_extractor_saved_path=None, ftmet=0):
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
        self.ftmet = ftmet

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

        self.run_id = np.random.randint(0, 10000, 1)[0]
        self.save_dir_path = 'pathSaving'
        self.da_spec = self.create_da_spec()
        self.optimizer = tf.keras.optimizers.SGD(lr) \
            if self.da_spec['use_optim'].lower() == "sgd" else tf.keras.optimizers.Adam(lr)

        from_logits = self.activation != "softmax"

        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
        self.metrics = [tf.keras.metrics.CategoricalAccuracy(),
                        tf.keras.metrics.CategoricalCrossentropy(from_logits=from_logits),
                        tfa.metrics.F1Score(num_classes=units, average='macro')
                        ]

        #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0001)
        #file_path = '/cluster/project/jbuhmann/alinadu/gdu_wildcam/SimulationExperiments/experiments_wilds/pathSaving/model' + str(self.run_id)
        #model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        #pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        #model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        #self.callback = [model_checkpoint]   #[EarlyStopping(patience=10, restore_best_weights=True), reduce_lr, model_checkpoint]

        print("\n FINISHED LOADING WILDS")

    def save_evaluation_files(self, model, fine_tune=False):
        method = self.da_spec["similarity_measure"]
        num_epochs = self.da_spec["epochs_FT"] if fine_tune else self.da_spec["epochs"]
        file_suffix = "_FT" if fine_tune else "E2E"
        run_start = datetime.now()

        hist = model.fit(x=self.train_generator,
                         epochs=1#num_epochs,
                         verbose=1,
                         validation_data=self.test_generator,
                         #callbacks=self.callback,
                         )
        run_end = datetime.now()
        predictions = model.predict(self.test_generator)
        y_pred = torch.tensor(np.argmax(predictions, axis=1))
        print(dataset.eval(y_pred, dataset.y_array, dataset.metadata_array))

        file_name_pred = "pred_camelyon_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run)
        pred_file_path = os.path.join(self.save_dir_path, file_name_pred)
        # TODO np.save(pred_file_path, predictions), don't need it?

        if self.save_file:
            hist_df = pd.DataFrame(hist.history)
            duration = run_end - run_start

            file_name_hist = "history_camelyon_{}_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run,
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

            file_name_eval = "spec_camelyon_{}_{}_{}_{}.csv".format(method.upper(), file_suffix, self.run, self.run_id)
            eval_file_path = os.path.join(self.save_dir_path, file_name_eval)
            print('EVAL_DF\n\n', eval_df)
            eval_df.to_csv(eval_file_path)

            if self.save_feature:
                df_file_path = os.path.join(self.save_dir_path,
                                            "{}_{}_{}_{}_feature_data_camelyon.csv".format(method.upper(), file_suffix,
                                                                                           self.run, self.run_id))
                pred_df = pd.DataFrame(predictions, columns=["x_{}".format(i) for i in range(1)])
                pred_df.to_csv(df_file_path)

    def create_da_spec(self):
        da_spec_dict = {"num_domains": 10, "domain_dim": 10, "sigma": 25, 'softness_param': 2,
                        "domain_reg_param": 1e-3, "batch_size": self.batch_size, "epochs": 12, "epochs_FT": 12,
                        "dropout": 0.5, "patience": 10, "use_optim": "adam", "orth_reg": "SRIP",
                        "source_sample_size": len(self.train_generator), "target_sample_size": len(self.test_generator),
                        "architecture": self.feature_extractor, "bias": self.bias, "similarity_measure": self.method, 'lr': self.lr,
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
        prediction_layer.add(tf.keras.layers.BatchNormalization())
        prediction_layer.add(
            DGLayer(domain_units=num_domains, N=domain_dim, softness_param=softness_param, units=units,
                    kernel=self.kernel, sigma=sigma, activation=self.activation, bias=self.bias,
                    similarity_measure=similarity_measure, orth_reg_method=reg_method,
                    lambda_orth=1e-6, lambda_sparse=0.0, lambda_OLS=1e-3))


    def build_model(self, feature_extractor, prediction_layer, ):
        model = DomainAdaptationModel(feature_extractor=feature_extractor, prediction_layer=prediction_layer)

        model.build(input_shape=(None, width, height, 3))
        model.feature_extractor.summary()
        model.prediction_layer.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, )
        return model

    def run_experiment(self) -> object:
        # Create output folder
        pathlib.Path(self.save_dir_path).mkdir(parents=True, exist_ok=True)

        # Define the feature extractor
        print("ResNet")
        feature_extractor = get_resnet((width, height, 3))

        # Define prediction layer
        prediction_layer = tf.keras.Sequential([], name='prediction_layer')
        if self.method == "SOURCE_ONLY":
            prediction_layer.add(Dense(units, activation=self.activation, use_bias=self.bias, ))
        else:
            self.add_da_layer(prediction_layer)

        # Initialize model
        model = self.build_model(feature_extractor, prediction_layer)
        print("\n\n\n BEGIN TRAIN:\t ")

        self.save_evaluation_files(model)

        if self.method == "SOURCE_ONLY" and self.fine_tune:

            feature_extractor_filepath = os.path.join(self.save_dir_path + str(self.run_id), 'feature_extractor_best')
            pathlib.Path(feature_extractor_filepath).mkdir(parents=True, exist_ok=True)
            feature_extractor.save(feature_extractor_filepath)
            if self.feature_extractor_saved_path is not None:
                feature_extractor = tf.keras.models.load_model(self.feature_extractor_saved_path)
            feature_extractor.trainable = False
            methods = ['projected']
            method = methods[self.ftmet]
            #for method in ['cs', 'mmd', 'projected']:
                # feature_extractor = tf.keras.models.load_model(feature_extractor_filepath)
                # feature_extractor.trainable = False

            self.da_spec["similarity_measure"] = method
            prediction_layer = tf.keras.Sequential([], name='prediction_layer')  # TODO: not sure about this
            self.add_da_layer(prediction_layer)

            model = self.build_model(feature_extractor, prediction_layer)

            print('\n BEGIN FINE TUNING:\t' + method.upper() + "\t\n")
            self.save_evaluation_files(model, fine_tune=True)

        tf.keras.backend.clear_session()


def initialize_transform():
    transform_steps = [transforms.Resize((448, 448))]
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]
    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    transform_steps.append(transforms.ToTensor())
    transform_steps.append(default_normalization)
    transform = transforms.Compose(transform_steps)

    return transform


def get_wilds_data():
    # Specify the wilds dataset
    dataset = get_dataset(dataset='iwildcam', download=True)


    #val_grouper.metadata_to_group(metadata)

    train_data = dataset.get_subset('train', transform=initialize_transform())
    valid_data = dataset.get_subset('val', transform=initialize_transform())
    test_data = dataset.get_subset('test', transform=initialize_transform())
    metadata = test_data.metadata_array
    y_true = test_data.y_array

    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size) #n_groups_per_batch=2)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size) #n_groups_per_batch=2)
    test_loader = get_eval_loader('standard', test_data, batch_size=batch_size)

    return DataGenerator(train_loader, batch_size=batch_size, one_hot=True, save_file=False, return_weights=False,
                         load_files=False), \
           DataGenerator(valid_loader, batch_size=batch_size, one_hot=True, save_file=False, load_files=False), \
           DataGenerator(test_loader, save_file=False, batch_size=batch_size, one_hot=True, load_files=False),\
           metadata, y_true, test_data

def parser_args():
    parser = argparse.ArgumentParser(description='Wilds classification')
    parser.add_argument('--method',
                        help='cosine_similarity, MMD, projected, None',
                        type=str,
                        default="projected")

    parser.add_argument('--lambda_sparse',
                        default=0,
                        type=float)

    parser.add_argument('--lambda_OLS',
                        type=float,
                        default=1e-3)

    parser.add_argument('--lambda_orth',
                        type=float,
                        default=1e-6)

    parser.add_argument('--early_stopping',
                        type=bool,
                        default=True)

    parser.add_argument('--fine_tune',
                        type=str,
                        default="True")

    parser.add_argument('--running',
                        type=int,
                        default=0)

    parser.add_argument('--ft',
                        type=bool,
                        default=True)

    parser.add_argument('--num_domains',
                        type=int,
                        default=5)

    parser.add_argument('--ftmet',
                        type=int,
                        default=0)

    parser.add_argument('--fe_path',
                        type=str,
                        default='')

    args = parser.parse_args()
    if args.method == 'None':
        args.method = None
    args.ft = True if args.fine_tune == "True" else False
    return args

if __name__ == "__main__":
    # load data once
    args = parser_args()
    train_generator, valid_generator, test_generator, metadata, y_true, dataset  = get_wilds_data()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    WildcamClassification(train_generator=train_generator,
                          valid_generator=test_generator,
                          test_generator=test_generator,
                          method=args.method, kernel=None, batch_norm=False, bias=False,
                          timestamp=timestamp, target_domain=None, save_file=True, save_plot=False,
                          save_feature=False, batch_size=batch_size, fine_tune=args.ft,
                          feature_extractor='ResNet', run=args.running,
                          only_fine_tune=False, activation='softmax',
                          metadata =metadata, y_true = y_true, dataset = dataset,
                          #feature_extractor_saved_path=args.fe_path
                          ftmet=args.ftmet,).run_experiment()

