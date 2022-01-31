import argparse
import copy
import torchvision
import warnings
import torch.nn as nn
import wilds
import tensorflow as tf
from utils_wilds import ParseKwargs, parse_bool
from SimulationExperiments.wilds.archive.wildcam_classification import WildcamClassification
from SimulationExperiments.wilds.archive.wildcam_generator import WildcamDataGenerator
from configs.algorithm import algorithm_defaults
from configs.model import model_defaults
from configs.scheduler import scheduler_defaults
from configs.data_loader import loader_defaults
from configs.datasets import dataset_defaults, split_defaults
from utils_wilds import load
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.random.set_seed(1234)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

'''
nohup /local/home/pokanovic/miniconda3/envs/gdu4dg/bin/python3.8 -u /local/home/pokanovic/project2/SimulationExperiments/wilds/get_coral_model.py --algorithm ERM --root_dir data --dataset iwildcam > /local/home/pokanovic/project2/SimulationExperiments/wilds/wildcamresnet1.log 2>&1 &

'''

def populate_defaults(config):
    """Populates hyperparameters with defaults implied by choices
    of other hyperparameters."""

    orig_config = copy.deepcopy(config)
    assert config.dataset is not None, 'dataset must be specified'
    assert config.algorithm is not None, 'algorithm must be specified'

    # implied defaults from choice of dataset
    config = populate_config(
        config,
        dataset_defaults[config.dataset]
    )

    # implied defaults from choice of split
    if config.dataset in split_defaults and config.split_scheme in split_defaults[config.dataset]:
        config = populate_config(
            config,
            split_defaults[config.dataset][config.split_scheme]
        )

    # implied defaults from choice of algorithm
    config = populate_config(
        config,
        algorithm_defaults[config.algorithm]
    )

    # implied defaults from choice of loader
    config = populate_config(
        config,
        loader_defaults
    )
    # implied defaults from choice of model
    if config.model: config = populate_config(
        config,
        model_defaults[config.model],
    )

    # implied defaults from choice of scheduler
    if config.scheduler: config = populate_config(
        config,
        scheduler_defaults[config.scheduler]
    )

    # misc implied defaults
    if config.groupby_fields is None:
        config.no_group_logging = True
    config.no_group_logging = bool(config.no_group_logging)

    # basic checks
    required_fields = [
        'split_scheme', 'train_loader', 'uniform_over_groups', 'batch_size', 'eval_loader', 'model', 'loss_function',
        'val_metric', 'val_metric_decreasing', 'n_epochs', 'optimizer', 'lr', 'weight_decay',
    ]
    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    # data loader validations
    # we only raise this error if the train_loader is standard, and
    # n_groups_per_batch or distinct_groups are
    # specified by the user (instead of populated as a default)
    if config.train_loader == 'standard':
        if orig_config.n_groups_per_batch is not None:
            raise ValueError(
                "n_groups_per_batch cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")
        if orig_config.distinct_groups is not None:
            raise ValueError(
                "distinct_groups cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")

    return config


def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = vars(config)
    for key, val in template.items():
        if not isinstance(val, dict):  # config[key] expected to be a non-index-able
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")

        else:  # config[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if kwargs_key not in d_config[key] or d_config[key][kwargs_key] is None:
                    d_config[key][kwargs_key] = kwargs_val
                elif d_config[key][kwargs_key] != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")
    return config


def get_config():
    ''' to see default hyperparams for each dataset/model, look at configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, )
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme',
                        help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to downloads the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str)

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')

    # Model
    parser.add_argument('--model')
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')

    # Transforms
    parser.add_argument('--transform')
    parser.add_argument('--target_resolution', nargs='+', type=int,
                        help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)

    # Objective
    parser.add_argument('--loss_function', )
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--algo_log_metric')

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', )
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={})

    # Scheduler
    parser.add_argument('--scheduler', )
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', )
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int,
                        help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False)

    config = parser.parse_args()
    config = populate_defaults(config)
    return config


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


def initialize_model(config, d_out, is_featurizer=False):
    model = initialize_torchvision_model(
        name=config.model,
        d_out=d_out,
        **config.model_kwargs)
    return model


def initialize_torchvision_model(name, d_out, **kwargs):
    constructor_name = 'resnet50'
    last_layer_name = 'fc'

    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    return model


def get_wilds_data(resnet):
    # Specify the wilds dataset
    dataset = get_dataset(dataset='iwildcam', download=True)

    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((
        width, height)), transforms.ToTensor()]))
    valid_data = dataset.get_subset('val', transform=transforms.Compose([transforms.Resize((
        width, height)), transforms.ToTensor()]))
    test_data = dataset.get_subset('test', transform=transforms.Compose([transforms.Resize((
        width, height)), transforms.ToTensor()]))

    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size)
    test_loader = get_train_loader('standard', test_data, batch_size=batch_size)

    return WildcamDataGenerator(train_loader, resnet, batch_size=batch_size, one_hot=True, save_file=False,
                                x_path='x_full_wildcam_resnet_train.npy', y_path='y_full_wildcam_resnet_train.npy',
                                return_weights=False, load_files=True, ), \
           WildcamDataGenerator(valid_loader, resnet, batch_size=batch_size, one_hot=True, save_file=False,
                                x_path='x_full_wildcam_resnet_valid.npy', y_path='y_full_wildcam_resnet_valid.npy',
                                load_files=True, ), \
           WildcamDataGenerator(test_loader, resnet, batch_size=batch_size, one_hot=True, save_file=False,
                                x_path='x_full_wildcam_resnet_test.npy', y_path='y_full_wildcam_resnet_test.npy',
                                load_files=True, )


if __name__ == '__main__':
    d_out = 182  # num of classes
    config = get_config()
    resnet = initialize_model(config, None)
    prev_epoch, best_val_metric = load(
        resnet,
        'ermbestmodel.pth',
        device=None)
    width, height = 448, 448
    batch_size = 256

    train_generator, valid_generator, test_generator = get_wilds_data(resnet)

    WildcamClassification(train_generator=train_generator,
                          valid_generator=valid_generator,
                          test_generator=test_generator,
                          method=None, kernel=None, batch_norm=False, bias=False,
                          target_domain=None, save_file=True, save_plot=False,
                          save_feature=False, batch_size=batch_size, fine_tune=True,
                          feature_extractor='ResNet', run=0,
                          only_fine_tune=False, activation='softmax',  # only for resnet
                          ).run_experiment()


