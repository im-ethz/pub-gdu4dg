
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../..')))
THIS_FILE = os.path.abspath(__file__)

import argparse
import torch
import copy
import pdb

from typing import Dict, List, Union

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.utils import get_counts
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from SimulationExperiments.wilds.rxrx1.rxrx1_classification import RXRX1Classification
from SimulationExperiments.wilds.rxrx1.rxrx1_dataloader import DataGenerator

import warnings
from datetime import datetime

import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.random.set_seed(1234)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

batch_size = 75


def parser_args():
    parser = argparse.ArgumentParser(description='Wilds classification')
    parser.add_argument('--method',
                        help='cosine_similarity, MMD, projected, None',
                        type=str,
                        default='projected')

    parser.add_argument('--lambda_sparse',
                        default=0,
                        type=float)

    parser.add_argument('--lambda_OLS',
                        type=float,
                        default=1e-3)

    parser.add_argument('--lambda_orth',
                        type=float,
                        default=0)

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

    parser.add_argument('--domain_dim',
                        type=int,
                        default=10)

    parser.add_argument('--sigma',
                        type=int,
                        default=20)

    parser.add_argument('--fe_path',
                        type=str,
                        default='')

    args = parser.parse_args()
    if args.method == 'None':
        args.method = None
    args.ft = True if args.fine_tune == "True" else False
    return args

def rxrx1_transform(is_training):
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]
    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]


    #transforms_ls.append(default_normalization)
    transform = transforms.Compose(transforms_ls)
    return transform


def get_wilds_data():
    # Specify the wilds dataset
    dataset = get_dataset(dataset='rxrx1', download=True)


    train_data = dataset.get_subset('train', transform=rxrx1_transform(is_training=True))
    valid_data = dataset.get_subset('val', transform=rxrx1_transform(is_training=False))
    test_data = dataset.get_subset('test', transform=rxrx1_transform(is_training=False))

    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size)
    test_loader = get_train_loader('standard', test_data, batch_size=batch_size)

    return DataGenerator(train_loader, save_file=False, batch_size=batch_size, one_hot=True), \
           DataGenerator(valid_loader, save_file=False, batch_size=batch_size, one_hot=True), \
           DataGenerator(test_loader, save_file=False, batch_size=batch_size, one_hot=True)


if __name__ == "__main__":
    # load data once
    args = parser_args()
    train_generator, valid_generator, test_generator = get_wilds_data()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    RXRX1Classification(train_generator=train_generator,
                        valid_generator=valid_generator,
                        test_generator=test_generator,
                        lambda_OLS=args.lambda_OLS,
                        lambda_orth=args.lambda_orth,
                        lambda_sparse=args.lambda_sparse,
                        num_domains=args.num_domains,
                        domain_dim=args.domain_dim,
                        sigma=args.sigma,
                        method=args.method,
                        kernel=None, batch_norm=False, bias=False,
                        timestamp=timestamp, target_domain=None, save_file=True, save_plot=False,
                        save_feature=False, batch_size=batch_size, fine_tune=True,
                        feature_extractor='resnet', run=args.running,
                        only_fine_tune=False, activation='softmax'  # only for resnet
                        ).run_experiment()
