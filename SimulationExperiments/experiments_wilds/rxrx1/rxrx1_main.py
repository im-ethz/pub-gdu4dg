# nohup /local/home/sfoell/anaconda3/envs/gdu4dg/bin/python3.8 -u /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiments_wilds/rxrx1/rxrx1_main.py > /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiments_wilds/rxrx1/wilds_play.log 2>&1 &
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

from SimulationExperiments.experiments_wilds.rxrx1.rxrx1_classification import RXRX1Classification
from SimulationExperiments.experiments_wilds.rxrx1.rxrx1_dataloader import DataGenerator

import warnings
from datetime import datetime

import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
#tf.random.set_seed(1234)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

batch_size = 72


def parser_args():
    parser = argparse.ArgumentParser(description='Wilds classification')
    parser.add_argument('--method',
                        help='cosine_similarity, MMD, projected, None',
                        type=str,
                        default='cosine_similarity')

    parser.add_argument('--lambda_sparse',
                        default=1e-3,
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

    parser.add_argument('--fe_path',
                        type=str,
                        default='')

    args = parser.parse_args()
    if args.method == 'None':
        args.method = None
    args.ft = True if args.fine_tune == "True" else False
    return args


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


    transforms_ls.append(default_normalization)
    transform = transforms.Compose(transforms_ls)
    return transform


def get_wilds_data():
    # Specify the wilds dataset
    dataset = get_dataset(dataset='rxrx1', download=True)

    train_grouper = CombinatorialGrouper(
        dataset=dataset,
        groupby_fields=['experiment']
    )

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
                           method="cs", kernel=None, batch_norm=False, bias=True,
                           timestamp=timestamp, target_domain=None, save_file=True, save_plot=False,
                           save_feature=False, batch_size=batch_size, fine_tune=True,
                           feature_extractor='resnet', run=args.running,
                           only_fine_tune=False, activation='softmax'  # only for resnet
                           ).run_experiment()
