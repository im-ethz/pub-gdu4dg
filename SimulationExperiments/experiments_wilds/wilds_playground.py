import argparse

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

from SimulationExperiments.experiments_wilds.camelyon_classification import CamelyonClassification
from SimulationExperiments.experiments_wilds.camelyon_dataloader import DataGenerator

import warnings
from datetime import datetime

import tensorflow as tf

warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.random.set_seed(1234)

batch_size = 1024

def parser_args():
    parser = argparse.ArgumentParser(description='Digits 5 classification')
    parser.add_argument('--method',
                        help='cosine_similarity, MMD, projected',
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

    args = parser.parse_args()
    if args.method == 'None':
        args.method = None
    args.ft = True if args.fine_tune == "True" else False
    return args


def get_wilds_data():
    dataset = get_dataset(dataset='camelyon17', download=True)

    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.ToTensor()]))
    valid_data = dataset.get_subset('val', transform=transforms.Compose([transforms.ToTensor()]))
    test_data = dataset.get_subset('test', transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size)
    test_loader = get_train_loader('standard', test_data, batch_size=batch_size)

    return DataGenerator(train_loader, x_path='x_full_train.npy', y_path='y_full_train.npy', batch_size=batch_size), \
           DataGenerator(valid_loader, x_path='x_full_valid.npy', y_path='y_full_valid.npy', batch_size=batch_size), \
           DataGenerator(test_loader, save_file=False, batch_size=batch_size)


if __name__ == "__main__":
    # load data once
    args = parser_args()
    train_generator, valid_generator, test_generator = get_wilds_data()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    CamelyonClassification(train_generator=train_generator,
                           valid_generator=valid_generator,
                           test_generator=test_generator,
                           method=args.method, kernel=None, batch_norm=False, bias=False,
                           timestamp=timestamp, target_domain=None, save_file=True, save_plot=False,
                           save_feature=False, batch_size=batch_size, fine_tune=args.ft,
                           feature_extractor='lenet', run=args.running
                           ).run_experiment()
