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


def get_wilds_data():
    dataset = get_dataset(dataset='camelyon17', download=True)

    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    valid_data = dataset.get_subset('val', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    test_data = dataset.get_subset('test', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))

    kwargs = {'drop_last':True}
    train_loader = get_train_loader('standard', train_data, batch_size=16, drop_last=True)
    valid_loader = get_train_loader('standard', valid_data, batch_size=16, drop_last=True)
    test_loader = get_train_loader('standard', test_data, batch_size=16, drop_last=True)

    return DataGenerator(train_loader), DataGenerator(valid_loader), DataGenerator(test_loader)


if __name__ == "__main__":
    # load data once
    train_generator, valid_generator, test_generator = get_wilds_data()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    CamelyonClassification(train_generator=train_generator,
                           valid_generator=valid_generator,
                           test_generator=test_generator,
                           data=None, method='cs', kernel=None, batch_norm=False, bias=False,
                           timestamp=timestamp, target_domain=None, save_file=True, save_plot=False,
                           save_feature=True, batch_size=64, fine_tune=False,
                           feature_extractor='resnet'
                           ).run_experiment()
