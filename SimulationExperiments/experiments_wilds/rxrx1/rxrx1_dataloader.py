# nohup /local/home/sfoell/anaconda3/envs/gdu4dg/bin/python3.8 -u /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiments_wilds/rxrx1/rxrx1_main.py > /local/home/sfoell/MTEC-IM-309/pub-gdu4dg/SimulationExperiments/experiments_wilds/rxrx1/wilds_play.log 2>&1 &
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find code directory relative to our directory
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
THIS_FILE = os.path.abspath(__file__)

import keras
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import rxrx1_classification


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_loader, x_path=None, y_path=None, batch_size=32, save_file=True, load_files=False,
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
                self.weights = np.zeros(rxrx1_classification.units)

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
                    y = one_hot(y, rxrx1_classification.units)
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
            x = self.x_full[index*self.batch_size:(index+1)*self.batch_size]
            y = self.y_full[index*self.batch_size:(index+1)*self.batch_size]
        else:
            try:
                x, y, metadata = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                x, y, metadata = next(self.iterator)
            y = y.numpy()
            if self.one_hot:
                y = one_hot(y, rxrx1_classification.units)
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