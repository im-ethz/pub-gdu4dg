import keras
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_loader, x_path=None, y_path=None, batch_size=32, save_file=True):
        super(DataGenerator, self).__init__()
        self.data_loader = data_loader
        self.iterator = iter(data_loader)
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.x_full = None
        self.y_full = None
        self.save_file = save_file
        self.load()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.y_full) / self.batch_size))

    def load(self):
        """Load the entire dataset into the memory"""
        if self.x_path is not None and self.y_path is not None:
            self.x_full = np.load(self.x_path)
            self.y_full = np.load(self.y_path)
            print('Loaded ', self.x_path, self.y_path)
        else:
            for x, y, metadata in tqdm(self.data_loader):
                y = y.numpy()
                x = x.numpy()
                B, C, W, H = x.shape
                x = x.reshape(B, W, H, C)
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

    def on_epoch_end(self):
        """Shuffle data at the end of every epoch"""
        self.x_full, self.y_full = shuffle(self.x_full, self.y_full)

    def __getitem__(self, index):
        x = self.x_full[index*self.batch_size:(index+1)*self.batch_size]
        y = self.y_full[index*self.batch_size:(index+1)*self.batch_size]
        return x, y


