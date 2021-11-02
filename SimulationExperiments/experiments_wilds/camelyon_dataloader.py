import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_loader):
        super(DataGenerator, self).__init__()
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, item):
        try:
            x, y, metadata = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            x, y, metadata = next(self.iterator)
        y = y.numpy()
        x = x.numpy()
        B, C, W, H = x.shape
        x = x.reshape(B, W, H, C)
        return x, y

