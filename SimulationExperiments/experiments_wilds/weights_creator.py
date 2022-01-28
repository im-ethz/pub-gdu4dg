from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    units = 182
    batch_size = 1024
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((
        96, 96)), transforms.ToTensor()]))
    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    weights = np.zeros(units)

    for x, y, metadata in tqdm(train_loader):
        y = y.numpy()
        unique, counts = np.unique(y, return_counts=True)
        weights[unique] += counts

    weights = weights.sum() / weights
    np.save('wildcam_weights.npy', weights)
