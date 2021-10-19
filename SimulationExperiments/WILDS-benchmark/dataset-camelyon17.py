import wilds
from wilds.common.data_loaders import get_train_loader

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dataset = wilds.get_dataset("camelyon17", download = False)

#train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor()]))
#train_loader = get_train_loader('standard', train_data, batch_size=16)


