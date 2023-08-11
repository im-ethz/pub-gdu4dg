import torch.nn.functional as F
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.io import loadmat
from os import path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

base_path = "/wave/odin/digitfive"


class DigitFiveDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(DigitFiveDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]


def load_mnist(base_path, test_size=25000):
    mnist_data = loadmat(path.join(base_path, "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:test_size]
    train_label = train_label[:test_size]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path, test_size=25000):
    mnistm_data = loadmat(path.join(base_path, "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train[:test_size]
    train_label = train_label[:test_size]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn(base_path, test_size=25000):
    svhn_train_data = loadmat(path.join(base_path, "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(path.join(base_path, "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]
    svhn_train = svhn_train[:test_size]
    train_label = train_label[:test_size]
    svhn_test = svhn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_test, test_label


def load_syn(base_path, test_size=25000):
    print("load syn train")
    syn_train_data = loadmat(path.join(base_path, "synth_train_32x32.mat"))
    print("load syn test")
    syn_test_data = loadmat(path.join(base_path, "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train[:test_size]
    syn_test = syn_test[:9000]
    train_label = train_label[:test_size]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_test, test_label


def load_usps(base_path, test_size=25000):
    usps_dataset = loadmat(path.join(base_path, "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train = np.tile(usps_train, (4, 1, 1, 1))
    train_label = np.tile(train_label, 4)
    usps_train = usps_train[:test_size]
    train_label = train_label[:test_size]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_test, test_label


def digit5_dataset_read(base_path, domain, batch_size, return_ds=False, test_size=25000):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path, test_size)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path, test_size)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path, test_size)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path, test_size)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path, test_size)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = DigitFiveDataset(data=train_image, labels=train_label, transform=transform)
    test_dataset = DigitFiveDataset(data=test_image, labels=test_label, transform=transform)
    if return_ds:
        return train_dataset, test_dataset
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


def load_digits(domains=["mnist", "mnistm", "svhn", "syn", "usps"], one_hot=True, test_size=25000):
    x_train_dict = {}
    y_train_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    for domain in domains:
        print(domain)
        train_loader, test_loader = digit5_dataset_read(base_path=base_path, domain=domain, batch_size=int(1e10),
                                                        test_size=test_size)

        x_train, y_train = next(iter(train_loader))
        x_train = np.transpose(x_train, (0, 2, 3, 1))

        x_test, y_test = next(iter(test_loader))
        x_test = np.transpose(x_test, (0, 2, 3, 1))

        if one_hot:
            y_train = F.one_hot(y_train.to(torch.int64), num_classes=10)
            y_test = F.one_hot(y_test.to(torch.int64), num_classes=10)

        x_train_dict[domain] = x_train
        y_train_dict[domain] = y_train
        x_test_dict[domain] = x_test
        y_test_dict[domain] = y_test

    return x_train_dict, y_train_dict, x_test_dict, y_test_dict


if __name__ == "__main__":
    load_usps(base_path)
    x_train, y_train, x_test, y_test = load_usps(base_path)
    x_train, y_train, x_test, y_test = load_mnist_m(base_path)
