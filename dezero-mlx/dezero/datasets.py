"""
DeZero Datasets Module
----------------------
Dataset 기본 클래스 및 MNIST, Spiral 등 데이터셋
"""
import os
import gzip
import numpy as np
from dezero.utils import get_file
from dezero.transforms import Compose, Flatten, ToFloat, Normalize


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


# =============================================================================
# Toy datasets
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


# =============================================================================
# MNIST-like dataset
# =============================================================================
class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data


class CIFAR10(Dataset):
    def __init__(self, train=True, transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        # CIFAR-10은 별도 다운로드 로직 필요
        raise NotImplementedError("CIFAR-10 데이터셋 로드는 아직 구현되지 않았습니다.")


class CIFAR100(Dataset):
    def __init__(self, train=True, transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        raise NotImplementedError("CIFAR-100 데이터셋 로드는 아직 구현되지 않았습니다.")


# =============================================================================
# Big dataset (ImageNet)
# =============================================================================
class ImageNet(Dataset):
    def __init__(self):
        raise NotImplementedError("ImageNet 데이터셋은 아직 구현되지 않았습니다.")


# =============================================================================
# Sequential datasets
# =============================================================================
class SinCurve(Dataset):
    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]
