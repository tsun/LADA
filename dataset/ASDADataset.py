import torch
from torchvision import datasets, transforms as T
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from .image_list import ImageList
from PIL import Image
import pickle as pkl
import copy

class Toy():
    def __init__(self):
        self.data = []
        self.targets = []
        self.gt_targets = []
        self.transform = None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        return img, target, index

    def __len__(self):
        return len(self.data)

    def add_item(self, add_data, add_target, add_gt_target):
        self.data.extend(add_data)
        self.targets.extend(add_target)
        self.gt_targets.extend(add_gt_target)

class SVHN(datasets.SVHN):
    def __init__(self, root, split = "train", transform = None, target_transform = None, download = False):
        super(SVHN, self).__init__(root, split = split, transform = transform, target_transform = target_transform, download = download)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def add_item(self, add_data, add_target):
        self.data.extend(add_data)
        self.targets.extend(add_target)


class MNIST(datasets.MNIST):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super(MNIST, self).__init__(root, train = train, transform = transform, target_transform = target_transform, download = download)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.numpy()
        if len(img.shape) == 2:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def add_item(self, add_data, add_target):
        self.data.extend(add_data)
        self.targets.extend(add_target)

class ASDADataset:
    # Active Semi-supervised DA Dataset class
    def __init__(self, dataset, domain, data_dir, num_classes, batch_size=128, num_workers=4, transforms=None):
        self.dataset = dataset
        self.domain = domain
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train_loader, self.valid_loader, self.test_loader = None, None, None

        self.build_dsets(transforms)

    def get_num_classes(self):
        return self.num_classes

    def get_dsets(self):
        return self.train_dataset, self.query_dataset, self.valid_dataset, self.test_dataset

    def build_dsets(self, transforms=None):
        if self.domain == "mnist":
            mean, std = 0.5, 0.5
            normalize_transform = T.Normalize((mean,), (std,))
            train_transforms = T.Compose([
                T.ToTensor(),
                normalize_transform
            ])
            test_transforms = T.Compose([
                T.ToTensor(),
                normalize_transform
            ])

            train_dataset = MNIST(self.data_dir, train=True, download=True, transform=train_transforms)
            valid_dataset = MNIST(self.data_dir, train=True, download=True, transform=test_transforms)
            query_dataset = valid_dataset
            test_dataset = MNIST(self.data_dir, train=True, download=True, transform=test_transforms)

        elif self.domain == "mnist-m":
            mean, std = 0.5, 0.5
            normalize_transform = T.Normalize((mean,), (std,))
            RGB2Gray = T.Lambda(lambda x: x.convert('L'))
            train_transforms = T.Compose([
                RGB2Gray,
                T.Resize((28, 28)),
                T.ToTensor(),
                normalize_transform
            ])
            test_transforms = T.Compose([
                RGB2Gray,
                T.Resize((28, 28)),
                T.ToTensor(),
                normalize_transform
            ])

            train_dataset = MNIST(self.data_dir, train=False, download=False, transform=train_transforms)
            valid_dataset = MNIST(self.data_dir, train=False, download=False, transform=test_transforms)
            query_dataset = valid_dataset
            test_dataset = MNIST(self.data_dir, train=False, download=False, transform=test_transforms)

            # load pickle
            with open('./data/MNIST-M/mnist_m_data.pkl', 'rb') as f:
                data = pkl.load(f, encoding='bytes')
            train_dataset.data = torch.tensor(data['train'])
            train_dataset.targets = torch.tensor(data['train_targets'])
            valid_dataset.data = torch.tensor(data['train'])
            valid_dataset.targets = torch.tensor(data['train_targets'])
            test_dataset.data = torch.tensor(data['test'])
            test_dataset.targets = torch.tensor(data['test_targets'])

        elif self.domain == "svhn":
            mean, std = 0.5, 0.5
            normalize_transform = T.Normalize((mean,), (std,))
            RGB2Gray = T.Lambda(lambda x: x.convert('L'))
            train_transforms = T.Compose([
                RGB2Gray,
                T.Resize((28, 28)),
                T.ToTensor(),
                normalize_transform
            ])
            test_transforms = T.Compose([
                RGB2Gray,
                T.Resize((28, 28)),
                T.ToTensor(),
                normalize_transform
            ])

            train_dataset = SVHN(self.data_dir, split='train', download=True, transform=train_transforms)
            valid_dataset = SVHN(self.data_dir, split='train', download=True, transform=test_transforms)
            query_dataset = valid_dataset
            test_dataset = SVHN(self.data_dir, split='test', download=True, transform=test_transforms)

        elif self.dataset == 'toy':
            train_dataset = Toy()
            valid_dataset = Toy()
            query_dataset = valid_dataset
            test_dataset = Toy()

            # load pickle
            with open('./data/toy/{}.pkl'.format(self.domain), 'rb') as f:
                data = pkl.load(f, encoding='bytes')
            train_dataset.data = torch.tensor(data['train']).to(torch.float32)
            train_dataset.targets = torch.tensor(data['train_targets']).to(torch.float32)
            valid_dataset.data = torch.tensor(data['train']).to(torch.float32)
            valid_dataset.targets = torch.tensor(data['train_targets']).to(torch.float32)
            test_dataset.data = torch.tensor(data['test']).to(torch.float32)
            test_dataset.targets = torch.tensor(data['test_targets']).to(torch.float32)

        elif self.dataset in ["officehome", "domainnet", "visda", "office31", "multi"]:
            assert transforms is not None

            if self.dataset == "domainnet":
                train_list = open(os.path.join(self.data_dir, "image_list", self.dataset, self.domain+"_train.txt")).readlines()
                test_list = open(os.path.join(self.data_dir, "image_list", self.dataset, self.domain+"_test.txt")).readlines()
                valid_list = train_list.copy()
            else:
                train_list = open(os.path.join(self.data_dir, "image_list", self.dataset, self.domain+".txt")).readlines()
                test_list = train_list.copy()
                valid_list = train_list.copy()

            train_dataset = ImageList(train_list, root=self.data_dir, transform=transforms['train'])
            query_dataset = ImageList(train_list, root=self.data_dir, transform=transforms['query'])
            valid_dataset = ImageList(valid_list, root=self.data_dir, transform=transforms['test'])
            test_dataset = ImageList(test_list, root=self.data_dir, transform=transforms['test'])


        self.train_dataset = train_dataset
        self.query_dataset = query_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset



    def get_loaders(self, valid_type='val', valid_ratio=1.0, rebuilt=False):

        if self.train_loader and self.valid_loader and self.test_loader and not rebuilt:
            return self.train_loader, self.valid_loader, self.test_loader

        num_train = len(self.train_dataset)
        self.train_size = num_train

        if valid_type == 'split':
            indices = list(range(num_train))
            split = int(np.floor(valid_ratio * num_train))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

        elif valid_type == 'val':
            train_idx = np.arange(len(self.train_dataset))
            if valid_ratio == 1.0:
                valid_idx = np.arange(len(self.valid_dataset))
            else:
                indices = list(range(len(self.valid_dataset)))
                split = int(np.floor(valid_ratio * num_train))
                np.random.shuffle(indices)
                valid_idx = indices[:split]
        else:
            raise NotImplementedError

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
                                                   batch_size=self.batch_size, num_workers=self.num_workers)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, sampler=valid_sampler, batch_size=self.batch_size,
                                                 num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        self.train_idx = train_idx
        self.valid_idx = valid_idx

        return self.train_loader, self.valid_loader, self.test_loader