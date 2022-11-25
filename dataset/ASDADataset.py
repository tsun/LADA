import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from .image_list import ImageList

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