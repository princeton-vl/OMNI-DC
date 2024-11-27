import os
import warnings

import numpy as np
import json
import h5py
import glob
from tqdm import tqdm
import re

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from importlib import import_module

def get_data(data_name):
    module_name = 'data.' + data_name.lower()
    module = import_module(module_name)

    return getattr(module, data_name)

warnings.filterwarnings("ignore", category=UserWarning)

class MultiDataset(BaseDataset):
    def __init__(self, args, mode="train"):
        # only support mixing training for now
        assert mode == "train"

        self.dataset_names = args.train_data_name.split('+')
        self.datasets = []
        for dataset_name in self.dataset_names:
            data = get_data(dataset_name)
            self.datasets.append(data(args, mode=mode))

        self.num_datasets = len(self.datasets)
        min_dataset_length = min([len(dataset) for dataset in self.datasets])

        if args.mixed_dataset_total_length > 0:
            self.subset_length = min(args.mixed_dataset_total_length // self.num_datasets, min_dataset_length)
        else:
            # if unspecified, use the shortest dataset
            self.subset_length = min_dataset_length

        self.total_length = self.num_datasets * self.subset_length

        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        for dataset in self.datasets:
            dataset_indices = np.random.choice(range(len(dataset)), size=self.subset_length, replace=False)
            indices.append(dataset_indices)
        return indices

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = idx % self.num_datasets
        sample_idx = idx // self.num_datasets
        sample =  self.datasets[dataset_idx][self.indices[dataset_idx][sample_idx]]

        if torch.sum(sample['dep'] > 0.0) < 20:
            return self.__getitem__(idx=np.random.randint(0, self.total_length))

        return sample

    def refresh_indices(self):
        self.indices = self._generate_indices()