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

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

class VKITTI(BaseDataset):
    def __init__(self, args, mode="train"):
        super(VKITTI, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)

        if mode != 'train':
            raise NotImplementedError

        self.sample_list = []

        train_list = '../data_json/marigold_data_split/vkitti/vkitti_train.txt'
        val_list = '../data_json/marigold_data_split/vkitti/vkitti_val.txt'

        # this dataset is not 4:3. maintain the original aspect ratio
        self.resize_height = args.resize_height
        self.resize_width = int(round(self.resize_height * 1242 / 375))

        print('Loading Virtual KITTI...')

        # combine all data
        with open(train_list, "r") as f:
            self.sample_list.extend([
                s.split() for s in f.readlines()
            ])
        with open(val_list, "r") as f:
            self.sample_list.extend([
                s.split() for s in f.readlines()
            ])

        # Filter out empty depth
        self.sample_list = [f for f in self.sample_list if "None" != f[1]]
        self.augment = self.args.augment

        print('Virtual KITTI total length: ', len(self.sample_list))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_file, depth_file = self.sample_list[idx]

        data_dir = str(dataset_folder / "vkitti")
        rgb = Image.open(os.path.join(data_dir, image_file), mode='r')

        depth_path = os.path.join(data_dir, depth_file)
        depth = Image.open(depth_path)  # [H, W, rgb]
        depth = np.asarray(depth).astype(np.float32)
        depth = depth / 100.0

        # filter out invalid depth values
        min_depth = 1e-5
        max_depth = 250.0

        dep_mask = np.logical_and(
            (depth > min_depth), (depth < max_depth)
        ).astype(np.float32)

        depth[dep_mask == 0.0] = 0.0

        dep = Image.fromarray(depth, mode='F')
        K = torch.eye(3)

        rgb, dep, K, rgb_np_raw = self.process(rgb, dep, K, normalize_median=self.args.data_normalize_median)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True, rgb_np=rgb_np_raw, input_noise=self.args.train_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output

