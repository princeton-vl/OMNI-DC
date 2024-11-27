import os
import warnings

import numpy as np
import json
import h5py
import glob
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# from azure.storage.blob import ContainerClient
import numpy as np
import io
import cv2
import time

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

class TartanAir(BaseDataset):
    def __init__(self, args, mode="train"):
        super(TartanAir, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)

        self.resize_height = args.resize_height
        self.resize_width = args.resize_width

        if mode != 'train':
            raise NotImplementedError

        self.sample_list = []
        pattern = str(dataset_folder / "tartanair/*/*/*/image_left/*_left.png")
        images = glob.glob(pattern)
        print('Loading TartanAir...')
        for image in tqdm(images):
            scene, difficulty, P = image.split("/")[-5: -2]
            i = image.split("/")[-1].split("_")[0]
            depth = str(dataset_folder / f"tartanair/{scene}/{difficulty}/{P}/depth_left/{i}_left_depth.npy")
            self.sample_list.append((image, depth))
        # self.K = torch.Tensor([
        #     320.0,
        #     320.0,
        #     320.0,
        #     240.0,
        # ])
        self.K = torch.eye(3)
        self.augment = self.args.augment



    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        max_depth = 100.0

        image_file, depth_file = self.sample_list[idx]

        dep_raw = np.load(depth_file).astype('float32')

        # tartanair has some crazily large depth values. clamp it:
        dep_raw[dep_raw > max_depth] = 0.0

        rgb = Image.open(image_file, mode='r')
        dep = Image.fromarray(dep_raw, mode='F')

        K = self.K

        rgb, dep, K, rgb_np_raw = self.process(rgb, dep, K, normalize_median=self.args.data_normalize_median)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True, rgb_np=rgb_np_raw, input_noise=self.args.train_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output