import os
import warnings

import numpy as np
import json
import h5py
import glob
from tqdm import tqdm
import pickle

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import OpenEXR
import Imath
import re

import torchvision

warnings.filterwarnings("ignore", category=UserWarning)

def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def load_exr(filename):
    hdr = exr2hdr(filename)
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr

from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets"/ "irs").resolve()

class IRS(BaseDataset):
    def __init__(self, args, mode="train"):
        super(IRS, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)

        # this dataset is not 4:3. maintain the original aspect ratio
        self.resize_height = args.resize_height
        self.resize_width = int(round(self.resize_height * 960 / 540))

        if mode != 'train':
            raise NotImplementedError

        print('Loading IRS...')
        self.sample_list = []

        # scene_types = ['Home', 'Office', 'Restaurant', 'Store']
        scene_types = ['Home', 'Restaurant', 'Store']
        for scene_type in scene_types:
            split_file = str(dataset_folder / "file_list" / f"{scene_type.lower()}_all.txt")
            with open(split_file, 'r') as file:
                for line in file:
                    image_path, _, depth_path, _ = line.strip().split()

                    if scene_type in ['Home', 'Office']:
                        image_path = re.sub(r'_(\d{5})\.png$', lambda x: f"_{int(x.group(1))}.png", image_path)
                        depth_path = re.sub(r'_(\d{5})\.exr$', lambda x: f"_{int(x.group(1))}.exr", depth_path)

                    self.sample_list.append((image_path, depth_path))

        # pattern = os.path.join(dataset_folder, '**', 'l_*.png')
        # matching_files = glob.glob(pattern, recursive=True)
        # for file in tqdm(matching_files):
        #     image_path = file
        #     dir, im_name = os.path.split('image_path')
        #     depth_path = os.path.join(dir, im_name.replace('l_', 'd_').replace('.png', '.exr'))
        #
        #     self.sample_list.append((image_path, depth_path))

        self.augment = self.args.augment

        self.focal = 480.0
        width, height = 960.0, 540.0
        # 0.1m: used to convert disparity to depth
        self.baseline = 0.1
        self.max_depth = 50.0

        self.K = torch.tensor([
            [self.focal, 0.0, width / 2.0],
            [0.0, self.focal, height / 2.0],
            [0.0, 0.0, 1.0]
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_file, depth_file = self.sample_list[idx]
        image_file = os.path.join(dataset_folder, image_file)
        depth_file = os.path.join(dataset_folder, depth_file)

        K = torch.clone(self.K)

        rgb = Image.open(image_file, mode='r').convert('RGB')

        disp = load_exr(depth_file)
        dep = self.baseline * self.focal / (disp + 1e-8)

        # mask out invalid values
        dep[dep < 0.0] = 0.0
        dep[dep > self.max_depth] = 0.0

        dep = Image.fromarray(dep, mode='F')

        rgb, dep, K, rgb_np_raw = self.process(rgb, dep, K, normalize_median=self.args.data_normalize_median)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True, rgb_np=rgb_np_raw, input_noise=self.args.train_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output