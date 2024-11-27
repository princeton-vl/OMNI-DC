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

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


class BlendedMVS(BaseDataset):
    def __init__(self, args, mode="train"):
        super(BlendedMVS, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)

        self.resize_height = args.resize_height
        self.resize_width = args.resize_width

        if mode != 'train':
            raise NotImplementedError

        self.sample_list = []

        pattern = str(dataset_folder / "blendedmvs" / "*" / "blended_images" / "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jpg")
        images = glob.glob(pattern)
        print('Loading BlendMVS...')
        for image in tqdm(images):
            scene = image.split("/")[-3]
            i = image.split("/")[-1][:-4]
            depth = str(dataset_folder / "blendedmvs" / scene / "rendered_depth_maps" / f"{i}.pfm")
            cam = str(dataset_folder / "blendedmvs" / scene / "cams" / f"{i}_cam.txt")
            self.sample_list.append((image, depth, cam))

        self.augment = self.args.augment

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_file, depth_file, cam_file = self.sample_list[idx]

        rgb = Image.open(image_file, mode='r')
        dep = Image.fromarray(readPFM(depth_file).astype('float32'), mode='F')
        # Camera intrinsics [fx, fy, cx, cy]
        # K = np.loadtxt(cam_file, skiprows=7, max_rows=3)
        # K = torch.Tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        K = torch.eye(3)

        rgb, dep, K, rgb_np_raw = self.process(rgb, dep, K, normalize_median=self.args.data_normalize_median)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True, rgb_np=rgb_np_raw, input_noise=self.args.train_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output

