import os
import warnings

import numpy as np
import json
import h5py
import glob
from tqdm import tqdm
import pickle

from . import BaseDataset, PATTERN_IDS

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

class Uniformat(BaseDataset):
    def __init__(self, args, mode="train"):
        super(Uniformat, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'test' and mode != 'val':
            raise NotImplementedError

        print('Loading Uniformat...')

        self.sample_list = sorted([os.path.join(args.dir_data, f) for f in os.listdir(args.dir_data) if f.endswith('.npy')])
        # self.sample_list = [self.sample_list[k] for k in np.random.choice(range(len(self.sample_list)), 10, replace=False)]


    def __len__(self):
        return len(self.sample_list)
        # return 10

    def __getitem__(self, idx):
        # image_file, depth_file, K = self.sample_list[idx]
        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        t_dep = T.Compose([
            T.ToTensor()
        ])

        filedir = self.sample_list[idx]
        data_dict = dict(np.load(filedir, allow_pickle=True).item())

        rgb = t_rgb(Image.fromarray(data_dict['rgb'], mode='RGB'))

        dep_sp = t_dep(data_dict['dep'])
        dep = t_dep(data_dict['gt'])

        if 'ETH3D' in filedir:
            K = torch.Tensor(
                [[425, 0, 320],
                 [0, 425, 240],
                 [0, 0, 1.0]]
            )
        elif 'iBims' in filedir:
            K = torch.Tensor(
                [[490, 0, 320],
                 [0, 490, 240],
                 [0, 0, 1.0]]
            )
        elif 'KITTI' in filedir:
            K = torch.Tensor(
                [[data_dict['K'][0], 0, data_dict['K'][2]],
                 [0, data_dict['K'][1], data_dict['K'][3]],
                 [0, 0, 1.0]]
            )
        else:
            K = torch.Tensor(data_dict['K'])

        dep_sp = dep_sp * self.args.depth_scale_multiplier
        dep = dep * self.args.depth_scale_multiplier

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': PATTERN_IDS[self.args.inference_pattern_type]}

        return output