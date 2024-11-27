import os
import tarfile
from io import BytesIO

import numpy as np

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

split_txt = '../data_json/marigold_data_split/nyu/labeled/filename_list_test.txt'
from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

class NYU_FULL_RES(BaseDataset):
    def __init__(self, args, mode):
        super(NYU_FULL_RES, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # only support testing for now
        if mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = 480
        self.width = 640

        print('Loading NYU Full Res...')
        with open(split_txt, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        min_depth = 1e-3
        max_depth = 10.0

        # data_dir = self.args.dir_data
        data_dir = str(dataset_folder / "marigold" / "nyuv2")
        rgb_rel_path, depth_rel_path, filled_depth_rel_path = self.filenames[idx]

        depth_path = os.path.join(data_dir, depth_rel_path)
        depth = Image.open(depth_path)  # [H, W, rgb]
        depth = np.asarray(depth).astype(np.float32)
        depth = depth / 1000.0

        dep_mask = np.logical_and(
            (depth > min_depth), (depth < max_depth)
        ).astype(np.float32)

        eigen_mask = np.zeros_like(dep_mask)
        eigen_mask[45:471, 41:601] = 1.0

        dep_mask = np.logical_and(dep_mask, eigen_mask)

        # perform masking
        depth[dep_mask == 0.0] = 0.0

        # dummy one
        K = torch.eye(3)

        rgb = Image.open(os.path.join(data_dir, rgb_rel_path))
        dep = Image.fromarray(depth, mode='F')

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_dep = T.Compose([
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb = t_rgb(rgb)
        dep = t_dep(dep)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output