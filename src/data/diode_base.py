import os
import tarfile
from io import BytesIO

import numpy as np

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Resize

from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

def shift_2d_replace(data, dx, dy, constant=0.0):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

class DIODE_Base(BaseDataset):
    def __init__(self, args, mode, split_txt, edge_noise_filter_radius=0, edge_noise_filter_thres=0.1):
        super(DIODE_Base, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.edge_noise_filter_radius = edge_noise_filter_radius
        self.edge_noise_filter_thres = edge_noise_filter_thres

        # only support testing for now
        # if mode != 'val' and mode != 'test':
        #     raise NotImplementedError
        if mode != 'test':
            raise NotImplementedError

        original_height, original_width = (768, 1024)

        self.height = self.args.resize_height
        self.width = self.args.resize_width

        self.crop_size = (750, 1000)

        self.augment = self.args.augment

        print('Loading DIODE...')
        with open(split_txt, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]

        # if 'indoor' in split_txt:
        #     if mode == 'val':
        #         self.filenames = self.filenames[:115]
        #     else:
        #         self.filenames = self.filenames[115:]
        # else:
        #     if mode == 'val':
        #         self.filenames = self.filenames[366:]
        #     else:
        #         self.filenames = self.filenames[:366]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # data_dir = self.args.dir_data
        data_dir = str(dataset_folder / "marigold" / "diode")

        rgb_rel_path, depth_rel_path, mask_rel_path = self.filenames[idx]

        dep_np = np.load(os.path.join(data_dir, depth_rel_path)).squeeze()
        dep_mask_np = np.load(os.path.join(data_dir, mask_rel_path)).squeeze()

        # perform masking
        dep_np[np.logical_not(dep_mask_np > 0.0)] = 0.0

        # min value from diode official report
        dep_np[dep_np < 0.1] = 0.0

        # filter out the noisy pixels
        radius = self.edge_noise_filter_radius
        depth_diff = []
        thres = self.edge_noise_filter_thres

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                depth_diff.append(np.abs(shift_2d_replace(dep_np, i, j) - dep_np))

        depth_diff = np.stack(depth_diff, axis=0)
        max_depth_diff = np.max(depth_diff, axis=0)

        dep_np[max_depth_diff > dep_np * thres] = 0.0

        # dummy one
        K = torch.eye(3)

        rgb = Image.open(os.path.join(data_dir, rgb_rel_path))
        dep = Image.fromarray(dep_np.astype('float32'), mode='F')

        # rgb, dep, K = self.process(rgb, dep, K, normalize_median=False)
        t_rgb = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize((480, 640)),
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize((480, 640), InterpolationMode.NEAREST),
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb_np_raw = t_rgb_np_raw(rgb)
        rgb = t_rgb(rgb)
        
        dep = t_dep(dep)

        dep_sp, pattern_id = self.get_sparse_depth(dep,
                                                   self.args.val_depth_pattern,
                                                   match_density=True,
                                                   rgb_np=rgb_np_raw,
                                                   input_noise=self.args.val_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output