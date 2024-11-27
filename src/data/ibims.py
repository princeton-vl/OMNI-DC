import os
import tarfile
from io import BytesIO

import numpy as np
from scipy import io

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()
split_txt = str(dataset_folder / "iBims" / "imagelist.txt")

class iBims(BaseDataset):
    def __init__(self, args, mode):
        super(iBims, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # only support testing for now
        # if mode != 'val' and mode != 'test':
        #     raise NotImplementedError
        if mode != 'test':
            raise NotImplementedError

        self.height = 480
        self.width = 640

        print('Loading iBims-1...')
        with open(split_txt, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # data_dir = self.args.dir_data
        data_dir = str(dataset_folder / "iBims" / "gt_depth")
        scene_name, = self.filenames[idx]

        data_path = os.path.join(data_dir, scene_name+".mat")
        data = io.loadmat(data_path)['data']

        image = data['rgb'][0][0]  # RGB image
        depth = data['depth'][0][0]  # Raw depth map
        # mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
        depth = np.asarray(depth).astype(np.float32)

        # print(np.max(depth), np.min(depth), depth.shape, depth)
        # assert False

        rgb = Image.fromarray(image)

        # perform masking
        # depth[mask_invalid > 0.0] = 0.0
        dep = Image.fromarray(depth, mode='F')

        # dummy one
        K = torch.eye(3)

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
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