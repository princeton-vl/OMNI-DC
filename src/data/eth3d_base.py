import os
import tarfile
from io import BytesIO

import numpy as np

from . import BaseDataset

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

split_txt = '../data_json/marigold_data_split/eth3d/eth3d_filename_list.txt'
from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

indoor_val_split = ['office']
indoor_test_split = ['delivery_area', 'kicker', 'pipes', 'relief', 'relief_2', 'terrains']

outdoor_val_split = ['courtyard']
outdoor_test_split = ['electro', 'facade', 'meadow', 'playground', 'terrace']

class ETH3D_Base(BaseDataset):
    def __init__(self, args, mode, scene='indoor'):
        super(ETH3D_Base, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # only support testing for now
        if mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.original_height, self.original_width = (4032, 6048)

        # self.height = self.args.resize_height
        # self.width = self.args.resize_width
        self.crop_size = (3840, 5120)
        self.target_size = (480, 640)

        self.augment = self.args.augment

        print('Loading ETH3D...')
        with open(split_txt, "r") as f:
            filenames = [
                s.split() for s in f.readlines()
            ]

        if mode == 'val':
            if scene == 'indoor':
                scene_list = indoor_val_split
            else:
                scene_list = outdoor_val_split
        else: # test
            if scene == 'indoor':
                scene_list = indoor_test_split
            else:
                scene_list = outdoor_test_split

        filtered_filenames = []
        for entry in filenames:
            scene_name = ((entry[0]).split('/'))[1]
            if scene_name in scene_list:
                filtered_filenames.append(entry)

        self.filenames = filtered_filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        min_depth = 1e-5
        max_depth = np.inf

        # data_dir = self.args.dir_data
        data_dir = str(dataset_folder / "marigold" / "eth3d")
        rgb_rel_path, depth_rel_path = self.filenames[idx]

        depth_path = os.path.join(data_dir, depth_rel_path)
        with open(depth_path, "rb") as file:
            binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats

        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        dep_np = depth_decoded.reshape((self.original_height, self.original_width))
        dep_mask_np = np.logical_and(
            (dep_np > min_depth), (dep_np < max_depth)
        ).astype(np.float32)

        # perform masking
        dep_np[dep_mask_np == 0.0] = 0.0

        # dummy one
        K = torch.eye(3)

        rgb = Image.open(os.path.join(data_dir, rgb_rel_path))
        dep = Image.fromarray(dep_np.astype('float32'), mode='F')

        t_rgb = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            T.CenterCrop(self.crop_size),
            T.Resize(self.target_size),
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
            T.CenterCrop(self.crop_size),
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb_np_raw = t_rgb_np_raw(rgb)
        rgb = t_rgb(rgb) # 3 x H x W

        dep = t_dep(dep) # 1 x H x W
        dep_mask = (dep > 0).float()

        dep = F.avg_pool2d(dep.unsqueeze(0), 8).squeeze(0)
        dep_mask = F.avg_pool2d(dep_mask.unsqueeze(0), 8).squeeze(0)

        dep[dep_mask > 0] = dep[dep_mask > 0] / dep_mask[dep_mask > 0]

        # rgb, dep, K = self.process(rgb, dep, K, normalize_median=False)

        dep_sp, pattern_id = self.get_sparse_depth(dep,
                                                   self.args.val_depth_pattern,
                                                   match_density=True,
                                                   rgb_np=rgb_np_raw,
                                                   input_noise=self.args.val_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output