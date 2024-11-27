import os
import os.path as osp

import warnings

import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore", category=UserWarning)

"""
Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
            https://github.com/bartn8/vppdc/blob/main/dataloaders/datasets.py
"""

def load_depth(path):
    '''
    Loads a depth map from a 16-bit PNG file

    Args:
    path : str
      path to 16-bit PNG file

    Returns:
    numpy : depth map
    '''
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z/256.0
    z[z <= 0] = 0.0
    return z

def read_gen(file_name, pil=False):
    ext = osp.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

class VOID(BaseDataset):
    def __init__(self, args, mode):
        super(VOID, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.augment = self.args.augment

        datapath = self.args.dir_data
        parent_dir = os.path.dirname(datapath)

        self.image_list = []
        self.extra_info = []

        appendix = "train" if mode == "train" else "test"

        # Glue code between persefone data and my shitty format
        intrinsics_txt = open(osp.join(datapath, f"{appendix}_intrinsics.txt"), 'r')
        rgb_txt = open(osp.join(datapath, f"{appendix}_image.txt"), 'r')
        hints_txt = open(osp.join(datapath, f"{appendix}_sparse_depth.txt"), 'r')
        gt_txt = open(osp.join(datapath, f"{appendix}_ground_truth.txt"), 'r')
        valid_txt = open(osp.join(datapath, f"{appendix}_validity_map.txt"))

        while True:

            i_path = intrinsics_txt.readline().strip()
            rgb_path = rgb_txt.readline().strip()
            hints_path = hints_txt.readline().strip()
            gt_path = gt_txt.readline().strip()
            valid_path = valid_txt.readline().strip()

            if not i_path or not rgb_path or not hints_path or not gt_path or not valid_path:
                break

            self.image_list += [[osp.join(parent_dir, i_path),
                                 osp.join(parent_dir, rgb_path),
                                 osp.join(parent_dir, hints_path),
                                 osp.join(parent_dir, gt_path),
                                 osp.join(parent_dir, valid_path)]]
            self.extra_info += [[rgb_path.split('/')[-1]]]

        intrinsics_txt.close()
        rgb_txt.close()
        hints_txt.close()
        gt_txt.close()
        valid_txt.close()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        K = np.loadtxt(self.image_list[index][0])
        rgb = read_gen(self.image_list[index][1])
        hints_depth = load_depth(self.image_list[index][2])
        gt_depth = load_depth(self.image_list[index][3])

        # print(np.min(gt_depth), np.max(gt_depth))
        # assert False

        # rgb = Image.fromarray(rgb, mode='RGB')
        dep_sp = Image.fromarray(hints_depth, mode='F')
        dep = Image.fromarray(gt_depth, mode='F')

        if self.augment and self.mode == 'train':
            raise NotImplementedError

        else:
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
            dep_sp = t_dep(dep_sp)

            normal = torch.ones_like(rgb) # dummy one

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'normal': normal}

        return output