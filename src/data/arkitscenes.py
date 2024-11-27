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

import pandas as pd

from pathlib import Path
dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()

class ARKitScenes(BaseDataset):
    def __init__(self, args, mode):
        super(ARKitScenes, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # only support testing for now
        # if mode != 'val' and mode != 'test':
        #     raise NotImplementedError
        if mode != 'test':
            raise NotImplementedError

        original_height, original_width = (1440, 1920)

        self.augment = self.args.augment

        print('Loading ARKitScenes...')

        self.filenames = []
        self.data_folder = str(dataset_folder / 'ARKitScenes' / 'depth' / 'upsampling' / 'Validation')
        scenes = [f.path for f in os.scandir(self.data_folder) if f.is_dir()]

        # num_scenes = len(scenes)
        # val_scenes = scenes[:int(0.2*num_scenes)]
        # test_scenes = scenes[int(0.2*num_scenes):]
        #
        # if self.mode == "val":
        #     scenes = val_scenes
        # else:
        #     scenes = test_scenes

        metadata = pd.read_csv(str(dataset_folder / 'ARKitScenes' / 'depth' / 'upsampling' / 'metadata.csv'))

        for scene in scenes:
            scene_info = metadata.loc[metadata['video_id'] == int(os.path.basename(scene))]
            sky_direction = scene_info.iloc[0][2]
            if sky_direction in 'Up':
                do_rot = False
            elif sky_direction in 'Down':
                do_rot = True
            else:
                continue

            image_names = [os.path.basename(f.path) for f in os.scandir(os.path.join(scene, 'wide'))]

            for image_name in image_names:
                self.filenames.append([os.path.join(scene, 'wide', image_name),
                                       os.path.join(scene, 'highres_depth', image_name),
                                       do_rot
                                       ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        rgb_path, depth_path, do_rot = self.filenames[idx]

        depth = Image.open(depth_path)  # [H, W, rgb]
        depth = np.asarray(depth).astype(np.float32)
        depth = depth / 1000.0

        # dummy one
        K = torch.eye(3)

        rgb = Image.open(rgb_path)
        dep = Image.fromarray(depth.astype('float32'), mode='F')

        if do_rot:
            rgb = rgb.transpose(Image.Transpose.ROTATE_180)
            dep = dep.transpose(Image.Transpose.ROTATE_180)

        # rgb, dep, K = self.process(rgb, dep, K, normalize_median=False)
        t_rgb = T.Compose([
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            T.Resize((480, 640)),
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
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