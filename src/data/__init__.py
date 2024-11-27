"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================
"""


import numpy as np
from importlib import import_module

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Resize
import cv2

from PIL import Image
from .NNfill import fill_in_fast

PATTERN_IDS = {
    'random': 0,
    'velodyne': 1,
    'sfm': 2
}

def add_noise(dep, input_noise):
    # add noise
    # the noise can be "0.1" (fixed probablity) or "0.0~0.1" (uniform in the range)
    if input_noise != "0.0":
        if '~' in input_noise:
            noise_prob_low, noise_prob_high = input_noise.split('~')
            noise_prob_low, noise_prob_high = float(noise_prob_low), float(noise_prob_high)
        else:
            noise_prob_low, noise_prob_high = float(input_noise), float(input_noise)

        noise_prob = np.random.uniform(noise_prob_low, noise_prob_high)
        noise_mask = torch.tensor(np.random.binomial(n=1, p=noise_prob, size=dep.shape))
        depth_min, depth_max = np.percentile(dep, 10), np.percentile(dep, 90)
        noise_values = torch.tensor(np.random.uniform(depth_min, depth_max, size=dep.shape)).float()

        dep[noise_mask == 1] = noise_values[noise_mask == 1]

    return dep

def get(args, mode):
    if mode == "train":
        data_name = args.train_data_name
    elif mode == "val" or mode == "test":
        data_name = args.val_data_name
    else:
        raise NotImplementedError

    data_names = data_name.split("+")
    if len(data_names) == 1: # use the original dataset
        module_name = 'data.' + data_name.lower()
        dataset_name = data_name
    else:
        module_name = 'data.multidataset'
        dataset_name = 'MultiDataset'

    module = import_module(module_name)
    return getattr(module, dataset_name)(args, mode=mode)


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.max_depth_range = 100.0

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    def process(self, rgb, dep, K, normalize_median=True, random_crop=True):
        args = self.args

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scaled_h = int(self.resize_height * _scale)
            scaled_w = int(self.resize_width * _scale)
            degree = np.random.uniform(-self.args.random_rot_deg, self.args.random_rot_deg)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.BILINEAR)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)

            resize_rgb = T.Resize((scaled_h, scaled_w))
            resize_dep = T.Resize((scaled_h, scaled_w), InterpolationMode.NEAREST)

            rgb = resize_rgb(rgb)
            dep = resize_dep(dep)

            if random_crop:
                # Random crop
                i, j, h, w = T.RandomCrop.get_params(
                    rgb, output_size=self.crop_size)
                rgb = TF.crop(rgb, i, j, h, w)
                dep = TF.crop(dep, i, j, h, w)
            else:
                cropping_op = T.CenterCrop(self.crop_size)
                rgb = cropping_op(rgb)
                dep = cropping_op(dep)

            t_rgb = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_rgb_np_raw = T.Compose([
                self.ToNumpy(),
            ])

            t_dep = T.Compose([
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb_final = t_rgb(rgb)
            dep = t_dep(dep)
            rgb_np_raw = t_rgb_np_raw(rgb)

            K = K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale

        else:
            t_rgb = T.Compose([
                T.Resize((self.height, self.width)),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_rgb_np_raw = T.Compose([
                T.Resize((self.height, self.width)),
                T.CenterCrop(self.crop_size),
                self.ToNumpy()
            ])

            t_dep = T.Compose([
                T.Resize((self.height, self.width), InterpolationMode.NEAREST),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb_final = t_rgb(rgb)
            dep = t_dep(dep)
            rgb_np_raw = t_rgb_np_raw(rgb)

            K = K.clone()

        max_depth = torch.max(dep)
        depth_min_bound = max_depth / self.max_depth_range
        dep[dep < depth_min_bound] = 0.0

        if normalize_median:
            # compute the median
            median = torch.median(dep[dep > 0.0])
            dep = dep / median # normalize so that median is 1

        dep = torch.nan_to_num(dep)
        
        return rgb_final, dep, K, rgb_np_raw

    def get_sparse_depth(self, dep, pattern_raw, match_density=True, rgb_np=None, input_noise="0.0"):
        dep = torch.clone(dep)

        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)

        # the patterns can have format "0.8*100~2000+0.2*sift"
        all_weights = []
        all_patterns = []
        for pattern_item in pattern_raw.split('+'):
            if '*' in pattern_item:
                weight, pattern = pattern_item.split('*')
            else:
                weight, pattern = 1.0, pattern_item

            all_weights.append(weight)
            all_patterns.append(pattern)
        pattern = np.random.choice(all_patterns, p=all_weights)
        # further parse if needed
        if '~' in pattern:
            num_start, num_end = pattern.split('~')
            num_start = int(num_start)
            num_end = int(num_end)
            pattern = str(np.random.randint(num_start, num_end))

        if pattern.isdigit():
            num_sample = int(pattern)

            if match_density:
                # we want a uniform density
                num_sample_normalized = max(int(round(num_idx * num_sample / (height * width))), 5)
                idx_sample = torch.randperm(num_idx)[:num_sample_normalized]
            else:
                idx_sample = torch.randperm(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample[:]]

            mask = torch.zeros((channel * height * width))
            mask[idx_nnz] = 1.0
            mask = mask.view((channel, height, width))

            dep = add_noise(dep, input_noise)
            dep_sp = dep * mask.type_as(dep)
            pattern_id = PATTERN_IDS['random']

        elif pattern == "velodyne":
            # sample a virtual baseline
            if self.args.train_depth_velodyne_random_baseline:
                baseline_horizontal = np.random.choice([1.0, -1.0]) * np.random.uniform(0.03, 0.06)
                baseline_vertical = np.random.uniform(-0.02, 0.02)
            else:
                baseline_horizontal = 0.0
                baseline_vertical = 0.0

            # the target view canvas need to be slightly bigger
            target_view_expand_factor = 1.5
            height_expanded = int(target_view_expand_factor * height)
            width_expanded = int(target_view_expand_factor * width)

            # sample a virtual intrinsics
            w_c = np.random.uniform(-0.5*width, 1.5*width)
            h_c = np.random.uniform(0.5*height, 0.7*height)
            # w_c = 0.5 * width
            # h_c = 0.5 * height
            focal = np.random.uniform(1.5*height, 2.0*height)
            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            Km_target = np.copy(Km)
            Km_target[0, 2] += (target_view_expand_factor - 1.0) / 2.0 * width
            Km_target[1, 2] += (target_view_expand_factor - 1.0) / 2.0 * height

            dep_np = dep.numpy()

            # unproject every depth to a virtual neighboring view
            _, v, u = np.nonzero(dep_np)
            z = dep_np[0, v, u]
            points3D_source = np.linalg.inv(Km) @ (np.vstack([u, v, np.ones_like(u)]) * z) # 3 x N
            points3D_target = np.copy(points3D_source)
            points3D_target[0] -= baseline_horizontal # move in the x direction
            points3D_target[1] -= baseline_vertical # move in the y direction

            points2D_target = Km_target @ points3D_target
            depth_target = points2D_target[2]
            points2D_target = points2D_target[0:2] / (points2D_target[2:3] + 1e-8)  # 2 x N

            # 2 x N_valid
            points2D_target = np.round(points2D_target).astype(int)
            points2D_target_valid = points2D_target[:, ((points2D_target[0] >= 0) & (points2D_target[0] < width_expanded) &
                                                        (points2D_target[1] >= 0) & (points2D_target[1] < height_expanded))]

            # N_valid
            depth_target_valid = depth_target[((points2D_target[0] >= 0) & (points2D_target[0] < width_expanded) &
                                                        (points2D_target[1] >= 0) & (points2D_target[1] < height_expanded))]

            # take the min of all values
            dep_map_target = np.full((height_expanded, width_expanded), np.inf)
            np.minimum.at(dep_map_target, (points2D_target_valid[1], points2D_target_valid[0]), depth_target_valid)
            dep_map_target[dep_map_target == np.inf] = 0.0

            dep_map_target = fill_in_fast(dep_map_target, max_depth=np.max(dep_map_target))
            dep_map_target = dep_map_target[None] # 1 x H x W

            # mask out boundaries
            # dep_map_target_mask = np.zeros_like(dep_map_target)
            # dep_map_target_mask[:, (points2D_target_valid[1].min()):(points2D_target_valid[1].max()+1),
            #         (points2D_target_valid[0].min()):(points2D_target_valid[0].max()+1)] = 1.0
            # dep_map_target = dep_map_target * dep_map_target_mask

            # return torch.tensor(dep_map_target).unsqueeze(0)

            # sample the lidar patterns
            pitch_max = np.random.uniform(0.25, 0.30)
            pitch_min = np.random.uniform(-0.15, -0.20)
            num_lines = np.random.randint(8, 64)
            num_horizontal_points = np.random.randint(400, 1000)

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi/2.1, np.pi/2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid) # assume the distace is unit
            z = np.sqrt(1. - x**2 - y**2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1) # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8) # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[:, ((points_2D[0]>=0) & (points_2D[0]<width_expanded) & (points_2D[1]>=0) & (points_2D[1]<height_expanded))]

            mask = np.zeros([channel, height_expanded, width_expanded])
            mask[:, points_2D_valid[1], points_2D_valid[0]] = 1.0

            dep_map_target_sampled = dep_map_target * mask

            # project it back to source
            _, v, u = np.nonzero(dep_map_target_sampled)
            if len(v) == 0:
                return self.get_sparse_depth(dep, "1000", match_density=match_density, rgb_np=rgb_np, input_noise=input_noise)
            
            z = dep_map_target_sampled[0, v, u]
            points3D_target = np.linalg.inv(Km_target) @ (np.vstack([u, v, np.ones_like(u)]) * z)  # 3 x N
            points3D_source = np.copy(points3D_target)
            points3D_source[0] += baseline_horizontal  # move in the x direction
            points3D_source[1] += baseline_vertical  # move in the y direction

            points2D_source = Km @ points3D_source
            depth_source = points2D_source[2]
            points2D_source = points2D_source[0:2] / (points2D_source[2:3] + 1e-8)  # 2 x N

            # 2 x N_valid
            points2D_source = np.round(points2D_source).astype(int)
            points2D_source_valid = points2D_source[:, ((points2D_source[0] >= 0) & (points2D_source[0] < width) &
                                                        (points2D_source[1] >= 0) & (points2D_source[1] < height))]

            # N_valid
            depth_source_valid = depth_source[((points2D_source[0] >= 0) & (points2D_source[0] < width) &
                                                        (points2D_source[1] >= 0) & (points2D_source[1] < height))]

            # take the min of all values
            dep_map_source = np.full((height, width), np.inf)
            np.minimum.at(dep_map_source, (points2D_source_valid[1], points2D_source_valid[0]), depth_source_valid)
            dep_map_source[dep_map_source == np.inf] = 0.0

            # only keep the orginal valid regions
            dep_map_source = dep_map_source * ((dep_np > 0.0).astype(float))

            # only allow deeper value to appear in shallower region
            dep_map_source[dep_map_source < dep_np] = 0.0

            dep_sp = torch.tensor(dep_map_source).float()
            pattern_id = PATTERN_IDS['velodyne']

        elif pattern == "sift" or pattern == "orb":
            assert rgb_np is not None
            gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

            if pattern == "sift":
                detector = cv2.SIFT.create()
            elif pattern == "orb":
                detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            else:
                raise NotImplementedError

            keypoints = detector.detect(gray)
            mask = torch.zeros([1, height, width])

            if len(keypoints) < 20:
                return self.get_sparse_depth(dep, "1000", match_density=match_density, rgb_np=rgb_np, input_noise=input_noise)
            
            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                mask[:, x, y] = 1.0

            if self.args.train_sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, self.args.train_sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(mask)
                mask_keep = torch.bernoulli(mask_keep)

                mask = mask * mask_keep

            dep = add_noise(dep, input_noise)
            dep_sp = dep * mask.type_as(dep)
            pattern_id = PATTERN_IDS['sfm']

        elif pattern == "LiDAR_64" or pattern == "LiDAR_32" or pattern == "LiDAR_16" or pattern == "LiDAR_8":
            baseline_horizontal = 0.0
            baseline_vertical = 0.0

            w_c = 0.5 * width
            h_c = 0.5 * height
            focal = height

            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            Km_target = np.copy(Km)

            dep_np = dep.numpy()

            # sample the lidar patterns
            pitch_max = 0.5
            pitch_min = -0.5
            num_lines = int(pattern.split('_')[1])
            num_horizontal_points = 200

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi / 2.1, np.pi / 2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid)  # assume the distace is unit
            z = np.sqrt(1. - x ** 2 - y ** 2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1)  # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8)  # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[:, ((points_2D[0] >= 0) & (points_2D[0] < width) & (
                        points_2D[1] >= 0) & (points_2D[1] < height))]

            mask = np.zeros([channel, height, width])
            mask[:, points_2D_valid[1], points_2D_valid[0]] = 1.0

            dep_map_target_sampled = dep_np * mask

            # only keep the orginal valid regions
            dep_map_target_sampled = dep_map_target_sampled * ((dep_np > 0.0).astype(float))

            dep_sp = torch.tensor(dep_map_target_sampled).float()
            pattern_id = PATTERN_IDS['velodyne']

        else:
            raise NotImplementedError

        dep_sp = torch.nan_to_num(dep_sp)

        return dep_sp, torch.tensor(pattern_id)

    def refresh_indices(self):
        pass