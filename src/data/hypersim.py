"""
    CompletionFormer
    ======================================================================

    NYU Depth V2 Dataset Helper
"""


import os
import warnings

import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

def getcamrays(width, height, K):
    u = np.linspace(0.5, width - 0.5, width).reshape(1, width).repeat(
        height, 0).astype(np.float32).reshape(-1, 1)
    v = np.linspace(0.5, height - 0.5, height).reshape(height, 1).repeat(
        width, 1).astype(np.float32).reshape(-1, 1)

    ones = np.ones_like(u)

    uv1 = np.concatenate([u, v, ones], -1)
    K_inv = np.linalg.inv(K)

    camrays = K_inv @ np.transpose(uv1)  # 3 x ?
    camrays = np.transpose(camrays).reshape(height, width, 3)

    return camrays

def distance2depth(distance, width, height, K):
    # modified from https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
    camrays = getcamrays(width, height, K)
    depth = distance / np.linalg.norm(camrays, 2, 2)

    return depth

class Hypersim(BaseDataset):
    def __init__(self, args, mode):
        super(Hypersim, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)
        self.resize_width = args.resize_width
        self.resize_height = args.resize_height

        if mode != 'train':
            raise NotImplementedError

        self.original_height, self.original_width = (768, 1024)

        self.dataset_max_depth_value = 50.0

        from pathlib import Path
        dataset_folder = (Path(__file__).parent.parent.parent / "datasets").resolve()
        self.dataset_basedir = str(dataset_folder / "hypersim" / "dataset")

        split_file = str(dataset_folder / "hypersim" / "ml-hypersim" / "evermotion_dataset" / "analysis" / "metadata_images_split_scene_v1.csv")
        split_data = pd.read_csv(split_file)
        self.split_data = split_data.to_dict()

        metadata_camera_parameters_csv_file =  str(dataset_folder / "hypersim" / "ml-hypersim" / "contrib" / "mikeroberts3000" / "metadata_camera_parameters.csv")
        self.df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")

        invalid_scenes = list()
        # filter out scenes that too far from pinhole camera
        focal_tol = 0.01
        focal_mean = 0.5773502886169104
        focal_values = self.df_camera_parameters['M_cam_from_uv_00'].to_dict()
        invalid_scenes_focal = [k for k, v in focal_values.items() if np.abs(v - focal_mean) > focal_tol]
        invalid_scenes.extend(invalid_scenes_focal)

        shift_tol = 0.01
        shift_mean = -1.0
        shift_values = self.df_camera_parameters['M_cam_from_uv_22'].to_dict()
        invalid_scenes_shift = [k for k, v in shift_values.items() if np.abs(v - shift_mean) > shift_tol]
        invalid_scenes.extend(invalid_scenes_shift)

        invalid_scenes = list(set(invalid_scenes))

        print('Loading Hypersim...')

        # print(f'removing {len(invalid_scenes)} scenes from dataset:', invalid_scenes)

        train_splits = np.where(
            np.logical_and(
                np.array([(v not in invalid_scenes) for v in list(self.split_data['scene_name'].values())])
            ,np.array(list(self.split_data['included_in_public_release'].values()))
            )
        )[0]

        self.sample_list = train_splits
        self.augment = self.args.augment

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        scene_idx = self.sample_list[idx]
        scene_name = self.split_data['scene_name'][scene_idx]
        cam_name = self.split_data['camera_name'][scene_idx]
        frame_id = self.split_data['frame_id'][scene_idx]

        scene_descriptor = scene_name + '_' + cam_name + '_' + str(frame_id).zfill(4)

        cam_info_raw = self.df_camera_parameters.loc[scene_name]

        # # method 1: assume a pinhole camera model
        #
        # focal_x_raw = cam_info_raw['M_cam_from_uv_00']
        # focal_x_px = (self.width / 2.0) / focal_x_raw
        #
        # focal_y_raw = cam_info_raw['M_cam_from_uv_11']
        # focal_y_px = (self.height / 2.0) / focal_y_raw
        #
        # cx_raw = cam_info_raw['M_cam_from_uv_02']
        # cx_px = (self.width / 2.0) - focal_x_px * cx_raw
        #
        # # the definition of camera coordinates in hypersim:
        # #      y
        # #      |
        # #      o -- x
        # #     /
        # #    z
        # # normal coordinates we want to use (OpenCV):
        # #        z
        # #       /
        # #      o -- x
        # #      |
        # #      y
        # # therefore, y is flipped. (we use + compared to - in x direction)
        #
        # cy_raw = cam_info_raw['M_cam_from_uv_12']
        # cy_px = (self.height / 2.0) + focal_y_px * cy_raw
        #
        # K = np.array([
        #     [focal_x_px, 0.0, cx_px],
        #     [0.0, focal_y_px, cy_px],
        #     [0.0, 0.0, 1.0]
        # ])

        # method 2:
        M_cam_T_hcam = np.array([[1., 0., 0.],
                                 [0., -1., 0.],
                                 [0., 0., -1.]])

        # hypersim_uv to hypersim cam
        # M_hcam_T_huv = np.array([[cam_info_raw["M_cam_from_uv_00"], cam_info_raw["M_cam_from_uv_01"], cam_info_raw["M_cam_from_uv_02"]],
        #                           [cam_info_raw["M_cam_from_uv_10"], cam_info_raw["M_cam_from_uv_11"], cam_info_raw["M_cam_from_uv_12"]],
        #                           [cam_info_raw["M_cam_from_uv_20"], cam_info_raw["M_cam_from_uv_21"], cam_info_raw["M_cam_from_uv_22"]]])

        # some intrinsics has non-zero 21 and 22 elements, which I don't know the meaning...
        # so just ignore them...
        M_hcam_T_huv = np.array(
            [[cam_info_raw["M_cam_from_uv_00"], 0.0, cam_info_raw["M_cam_from_uv_02"]],
             [0.0, cam_info_raw["M_cam_from_uv_11"], cam_info_raw["M_cam_from_uv_12"]],
             [0.0, 0.0, -1.0]])

        # uv to hypersim_uv
        M_huv_T_uv = np.array([[1. / (self.original_width / 2.0),  0.0, -1.],
                               [0.0,  -1. / (self.original_height / 2.0), 1.],
                               [0.0, 0.0, 1.0]
                               ])

        # uv to cam
        M_cam_T_uv = M_cam_T_hcam @ M_hcam_T_huv @ M_huv_T_uv

        # K is cam to uv
        K = np.linalg.inv(M_cam_T_uv)

        depth_h5fn = "frame." + str(frame_id).zfill(4) + ".depth_meters.hdf5"
        depth_h5fn = os.path.join(self.dataset_basedir, scene_name, "images", f"scene_{cam_name}_geometry_hdf5", depth_h5fn)

        with h5py.File(depth_h5fn, "r") as f:
            distance_meters = f["dataset"][:].astype(np.float32)
        # depth = distance2depth(distance_meters, self.width, self.height, focal_x_px, focal_y_px, cx_px, cy_px)
        depth = distance2depth(distance_meters, self.original_width, self.original_height, K)

        rgb_fn = "frame." + str(frame_id).zfill(4) + ".tonemap.jpg"
        rgb_fn = os.path.join(self.dataset_basedir, scene_name, "images", f"scene_{cam_name}_final_preview", rgb_fn)
        # rgb = cv2.cvtColor(cv2.imread(rgb_fn), cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)
        # rgb = rgb / 255.0 # [0,1]
        rgb = cv2.cvtColor(cv2.imread(rgb_fn), cv2.COLOR_BGR2RGB).astype(np.uint8)
        rgb = Image.fromarray(rgb, mode='RGB')

        # some pixels may be invalid
        depth_valid_mask = np.ones_like(depth)

        infinite_vals_mask = np.logical_not(np.isfinite(distance_meters))
        depth_valid_mask[infinite_vals_mask] = False

        depth[np.logical_not(depth_valid_mask)] = 0.0

        # hypersim contains some very big depth values. mark those as invalid
        depth[depth > self.dataset_max_depth_value] = 0.0

        K = torch.Tensor(K)

        dep = Image.fromarray(depth.astype(np.float32), mode='F')

        rgb, dep, K, rgb_np_raw = self.process(rgb, dep, K, normalize_median=self.args.data_normalize_median)

        if self.mode == "train":
            depth_pattern_raw = self.args.train_depth_pattern
        else:
            depth_pattern_raw = self.args.val_depth_pattern
        dep_sp, pattern_id = self.get_sparse_depth(dep, depth_pattern_raw, match_density=True, rgb_np=rgb_np_raw, input_noise=self.args.train_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output