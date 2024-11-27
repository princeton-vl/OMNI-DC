import sys
sys.path.append('..')
sys.path.append('.')

from config import args
import os
from data import get as get_data
from summary import save_ply, PtsUnprojector
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random

from torch.utils.data import DataLoader

# ImageNet normalization
img_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
img_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

# Minimize randomness
def init_seed(args, seed=None):
    if seed is None:
        seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


save_base_path = '../../datasets/uniformat_release'
dataset_name = args.val_data_name
pattern_raw = args.val_depth_pattern
noise_level = args.val_depth_noise
split = args.benchmark_gen_split

print('dataset name: ', dataset_name, 'pattern: ', pattern_raw, 'noise: ', noise_level, 'split: ', split)

# save_name = dataset_name + '_sample' + str(args.train_depth_pattern)
save_name = args.benchmark_save_name

save_path = os.path.join(save_base_path, save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == "__main__":
    tgt_dataset = get_data(args, split)
    print('dataset length: ', len(tgt_dataset))

    init_seed(args)
    
    if args.save_uniformat_max_dataset_length < len(tgt_dataset):
        sample_idxs = np.random.choice(np.arange(len(tgt_dataset)), args.save_uniformat_max_dataset_length)
    else:
        sample_idxs = np.arange(len(tgt_dataset))

    dataset_sub = torch.utils.data.Subset(tgt_dataset, sample_idxs)
    loader = DataLoader(dataset=dataset_sub, batch_size=1,
                        shuffle=False, num_workers=args.num_threads)

    sid = 0
    for sample in tqdm(loader):
        depth = sample['gt'][0].cpu().numpy()[0] # H x W
        rgb = sample['rgb'][0].cpu().numpy().transpose(1, 2, 0) # H x W x 3
        sparse_depth = sample['dep'][0].cpu().numpy()[0] # H x W
        K = sample['K'][0].cpu().numpy() # 3 x 3

        rgb = (((rgb * img_std) + img_mean) * 255.0).astype(np.uint8) # 0-255 range

        save_dict = {
            'gt': depth,
            'rgb': rgb,
            'dep': sparse_depth,
            'K': K
        }

        np.save(os.path.join(save_path, str(sid).zfill(6) + '.npy'), save_dict)
        sid += 1
