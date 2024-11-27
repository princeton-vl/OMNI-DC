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

def save_pointcloud_visualization(save_path, depth, rgb, K):
    # depth has shape B x 1 x H x W
    mask = (depth > 0.0).float()

    unprojector = PtsUnprojector(device=torch.device('cpu'))
    xyz = unprojector(depth, K, mask=mask)  # N x 3
    colors = unprojector.apply_mask(rgb, mask=mask)

    save_ply(save_path, xyz, colors)

cmap = 'jet'
cm = plt.get_cmap(cmap)

# ImageNet normalization
img_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
img_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

num_image_samples = 10

# split = 'train'
# split = 'val'
split = 'test'
save_pointcloud = False

debug_path = './dataset_debug'
if split == "train":
    dataset_name = args.train_data_name
    pattern_raw = args.train_depth_pattern
    noise_level = args.train_depth_noise
else:
    dataset_name = args.val_data_name
    pattern_raw = args.val_depth_pattern
    noise_level = args.val_depth_noise
print('dataset name: ', dataset_name)
print('depth pattern: ', pattern_raw)
print('noise level: ', noise_level)

save_path = os.path.join(debug_path, dataset_name + '_' + split)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# vis_space = 'linear'
vis_space = 'log'

if __name__ == "__main__":
    class Object(object):
        pass

    tgt_dataset = get_data(args, split)
    print('dataset length: ', len(tgt_dataset))

    np.random.seed(0)
    torch.manual_seed(0)

    for sid, idx in tqdm(enumerate(np.random.choice(np.arange(len(tgt_dataset)), num_image_samples))):
        sample = tgt_dataset.__getitem__(idx)

        depth = sample['gt'].cpu().numpy()[0]
        rgb = sample['rgb'].cpu().numpy().transpose(1, 2, 0) # H x W x 3
        sparse_depth = sample['dep'].cpu().numpy()[0]

        gt_noise_postive = np.abs((sparse_depth - depth) > 0.1).astype(float)
        gt_noise_negative = 1.0 - gt_noise_postive

        # only compute loss in these areas: sparse noise are not empty
        gt_noise_valid_area = (sparse_depth > 0.001).astype(float)
        noise_mask_vis = cm(gt_noise_negative)[..., :3]
        noise_mask_vis[np.repeat(gt_noise_valid_area[..., None], 3, -1) == 0.0] = 1.0

        # print(sample['pattern'].numpy())

        kernel = np.ones((2, 2))
        sparse_depth = cv2.dilate(sparse_depth, kernel, iterations=1)

        h, w, _ = rgb.shape

        depth_nonzero = depth[depth!=0.0]
        # depth_min, depth_max = depth_nonzero.min(), depth_nonzero.max()
        depth_min = np.percentile(depth_nonzero, 5)
        depth_max = np.percentile(depth_nonzero, 95)

        vis_to_save = None

        # compute the bar range
        depth_normalizer = plt.Normalize(vmin=depth_min, vmax=depth_max)
        depth_vis = cm(depth_normalizer(depth))[..., :3]
        sparse_depth_vis = cm(depth_normalizer(sparse_depth))[..., :3]

        # mark invalid px as white
        depth_vis[np.repeat(depth[..., None], 3, -1) == 0.0] = 1.0

        err = np.abs(depth - sparse_depth)
        err_normalizer = plt.Normalize(vmin=0.0, vmax=0.01)
        err_vis = cm(err_normalizer(err))[..., :3]
        err_vis[np.repeat(sparse_depth[..., None], 3, -1) == 0.0] = 1.0

        # superimpose
        rgb_vis = rgb.copy()
        rgb_vis = (rgb_vis * img_std) + img_mean
        valid_depth_px = np.repeat(sparse_depth[..., None], 3, -1) != 0 # H x W x 1
        rgb_vis[valid_depth_px] = sparse_depth_vis[valid_depth_px]
        rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

        # concate
        # vis_to_save = np.concatenate([rgb_vis, depth_vis, err_vis], axis=1)
        # vis_to_save = np.concatenate([rgb_vis, depth_vis, noise_mask_vis], axis=1)
        vis_to_save = np.concatenate([rgb_vis, depth_vis], axis=1)

        # Create a figure and axis for the image
        fig, ax = plt.subplots()

        # Display the image without applying any colormap
        ax.imshow(vis_to_save)

        plt.subplots_adjust(right=0.85)

        # Create a separate axis for the color bar
        cax = fig.add_axes([0.9, 0.25, 0.02, 0.5])

        # Create a dummy image with the desired colormap for the color bar
        dummy_image = np.linspace(1, 0, 100).reshape(1, 100)

        # Display the dummy image in the color bar axis
        cbar = cax.imshow(dummy_image, aspect='auto', cmap=cmap)

        # Add the color bar to the dummy image
        cbar = fig.colorbar(cbar, cax=cax, orientation='vertical')

        # Add ticks
        num_ticks = 5
        ticks = np.linspace(1, 0, num_ticks)
        tick_labels = [f"{v:.2f}" for v in ticks * (depth_max - depth_min) + depth_min]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)

        # # add a scale bar
        # scale_bar_width = 20
        # num_ticks = 5
        #
        # scale_bar = np.repeat(np.linspace(0, 1, h).reshape(h, 1), scale_bar_width, axis=1)
        # scale_bar_map = plt.cm.get_cmap(cmap)(scale_bar)[:, :, :3]
        #
        #
        # plt.yticks(ticks=ticks, labels=tick_labels)
        # plt.axis('on')

        # save image
        # plt.imsave(os.path.join(debug_path, f'{dataset_name}_{sid}_img{idx}.png'), vis_to_save)

        plt.savefig(os.path.join(save_path, f'samples{pattern_raw}_noise{noise_level}_{dataset_name}_{sid}_img{idx}.png'), format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

        if save_pointcloud:
            depth_torch = sample['gt'].reshape(1, 1, h, w)
            rgb_torch = torch.tensor(rgb_vis).permute(2, 0, 1).reshape(1, 3, h, w)
            K_torch = sample['K'].reshape(1, 3, 3)
            save_pointcloud_visualization(os.path.join(save_path, f'{dataset_name}_{sid}_img{idx}.ply'),
                                          depth_torch,
                                          rgb_torch,
                                          K_torch)
