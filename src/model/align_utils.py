import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from optim_layer.optim_layer import DepthGradOptimLayer

def align_single_res(
        sparse_depth,
        pred_depth,
        patch_size
):
    B, _, H, W = sparse_depth.shape
    assert H % patch_size == 0 and W % patch_size == 0

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    # split into patches
    pred_patches = F.unfold(pred_depth, kernel_size=patch_size, stride=patch_size)
    pred_patches = pred_patches.permute(0, 2, 1).reshape(B, -1, 1, patch_size, patch_size)

    sparse_patches = F.unfold(sparse_depth, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    sparse_patches = sparse_patches.permute(0, 2, 1).reshape(B, -1, 1, patch_size, patch_size)

    # align each. combine batch and num_patch dimensions
    pred_patches = pred_patches.reshape(-1, 1, patch_size, patch_size)
    sparse_patches = sparse_patches.reshape(-1, 1, patch_size, patch_size)

    _, scale_arr = align_least_square(sparse_patches, pred_patches, align_shift=False)
    scale_field_prior = scale_arr.reshape(B, 1, n_patches_h, n_patches_w)

    # smoothness term
    scale_field_gradients = torch.zeros_like(scale_field_prior).repeat(1, 2, 1, 1)

    # compute full scale field
    scale_field, _ = DepthGradOptimLayer.apply(scale_field_gradients,
                                               scale_field_prior,
                                               (scale_field_prior > 0.0).float(),
                                               torch.ones_like(scale_field_gradients),  # conf depth gradients
                                               torch.ones_like(scale_field_prior),  # conf_input
                                               1,  # resolution
                                               None,
                                               None,
                                               5.0,  # integ_alpha
                                               1e-5, 5000)

    # bi-linear upsample
    scale_field_upsampled = TF.resize(scale_field, (H, W))
    aligned_pred = scale_field_upsampled * pred_depth

    return aligned_pred

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)

def constrain_to_multiple_of(x, __multiple_of, min_val=0, max_val=None):
    y = (np.round(x / __multiple_of) * __multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / __multiple_of) * __multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / __multiple_of) * __multiple_of).astype(int)

    return y

def resize_image(image, size=518, multiply_of=14):
    B, _, H, W = image.shape

    # resize to 518 (short edge)
    scale_height = size / H
    scale_width = size / W

    # keep aspect ratio
    if scale_width > scale_height:
        # fit width
        scale_height = scale_width
    else:
        # fit height
        scale_width = scale_height

    h = constrain_to_multiple_of(scale_height * H, multiply_of)
    w = constrain_to_multiple_of(scale_width * W, multiply_of)

    image_resized = F.interpolate(image, (h, w), mode="bicubic", align_corners=True)

    return image_resized

def align_least_square(
    sparse_depth,
    pred_depth,
    align_shift=False
):
    # sparse depth and pred_depth both have shape B x 1 x H x W
    B, _, H, W = sparse_depth.shape
    sparse_depth_mask = (sparse_depth > 0.0)

    # assume pred is always dense
    # need at least one point to align.
    if align_shift:
        min_pts = 2
    else:
        min_pts = 1

    scale_arr, shift_arr = torch.zeros(B).to(sparse_depth.device), torch.zeros(B).to(sparse_depth.device)
    aligned_pred = torch.zeros_like(pred_depth)

    for b in range(B):
        # all 1 x H x W
        sparse_depth_ = sparse_depth[b]
        pred_depth_ = pred_depth[b]
        sparse_depth_mask_ = sparse_depth_mask[b]

        if torch.sum(sparse_depth_mask_) < min_pts:
            continue

        else:
            gt_masked = sparse_depth_[sparse_depth_mask_].reshape((-1, 1)) # N_valid x 1
            pred_masked = pred_depth_[sparse_depth_mask_].reshape((-1, 1)) # N_valid x 1

            if align_shift:
                _ones = torch.ones_like(pred_masked)
                A = torch.cat([pred_masked, _ones], axis=-1) # N_valid x 2
                X = torch.linalg.lstsq(A, gt_masked, rcond=None).solution
                scale, shift = X
                scale_arr[b] = scale
                shift_arr[b] = shift

            else:
                A = pred_masked
                X = torch.linalg.lstsq(A, gt_masked, rcond=None).solution
                scale = X
                shift = 0.0
                scale_arr[b] = scale

            aligned_pred[b] = scale * pred_depth[b] + shift

    if align_shift:
        return aligned_pred, scale_arr, shift_arr
    else:
        return aligned_pred, scale_arr