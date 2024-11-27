import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConfInputLoss(nn.Module):
    def __init__(self, args):
        super(ConfInputLoss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, conf_input_pred, sparse_depth, gt):
        _, _, H_pred, W_pred = conf_input_pred.shape
        _, _, H_gt, W_gt = gt.shape
        down_rate_h = H_gt / H_pred
        down_rate_w = W_gt / W_pred
        assert np.isclose(down_rate_h, down_rate_w)

        down_rate = int(np.round(down_rate_h))

        # generate the gt noise mask
        # these areas has noise
        gt_noise_postive = torch.abs((sparse_depth - gt) > 0.1).float()
        gt_noise_postive = F.max_pool2d(gt_noise_postive, down_rate)

        # noise-free areas should have a 1.0 confidence
        gt_noise_negative = 1.0 - gt_noise_postive

        # only compute loss in these areas: sparse noise are not empty
        gt_noise_valid_area = (sparse_depth > self.t_valid).float()
        gt_noise_valid_area = F.max_pool2d(gt_noise_valid_area, down_rate)
        num_valid = torch.sum(gt_noise_valid_area, dim=[1, 2, 3])

        loss = F.binary_cross_entropy(conf_input_pred, gt_noise_negative, reduction='none') * gt_noise_valid_area
        loss = torch.nan_to_num(loss)
        loss = torch.sum(loss, dim=[1, 2, 3]) / (num_valid + 1e-8)

        return loss.sum()
