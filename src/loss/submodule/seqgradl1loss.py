import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SeqGradL1Loss(nn.Module):
    def __init__(self, args):
        super(SeqGradL1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.gamma = args.sequence_loss_decay
        self.resolution = args.num_resolution

    def forward(self, seq_pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        mask = (gt > self.t_valid).type_as(seq_pred[0]).detach()

        _, _, H_pred, W_pred = seq_pred[0].shape
        _, _, H_gt, W_gt = gt.shape
        down_rate_h = H_gt / H_pred
        down_rate_w = W_gt / W_pred
        assert np.isclose(down_rate_h, down_rate_w)

        down_rate = int(np.round(down_rate_h))

        if down_rate > 1:
            assert down_rate in [2, 4]
            # downsample normal
            gt = F.avg_pool2d(gt, down_rate)
            mask = F.avg_pool2d(mask, down_rate)

            gt[mask > 0.0] = gt[mask > 0.0] / mask[mask > 0.0]
            mask[mask > 0.0] = 1.0

        _, _, H, W = gt.shape

        if self.args.depth_activation_format == "exp":
            log_gt = torch.log(gt) # B x 1 x H x W
        else:
            log_gt = gt
            
        log_gt[mask == 0.0] = 0.0

        log_grad_gt = torch.zeros_like(seq_pred[0])
        log_grad_mask = torch.zeros_like(seq_pred[0])

        for i in range(self.resolution):
            r = 2 ** i

            log_gt_this_res = F.avg_pool2d(log_gt, r)
            mask_this_res = F.avg_pool2d(mask, r)

            log_gt_this_res[mask_this_res > 0.0] = log_gt_this_res[mask_this_res > 0.0] / mask_this_res[mask_this_res > 0.0]
            mask_this_res[mask_this_res > 0.0] = 1.0

            x_grad_u = log_gt_this_res[:, 0, :, 1:] - log_gt_this_res[:, 0, :, :-1]
            mask_u = mask_this_res[:, 0, :, 1:] * mask_this_res[:, 0, :, :-1]

            x_grad_v = log_gt_this_res[:, 0, 1:, :] - log_gt_this_res[:, 0, :-1, :]
            mask_v = mask_this_res[:, 0, 1:, :] * mask_this_res[:, 0, :-1, :]

            log_grad_gt[:, i*2, :H//r, 1:W//r] = x_grad_u
            log_grad_mask[:, i*2, :H//r, 1:W//r] = mask_u

            log_grad_gt[:, i*2+1, 1:H//r, :W//r] = x_grad_v
            log_grad_mask[:, i*2+1, 1:H//r, :W//r] = mask_v

        num_valid = torch.sum(log_grad_mask, dim=[1, 2, 3])
        n_predictions = len(seq_pred)
        loss = 0.0

        for i in range(n_predictions):
            i_weight = self.gamma ** ((n_predictions - 1) - i)
            i_loss = torch.abs(seq_pred[i] - log_grad_gt) * log_grad_mask
            i_loss = torch.nan_to_num(i_loss)
            i_loss = torch.sum(i_loss, dim=[1, 2, 3]) / (num_valid + 1e-8)
            loss += i_weight * i_loss.sum()

        return loss
