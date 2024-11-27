import torch
import torch.nn as nn
import torch.nn.functional as F

class GradMatchingScaleLoss(nn.Module):
    def __init__(self, args):
        super(GradMatchingScaleLoss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.gamma = args.sequence_loss_decay
        self.scale_level = args.scale_level

    def forward(self, seq_pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)

        seq_pred = [torch.clamp(pred, min=0, max=self.args.max_depth) for pred in seq_pred]
        n_predictions = len(seq_pred)

        mask = (gt > self.t_valid).type_as(seq_pred[0]).detach()

        loss = 0.0

        for scale in range(self.scale_level):
            down_factor = 2 ** scale

            if down_factor > 1:
                # divisible by down_factor
                pad_h = (down_factor - gt.shape[-2] % down_factor) % down_factor
                pad_w = (down_factor - gt.shape[-1] % down_factor) % down_factor
                padding = (0, pad_w, 0, pad_h)
                gt_padded = F.pad(gt, padding, mode="replicate")
                mask_padded = F.pad(mask, padding, mode="replicate")

                gt_scaled = F.avg_pool2d(gt_padded, down_factor)
                mask_scaled = F.avg_pool2d(mask_padded, down_factor)
            else:
                gt_scaled = gt
                mask_scaled = mask

            gt_scaled[mask_scaled > 0.0] = gt_scaled[mask_scaled > 0.0] / mask_scaled[mask_scaled > 0.0]
            mask_scaled[mask_scaled > 0.0] = 1.0

            mask_u = mask_scaled[:, :, :, 1:] * mask_scaled[:, :, :, :-1]
            mask_v = mask_scaled[:, :, 1:, :] * mask_scaled[:, :, :-1, :]

            num_valid = torch.sum(mask_u, dim=[1, 2, 3]) + torch.sum(mask_v, dim=[1, 2, 3])

            for i in range(n_predictions):
                if down_factor > 1:
                    pred_padded = F.pad(seq_pred[i], padding, mode="replicate")
                    pred_scaled = F.avg_pool2d(pred_padded, down_factor)
                else:
                    pred_scaled = seq_pred[i]

                i_weight = self.gamma ** ((n_predictions - 1) - i)
                residual = pred_scaled - gt_scaled

                gradu_residual = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
                gradv_residual = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])

                loss_u = mask_u * gradu_residual
                loss_v = mask_v * gradv_residual

                loss_u, loss_v = torch.nan_to_num(loss_u), torch.nan_to_num(loss_v)
                i_loss_u = torch.sum(loss_u, dim=[1, 2, 3]) / (num_valid + 1e-8)
                i_loss_v = torch.sum(loss_v, dim=[1, 2, 3]) / (num_valid + 1e-8)
                i_loss = i_loss_u + i_loss_v

                loss += i_weight * i_loss.sum()

        return loss