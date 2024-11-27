import torch
import torch.nn as nn

class GradMatchingLoss(nn.Module):
    def __init__(self, args):
        super(GradMatchingLoss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.gamma = args.sequence_loss_decay

    def forward(self, seq_pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)

        seq_pred = [torch.clamp(pred, min=0, max=self.args.max_depth) for pred in seq_pred]

        mask = (gt > self.t_valid).type_as(seq_pred[0]).detach()
        mask_u = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_v = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        num_valid = torch.sum(mask_u, dim=[1, 2, 3]) + torch.sum(mask_v, dim=[1, 2, 3])

        n_predictions = len(seq_pred)
        loss = 0.0

        for i in range(n_predictions):
            i_weight = self.gamma ** ((n_predictions - 1) - i)
            residual = seq_pred[i] - gt

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