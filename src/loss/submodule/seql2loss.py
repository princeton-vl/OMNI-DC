import torch
import torch.nn as nn

class SeqL2Loss(nn.Module):
    def __init__(self, args):
        super(SeqL2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.gamma = args.sequence_loss_decay

    def forward(self, seq_pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)

        seq_pred = [torch.clamp(pred, min=0, max=self.args.max_depth) for pred in seq_pred]

        mask = (gt > self.t_valid).type_as(seq_pred[0]).detach()
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        n_predictions = len(seq_pred)
        loss = 0.0

        for i in range(n_predictions):
            i_weight = self.gamma ** ((n_predictions - 1) - i)
            i_loss = torch.pow(seq_pred[i] - gt, 2) * mask
            i_loss = torch.nan_to_num(i_loss)
            i_loss = torch.sum(i_loss, dim=[1, 2, 3]) / (num_valid + 1e-8)
            loss += i_weight * i_loss.sum()

        return loss
