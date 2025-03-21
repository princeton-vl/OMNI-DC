import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backbone import Backbone
from convgru import BasicUpdateBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from optim_layer.optim_layer import DepthGradOptimLayer

from depth_models.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from align_utils import resize_image, depth2disparity, disparity2depth, align_least_square, align_single_res

dav2_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def upsample_depth(depth, mask, r=8):
    """ Upsample depth field [H/r, W/r, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape  # B x 1 x H x W
    mask = mask.view(N, 1, 9, r, r, H, W)
    mask = torch.softmax(mask, dim=2)

    up_depth = F.unfold(depth, [3, 3], padding=1)
    up_depth = up_depth.view(N, 1, 9, 1, 1, H, W)

    up_depth = torch.sum(mask * up_depth, dim=2)
    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
    return up_depth.reshape(N, 1, r * H, r * W)


class OGNIDC(nn.Module):
    def __init__(self, args):
        super(OGNIDC, self).__init__()

        self.args = args
        self.GRU_iters = self.args.GRU_iters

        if args.load_dav2:
            depth_input_channels = 2
        else:
            depth_input_channels = 1

        self.backbone = Backbone(args, mode=self.args.backbone_mode, depth_input_channels=depth_input_channels)

        self.hdim = args.gru_hidden_dim
        self.cdim = args.gru_context_dim

        self.resolution = args.num_resolution

        encoder = 'vitl'

        if self.args.load_dav2:

            self.depth_module = DepthAnythingV2(**dav2_model_configs[encoder])
            self.depth_module.load_state_dict(
                torch.load(f'./depth_models/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth',
                           map_location='cpu'))
            self.depth_module = self.depth_module.eval()

            # freeze foundation model
            for param in self.depth_module.parameters():
                param.requires_grad = False

            self.depth_module_input_size = 518

        # NLSPN
        self.prop_time = args.prop_time
        if args.spn_type == "nlspn":
            from nlspn_module import NLSPN

            self.num_neighbors = args.prop_kernel * args.prop_kernel - 1
            if self.prop_time > 0:
                self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                        self.args.prop_kernel)
        elif args.spn_type == "dyspn":
            from dyspn_module import DySPN_Module

            self.num_neighbors = 5
            if self.prop_time > 0:
                assert self.prop_time == 6
                self.prop_layer = DySPN_Module(iteration=self.prop_time,
                                               num=self.num_neighbors,
                                               mode='yx')
        else:
            raise NotImplementedError

        # DySPN

        self.downsample_rate = args.backbone_output_downsample_rate
        self.update_block = BasicUpdateBlock(args=self.args, resolution=self.resolution, hidden_dim=self.hdim,
                                             mask_r=self.downsample_rate,
                                             conf_min=self.args.conf_min)

    def initialize_depth(self, sparse_depth):
        log_depth_init = torch.zeros_like(sparse_depth)
        log_depth_grad_init = torch.zeros_like(sparse_depth).repeat(1, 2 * self.resolution, 1, 1)  # B x 2 x H x W

        return log_depth_init, log_depth_grad_init

    def forward(self, sample):
        rgb = sample['rgb']
        dep = torch.clone(sample['dep'])
        dep_original = torch.clone(dep)
        K = sample['K']
        depth_pattern = sample['pattern']

        B, _, H, W = rgb.shape

        # there are two sparse depths:
        # dep_integrator is scale-senstive, bringing the actual scale values to the depth integrator
        # dep_network_input is scale-agnostic, making the network invariant to depth scale changes
        # if you multiply the sparse depth by a factor s, the network is guaranteed to produce a
        # dense depth also multiplied by the factor s.

        valid_sparse_mask = (dep > 0.0).float()
        valid_sparse_mask_network_input = torch.clone(valid_sparse_mask)

        K_downsampled = torch.clone(K)
        if self.downsample_rate > 1:
            K_downsampled[:, 0] /= float(self.downsample_rate)
            K_downsampled[:, 1] /= float(self.downsample_rate)

        # this is full-res depth
        if self.args.whiten_sparse_depths:
            medians = torch.ones(B, device=rgb.device)
            for b in range(B):
                nonzeros = dep_original[b] > 0.0
                if len(nonzeros) > 0:
                    medians[b] = torch.median(dep_original[b][nonzeros])

            dep_network_input = dep_original / medians.reshape(B, 1, 1, 1)  # make the median to be always 1.0
        else:
            dep_network_input = torch.clone(dep_original)

        # sparse depth needs downsample before feeding into the optim layer
        if self.downsample_rate > 1:
            if self.args.depth_downsample_method == "mean":
                dep = F.avg_pool2d(dep, self.downsample_rate)
                valid_sparse_mask = F.avg_pool2d(valid_sparse_mask, self.downsample_rate)
                dep[valid_sparse_mask > 0.0] = dep[valid_sparse_mask > 0.0] / valid_sparse_mask[valid_sparse_mask > 0.0]
                valid_sparse_mask[valid_sparse_mask > 0.0] = 1.0
            elif self.args.depth_downsample_method == "min":
                dep[dep == 0.0] = 100000.0  # set the invalid values to inf
                dep = -F.max_pool2d(-dep, self.downsample_rate)  # trick to do min-pooling
                valid_sparse_mask = F.max_pool2d(valid_sparse_mask,
                                                 self.downsample_rate)  # mask is 1 if at least one pt in neighbor
                dep[valid_sparse_mask == 0.0] = 0.0  # set invalid value back to 0.0, for safety
            else:
                raise NotImplementedError

        if self.args.depth_activation_format == "exp":
            dep_integrator = torch.log(dep)
            dep_network_input = torch.log(dep_network_input)
        else:
            dep_integrator = dep
            dep_network_input = dep_network_input

        dep_integrator[valid_sparse_mask == 0.0] = 0.0
        dep_network_input[valid_sparse_mask_network_input == 0.0] = 0.0

        if self.args.training_depth_random_shift_range > 0.0 and self.training:
            batch_size = rgb.shape[0]
            random_shift = torch.empty(batch_size).uniform_(-0.5,
                                                            0.5).cuda() * self.args.training_depth_random_shift_range
            dep_network_input = dep_network_input + random_shift.reshape(batch_size, 1, 1, 1)

        if self.args.load_dav2:
            if self.training:
                dav2_depth = sample['mono_dep']
            else:
                rgb_resized = resize_image(rgb, size=self.depth_module_input_size)  # B x 3 x 518 x W_resized

                # relative depth
                depth_pred_raw = self.depth_module.forward(rgb_resized).unsqueeze(1)  # B x 1 x 518 x W_resized
                depth_pred_raw = F.relu(depth_pred_raw)

                # resize back
                depth_pred_raw = F.interpolate(depth_pred_raw, (H, W), mode="bilinear",
                                               align_corners=True)  # B x 1 x H x W

                # normalize to [0,1]
                _min = torch.quantile(depth_pred_raw.reshape(B, -1), 0.02, dim=1).reshape(B, 1, 1, 1)
                _max = torch.quantile(depth_pred_raw.reshape(B, -1), 0.98, dim=1).reshape(B, 1, 1, 1)

                dav2_depth = 1.0 * (depth_pred_raw - _min) / (_max - _min)

            dep_network_input = torch.cat([dep_network_input, dav2_depth], dim=1)
        else:
            dep_network_input = dep_network_input

        # backbone
        assert self.args.pred_context_feature
        _, spn_guide, spn_confidence, context, confidence_input, confidence_output = self.backbone(rgb,
                                                                                                   dep_network_input,
                                                                                                   depth_pattern)

        if confidence_input is None:
            confidence_input = torch.ones_like(dep)  # B x 1 x H x W

        net, inp = torch.split(context, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initialization
        log_depth_pred, log_depth_grad_pred_init = self.initialize_depth(dep)
        log_depth_grad_pred = log_depth_grad_pred_init

        # dummy variable fpr recording gradients
        b_init = torch.zeros_like(dep, requires_grad=True)

        log_depth_grad_predictions = []  # record the init value also
        confidence_predictions = []
        depth_predictions_up = []
        depth_predictions_up_initial = []

        resolution = self.resolution

        for itr in range(self.GRU_iters):
            log_depth_pred = log_depth_pred.detach()
            log_depth_grad_pred = log_depth_grad_pred.detach()

            # ideally, we should whiten the log_depth_pred, so that the input to gru is always invariant to depth scale.

            if self.args.gru_internal_whiten_method == "mean":
                log_depth_pred_mean = torch.mean(log_depth_pred, dim=(1, 2, 3), keepdim=True)
                log_depth_pred_whitened = log_depth_pred - log_depth_pred_mean
            else:
                log_depth_pred_median = torch.median(log_depth_pred.reshape(B, -1), dim=1)[0]
                log_depth_pred_whitened = log_depth_pred - log_depth_pred_median.reshape(B, 1, 1, 1)

            net, up_mask, delta_log_depth_grad, weights_depth_grad, weights_input = self.update_block(net, inp,
                                                                                                      log_depth_pred_whitened,
                                                                                                      log_depth_grad_pred,
                                                                                                      K_downsampled
                                                                                                      )
            # print('depth grad pred', log_depth_grad_pred)
            log_depth_grad_pred = log_depth_grad_pred + delta_log_depth_grad

            # numerical stability
            thres = self.args.optim_layer_input_clamp
            log_depth_grad_pred = torch.clamp(log_depth_grad_pred, min=-thres, max=thres)

            # the optimization layer use the prediction from last round to accelerate convergence
            log_depth_pred, b_init = DepthGradOptimLayer.apply(log_depth_grad_pred,
                                                               dep_integrator,
                                                               valid_sparse_mask,
                                                               weights_depth_grad,
                                                               confidence_input * weights_input,
                                                               resolution,
                                                               log_depth_pred,
                                                               b_init,
                                                               self.args.integration_alpha,
                                                               1e-5, 5000)

            log_depth_grad_predictions.append(log_depth_grad_pred)
            confidence_predictions.append(weights_depth_grad)

            # convex upsample
            if self.downsample_rate > 1:
                log_depth_up = upsample_depth(log_depth_pred, up_mask, r=self.downsample_rate)
            else:
                log_depth_up = log_depth_pred

            # in case where Hrgb / downsample_rate is not integer, extra interpolation is needed
            _, _, Hrgb, Wrgb = rgb.shape
            _, _, Hd, Wd = log_depth_up.shape
            if Hd != Hrgb or Wd != Wrgb:
                print('warning: dim mismatch!')
                log_depth_up = F.interpolate(log_depth_up, size=(Hrgb, Wrgb), mode='bilinear', align_corners=True)

            if self.args.depth_activation_format == "exp":
                depth_pred_up_init = torch.exp(log_depth_up)
            else:
                depth_pred_up_init = log_depth_up

            depth_predictions_up_initial.append(depth_pred_up_init)

            # SPN
            if self.prop_time > 0 and (self.training or itr == self.GRU_iters - 1):
                if self.args.spn_type == "dyspn":
                    spn_out = self.prop_layer(depth_pred_up_init,
                                              spn_guide,
                                              dep_original,
                                              spn_confidence)
                    depth_pred_up_final = spn_out['pred']
                    dyspn_offset = spn_out['offset']
                elif self.args.spn_type == "nlspn":
                    depth_pred_up_final, _, _, _, _ = self.prop_layer(depth_pred_up_init, spn_guide, spn_confidence,
                                                                      None)
                    dyspn_offset = None
            else:
                depth_pred_up_final = depth_pred_up_init
                dyspn_offset = None

            depth_predictions_up.append(depth_pred_up_final)

        output = {'pred': depth_predictions_up[-1], 'pred_inter': depth_predictions_up,
                  'depth_predictions_up_initial': depth_predictions_up_initial,
                  'log_depth_grad_inter': log_depth_grad_predictions,
                  'log_depth_grad_init': log_depth_grad_pred_init,
                  'confidence_depth_grad_inter': confidence_predictions,
                  'dep_down': dep,
                  'confidence_input': confidence_input,
                  'confidence_output': confidence_output,
                  'dyspn_offset': dyspn_offset
                  }

        return output