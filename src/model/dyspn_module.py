# adapted from: https://github.com/Kyakaka/DySPN/blob/main/DySPN/module.py

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
from torchvision.ops import deform_conv2d
import math


# naive dynamic spatial propagation attention module
def edge_sum(k_f, dilation=1):
    ch_g = 4 * (k_f - 1)
    edge_range = []
    for i in range(k_f):
        for j in range(k_f):
            if i == 0 or i == k_f - 1 or j == 0 or j == k_f - 1:
                edge_range.append(i * k_f + j)
    weight = torch.zeros(1, ch_g, k_f, k_f)
    for i in range(ch_g):
        weight[:, i, -int(edge_range[i] / k_f) - 1, -edge_range[i] % k_f - 1] = 1
    sum_conv = nn.Conv2d(in_channels=ch_g,
                         out_channels=1,
                         kernel_size=(k_f, k_f),
                         stride=1,
                         dilation=dilation,
                         padding=int(k_f / 2) * dilation,
                         bias=False, )
    sum_conv.weight = nn.Parameter(weight)
    for param in sum_conv.parameters():
        param.requires_grad = False
    return sum_conv


# implement of cspn
class cspn_3x3_naive(nn.Module):
    def __init__(self, iteration=18):
        super(cspn_3x3_naive, self).__init__()

        self.prop_time = iteration
        self.sum_conv_3x3 = edge_sum(3)

    def _get_affinity_sum(self, guidance):
        B, _, H, W = guidance.shape

        aff_abs_sum_1 = self.sum_conv_3x3(torch.abs(guidance)) + 1e-4

        aff_sum_1 = self.sum_conv_3x3(guidance)

        return aff_sum_1, aff_abs_sum_1

    def forward(self, feat_init, guidance, confidence, feat_fix):
        mask_fix = feat_fix.sign()
        feat_result = feat_init.contiguous()
        list_feat = []
        aff_sum, aff_abs_sum = self._get_affinity_sum(guidance)

        for k in range(self.prop_time):
            feat_result_g = feat_result * guidance
            feat_result = (self.sum_conv_3x3(feat_result_g[:, 0:8, :, :]) +
                           (aff_abs_sum - torch.sum(aff_sum, dim=1, keepdim=True)) * feat_init) / aff_abs_sum
            feat_result = (1 - mask_fix * confidence) * feat_result + mask_fix * confidence * feat_fix
            list_feat.append(feat_result)
        return feat_result, list_feat, [], []


# implement of deformable spn
class dspn_3x3_naive(nn.Module):
    def __init__(self, iteration=18):
        super(dspn_3x3_naive, self).__init__()
        self.prop_time = iteration
        self.ch_g = 9
        self.ch_f = 1
        self.k_g = 3
        self.k_f = 3
        pad_g = int((self.k_g - 1) / 2)
        pad_f = int((self.k_f - 1) / 2)
        self.conv_offset_aff = nn.Conv2d(
            self.ch_g, 3 * self.ch_g, kernel_size=3, stride=1,
            padding=pad_g, bias=True,
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        self.alpha = nn.Parameter(torch.ones(1))

    def _get_offset_affinity(self, guidance):
        B, _, H, W = guidance.shape
        offset_aff = self.conv_offset_aff(guidance)
        section = [self.ch_g * 2, self.ch_g]
        offset_1, aff_1 = torch.split(offset_aff, section, dim=1)
        aff_1 = torch.softmax(aff_1, dim=1)
        return offset_1.half(), aff_1.half()
        # return offset_1, aff_1

    def _propagate_once(self, feat, offset, aff):
        return deform_conv2d(feat, offset, self.w, self.b, (self.stride, self.stride), (self.padding, self.padding),
                             (self.dilation, self.dilation), mask=aff)  # torch>=1.8

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, feat_init, guidance, confidence, feat_fix):
        assert self.ch_f == feat_init.shape[1]
        confidence = feat_fix.sign() * torch.sigmoid(confidence)
        feat_result = feat_init.float().contiguous()
        list_feat = []
        offset1, aff1 = self._get_offset_affinity(guidance)
        for k in range(self.prop_time):  # 0.002
            feat_result = (1 - confidence) * self._propagate_once(feat_result, offset1.float(),
                                                                  aff1.float()) + confidence * feat_fix

            list_feat.append(feat_result)
        return {'pred': feat_result,
                "list_feat": list_feat,
                "offset": offset1,
                "aff": aff1,
                }


# implement of 7x7 dyspn
class Dynamic_7x7_naivev2(nn.Module):
    # def __init__(self, args, ch_g, ch_f, k_g, k_f):
    def __init__(self):
        super(Dynamic_7x7_naivev2, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        self.prop_time = 6

        # Assume zero offset for center pixels

        self.sum_conv_3x3 = edge_sum(3)
        self.sum_conv_5x5 = edge_sum(5)
        self.sum_conv_7x7 = edge_sum(7)
        self.aff_scale_const = nn.Parameter(torch.ones(1))
        self.aff_scale_const.requires_grad = False

    def _get_affinity_sum(self, guidance):
        B, _, H, W = guidance.shape

        aff_abs_sum_1 = self.sum_conv_3x3(torch.abs(guidance[:, 0:8, :, :]))
        aff_abs_sum_2 = self.sum_conv_5x5(torch.abs(guidance[:, 8:24, :, :]))
        aff_abs_sum_3 = self.sum_conv_7x7(torch.abs(guidance[:, 24:48, :, :]))

        aff_sum_1 = self.sum_conv_3x3(guidance[:, 0:8, :, :])
        aff_sum_2 = self.sum_conv_5x5(guidance[:, 8:24, :, :])
        aff_sum_3 = self.sum_conv_7x7(guidance[:, 24:48, :, :])

        list_aff_sum = torch.cat((aff_sum_1, aff_sum_2, aff_sum_3, torch.ones(B, 1, H, W).cuda()), dim=1)
        list_aff_abs_sum = torch.cat((aff_abs_sum_1, aff_abs_sum_2, aff_abs_sum_3, torch.ones(B, 1, H, W).cuda()),
                                     dim=1)
        return list_aff_sum, list_aff_abs_sum

    def forward(self, feat_init, guidance, dynamic, confidence, feat_fix):
        mask_fix = feat_fix.sign()
        feat_result = feat_init.contiguous()
        list_feat = []
        aff_sum, aff_abs_sum = self._get_affinity_sum(guidance)
        dynamic = torch.sigmoid(dynamic)
        for k in range(self.prop_time):
            attention = dynamic.narrow(1, 4 * k, 4)
            gate_sum_abs = torch.sum(attention * aff_abs_sum, dim=1, keepdim=True) + 1e-4
            feat_result_g = feat_result * guidance
            feat_result = (attention[:, 0:1, :, :] * self.sum_conv_3x3(feat_result_g[:, 0:8, :, :]) +
                           attention[:, 1:2, :, :] * self.sum_conv_5x5(feat_result_g[:, 8:24, :, :]) +
                           attention[:, 2:3, :, :] * self.sum_conv_7x7(feat_result_g[:, 24:48, :, :]) +
                           attention[:, 3:4, :, :] * feat_result +
                           (gate_sum_abs - torch.sum(attention * aff_sum, dim=1,
                                                     keepdim=True)) * feat_init) / gate_sum_abs
            list_feat.append(feat_result)
            feat_result = (1 - mask_fix * confidence) * feat_result + mask_fix * confidence * feat_fix
        return {'pred': feat_result,
                "list_feat": list_feat,
                "aff": guidance,
                "dynamic": dynamic
                }


# DySPN with dynamic edge
class DySPN_Module(nn.Module):
    def __init__(self, iteration=3, num=9, mode='xy'):
        super().__init__()
        assert num in [1, 3, 5, 9], 'only sample num [1,3,5,9] supported but num = {}'.format(num)
        self.num = num  # sample num
        self.iteration = iteration  # iteration num
        self.mode = mode
        self.ch = self.num * self.iteration
        self.conv_offset_aff = nn.Conv2d(
            self.ch, 3 * self.ch, kernel_size=3, stride=1,
            padding=1, bias=True
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()
        # Dynamic spn attention
        if self.num == 9:
            self.offset_3x3 = nn.Parameter(torch.Tensor([[-1, -1], [0, -1], [1, -1],
                                                         [-1, 0], [0, 0], [1, 0],
                                                         [-1, 1], [0, 1], [1, 1]]).view(self.num, 1, 1, 2))  # BHWN2
        elif self.num == 5:
            self.offset_3x3 = nn.Parameter(torch.Tensor([[0, -1],
                                                         [-1, 0], [0, 0], [1, 0],
                                                         [0, 1]]).view(self.num, 1, 1, 2))  # BHWN2
        elif self.num == 3:
            self.offset_3x3 = nn.Parameter(torch.Tensor([[-1, 0], [0, 0], [1, 0]]).view(self.num, 1, 1, 2))  # BHWN2
        elif self.num == 1:
            self.offset_3x3 = nn.Parameter(torch.Tensor([[0, 0]]).view(self.num, 1, 1, 2))  # BHWN2
        else:
            raise NotImplementedError
        self.offset_3x3.requires_grad = False

    def get_refgrid(self, B, H, W, offset):
        offset = rearrange(offset, 'b (iter c n) h w -> b iter c h w n', iter=self.iteration, c=self.num)
        # Deformable cnn (y,x) is different from grid_sampler (x,y)
        ref_y = torch.linspace(-H + 1, H - 1, H, device=torch.device(offset.device))
        ref_x = torch.linspace(-W + 1, W - 1, W, device=torch.device(offset.device))
        if self.mode == 'yx':
            offset_ = offset.clone()
            offset_[..., 0] = ((offset[..., 1] + self.offset_3x3[..., 0]) * 2 + ref_x.view(1, 1, 1, W)) / W
            offset_[..., 1] = ((offset[..., 0] + self.offset_3x3[..., 1]) * 2 + ref_y.view(1, 1, H, 1)) / H
            return offset_
        elif self.mode == 'xy':
            offset[..., 0] = ((offset[..., 0] + self.offset_3x3[..., 0]) * 2 + ref_x.view(1, 1, 1, W)) / W
            offset[..., 1] = ((offset[..., 1] + self.offset_3x3[..., 1]) * 2 + ref_y.view(1, 1, H, 1)) / H
            return offset

    def forward(self,
                input: torch.Tensor,
                guide: torch.Tensor,
                sp_dep: torch.Tensor,
                confidence: torch.Tensor):

        B, C, H, W = input.shape
        offset, aff = torch.split(self.conv_offset_aff(guide), [2 * self.ch, self.ch], dim=1)
        confidence = torch.sigmoid(confidence) * sp_dep.sign()
        aff = rearrange(aff, 'b (iter c) h w -> b iter c h w', iter=self.iteration)
        offset = torch.unbind(self.get_refgrid(B, H, W, offset).float(), dim=1)
        aff = torch.chunk(torch.softmax(aff, dim=2), self.iteration, dim=1)
        inter = []
        input = input.float()
        feat_init = input
        for i in range(self.iteration):
            output = 0
            for j in range(self.num):
                output += F.grid_sample(input,
                                        offset[i][:, j, :, :, :],
                                        align_corners=False,
                                        padding_mode="zeros",
                                        mode="bilinear") * aff[i][:, :, j, :, :]
            input = (1 - confidence) * output + confidence * sp_dep
            inter.append(input)
        return {'pred': input,
                'pred_init': feat_init,
                "list_feat": inter,
                "offset": offset,
                "aff": aff
                }


class Dynamic_deformablev2(nn.Module):
    def __init__(self, iteration=6):
        super(Dynamic_deformablev2, self).__init__()
        self.prop_time = iteration
        self.ch_g = 8
        self.ch_f = 1
        self.k_g = 3
        self.k_f = 3
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2
        pad_g = int((self.k_g - 1) / 2)
        pad_f = int((self.k_f - 1) / 2)
        self.conv_offset_aff1 = nn.Conv2d(
            self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
            padding=pad_g, bias=True
        )
        self.conv_offset_aff1.weight.data.zero_()
        self.conv_offset_aff1.bias.data.zero_()
        self.conv_offset_aff2 = nn.Conv2d(
            self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
            padding=pad_g, bias=True
        )
        self.conv_offset_aff2.weight.data.zero_()
        self.conv_offset_aff2.bias.data.zero_()

        self.sum_conv = nn.Conv2d(in_channels=self.ch_g,
                                  out_channels=1,
                                  kernel_size=(self.k_f, self.k_f),
                                  stride=1,
                                  padding=int(self.k_f / 2),
                                  bias=False)
        weight = torch.zeros(1, self.ch_g, self.k_f, self.k_f).cuda()
        for i in range(self.ch_g):
            if i < self.ch_g / 2:
                weight[:, i, -int(i / self.k_f) - 1, -i % self.k_f - 1] = 1
            else:
                weight[:, i, -int((i + 1) / self.k_f) - 1, -(i + 1) % self.k_f - 1] = 1
        self.sum_conv.weight = nn.Parameter(weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False

        self.aff_scale_const = nn.Parameter(torch.ones(1))
        self.aff_scale_const.requires_grad = False

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    # @torchsnooper.snoop()
    def _get_offset_affinity(self, guidance):
        B, _, H, W = guidance.shape
        guidance1, guidance2, guidance3 = torch.chunk(guidance, 3, dim=1)
        offset_aff1 = self.conv_offset_aff1(guidance1)
        o1_1, o2_1, aff_1 = torch.chunk(offset_aff1, 3, dim=1)
        # Add zero reference offset
        offset1 = torch.cat((o1_1, o2_1), dim=1).view(B, self.num, 2, H, W)
        list_offset1 = list(torch.chunk(offset1, self.num, dim=1))
        list_offset1.insert(self.idx_ref,
                            torch.zeros((B, 1, 2, H, W)).type_as(offset1))
        offset1 = torch.cat(list_offset1, dim=1).view(B, -1, H, W)

        offset_aff2 = self.conv_offset_aff2(guidance2)
        o1_2, o2_2, aff_2 = torch.chunk(offset_aff2, 3, dim=1)
        # Add zero reference offset
        offset2 = torch.cat((o1_2, o2_2), dim=1).view(B, self.num, 2, H, W)
        list_offset2 = list(torch.chunk(offset2, self.num, dim=1))
        list_offset2.insert(self.idx_ref,
                            torch.zeros((B, 1, 2, H, W)).type_as(offset2))
        offset2 = torch.cat(list_offset2, dim=1).view(B, -1, H, W)
        aff_abs_1 = torch.abs(aff_1)
        aff_abs_2 = torch.abs(aff_2)
        aff_abs_3 = self.sum_conv(torch.abs(guidance3))
        aff_abs_sum_1 = torch.sum(aff_abs_1, dim=1, keepdim=True)
        aff_abs_sum_2 = torch.sum(aff_abs_2, dim=1, keepdim=True)
        aff_abs_sum_3 = aff_abs_3
        aff_sum_1 = torch.sum(aff_1, dim=1, keepdim=True)
        aff_sum_2 = torch.sum(aff_2, dim=1, keepdim=True)
        aff_sum_3 = self.sum_conv(guidance3)
        aff_ref = torch.zeros((B, 1, H, W)).type_as(aff_sum_1)
        list_aff_1 = list(torch.chunk(aff_1, self.num, dim=1))
        list_aff_1.insert(self.idx_ref, aff_ref)
        aff_1 = torch.cat(list_aff_1, dim=1)
        list_aff_2 = list(torch.chunk(aff_2, self.num, dim=1))
        list_aff_2.insert(self.idx_ref, aff_ref)
        aff_2 = torch.cat(list_aff_2, dim=1)
        list_aff_sum = torch.cat((aff_sum_1, aff_sum_2, aff_sum_3, torch.ones(B, 1, H, W).cuda()), dim=1)
        list_aff_abs_sum = torch.cat((aff_abs_sum_1, aff_abs_sum_2, aff_abs_sum_3, torch.ones(B, 1, H, W).cuda()),
                                     dim=1)

        return offset1, aff_1, offset2, aff_2, guidance3, list_aff_sum, list_aff_abs_sum

    def _propagate_once(self, feat, offset, aff):
        return deform_conv2d(feat, offset, self.w, self.b, (self.stride, self.stride), (self.padding, self.padding),
                             (self.dilation, self.dilation), mask=aff)  # torch>=1.8

    def forward(self, feat_init, guidance, dynamic, confidence, feat_fix):
        assert self.ch_f == feat_init.shape[1]
        confidence = torch.sigmoid(confidence)
        # dynamic = torch.sigmoid(dynamic)
        dynamic=torch.softmax(einops.rearrange(dynamic,'b (n c) h w -> b n c h w',n=self.prop_time),dim=2)
        dynamic = einops.rearrange(dynamic, 'b n c h w -> b (n c) h w')
        mask_fix = feat_fix.sign()
        feat_result = feat_init.contiguous()
        list_feat = []
        offset1, aff1, offset2, aff2, aff3, aff_sum, aff_abs_sum = self._get_offset_affinity(guidance)
        for k in range(self.prop_time):
            attention = dynamic.narrow(1, 4 * k, 4)
            gate_sum_abs = torch.sum(attention * aff_abs_sum, dim=1, keepdim=True) + 1e-4
            feat_result = (attention[:, 0:1, :, :] * self._propagate_once(feat_result, offset1, aff1) +
                           attention[:, 1:2, :, :] * self._propagate_once(feat_result, offset2, aff2) +
                           attention[:, 2:3, :, :] * self.sum_conv(aff3 * feat_result) +
                           attention[:, 3:4, :, :] * feat_result +
                           (gate_sum_abs - torch.sum(attention * aff_sum, dim=1,
                                                     keepdim=True)) * feat_init) / gate_sum_abs
            list_feat.append(feat_result)

            feat_result = (1 - mask_fix * confidence) * feat_result + mask_fix * confidence * feat_fix
        return {'pred': feat_result,
                'pred_init': feat_init,
                "list_feat": list_feat,
                "offset": offset1,
                "offset2": offset2,
                "aff": aff1,
                "aff2": aff2,
                "aff3": aff3,
                "dynamic": dynamic
                }


# DySPN with dynamic edge
class DySPN_Modulev2(nn.Module):
    def __init__(self, iteration=3, num=8, mode='xy'):
        super().__init__()
        # assert num in [1,3,5,9], 'only sample num [1,3,5,9] supported but num = {}'.format(num)
        self.num = num  # sample num
        self.iteration = iteration  # iteration num
        self.mode = mode
        self.ch = self.num * self.iteration
        self.conv_offset = nn.Conv2d(
            self.ch, 2 * self.ch, kernel_size=3, stride=1,
            padding=1, bias=True
        )
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

        self.conv_aff = nn.Conv2d(
            self.ch, self.ch + 1, kernel_size=3, stride=1,
            padding=1, bias=True
        )
        self.conv_aff.weight.data.zero_()
        self.conv_aff.bias.data.zero_()

    def get_refgrid(self, B, H, W, offset):
        offset = rearrange(offset, 'b (iter c n) h w -> b iter c h w n', iter=self.iteration, c=self.num)
        # Deformable cnn (y,x) is different from grid_sampler (x,y)
        ref_y = torch.linspace(-H + 1, H - 1, H, device=torch.device(offset.device))
        ref_x = torch.linspace(-W + 1, W - 1, W, device=torch.device(offset.device))
        if self.mode == 'yx':
            offset_ = offset.clone()
            offset_[..., 0] = ((offset[..., 1]) * 2 + ref_x.view(1, 1, 1, W)) / W
            offset_[..., 1] = ((offset[..., 0]) * 2 + ref_y.view(1, 1, H, 1)) / H
            return offset_
        elif self.mode == 'xy':
            offset[..., 0] = ((offset[..., 0]) * 2 + ref_x.view(1, 1, 1, W)) / W
            offset[..., 1] = ((offset[..., 1]) * 2 + ref_y.view(1, 1, H, 1)) / H
            return offset

    def forward(self,
                input: torch.Tensor,
                guide: torch.Tensor,
                sp_dep: torch.Tensor,
                confidence: torch.Tensor):

        B, C, H, W = input.shape
        offset = self.conv_offset(guide)
        aff = self.conv_aff(guide)

        confidence = torch.sigmoid(confidence) * sp_dep.sign()
        aff = rearrange(aff, 'b (iter c) h w -> b iter c h w', iter=self.iteration)

        offset = torch.unbind(self.get_refgrid(B, H, W, offset).float(), dim=1)
        aff = torch.chunk(torch.softmax(aff, dim=2), self.iteration, dim=1)
        inter = []
        input = input.float()
        for i in range(self.iteration):
            output = 0
            for j in range(self.num):
                output += F.grid_sample(input,
                                        offset[i][:, j, :, :, :],
                                        align_corners=False,
                                        padding_mode="zeros",
                                        mode="bilinear") * aff[i][:, :, j, :, :]
            input = (1 - confidence) * output + confidence * sp_dep
            inter.append(input)
        return input, inter, offset, aff


class Dynamic_deformable_DySamplev6(nn.Module):
    # def __init__(self, args, ch_g, ch_f, k_g, k_f):
    def __init__(self, prop_time=3):
        super(Dynamic_deformable_DySamplev6, self).__init__()
        self.prop_time = prop_time
        self.ch_g = 9
        self.ch_f = 1
        self.k_g = 3
        self.k_f = 3
        pad_g = int((self.k_g - 1) / 2)
        pad_f = int((self.k_f - 1) / 2)
        self.num = self.k_f * self.k_f
        self.conv_offset_aff = nn.Conv2d(
            self.ch_g * self.prop_time, 3 * self.ch_g * self.prop_time, kernel_size=3, stride=1,
            padding=pad_g, bias=True,
            # groups=self.prop_time
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        self.alpha = nn.Parameter(torch.ones(1))

    def _get_offset_affinity(self, guidance):
        B, _, H, W = guidance.shape
        offset_aff = self.conv_offset_aff(guidance).view(B, self.prop_time, self.num * 3, H, W)
        section = [self.num * 2, self.num]
        offset_1, aff_1 = torch.split(offset_aff, section, dim=2)
        aff_1 = torch.softmax(aff_1, dim=2)
        return torch.unbind(offset_1.half(), dim=1), torch.unbind(aff_1.half(), dim=1)

    def _propagate_once(self, feat, offset, aff):
        return deform_conv2d(feat, offset, self.w, self.b, (self.stride, self.stride), (self.padding, self.padding),
                             (self.dilation, self.dilation), mask=aff)  # torch>=1.8

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, feat_init, guidance, confidence, feat_fix):
        assert self.ch_f == feat_init.shape[1]
        confidence = feat_fix.sign() * torch.sigmoid(confidence)
        feat_result = feat_init.float().contiguous()
        list_feat = []
        # offset1, aff1,aff0 = self._get_offset_affinity(guidance)
        offset1, aff1 = self._get_offset_affinity(guidance)
        # offset1, aff1,aff_sum,aff_abs_sum = self._get_offset_affinity(guidance)
        for k in range(self.prop_time):  # 0.002
            feat_result = (1 - confidence) * self._propagate_once(feat_result, offset1[k].float(),
                                                                  aff1[k].float()) + confidence * feat_fix
            list_feat.append(feat_result)

        return feat_result, list_feat, offset1, aff1


class Dynamic_deformable_DySample_restart(nn.Module):
    # def __init__(self, args, ch_g, ch_f, k_g, k_f):
    def __init__(self, prop_time=3):
        super(Dynamic_deformable_DySample_restart, self).__init__()
        self.prop_time = prop_time
        self.ch_g = 9
        self.ch_f = 1
        self.k_g = 3
        self.k_f = 3
        pad_g = int((self.k_g - 1) / 2)
        pad_f = int((self.k_f - 1) / 2)
        self.num = self.k_f * self.k_f
        self.conv_offset_aff = nn.Conv2d(
            (self.ch_g + 1) * self.prop_time, (3 * self.ch_g + 1) * self.prop_time, kernel_size=3, stride=1,
            padding=pad_g, bias=True,
            # groups=self.prop_time
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        self.alpha = nn.Parameter(torch.ones(1))

    def _get_offset_affinity(self, guidance):
        B, _, H, W = guidance.shape

        offset_aff = self.conv_offset_aff(guidance).view(B, self.prop_time, self.num * 3 + 1, H, W)
        # # aff_0 = self.conv_aff0(guidance).view(B, self.prop_time, 1, H, W)
        section = [self.num * 2, self.num + 1]
        offset_1, aff_0 = torch.split(offset_aff, section, dim=2)
        aff_0 = torch.softmax(aff_0, dim=2)
        section = [self.num, 1]
        aff_1, aff_restart = torch.split(aff_0, section, dim=2)
        return torch.unbind(offset_1.half(), dim=1), torch.unbind(aff_1.half(), dim=1), torch.unbind(aff_restart.half(),
                                                                                                     dim=1)

    def _propagate_once(self, feat, offset, aff):
        return deform_conv2d(feat, offset, self.w, self.b, (self.stride, self.stride), (self.padding, self.padding),
                             (self.dilation, self.dilation), mask=aff)  # torch>=1.8

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, feat_init, guidance, confidence, feat_fix):
        assert self.ch_f == feat_init.shape[1]
        confidence = feat_fix.sign() * torch.sigmoid(confidence)
        feat_result = feat_init.float().contiguous()
        list_feat = []
        offset1, aff1, restart = self._get_offset_affinity(guidance)
        for k in range(self.prop_time):  # 0.002
            feat_result = (1 - confidence) * (
                    self._propagate_once(feat_result, offset1[k].float(), aff1[k].float()) + restart[
                k].float() * feat_result) + confidence * feat_fix
            list_feat.append(feat_result)
        return {'pred': feat_result,
                "list_feat": list_feat,
                "offset": offset1,
                "aff": aff1,
                "restart": restart,
                }


if __name__ == '__main__':
    # spn = DySPA()
    # ref = spn.get_refgrid(1, 352, 1216)  # BHW2N
    print(" ")