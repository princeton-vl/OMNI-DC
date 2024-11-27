import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_cbam import BasicBlock
from pvt import PVT
from pathlib import Path


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class Backbone(nn.Module):
    def __init__(self, args, mode='rgbd'):
        super(Backbone, self).__init__()
        self.args = args
        self.mode = mode
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # Encoder
        if mode == 'rgbd':
            self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        elif mode == 'rgb':
            self.conv1 = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        elif mode == 'd':
            self.conv1 = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        else:
            raise TypeError(mode)

        if args.backbone == "cformer":
            self.former = PVT(in_chans=64, patch_size=2, pretrained=Path(__file__).parent / '../pretrained/pvt.pth',
                              backbone_pattern_condition_format=args.backbone_pattern_condition_format, num_pattern_types=args.num_pattern_types)
            channels = [64, 128, 64, 128, 320, 512]
        else:
            raise NotImplementedError

        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(channels[5], 256, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(256+channels[4], 128, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(128, 128, stride=1, downsample=None, ratio=8),

        )
        # 1/4
        self.dec4 = nn.Sequential(
            convt_bn_relu(128 + channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        self.output_resolution = args.backbone_output_downsample_rate

        if self.output_resolution <= 2 or self.args.prop_time > 0:
            # 1/2
            self.dec3 = nn.Sequential(
                convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
                              padding=1, output_padding=1),
                BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
            )

        if self.output_resolution <= 1 or self.args.prop_time > 0:
            # 1/1
            self.dec2 = nn.Sequential(
                convt_bn_relu(64 + channels[1], 64, kernel=3, stride=2,
                              padding=1, output_padding=1),
                BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
            )

        if self.output_resolution == 1:
            decoder_input_channel = channels[0]
            additional_decoder_channels = 64
        elif self.output_resolution == 2:
            decoder_input_channel = channels[1]
            additional_decoder_channels = 0
        elif self.output_resolution == 4:
            decoder_input_channel = channels[2]
            additional_decoder_channels = 0
        else:
            raise NotImplementedError

        # Guidance Branch
        # 1/1
        if self.args.prop_time > 0:
            self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                        padding=1)

            if self.args.spn_type == "dyspn":
                self.gd_dec0 = conv_bn_relu(64+64, 5*self.args.prop_time, kernel=3, stride=1,
                                            padding=1, bn=False, relu=False)
            elif self.args.spn_type == "nlspn":
                self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                        padding=1, bn=False, relu=False)
            else:
                raise NotImplementedError

        if self.args.prop_time > 0 and self.args.conf_prop:
            self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)

            if self.args.spn_type == "dyspn":
                # sigmoid will be applied later in the module
                self.cf_dec0 = nn.Conv2d(32+64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            elif self.args.spn_type == "nlspn":
                self.cf_dec0 = nn.Sequential(
                    nn.Conv2d(32+64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.Sigmoid()
                )
            else:
                raise NotImplementedError

        if self.args.pred_depth:
            # Depth Branch (Not used in OGNI-DC. Keep for compatibility)
            # 1/1
            self.dep_dec1 = conv_bn_relu(64+decoder_input_channel, 64, kernel=3, stride=1,
                                         padding=1)
            self.dep_dec0 = conv_bn_relu(64+additional_decoder_channels, 1, kernel=3, stride=1,
                                         padding=1, bn=False, relu=True)

        if self.args.pred_context_feature:
            out_dim = self.args.gru_hidden_dim + self.args.gru_hidden_dim
            self.ctx_dec1 = conv_bn_relu(64+decoder_input_channel, 64, kernel=3, stride=1, padding=1)
            self.ctx_dec0 = conv_bn_relu(64, out_dim, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.pred_confidence_input:
            self.cfi_dec1 = conv_bn_relu(64+decoder_input_channel, 32, kernel=3, stride=1, padding=1)
            self.cfi_dec0 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
            )

        if "Laplace" in self.args.loss:
            self.cfo_dec1 = conv_bn_relu(64 + 64, 32, kernel=3, stride=1, padding=1)
            self.cfo_dec0 = nn.Conv2d(32 + 64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb=None, depth=None, depth_pattern=None):
        # Encoding
        if self.mode == 'rgbd':
            fe1_rgb = self.conv1_rgb(rgb)
            fe1_dep = self.conv1_dep(depth)
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
            fe1 = self.conv1(fe1)
        elif self.mode == 'rgb':
            fe1 = self.conv1(rgb)
        elif self.mode == 'd':
            fe1 = self.conv1(depth)
        else:
            raise TypeError(self.mode)

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1, depth_pattern)
        # print(fe2.shape, fe3.shape, fe4.shape, fe5.shape, fe6.shape, fe7.shape)
        # assert False
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        if self.output_resolution <= 2 or self.args.prop_time > 0:
            fd3 = self.dec3(self._concat(fd4, fe4))
        if self.output_resolution <= 1 or self.args.prop_time > 0:
            fd2 = self.dec2(self._concat(fd3, fe3))

        if self.output_resolution == 1:
            decoder_feature = self._concat(fd2, fe2)
        elif self.output_resolution == 2:
            decoder_feature = self._concat(fd3, fe3)
        elif self.output_resolution == 4:
            decoder_feature = self._concat(fd4, fe4)
        else:
            raise NotImplementedError

        # Guidance Decoding
        # always works at full resolution
        if self.args.prop_time > 0:
            gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
            guide = self.gd_dec0(self._concat(gd_fd1, fe1))
        else:
            guide = None

        # Conf Decoding
        # always works at full resolution
        if self.args.prop_time > 0 and self.args.conf_prop:
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence_spn = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence_spn = None

        if self.args.pred_depth:
            # Init Depth Decoding
            dep_fd1 = self.dep_dec1(decoder_feature)
            if self.output_resolution == 1:
                init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))
            else:
                init_depth = self.dep_dec0(dep_fd1)

        else:
            init_depth = None

        if self.args.pred_context_feature:
            ctx_fd1 = self.ctx_dec1(decoder_feature)
            context = self.ctx_dec0(ctx_fd1)
        else:
            context = None

        if self.args.pred_confidence_input:
            cfi_fd1 = self.cfi_dec1(decoder_feature)
            confidence_input = self.cfi_dec0(cfi_fd1)
        else:
            confidence_input = None

        if "Laplace" in self.args.loss:
            cfo_fd1 = self.cfo_dec1(self._concat(fd2, fe2))
            confidence_output = self.cfo_dec0(self._concat(cfo_fd1, fe1))
        else:
            confidence_output = None

        return init_depth, guide, confidence_spn, context, confidence_input, confidence_output

