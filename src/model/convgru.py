# adapted from https://github.com/princeton-vl/RAFT/blob/master/core/update.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_to_log_depth_gradient(batched_K, batched_normal_map):
    # batched_K: B x 3 x 3
    # batched_normal_map: B x 3 x H x W
    device = batched_normal_map.device

    focal_x = batched_K[:, 0, 0].reshape(-1, 1, 1)
    focal_y = batched_K[:, 1, 1].reshape(-1, 1, 1)
    principal_x = batched_K[:, 0, 2].reshape(-1, 1, 1)
    principal_y = batched_K[:, 1, 2].reshape(-1, 1, 1)

    # assert (torch.abs(focal_x - focal_y) < 1e-3).all() # only supports single focal length
    focal = (focal_x + focal_y) / 2.0

    batch_size, _, H, W = batched_normal_map.shape
    nx, ny, nz = batched_normal_map[:, 0], batched_normal_map[:, 1], batched_normal_map[:, 2] # B x H x W each

    v, u = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing='ij')
    v, u = v.unsqueeze(0) + 0.5, u.unsqueeze(0) + 0.5 # 1 x H x W each

    denominator = nx * (u - principal_x) + ny * (v - principal_y) + nz * focal # B x H x W

    inv_denominator = 1.0 / (denominator + 1e-8)
    # inv_denominator = torch.clamp(inv_denominator, min=-1.0, max=1.0)

    log_depth_gradient_x = -nx * inv_denominator
    log_depth_gradient_y = -ny * inv_denominator

    log_depth_gradient = torch.stack([log_depth_gradient_x, log_depth_gradient_y], dim=1) # B x 2 x H x W

    return log_depth_gradient

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out = self.block(x)
        out += self.skip(x)

        out = F.relu(out)
        return out

class DepthGradHead(nn.Module):
    def __init__(self, resolution, input_dim=64, hidden_dim=128):
        super(DepthGradHead, self).__init__()

        self.blocks = nn.ModuleList([Block(input_dim, hidden_dim), ])
        for _ in range(1, resolution):
            self.blocks.append(Block(hidden_dim, hidden_dim))

        self.decoders = nn.ModuleList([nn.Conv2d(hidden_dim, 2, 1, padding=0) for _ in range(resolution)])

        self.resolution = resolution

    def forward(self, x):
        out = []

        for r in range(self.resolution):
            x = self.blocks[r](x) # encode feature
            out.append(self.decoders[r](x)) # decode to depthgrad

            x = F.max_pool2d(x, 2)

        return out

class GlobalWeightsHead(nn.Module):
    def __init__(self, resolution, input_dim=64, hidden_dim=128):
        super(GlobalWeightsHead, self).__init__()

        self.block = Block(input_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, resolution)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block(x)

        # global pooling
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, (H, W)).reshape(B, C)
        x = self.decoder(x)
        x = self.softmax(x)

        return x

class DepthDirectHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(DepthDirectHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DepthDirectEncoder(nn.Module):
    def __init__(self):
        super(DepthDirectEncoder, self).__init__()
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64, 64-1, 3, padding=1)

    def forward(self, depth):
        dep = F.relu(self.convd1(depth))
        dep = F.relu(self.convd2(dep))
        
        out = F.relu(self.conv(dep))

        return torch.cat([out, depth], dim=1)

class ConfidenceHead(nn.Module):
    def __init__(self, resolution, input_dim=64, hidden_dim=32):
        super(ConfidenceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2 * resolution, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.resolution = resolution

    def forward(self, x):
        batch_size, _, H, W = x.shape
        batched_confidence = self.conv2(self.relu(self.conv1(x))) # B x 2*r x H x W
        batched_confidence = batched_confidence.reshape(batch_size, self.resolution, 2, H, W)
        batched_confidence = float(self.resolution) * F.softmax(batched_confidence, dim=1)
        batched_confidence = batched_confidence.reshape(batch_size, -1, H, W)
        return batched_confidence

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64+64):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class BasicDepthEncoder(nn.Module):
    def __init__(self):
        super(BasicDepthEncoder, self).__init__()
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 32, 3, padding=1)

        self.convg1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convg2 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv = nn.Conv2d(64, 64-(1 + 2), 3, padding=1)

    def forward(self, depth, depth_grad):
        dep = F.relu(self.convd1(depth))
        dep = F.relu(self.convd2(dep))

        gra = F.relu(self.convg1(depth_grad))
        gra = F.relu(self.convg2(gra))

        out = F.relu(self.conv(torch.cat([dep, gra], dim=1)))
        return torch.cat([out, depth, depth_grad], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, resolution=1, hidden_dim=64, input_dim=64, mask_r=8, conf_min=0.01):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicDepthEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=64+hidden_dim)
        self.depth_grad_head = DepthGradHead(resolution, input_dim=hidden_dim)
        self.resolution = resolution
        if self.args.multi_resolution_learnable_gradients_weights:
            self.grad_weights_head = GlobalWeightsHead(resolution, input_dim=hidden_dim)
        if self.args.multi_resolution_learnable_input_weights:
            self.inputs_weights_head = GlobalWeightsHead(resolution, input_dim=hidden_dim)

        self.mask_r = mask_r
        if self.mask_r > 1:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, mask_r * mask_r * 9, 1, padding=0))

    def forward(self, net, inp, depth, depth_grad, K):
        # net: hidden; inp: ctx
        # depth and depth_grad should be detached before feeding into this layer
        B, _, H, W = depth_grad.shape

        depth_features = self.encoder(depth, depth_grad[:, 0:2]) # B x 64 x H x W
        inp = torch.cat([inp, depth_features], dim=1) # B x 128 x H x W

        net = self.gru(net, inp)

        delta_depth_grad_raw = self.depth_grad_head(net)

        # align to top-left corner
        delta_depth_grad = torch.zeros_like(depth_grad)
        for i in range(self.resolution):
            r = 2 ** i
            delta_depth_grad[:, i*2:(i+1)*2, :H//r, :W//r] += delta_depth_grad_raw[i]

        if self.args.multi_resolution_learnable_gradients_weights == "learnable":
            weights_depth_grad = self.grad_weights_head(net)
            weights_depth_grad = weights_depth_grad.reshape(B, self.resolution, 1, 1).repeat(1, 1, H, W).repeat_interleave(2, dim=1)
        elif self.args.multi_resolution_learnable_gradients_weights == "uniform":
            weights_depth_grad = torch.tensor(([1.0] * self.resolution), device='cuda') / float(self.resolution)
            weights_depth_grad = weights_depth_grad.reshape(1, self.resolution, 1, 1).repeat(B, 2, H, W)
        elif self.args.multi_resolution_learnable_gradients_weights == "uniform_area":
            weights_depth_grad = torch.tensor(([1.0, 4.0, 16.0]), device='cuda')[:self.resolution]
            weights_depth_grad = weights_depth_grad / torch.sum(weights_depth_grad)
            weights_depth_grad = weights_depth_grad.reshape(1, self.resolution, 1, 1).repeat(B, 2, H, W)
        else:
            raise NotImplementedError

        if self.args.multi_resolution_learnable_input_weights:
            weights_input = self.inputs_weights_head(net)
            weights_input = weights_input.reshape(B, self.resolution, 1, 1).repeat(1, 1, H, W)
        else:
            weights_input = torch.tensor(([1.0] + [0.0] * (self.resolution-1)), device='cuda')
            weights_input = weights_input.reshape(1, self.resolution, 1, 1).repeat(B, 1, 1, 1)

        if self.mask_r > 1:
            mask = self.mask(net)
        else:
            mask = None
            
        return net, mask, delta_depth_grad, weights_depth_grad, weights_input

class DirectDepthUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, mask_r=8):
        super(DirectDepthUpdateBlock, self).__init__()
        self.encoder = DepthDirectEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=64 + hidden_dim)
        self.depth_head = DepthDirectHead(input_dim=hidden_dim)

        self.mask_r = mask_r
        if self.mask_r > 1:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, mask_r * mask_r * 9, 1, padding=0))

    def forward(self, net, inp, depth):
        # net: hidden; inp: ctx
        # depth and depth_grad should be detached before feeding into this layer
        depth_features = self.encoder(depth)  # B x 64 x H x W
        inp = torch.cat([inp, depth_features], dim=1)  # B x 128 x H x W

        net = self.gru(net, inp)

        delta_depth = self.depth_head(net)

        if self.mask_r > 1:
            mask = self.mask(net)
        else:
            mask = None

        return net, mask, delta_depth