"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    BaseSummary implementation

    If you want to implement a new summary interface,
    it should inherit from the BaseSummary class.
"""


from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement


def get(args):
    summary_name = args.model_name + 'Summary'
    module_name = 'summary.' + summary_name.lower()
    module = import_module(module_name)

    return getattr(module, summary_name)


class BaseSummary(SummaryWriter):
    def __init__(self, log_dir, mode, args):
        super(BaseSummary, self).__init__(log_dir=log_dir + '/' + mode)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.f_loss = '{}/loss_{}.txt'.format(log_dir, mode)
        self.f_metric = '{}/metric_{}.txt'.format(log_dir, mode)

        f_tmp = open(self.f_loss, 'w')
        f_tmp.close()
        f_tmp = open(self.f_metric, 'w')
        f_tmp.close()

    def add(self, loss=None, metric=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            self.loss.append(loss.data.cpu().numpy())
        if metric is not None:
            self.metric.append(metric.data.cpu().numpy())

    def update(self, global_step, sample, output):
        self.loss = np.concatenate(self.loss, axis=0)
        self.metric = np.concatenate(self.metric, axis=0)

        self.loss = np.mean(self.loss, axis=0)
        self.metric = np.mean(self.metric, axis=0)

        # Do update

        self.reset()

    def reset(self):
        self.loss = []
        self.metric = []

    def make_dir(self, epoch, idx):
        pass

    def save(self, epoch, idx, sample, output):
        pass

def save_ply(plyfilename, vertexs, vertex_colors, vertex_normals=None):
    # vert pos has shape N x 3
    # vert_colors has shape N x 3
    # vert_normals has shape N x 3
    # save
    if torch.is_tensor(vertex_normals):
        vertexs = vertexs.cpu().numpy()
    if torch.is_tensor(vertex_colors):
        # vertex_colors = ((vertex_colors.cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)[...,[2,1,0]]
        vertex_colors = (vertex_colors.cpu().numpy() * 255.0).astype(np.uint8)


    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    if vertex_normals is not None:
        if torch.is_tensor(vertex_normals):
            vertex_normals = vertex_normals.cpu().numpy()
        vertex_normals = np.array([tuple(v) for v in vertex_normals], dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_normals.dtype.descr + vertex_colors.dtype.descr)

    else:
        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)

    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    if vertex_normals is not None:
        for prop in vertex_normals.dtype.names:
            vertex_all[prop] = vertex_normals[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    # print("saving the final model to", plyfilename)

class PtsUnprojector(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(PtsUnprojector, self).__init__()
        self.device=device

    def forward(self, depth, intrinsics, pose=None, mask=None, return_coord=False):
        # take depth and convert into world pts
        # depth: B x 1 x H x W
        # pose: B x 4 x 4
        # intrinsics: B x 3 x 3
        # mask: B x 1 x H x W
        # return coord: return the corresponding [b,y,x] coord for each point, so that we can index into the vertex feature

        B, _, H, W = depth.shape

        # assert(h==self.H)
        # assert(w==self.W)
        xs = torch.linspace(0, W - 1, W).float() + 0.5
        ys = torch.linspace(0, H - 1, H).float() + 0.5

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1).to(self.device)  # 1 x 3 x N

        depth = depth.reshape(B, 1, -1)

        projected_coors = xyzs * depth # B x 3 x N

        xyz_source = torch.inverse(intrinsics).bmm(projected_coors)  # B x 3 x N, xyz in cam1 space
        xyz_source = torch.cat((xyz_source, torch.ones_like(xyz_source[:, 0:1])), dim=1) # B x 4 x N

        if pose is not None:
            # pose is cam_T_world
            xyz_world = torch.inverse(pose).bmm(xyz_source) # B x 4 x N
        else:
            xyz_world = xyz_source

        xyz_world = xyz_world[:, 0:3]  # B x 3 x N, discard homogeneous dimension
        xyz_world = xyz_world.permute(0, 2, 1).reshape(-1, 3)  # B*N x 3

        if return_coord:
            bs = torch.linspace(0, B-1, B).float()
            bs = bs.view(B, 1, 1).repeat(1, H, W)
            xs = xs.view(1, H, W).repeat(B, 1, 1)
            ys = ys.view(1, H, W).repeat(B, 1, 1)

            buvs = torch.stack((bs, ys, xs), dim=-1).view(-1, 3).to(self.device) # B*N x 3

        # if mask not none, we prune the xyzs by only selecting the valid ones
        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask>0.5)[0]
            xyz_world = xyz_world[nonzeros, :] # n_valid x 3
            if return_coord:
                buvs = buvs[nonzeros, :]

        if return_coord:
            return xyz_world, buvs.to(torch.long)
        else:
            return xyz_world

    def apply_mask(self, feat, mask=None):
        # feat: B x C x H x W
        # mask: B x 1 x H x W
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, -1)
        feat = feat.permute(0, 2, 1).reshape(-1, C)  # B*N x C

        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask > 0.5)[0]
            feat = feat[nonzeros, :]  # n_valid x C

        return feat



