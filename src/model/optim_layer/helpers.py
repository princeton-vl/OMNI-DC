"""Implements helper functions."""

import torch
import torch.nn.functional as F

def multires_sparse_depth_downsample(batched_sparse_depth, batched_valid_sparse_mask, resolution):
    # batched_sparse_depth: B x 1 x H x W
    multires_sparse_depth_outputs = []
    multires_valid_mask_outputs = []

    batched_sparse_depth = batched_sparse_depth.clone()
    batched_sparse_depth[batched_valid_sparse_mask == 0.0] = 0.0

    for i in range(resolution):
        r = 2**i
        sparse_depth_downsampled = F.avg_pool2d(batched_sparse_depth, r)
        valid_mask_downsampled = F.avg_pool2d(batched_valid_sparse_mask, r)

        # normalize depth and binarize the mask
        sparse_depth_downsampled[valid_mask_downsampled > 0.0] /= valid_mask_downsampled[valid_mask_downsampled > 0.0]
        valid_mask_downsampled[valid_mask_downsampled > 0.0] = 1.0

        multires_sparse_depth_outputs.append(sparse_depth_downsampled)
        multires_valid_mask_outputs.append(valid_mask_downsampled)

    return multires_sparse_depth_outputs, multires_valid_mask_outputs

def batched_matrix_to_trucated_flattened(batched_matrix, resolution):
    # batched matrix has shape B x 2r x H x W
    max_patch_size = 2 ** (resolution - 1)

    batch_size, C, H, W = batched_matrix.shape
    assert C == 2 * resolution and H % max_patch_size == 0 and W % max_patch_size == 0

    truncated = []

    for i in range(resolution):
        r = 2**i
        # for each res, only the top-left corner is valid
        # additionally discard the first column and first row
        x_truncated = batched_matrix[:, i*2, 0:H//r, 1:W//r].reshape(batch_size, -1)  # B x ((H/r)*(W/r-1))
        y_truncated = batched_matrix[:, i*2+1, 1:H//r, 0:W//r].reshape(batch_size, -1)  # B x ((H/r-1)*(W/r))

        truncated.append(x_truncated)
        truncated.append(y_truncated)

    return torch.cat(truncated, dim=-1).unsqueeze(-1)  # B x num_eqns x 1

def trucated_flattened_to_batched_matrix(truncated_matrix, resolution, H, W):
    batch_size, num_eqns, _ = truncated_matrix.shape

    batched_matrix = torch.zeros(batch_size, 2*resolution, H, W, device=truncated_matrix.device)

    shift = 0
    for i in range(resolution):
        r = 2**i

        num_elements = H//r * (W//r - 1)
        x_truncated = truncated_matrix[:, shift:shift+num_elements, :].reshape(batch_size, H//r, W//r-1)
        batched_matrix[:, i*2, 0:H//r, 1:W//r] = x_truncated
        shift += num_elements

        num_elements = (H//r - 1) * W//r
        y_truncated = truncated_matrix[:, shift:shift+num_elements, :].reshape(batch_size, H//r-1, W//r)
        batched_matrix[:, i*2+1, 1:H//r, 0:W//r] = y_truncated
        shift += num_elements

    assert shift == num_eqns

    return batched_matrix

def sparse_dense_mul(s, d):
    """Sparse dense element-wise mul."""
    i = s._indices()
    v = s._values()
    # get values from relevant entries of dense matrix
    dv = d[i[0, :], i[1, :], i[2, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())

class FastFiniteDiffMatrix:
    def __init__(self, H, W, resolution, device='cuda', dtype=torch.float):
        self.H = H
        self.W = W
        self.resolution = resolution

    def bmm(self, x):
        # x has shape B x (H*W) x 1.
        # return value has shape B x n_eqn x 1
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self.H, self.W)

        gradients = []

        for i in range(self.resolution):
            r = 2**i
            x_grad_u = (x[:, :, :, r:] - x[:, :, :, :-r])
            x_grad_u = F.avg_pool2d(x_grad_u, kernel_size=r) # B x 1 x H/r x W/r-1
            gradients.append(x_grad_u.reshape(batch_size, -1))

            x_grad_v = (x[:, :, r:, :] - x[:, :, :-r, :])
            x_grad_v = F.avg_pool2d(x_grad_v, kernel_size=r)  # B x 1 x H/r-1 x W/r
            gradients.append(x_grad_v.reshape(batch_size, -1))

        return torch.cat(gradients, dim=1).unsqueeze(-1) # B x n_eqn x 1

    def bmm_transposed(self, x):
        # x has shape B x n_eqn x 1
        # return value has shape B x (H*W) x 1
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, 1, self.H, self.W, device=x.device, dtype=x.dtype)
        shift = 0

        for i in range(self.resolution):
            r = 2**i
            num_elements = self.H//r * (self.W//r - 1)
            # x_grad_u = x[:, shift:shift+num_elements].reshape(batch_size, 1, self.H//r, self.W//r - 1)
            # x_grad_u = x_grad_u.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3) / float(r**2)

            x_grad_u = x[:, shift:shift+num_elements].reshape(batch_size, 1, self.H//r, 1, self.W//r-1, 1)
            x_grad_u = x_grad_u.expand(-1, -1, -1, r, -1, r)
            x_grad_u = x_grad_u.reshape(batch_size, 1, self.H, self.W-r) / float(r**2)

            output[:, :, :, r:] += x_grad_u
            output[:, :, :, :-r] -= x_grad_u
            shift += num_elements

            num_elements = (self.H//r - 1) * self.W//r
            # x_grad_v = x[:, shift:shift+num_elements].reshape(batch_size, 1, self.H//r - 1, self.W//r)
            # x_grad_v = x_grad_v.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3) / float(r ** 2)

            x_grad_v = x[:, shift:shift+num_elements].reshape(batch_size, 1, self.H//r-1, 1, self.W//r, 1)
            x_grad_v = x_grad_v.expand(-1, -1, -1, r, -1, r)
            x_grad_v = x_grad_v.reshape(batch_size, 1, self.H-r, self.W) / float(r ** 2)

            output[:, :, r:, :] += x_grad_v
            output[:, :, :-r, :] -= x_grad_v

            shift += num_elements

        return output.reshape(batch_size, -1, 1)
    
def construct_diff_matrix_sparse(H, W, resolution, device='cuda', dtype=torch.float):
    total_px = H * W

    total_idx = torch.arange(total_px).reshape(H, W)

    indices = []
    one_minus_one = []
    shift = 0

    for i in range(resolution):
        r = 2**i

        x_ind_minus_0 = (shift + torch.arange(H//r * (W//r-1)))
        x_ind_minus_0 = x_ind_minus_0.reshape(H//r, (W//r-1)).repeat_interleave(r, dim=0).repeat_interleave(r, dim=1)
        x_ind_minus_0 = x_ind_minus_0.flatten()

        x_ind_minus_1 = total_idx[:, :-r].flatten()
        x_ind_minus = torch.stack([x_ind_minus_0, x_ind_minus_1], dim=-1)  # (H/r*(W/r-1)) x 2

        x_ind_plus_0 = x_ind_minus_0
        x_ind_plus_1 = total_idx[:, r:].flatten()
        x_ind_plus = torch.stack([x_ind_plus_0, x_ind_plus_1], dim=-1)  # (H/r*(W/r-1)) x 2

        shift += H//r * (W//r-1)

        y_ind_minus_0 = (shift + torch.arange((H//r - 1) * (W//r)))
        y_ind_minus_0 = y_ind_minus_0.reshape((H//r - 1), (W//r)).repeat_interleave(r, dim=0).repeat_interleave(r, dim=1)
        y_ind_minus_0 = y_ind_minus_0.flatten()

        y_ind_minus_1 = total_idx[:-r, :].flatten()
        y_ind_minus = torch.stack([y_ind_minus_0, y_ind_minus_1], dim=-1)  # ((H/r-1)*W/r) x 2

        y_ind_plus_0 = y_ind_minus_0
        y_ind_plus_1 = total_idx[r:, :].flatten()
        y_ind_plus = torch.stack([y_ind_plus_0, y_ind_plus_1], dim=-1)  # ((H/r-1)*W/r) x 2

        shift += (H//r - 1) * (W//r)

        indices.append(x_ind_plus)
        indices.append(y_ind_plus)
        indices.append(x_ind_minus)
        indices.append(y_ind_minus)

        num_vars = len(x_ind_plus) + len(y_ind_plus)

        one_minus_one.append((1.0 / r**2) * torch.ones(num_vars*2, device=device))
        one_minus_one[-1][num_vars:] *= -1
    
    indices = torch.cat(indices, dim=0) # num_vars x 2
    one_minus_one = torch.cat(one_minus_one)

    num_eqns = shift

    d_diff = torch.sparse_coo_tensor(indices.transpose(0, 1), one_minus_one,
                                     [num_eqns, H * W],
                                     device=device, dtype=dtype)

    return d_diff

def sparse_dense_mul_prod(s, d1, d2, d3, d4):
    """Sparse dense element-wise mul with a lookup."""
    i = s._indices()
    v = s._values()
    # get values from relevant entries of dense matrix
    dv1 = d1[i[0, :], i[1, :], 0]
    dv2 = d2[i[0, :], i[1, :], 0]
    dv3 = d3[i[0, :], 0, i[2, :]]
    dv4 = d4[i[0, :], 0, i[2, :]]
    out = v * (dv1*dv3 - dv2*dv4)
    return torch.sparse.FloatTensor(i, out, s.size())

def normal_to_log_depth_gradient(batched_K, batched_normal_map):
    # batched_K: B x 3 x 3
    # batched_normal_map: B x 3 x H x W
    device = batched_normal_map.device

    focal_x = batched_K[:, 0, 0].reshape(-1, 1, 1)
    focal_y = batched_K[:, 1, 1].reshape(-1, 1, 1)
    principal_x = batched_K[:, 0, 2].reshape(-1, 1, 1)
    principal_y = batched_K[:, 1, 2].reshape(-1, 1, 1)

    # assert (torch.abs(focal_x - focal_y) < 1e-3).all() # only supports single focal length
    # focal = focal_x
    focal = (focal_x + focal_y) / 2.0

    batch_size, _, H, W = batched_normal_map.shape
    nx, ny, nz = batched_normal_map[:, 0], batched_normal_map[:, 1], batched_normal_map[:, 2] # B x H x W each

    # nz = torch.clamp(nz, max=-1e-2)
    #
    # print('nz:', nz.min(), nz.max(), nz.mean())

    v, u = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing='ij')
    v, u = v.unsqueeze(0) + 0.5, u.unsqueeze(0) + 0.5 # 1 x H x W each

    denominator = nx * (u - principal_x) + ny * (v - principal_y) + nz * focal # B x H x W

    inv_denominator = 1.0 / denominator

    # sign_denominator = torch.sign(denominator)
    # abs_denominator = torch.abs(denominator)
    #
    # # denominator = sign_denominator * torch.clamp(abs_denominator, min=1e-1)
    # denominator = sign_denominator * torch.clamp(abs_denominator, min=1.0)
    inv_denominator = torch.clamp(inv_denominator, min=-1.0, max=1.0)

    # log_depth_gradient_x = - nx / denominator
    # log_depth_gradient_y = - ny / denominator

    log_depth_gradient_x = -nx * inv_denominator
    log_depth_gradient_y = -ny * inv_denominator

    log_depth_gradient = torch.stack([log_depth_gradient_x, log_depth_gradient_y], dim=1) # B x 2 x H x W

    # abs_log_depth_gradient = torch.abs(log_depth_gradient)
    # print('grad:', abs_log_depth_gradient.min(), abs_log_depth_gradient.max(), abs_log_depth_gradient.mean())

    return log_depth_gradient

def log_depth_gradient_to_normal(batched_K, batched_log_depth_gradients):
    # batched_K: B x 3 x 3
    # batched_log_depth_gradients: B x 2 x H x W
    device = batched_log_depth_gradients.device

    focal_x = batched_K[:, 0, 0].reshape(-1, 1, 1)
    focal_y = batched_K[:, 1, 1].reshape(-1, 1, 1)
    principal_x = batched_K[:, 0, 2].reshape(-1, 1, 1)
    principal_y = batched_K[:, 1, 2].reshape(-1, 1, 1)

    # assert (torch.abs(focal_x - focal_y) < 1e-3).all()  # only supports single focal length
    # focal = focal_x
    focal = (focal_x + focal_y) / 2.0

    batch_size, _, H, W = batched_log_depth_gradients.shape

    plogzpu = batched_log_depth_gradients[:, 0]
    plogzpv = batched_log_depth_gradients[:, 1]

    v, u = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing='ij')
    v, u = v.unsqueeze(0) + 0.5, u.unsqueeze(0) + 0.5  # 1 x H x W each

    pup = torch.stack([1. / focal * ((u - principal_x) * plogzpu + 1.), 1. / focal * (v - principal_y) * plogzpu, plogzpu], dim=1)
    pvp = torch.stack([1. / focal * (u - principal_x) * plogzpv, 1. / focal * ((v - principal_y) * plogzpv + 1.), plogzpv], dim=1)

    normal_from_depth = torch.cross(pvp, pup, dim=1) # B x 3 x H x W
    normalized_normal_from_depth = normal_from_depth / torch.linalg.norm(normal_from_depth, ord=2, dim=1, keepdim=True)

    return normalized_normal_from_depth



