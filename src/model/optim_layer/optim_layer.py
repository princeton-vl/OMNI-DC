import numpy as np

import torch
import torch.nn.functional as F

# import sys
# sys.path.append('.')

from .cg_batch import cg_batch
from .helpers import (
  FastFiniteDiffMatrix,
  construct_diff_matrix_sparse,
  sparse_dense_mul,
  sparse_dense_mul_prod,
  batched_matrix_to_trucated_flattened,
  trucated_flattened_to_batched_matrix,
  multires_sparse_depth_downsample
)

class DepthGradOptimLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batched_image_gradients, batched_sparse_depth,
                batched_valid_sparse_mask, batched_confidence, batched_input_confidence, resolution=1, x_init=None,
                b_init=None, lamda=0.1, rtol=1e-5, max_iter=1000):
        """Performs Surface Snapping.
          batched_image_gradients: B x (2*r) x H x W, odd feature channel stores x gradients
          for finer res, the values are stored in the upper-left corner of the matrix
          batched_sparse_depth: B x 1 x H x W
          batched_valid_sparse_mask: B x 1 x H x W, each value is either 0 or 1
          batched_confidence: B x (2*r) x H x W, corresponding to batched_image_gradients.
          batched_input_confidence: B x r x H x W, the confidence on batched_sparse_depth.
          b_init is a dummy variable that we use to initialize the gradient of b in backward path.
          it has shape B x 1 x H x W. you can pass an all-zero tensor if you want to use the functionality
        """
        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        if not torch.is_tensor(lamda):
            lamda = torch.ones(batch_size, resolution, 1, device=device, dtype=dtype, requires_grad=False) * lamda
        else:
            assert lamda.shape == (batch_size, resolution, 1) or lamda.shape == (1, resolution, 1)

        ctx.max_iter = max_iter
        ctx.rtol = rtol
        ctx.resolution = resolution

        batched_confidence = torch.clamp(batched_confidence, min=1e-4)

        A = FastFiniteDiffMatrix(H, W, resolution, device=device, dtype=dtype)

        # compute A^T @ b
        # split this into grad terms (bottom) and sparse depth terms (top)
        rhs = batched_matrix_to_trucated_flattened(batched_image_gradients, resolution)  # B x num_eqns x 1
        confidence = batched_matrix_to_trucated_flattened(batched_confidence, resolution) # B x num_eqns x 1

        b_bottom = A.bmm_transposed(confidence * rhs)

        # Compute b = A^T @ rhs
        # TODO: value constraints for other resolutions
        multires_batched_sparse_depth, multires_batched_valid_sparse_mask = \
            multires_sparse_depth_downsample(batched_sparse_depth, batched_valid_sparse_mask, resolution)

        b_top = torch.zeros_like(b_bottom)
        # for i in range(resolution):
        for i in range(1):
            r = 2**i

            # all have shape B x 1 x H/r x W/r
            batched_input_confidence_this_res = batched_input_confidence[:, i:i+1, :H//r, :W//r]
            batched_valid_sparse_mask_this_res = multires_batched_valid_sparse_mask[i]
            batched_sparse_depth_this_res = multires_batched_sparse_depth[i]

            # A^T @ b for the sparse depth terms
            b_top = b_top + (1.0 / r**2) * lamda[:, i:i+1] * \
                     (batched_input_confidence_this_res *
                      batched_valid_sparse_mask_this_res *
                      batched_sparse_depth_this_res).repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)\
                         .reshape(batch_size, -1, 1)  # B x (H*W) x 1

        b = b_top + b_bottom

        def batched_RTRp(p):
            # A^T @ A for the gradient terms
            ret = A.bmm_transposed(confidence * A.bmm(p))

            # A^T @ A for the sparse depth terms
            # for i in range(resolution):
            for i in range(1):
                r = 2**i
                batched_input_confidence_this_res = batched_input_confidence[:, i:i + 1, :H//r, :W//r]
                batched_valid_sparse_mask_this_res = multires_batched_valid_sparse_mask[i]

                if i == 0:
                    Ap = p.reshape(batch_size, 1, H, W) # B x 1 x H x W
                else:
                    Ap = F.avg_pool2d(p.reshape(batch_size, 1, H, W), kernel_size=r) # B x 1 x H x W

                ATAp = batched_input_confidence_this_res * batched_valid_sparse_mask_this_res * Ap

                if i == 0:
                    pass
                else:
                    ATAp = ATAp.view(batch_size, 1, H//r, 1, W//r, 1).expand(-1, -1, -1, r, -1, r)
                    ATAp = ATAp.reshape(batch_size, 1, H, W)
                    # ATAp = ATAp.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)

                ATAp = (1.0 / r**2) * lamda[:, i:i+1] * ATAp.reshape(batch_size, -1, 1)
                ret += ATAp

            return ret

        if x_init is not None:
            x_init = x_init.reshape(batch_size, -1, 1)

        x, info = cg_batch(batched_RTRp, b, X0=x_init, rtol=rtol, maxiter=max_iter, verbose=False)
        # print('forward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W, H))

        ctx.save_for_backward(b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth,
                              batched_valid_sparse_mask, batched_confidence, batched_input_confidence)

        return x.reshape(batched_sparse_depth.shape), b_init

    @staticmethod
    def backward(ctx, gradx, grad_b_init):
        b, rhs, x, lamda, batched_image_gradients, batched_sparse_depth, \
            batched_valid_sparse_mask, batched_confidence, batched_input_confidence = ctx.saved_tensors
        batch_size, _, H, W = batched_image_gradients.shape
        device = batched_image_gradients.device
        dtype = batched_image_gradients.dtype

        gradx = gradx.reshape(x.shape)

        resolution = ctx.resolution

        A = FastFiniteDiffMatrix(H, W, resolution, device=device, dtype=dtype)
        A_sparse = construct_diff_matrix_sparse(H, W, resolution, device=device, dtype=dtype).unsqueeze(0)  # B x num_eqns x (H*W)

        # valid_sparse_mask = (batched_sparse_depth > -1e9).float()  # [0, 1]
        confidence = batched_matrix_to_trucated_flattened(batched_confidence, resolution)

        multires_batched_sparse_depth, multires_batched_valid_sparse_mask = \
            multires_sparse_depth_downsample(batched_sparse_depth, batched_valid_sparse_mask, resolution)

        def batched_RTRp(p):
            # A^T @ A for the gradient terms
            ret = A.bmm_transposed(confidence * A.bmm(p))

            # A^T @ A for the sparse depth terms
            # for i in range(resolution):
            for i in range(1):
                r = 2 ** i
                batched_input_confidence_this_res = batched_input_confidence[:, i:i + 1, :H // r, :W // r]
                batched_valid_sparse_mask_this_res = multires_batched_valid_sparse_mask[i]

                if i == 0:
                    Ap = p.reshape(batch_size, 1, H, W)  # B x 1 x H x W
                else:
                    Ap = F.avg_pool2d(p.reshape(batch_size, 1, H, W), kernel_size=r)  # B x 1 x H x W

                ATAp =  batched_input_confidence_this_res * batched_valid_sparse_mask_this_res * Ap

                if i == 0:
                    pass
                else:
                    ATAp = ATAp.view(batch_size, 1, H//r, 1, W//r, 1).expand(-1, -1, -1, r, -1, r)
                    ATAp = ATAp.reshape(batch_size, 1, H, W)
                    # ATAp = ATAp.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)

                ATAp = (1.0 / r ** 2) * lamda[:, i:i + 1] * ATAp.reshape(batch_size, -1, 1)

                ret += ATAp

            return ret

        # gradb has shape B x (H*W) x 1
        if grad_b_init is not None:
            grad_b_init = grad_b_init.reshape(batch_size, -1, 1)

        gradb, info = cg_batch(batched_RTRp, gradx, X0=grad_b_init, rtol=ctx.rtol, maxiter=ctx.max_iter,
                                     verbose=False)
        # print('backward:', 'stopped in %d steps' % info['niter'], 'resolution %dx%d' % (W, H))
        
        if torch.isnan(gradb).any():
            return torch.zeros_like(batched_image_gradients), None, None, torch.zeros_like(batched_image_gradients), \
                   None, None, None, None, None, None, None

        grad_batched_input_confidence = torch.zeros_like(batched_input_confidence)
        # Compute dL/d(batched_input_confidence)
        if lamda.requires_grad:
            grad_lamda = torch.zeros_like(lamda) # B x r x 1
        else:
            grad_lamda = None

        # for i in range(resolution):
        for i in range(1):
            r = 2**i
            # dL/dinput_conf through b
            # gradb has shape B x (H*W) x 1
            batched_valid_sparse_mask_this_res = multires_batched_valid_sparse_mask[i]
            batched_sparse_depth_this_res = multires_batched_sparse_depth[i]
            batched_input_confidence_this_res = batched_input_confidence[:, i:i + 1, :H//r, :W//r]

            term1 = (F.avg_pool2d((gradb).reshape(batch_size, 1, H, W), kernel_size=r)
                    * batched_valid_sparse_mask_this_res
                    * batched_sparse_depth_this_res) # B x 1 x H/r x W/r

            term1_conf = term1 * lamda[:, i].reshape(batch_size, 1, 1, 1)
            term1_lambda = (term1 * batched_input_confidence_this_res).sum(dim=(1,2,3)).reshape(batch_size, 1, 1)

            # dL/dinput_conf through A
            # x has shape B x (H*W) x 1
            # cross_term = F.avg_pool2d((gradb).reshape(batch_size, 1, H, W), kernel_size=r)
            # cross_term = cross_term.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)
            # cross_term = cross_term * x.reshape(batch_size, 1, H, W)
            # cross_term = F.avg_pool2d((cross_term).reshape(batch_size, 1, H, W), kernel_size=r)

            term2 = - (F.avg_pool2d(gradb.reshape(batch_size, 1, H, W), kernel_size=r)
                    * F.avg_pool2d(x.reshape(batch_size, 1, H, W), kernel_size=r)
                    * batched_valid_sparse_mask_this_res)
            term2_conf = term2 * lamda[:, i].reshape(batch_size, 1, 1, 1)
            term2_lambda = (term2 * batched_input_confidence_this_res).sum(dim=(1,2,3)).reshape(batch_size, 1, 1)

            grad_batched_input_confidence[:, i:i+1, :H//r, :W//r] = term1_conf + term2_conf

            if lamda.requires_grad:
                grad_lamda[:, i:i+1] = term1_lambda + term2_lambda

        grad_confrhs = A.bmm(gradb)
        # compute dL/d(batched_image_gradients)
        # A has shape B x num_eqns x (H*W)
        grad_rhs = confidence * grad_confrhs # B x num_eqns x 1

        grad_batched_image_gradients = trucated_flattened_to_batched_matrix(grad_rhs, resolution, H, W)

        # compute dL/d(batched_confidence)
        # dL/dconf through b
        term1 = (grad_confrhs * rhs) # B x num_eqns x 1

        # dL/dconf through A
        tmp1 = torch.sqrt(confidence) * A.bmm(-gradb)
        tmp2 = torch.sqrt(confidence) * A.bmm(x)
        Nx = sparse_dense_mul_prod(A_sparse, tmp1, tmp2, x.transpose(1, 2), gradb.transpose(1, 2))
        term2 = 0.5 / torch.sqrt(confidence) * torch.sparse.sum(Nx, -1).unsqueeze(-1).to_dense()

        grad_conf = (term1 + term2)

        grad_batched_confidence = trucated_flattened_to_batched_matrix(grad_conf, resolution, H, W)
        
        grad_b_init_to_return = gradb.reshape(batched_sparse_depth.shape) if grad_b_init is not None else None

        # debug only
        # grad_batched_confidence[:, 1:] = 0.0
        # grad_batched_input_confidence[:, 1:] = 0.0

        return grad_batched_image_gradients, None, None, grad_batched_confidence, grad_batched_input_confidence, \
                None, None, grad_b_init_to_return, \
                grad_lamda, None, None