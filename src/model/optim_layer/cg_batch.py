"""Conjugate Gradient from https://github.com/sbarratt/torch_cg"""

import torch
import time
import numpy as np

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-12, atol=0.,
             maxiter=None, early_stopping_improvement_tol=1e-2, early_stopping_consecutive_steps=10, verbose=False):
  """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

  This function solves a batch of matrix linear systems of the form

      A_i X_i = B_i,  i=1,...,K,

  where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
  and X_i is the n x m matrix representing the solution for the ith system.

  Args:
      A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
      B: A K x n x m matrix representing the right hand sides.
      M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
          matrices M and a K x n x m matrix. (default=identity matrix)
      X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
      rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
      atol: (optional) Absolute tolerance for norm of residual. (default=0)
      maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
      verbose: (optional) Whether or not to print status messages. (default=False)
  """
  K, n, m = B.shape

  if M_bmm is None:
    def M_bmm(x): return x
  if X0 is None:
    X0 = M_bmm(B)
  if maxiter is None:
    maxiter = 5 * n

  assert B.shape == (K, n, m)
  assert X0.shape == (K, n, m)
  assert rtol > 0 or atol > 0
  assert isinstance(maxiter, int)

  X_k = X0
  A_bmmXk = A_bmm(X_k)
  R_k = B - A_bmmXk
  Z_k = M_bmm(R_k)

  P_k = torch.zeros_like(Z_k)

  P_k1 = P_k
  R_k1 = R_k
  R_k2 = R_k
  X_k1 = X0
  Z_k1 = Z_k
  Z_k2 = Z_k


  B_norm = torch.norm(B, dim=1)
  stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

  residual_norm_last = torch.norm(R_k, dim=1)
  max_relative_improvement_list = np.zeros(maxiter)

  if verbose:
    print("%03s | %010s %06s" % ("it", "dist", "it/s"))

  optimal = False
  start = time.perf_counter()
  for k in range(1, maxiter + 1):
    start_iter = time.perf_counter()
    Z_k = M_bmm(R_k)

    if k == 1:
      P_k = Z_k
      R_k1 = R_k
      X_k1 = X_k
      Z_k1 = Z_k
    else:
      R_k2 = R_k1
      Z_k2 = Z_k1
      P_k1 = P_k
      R_k1 = R_k
      Z_k1 = Z_k
      X_k1 = X_k
      denominator = (R_k2 * Z_k2).sum(1)
      denominator[denominator == 0] = 1e-8
      beta = (R_k1 * Z_k1).sum(1) / denominator
      P_k = Z_k1 + beta.unsqueeze(1) * P_k1

    A_bmmPk = A_bmm(P_k)
    denominator = (P_k * A_bmmPk).sum(1)
    denominator[denominator == 0] = 1e-8
    alpha = (R_k1 * Z_k1).sum(1) / denominator
    X_k = X_k1 + alpha.unsqueeze(1) * P_k
    R_k = R_k1 - alpha.unsqueeze(1) * A_bmmPk
    end_iter = time.perf_counter()

    A_bmmXk = A_bmmXk + alpha.unsqueeze(1) * A_bmmPk

    residual_norm = torch.norm(A_bmmXk - B, dim=1) # B x m

    relative_improvement = (residual_norm_last - residual_norm) / residual_norm_last
    max_relative_improvement_list[k-1] = relative_improvement.max().item()
    residual_norm_last = residual_norm

    # stop if no improvement for some iters
    if k > early_stopping_consecutive_steps and \
            np.all(max_relative_improvement_list[k-early_stopping_consecutive_steps:k] < early_stopping_improvement_tol):
      if verbose:
        print('stop because no improvement')
      optimal = True
      break

    if verbose and k % 10 == 1:
      print(residual_norm, max_relative_improvement_list[k - 1])

    if verbose and k % 10 == 1:
      print("%03d | %8.4e %4.2f" %
            (k, torch.max(residual_norm-stopping_matrix),
             1. / (end_iter - start_iter)))

    if (residual_norm <= stopping_matrix).all():
      if verbose:
        print('stop because stopping_matrix achieved')
      optimal = True
      break

  end = time.perf_counter()

  if verbose:
    if not optimal:
      print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
            (k, (end - start) * 1000))
    else:
      print("Terminated in %d steps (optimal). Took %.3f ms." %
            (k, (end - start) * 1000))

  info = {
      "niter": k,
      "optimal": optimal
  }

  return X_k, info
