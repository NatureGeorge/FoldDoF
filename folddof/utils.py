# Copyright 2022 Zefeng Zhu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Created Date: 2022-02-15 08:20:53 pm
# @Filename: frame.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-05-04 06:49:10 pm
import torch
import roma
import math
import numpy as np
from itertools import accumulate
import sys
from roma.internal import svd


def get_dist6d(coords: torch.Tensor, dmax: float, dist_fill_value: float = 0.0, dtype=torch.float):
    import scipy.spatial
    nres = coords.shape[0]
    kd = scipy.spatial.cKDTree(coords)
    indices = kd.query_ball_tree(kd, dmax)
    idx = torch.tensor([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    mask_for_sym = idx0 > idx1
    idx0_sym = idx0[mask_for_sym]
    idx1_sym = idx1[mask_for_sym]
    dist6d = torch.full((nres, nres), dist_fill_value, dtype=dtype, device=coords.device)
    dist6d[idx1_sym, idx0_sym] = dist6d[idx0_sym, idx1_sym] = torch.linalg.vector_norm(coords[idx1_sym]-coords[idx0_sym], dim=1)
    return idx, dist6d


def dihedral(point0: torch.Tensor, point1: torch.Tensor, point2: torch.Tensor, point3: torch.Tensor):
    b = point2 - point1
    u = torch.linalg.cross(b, point1 - point0)
    w = torch.linalg.cross(b, point2 - point3)
    if len(b.shape) > 1:
        return torch.atan2(
            torch.einsum('...km,...km->...k', torch.linalg.cross(u, w), b),
            torch.mul(torch.einsum('...km,...km->...k', u, w), b.norm(dim=-1)))
    else:
        return torch.atan2(torch.linalg.cross(u, w).dot(b), u.dot(w) * torch.linalg.norm(b))


def planar_angle(point0: torch.Tensor, point1: torch.Tensor, point2: torch.Tensor):
    b1 = point0 - point1
    b2 = point2 - point1
    return torch.atan2(
        torch.linalg.cross(b1, b2).norm(dim=-1),
        torch.einsum('...km,...km->...k', b1, b2))


def get_internal_coordinates(backbone_coords):
        '''
        shape: ...x4xLx3
        '''
        c_n = (backbone_coords[..., 0, 1:, :] - backbone_coords[..., 2, :-1, :]).norm(dim=-1)
        ca_c = (backbone_coords[..., 2, :, :] - backbone_coords[..., 1, :, :]).norm(dim=-1)
        n_ca = (backbone_coords[..., 1, :, :] - backbone_coords[..., 0, :, :]).norm(dim=-1)

        phi = dihedral(
            backbone_coords[..., 2, :-1, :],
            backbone_coords[..., 0, 1:, :],
            backbone_coords[..., 1, 1:, :],
            backbone_coords[..., 2, 1:, :])
        psi = dihedral(
            backbone_coords[..., 0, :-1, :],
            backbone_coords[..., 1, :-1, :],
            backbone_coords[..., 2, :-1, :],
            backbone_coords[..., 0, 1:, :],
        )
        omega = dihedral(
            backbone_coords[..., 1, :-1, :],
            backbone_coords[..., 2, :-1, :],
            backbone_coords[..., 0, 1:, :],
            backbone_coords[..., 1, 1:, :],
        )

        n_ca_c_angle = planar_angle(backbone_coords[..., 0, :, :], backbone_coords[..., 1, :, :], backbone_coords[..., 2, :, :])
        c_n_ca_anlge = planar_angle(backbone_coords[..., 2, :-1, :], backbone_coords[..., 0, 1:, :], backbone_coords[..., 1, 1:, :])
        ca_c_n_angle = planar_angle(backbone_coords[..., 1, :-1, :], backbone_coords[..., 2, :-1, :], backbone_coords[..., 0, 1:, :])
        return c_n, ca_c, n_ca, phi, psi, omega, n_ca_c_angle, c_n_ca_anlge, ca_c_n_angle


def quat_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Returns the rotated points by quaternions.
    
    Args:
        quat (torch.Tensor): shape(...x4, XYZW convention) batch of quaternions.
        point (torch.Tensor): shape(...x3) batch of points.
    Returns:
        rpoint (torch.Tensor): shape(...x3 tensor) batch of points.
    TODO: allow to rotate batch points by a quaternion rotation (related to `roma.quat_product`); via `expand_dim: Optional[int] = None`
    """
    #if point.shape[0] == 0:
    #    return point.reshape(0, 3)
    assert point.shape[-1] == 3, "Expecting a ...x3 batch of points"
    assert quaternion.shape[:-1] == point.shape[:-1], (quaternion.shape, point.shape)
    rpoint = roma.quat_product(
        roma.quat_product(quaternion, torch.cat((point, point.new_zeros(point.shape[:-1] + (1,))), -1)),
        roma.quat_conjugation(quaternion),
    )
    return rpoint[..., :-1]


def quat_cumprod_sequential_since_py38(quaternion: torch.Tensor, add_head: bool = False, normalize: bool = False) -> torch.Tensor:
    '''
    Returns the cumulative product of a sequence of quaternions.

    Args:
        quaternion (sequence of ...x4 tensors, XYZW convention): sequence of batches of quaternions.
        add_head (bool): it True, add [0., 0., 0., 1.] as the head of the returned quaternion.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    '''
    prod_res = list(accumulate(quaternion, lambda x, y: torch.nn.functional.normalize(roma.quat_product(x, y), dim=-1) if normalize else roma.quat_product, initial=torch.tensor([0., 0., 0., 1.], dtype=quaternion.dtype, device=quaternion.device) if add_head else None))
    return torch.stack(prod_res, dim=0)


def quat_cumprod_sequential_before_py38(quaternion: torch.Tensor, add_head: bool = False, normalize: bool = False) -> torch.Tensor:
    '''
    Returns the cumulative product of a sequence of quaternions.

    Args:
        quaternion (sequence of ...x4 tensors, XYZW convention): sequence of batches of quaternions.
        add_head (bool): it True, add [0., 0., 0., 1.] as the head of the returned quaternion.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    '''
    prod_res = list(accumulate(quaternion, lambda x, y: torch.nn.functional.normalize(roma.quat_product(x, y), dim=-1) if normalize else roma.quat_product))
    if add_head:
        prod_res.insert(0, torch.tensor([0., 0., 0., 1.], dtype=quaternion.dtype, device=quaternion.device))
    return torch.stack(prod_res, dim=0) 


quat_cumprod_sequential = quat_cumprod_sequential_before_py38 if sys.version_info.minor < 8 else quat_cumprod_sequential_since_py38


def quat_cumprod_(quaternion: torch.Tensor, add_head: bool = True, normalize: bool = True) -> torch.Tensor:
    '''
    Returns the cumulative product of a sequence of quaternions. Parallelized via binary merge.

    Args:
        quaternion (L(xB)x4 tensors, XYZW convention): sequence of batches of quaternions.
        add_head (bool): it True, add [0., 0., 0., 1.] as the head of the returned quaternion.
        normalize (bool): it True, normalize the returned quaternion.
    Returns:
        sequence of batches of quaternions (L(xB)x4 tensor, XYZW convention).
    '''
    reconstruct_quaternion = quaternion.clone()
    first_part = np.floor(np.log2(reconstruct_quaternion.shape[0])).astype(int)
    idx = torch.arange(1, reconstruct_quaternion.shape[0], 2)
    reconstruct_quaternion[idx] = roma.quat_product(reconstruct_quaternion[idx-1], reconstruct_quaternion[idx])
    complete_progress = 2**first_part
    for blevel in range(1, first_part):
        idx = torch.arange(2**blevel, reconstruct_quaternion.shape[0], 2**(blevel+1))
        delta = reconstruct_quaternion[idx-1]
        reconstruct_quaternion[idx] = roma.quat_product(reconstruct_quaternion[idx-1], reconstruct_quaternion[idx])
        idx_cat = torch.cat([idx+step for step in range(1, 2**blevel)])
        idx_cat_mask = idx_cat < reconstruct_quaternion.shape[0]
        use_idx_cat = idx_cat[idx_cat_mask]
        if len(quaternion.shape) == 2:
            delta_repeat = delta.repeat(2**blevel-1, 1)
        else:
            delta_repeat = delta.repeat(2**blevel-1, 1, 1)
        reconstruct_quaternion[use_idx_cat] = roma.quat_product(delta_repeat[idx_cat_mask], reconstruct_quaternion[use_idx_cat])  # TODO: check whether `repeat` would affect backprop
    counter = 0
    while complete_progress < reconstruct_quaternion.shape[0]:
        remain_part = np.floor(np.log2(reconstruct_quaternion.shape[0] - complete_progress)).astype(int)
        idx = complete_progress
        delta = reconstruct_quaternion[[idx-1-counter]]
        reconstruct_quaternion[[idx]] = roma.quat_product(reconstruct_quaternion[[idx-1-counter]], reconstruct_quaternion[[idx]])
        idx_range = torch.arange(idx+1, idx+2**remain_part)
        if len(quaternion.shape) == 2:
            delta_expand = delta.expand(2**remain_part-1, -1)
        else:
            delta_expand = delta.expand(2**remain_part-1, -1, -1)
        reconstruct_quaternion[idx_range] = roma.quat_product(delta_expand, reconstruct_quaternion[idx_range])
        complete_progress += 2**remain_part
        counter += 2**remain_part
    if normalize:
        reconstruct_quaternion = torch.nn.functional.normalize(reconstruct_quaternion, dim=-1)
    if add_head:
        if len(quaternion.shape) == 2:
            reconstruct_quaternion = torch.nn.functional.pad(reconstruct_quaternion, (0,0,1,0))
            reconstruct_quaternion[0, 3] = 1
        else:
            reconstruct_quaternion = torch.nn.functional.pad(reconstruct_quaternion, (0,0,0,0,1,0))
            reconstruct_quaternion[0, :, 3] = 1
    return reconstruct_quaternion


def quat_cumprod(input: torch.Tensor, dim: int, normalize: bool = True):
    L, v = input.shape[dim], input.clone()
    n_func = lambda x: torch.nn.functional.normalize(x, dim=-1) if normalize else lambda x: x
    assert dim not in (-1, v.shape[-1]), "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        if i > L: break
        index = torch.arange(i, L, device=v.device)
        if index.numel() == 0: continue
        v.index_copy_(dim, index, n_func(roma.quat_product(v.index_select(dim, index - i), v.index_select(dim, index)))) # related to: https://github.com/pypose/pypose/issues/346
    return v


def mat_cumops(input: torch.Tensor, dim: int, ops = lambda a, b : b @ a):
    '''modified from https://github.com/pypose/pypose/blob/main/pypose/lietensor/basics.py'''
    L, v = input.shape[dim], input.clone()
    assert dim not in (-1, v.shape[-1]), "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(i, L, device=v.device, dtype=torch.int64)
        v.index_copy_(dim, index, ops(v.index_select(dim, index), v.index_select(dim, index-i)))
    return v


def parallel_prefix_sum(deltas: torch.Tensor, dim: int):
    L, v = deltas.shape[dim], deltas.clone()
    for stride in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(stride, L, device=v.device)
        if index.numel() == 0: continue
        v.index_copy_(dim, index, (v.index_select(dim, index - stride) + v.index_select(dim, index)))
    return torch.cat((torch.zeros_like(v.index_select(dim=dim, index=torch.tensor([0]))), v), dim=dim)


def unitquat_slerp_fast(q0, q1, steps, shortest_arc=True, align_batch=False):
    r"""
    Spherical linear interpolation between two unit quaternions.
    This function requires less computations than :func:`roma.utils.unitquat_slerp`,
    but is **unsuitable for extrapolation (i.e.** ``steps`` **must be within [0,1])**.

    Args: 
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
        align_batch (boolean): if True, assumes `steps` and `q0/q1` share batch dimensions (result shape: Bx4). if False, treats them as separate dimensions (result shape: BxAx4).
    Returns: 
        batch of interpolated quaternions (BxAx4 tensor if align_batch=False, else Bx4 tensor).
    """
    # Flatten batch dimensions of q0 and q1
    q0, batch_shape = roma.internal.flatten_batch_dims(q0, end_dim=-2)
    q1, batch_shape1 = roma.internal.flatten_batch_dims(q1, end_dim=-2)
    assert batch_shape == batch_shape1, "q0 and q1 must have the same batch dimensions"
    
    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(q0 * q1, dim=-1)
    if shortest_arc:
        # Flip some quaternions to perform shortest arc interpolation.
        q1 = q1.clone()
        q1[cos_omega < 0,:] *= -1
        cos_omega = torch.abs(cos_omega)
    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    if align_batch:
        # Ensure steps can broadcast to q0's batch dimension
        if steps.dim() == 0:  # Check if steps is a scalar tensor
            steps = steps.unsqueeze(0).expand_as(cos_omega)
        elif steps.dim() == 1 and steps.shape[0] == 1:
            steps = steps.expand_as(cos_omega)
        else:
            steps = steps.reshape_as(cos_omega)
        
        # Reshape tensors for element-wise operations
        cos_omega = cos_omega.unsqueeze(-1)  # [A, 1]
        s = steps.unsqueeze(-1)              # [A, 1]
        
        # Compute interpolation coefficients
        omega = torch.acos(cos_omega)
        alpha = torch.sin((1 - s) * omega)
        beta = torch.sin(s * omega)
        
        # Apply linear interpolation fallback
        alpha = alpha.clone()
        beta = beta.clone()
        alpha[nearby_quaternions] = 1 - s[nearby_quaternions]
        beta[nearby_quaternions] = s[nearby_quaternions]
        
        # Element-wise interpolation
        q = alpha * q0 + beta * q1
        
    else:
        # Reshape for broadcasting
        cos_omega = cos_omega.reshape((1,) * steps.dim() + (-1, 1))  # [1, ..., A, 1]
        s = steps.reshape(steps.shape + (1, 1))                      # [B, ..., 1, 1]
        
        # Compute interpolation coefficients
        omega = torch.acos(cos_omega)
        alpha = torch.sin((1 - s) * omega)
        beta = torch.sin(s * omega)
        
        # Apply linear interpolation fallback
        alpha = alpha.clone()
        beta = beta.clone()
        alpha[..., nearby_quaternions, :] = 1 - s
        beta[..., nearby_quaternions, :] = s
        
        # Broadcast and interpolate
        q0_bc = q0.reshape((1,) * steps.dim() + q0.shape)  # [1, ..., A, 4]
        q1_bc = q1.reshape((1,) * steps.dim() + q1.shape)  # [1, ..., A, 4]
        q = alpha * q0_bc + beta * q1_bc
    
    # Normalize and reshape
    q = torch.nn.functional.normalize(q, dim=-1)
    return q.reshape(batch_shape + (4,)) if align_batch else q.reshape(steps.shape + batch_shape + (4,))

def pdist_(X: torch.Tensor, squared: bool = False, eps_in_sqrt: float = 1e-8):
    assert len(X.shape) == 2
    B = X @ X.T
    c = torch.diag(B).expand(X.shape[0], X.shape[0])
    D2 = -2 * B + c + c.T
    if not squared:
        return torch.sqrt(D2 + eps_in_sqrt)
    else:
        return D2


def pdist(X: torch.Tensor, squared: bool = False, eps_in_sqrt: float = 1e-8):
    '''alternative to `torch.nn.functional.pdist`'''
    B = torch.einsum('...ij,...kj -> ...ik', X, X)
    c = torch.diagonal(B, dim1=-2, dim2=-1)
    D2 = -2 * B + c.unsqueeze(-1) + c.unsqueeze(-2)
    if not squared:
        return torch.sqrt(D2 + eps_in_sqrt)
    else:
        return D2


def cmds(D: torch.Tensor, squared: bool = False, algo = '1', DIMs: int = 3, eps_in_sqrt: float = 1e-8):
    r'''
    Implementation of the Classical Multidimensional Scaling (Principal Coordinates Analysis, PCoA)
    which converts a matrix of interpoint Euclidean distances back to Euclidean points
    that approximately reproduces the distance matrix.

    Reference:
    
    G. Marion Young, and Alston S. Householder 1938.
    Discussion of a set of points in terms of their mutual distances.
    Psychometrika, 3, p.19-22.
    
    Torgerson, W. 1952.
    Multidimensional Scaling: I. Theory and Method.
    Psychometrika, 17, p.401-419.

    Algorithm 1:

    Zhang Xuegong
    Pattern Recognition (3rd edition, Chinese)
    January 1, 1991.
    
    $$
    \begin{aligned}
        J &= I - 1/n 11^{\intercal} \\
        B &= -1/2 JD^{2}J \\
        B &= USU^{\intercal} \\
        X &= U\sqrt{S}
    \end{aligned}
    $$

    Algorithm 2:

    Legendre17 (https://math.stackexchange.com/users/82976/legendre17),
    Finding the coordinates of points from distance matrix,
    URL (version: 2013-06-18): https://math.stackexchange.com/q/423898
    
    $$
    \begin{aligned}
        M_{ij} &= \frac{D_{1j}^{2}+D_{i1}^{2}-D_{ij}^{2}}{2} \\
        M      &= USU^{\intercal} \\
        X      &= U\sqrt{S}
    \end{aligned}
    $$

    NOTE: the returned tensor may need to be flipped, e.g. `ret.flip(-1)`
    NOTE: this implementation use `svd` instead of `torch.linalg.eigh`
    '''
    assert (len(D.shape) == 2) and (D.shape[0] == D.shape[1]) and (D >= 0).all()
    size = D.shape[0]
    if squared:
        D2 = D
    else:
        D2 = torch.square(D)
    
    if algo == '1':
        # Algorithm 1
        J = torch.eye(size, device=D.device) - torch.full_like(D, 1/size)
        B = -1/2 * J @ D2 @ J
        _, evals, evecs = svd(B)
    else:
        # Algorithm 2
        M = -1/2 * D2
        idx0, idx1 = torch.triu_indices(size, size)
        M[idx0, idx1] = M[idx1, idx0] = M[idx0, idx1] + 1/2 * D2[idx0, 0] + 1/2 * D2[idx1, 0]
        _, evals, evecs = svd(M)
    
    return evecs[:, :DIMs] * torch.sqrt(evals[:DIMs] + eps_in_sqrt)


def batch_cmds(D: torch.Tensor, squared: bool = False, DIMs: int = 3, eps_in_sqrt: float = 1e-8):
    r'''
    Implementation of the Classical Multidimensional Scaling (Principal Coordinates Analysis, PCoA)
    which converts a matrix of interpoint Euclidean distances back to Euclidean points
    that approximately reproduces the distance matrix.

    Reference:
    
    G. Marion Young, and Alston S. Householder 1938.
    Discussion of a set of points in terms of their mutual distances.
    Psychometrika, 3, p.19-22.
    
    Torgerson, W. 1952.
    Multidimensional Scaling: I. Theory and Method.
    Psychometrika, 17, p.401-419.

    Algorithm 1:

    Zhang Xuegong
    Pattern Recognition (3rd edition, Chinese)
    January 1, 1991.
    
    $$
    \begin{aligned}
        J &= I - 1/n 11^{\intercal} \\
        B &= -1/2 JD^{2}J \\
        B &= USU^{\intercal} \\
        X &= U\sqrt{S}
    \end{aligned}
    $$

    NOTE: the returned tensor may need to be flipped, e.g. `ret.flip(-1)`
    NOTE: this implementation use `svd` instead of `torch.linalg.eigh`
    '''
    assert (len(D.shape) == 3) and (D.shape[1] == D.shape[2]) and (D >= 0).all()
    batch_size, size, _ = D.shape
    if squared:
        D2 = D
    else:
        D2 = torch.square(D)
    # Algorithm 1
    J = torch.eye(size, device=D.device).expand(batch_size, size, size) - torch.full_like(D, 1/size)
    B = -1/2 * torch.einsum('bij,bjk,bkl->bil', J, D2, J)
    _, evals, evecs = svd(B)
    return torch.einsum('bij,bj->bij', evecs[:, :, :DIMs], torch.sqrt(evals[:, :DIMs] + eps_in_sqrt))


def cartesian2spherical(xyz, backend=torch):
    '''
    modified from https://stackoverflow.com/a/4116899
    '''
    assert xyz.shape[1] == 3
    sphe = backend.zeros_like(xyz)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    # r
    sphe[:, 0] = backend.sqrt(xy + xyz[:, 2]**2)
    # theta
    ## for elevation angle defined from Z-axis down
    sphe[:, 1] = backend.arctan2(backend.sqrt(xy), xyz[:, 2])
    ## for elevation angle defined from XY-plane up
    ### sphe[:, 1] = backend.arctan2(xyz[:, 2], backend.sqrt(xy))
    # tau
    sphe[:, 2] = backend.arctan2(xyz[:, 1], xyz[:, 0])
    return sphe


def spherical2cartesian(r, theta, tau, backend=torch):
    return backend.stack([
        r * backend.sin(theta) * backend.cos(tau),
        r * backend.sin(theta) * backend.sin(tau),
        r * backend.cos(theta)], axis=1)


def rad_unwrap(x, backend=torch):
    y = x % (2 * backend.pi)
    return backend.where(y > backend.pi, y-2*backend.pi, y)


def batch_expand_transform_rotmat(WF_rotmat: torch.Tensor, WF_ori: torch.Tensor, W_expanded_atoms: torch.Tensor):
    '''
    WF_rotmat:        batch x frame x 3 x 3
    WF_ori:           batch x frame x 3
    W_expanded_atoms: batch x atom  x frame x 3

    -> batch x frame x atom x 3
    '''
    return torch.einsum('...bij,...abj -> ...bai', WF_rotmat, W_expanded_atoms) + WF_ori.unsqueeze(-2)


def batch_expand_transform_unitquat(WF_q: torch.Tensor, WF_ori: torch.Tensor, W_expanded_atoms: torch.Tensor):
    # TODO: optimize
    # return roma.quat_action(roma.quat_conjugation(WF_q), W_expanded_atoms - WF_ori)
    return batch_expand_transform_rotmat(roma.unitquat_to_rotmat(WF_q), WF_ori, W_expanded_atoms)


def rbf(D: torch.Tensor, num_rbf: int = 16, dmax: float = 20., dmin: float = 0.):
    '''
    https://github.com/jingraham/neurips19-graph-protein-design

    D: batch x length1 x length2
    '''
    D_mu = torch.linspace(dmin, dmax, num_rbf, device=D.device).view([1, 1, 1, -1])
    D_sigma = (dmax - dmin) / num_rbf
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma)**2)

