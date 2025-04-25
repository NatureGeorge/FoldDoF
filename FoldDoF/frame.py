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
# @Last Modified: 2025-04-25 05:23:34 pm
from typing import Union, List, Optional
import math
import torch
import roma
import numpy as np
from .utils import quat_apply, quat_cumprod_sequential, quat_cumprod, dihedral, rad_unwrap
from .data import DEF_LOC, CB_LOC, SC_F_ANCHOR_LOC, AA_SIDECHAIN_ATOMS, SC_F_REMAIN_LOC, SC_F_LOC


class FrameClass:
    
    def __len__(self):
        return self.frame_q.shape[0]
    
    def __getitem__(self, index):
        return self.frame_q[index]
    
    def to_local_pos(self, some_W_coords: torch.Tensor):
        return quat_apply(roma.quat_conjugation(self.frame_q), some_W_coords-self.ori)
    
    def to_local_vec(self, some_W_vecs: torch.Tensor):
        return quat_apply(roma.quat_conjugation(self.frame_q), some_W_vecs)
    
    def to_local_pos_i(self, some_W_coord: torch.Tensor, idx: Union[int, torch.Tensor, List]):
        return quat_apply(roma.quat_conjugation(self.frame_q[idx]), some_W_coord-self.ori[idx])
    
    def to_W_pos(self, some_loc_coords: torch.Tensor):
        return quat_apply(self.frame_q, some_loc_coords) + self.ori
    
    def to_W_pos_i(self, some_loc_coord: torch.Tensor, idx: Union[int, torch.Tensor, List]):
        return quat_apply(self.frame_q[idx], some_loc_coord) + self.ori[idx]


class PeptideUnitFrame(FrameClass):

    """
    Pytorch implementation of `PeptideUnitFrame` based on the SO(3) connection model on backbone defined in:

    Penner, R.C., Knudsen, M., Wiuf, C. and Andersen, J.E. (2010),
    Fatgraph models of proteins.
    Comm. Pure Appl. Math., 63: 1249-1297. https://doi.org/10.1002/cpa.20340
    """

    @staticmethod
    def get_peptide_unit(n_coords: torch.Tensor, ca_coords: torch.Tensor, c_coords: torch.Tensor):
        #assert n_coords.shape[1] == 3 and len(n_coords.shape) == 2
        #assert n_coords.shape == ca_coords.shape == c_coords.shape
        peptide_unit = torch.zeros((n_coords.shape[0]-1, 4, 3), device=n_coords.device, dtype=n_coords.dtype)
        peptide_unit[:, 0] = ca_coords[:-1]  # ca_i
        peptide_unit[:, 1] = c_coords[:-1]   # c_i
        peptide_unit[:, 2] = n_coords[1:]    # n_ia1
        peptide_unit[:, 3] = ca_coords[1:]   # ca_ia1
        return peptide_unit

    @classmethod
    def from_W_n_ca_c(cls, n_coords: torch.Tensor, ca_coords: torch.Tensor, c_coords: torch.Tensor):
        #return cls.from_W_peptide_unit(cls.get_peptide_unit(n_coords, ca_coords, c_coords))
        coords_1, coords_2, coords_3 = ca_coords[:-1], c_coords[:-1], n_coords[1:]
        frame_q = roma.rotmat_to_unitquat(roma.special_gramschmidt(torch.stack([coords_3 - coords_2, coords_2 - coords_1], dim=2)))
        with torch.no_grad():
            coords_4 = ca_coords[1:]
            is_trans = torch.einsum('km,km->k', coords_2 - coords_1, coords_4 - coords_3).ge(0)
        return cls(coords_2, frame_q, is_trans)

    @classmethod
    def from_W_peptide_unit(cls, peptide_unit: torch.Tensor):
        xyz = peptide_unit[:, [2, 1, 3]]-peptide_unit[:, [1, 0, 2]]
        ori = peptide_unit[:, 1]
        frame = roma.special_gramschmidt(xyz[:, :2].transpose(-1, -2)) # R_{WF}
        #v = frame[:, :, 1]  # NOTE: for calculating is_twisted
        #w = frame[:, :, 2]  # NOTE: for calculating is_twisted
        #vvww = (torch.einsum('km,km->k', v[:-1], v[1:]) + torch.einsum('km,km->k', w[:-1], w[1:]))  # NOTE: for calculating is_twisted
        frame_q = roma.rotmat_to_unitquat(frame)
        #frame_i = frame.transpose(-1, -2)                             # R_{FW}
        #frame_i_q = roma.quat_conjugation(frame_q)
        is_trans = torch.einsum('km,km->k', xyz[:, 1], xyz[:, 2]).ge(0)
        #is_twisted = torch.logical_or(
        #    torch.logical_and(is_trans[:-1], vvww.le(0)),
        #    torch.logical_and(~is_trans[:-1], vvww.ge(0))
        #)
        return cls(ori, frame_q, is_trans)
    
    def __init__(self, ori: Optional[torch.Tensor], frame_q: torch.Tensor, is_trans: Optional[torch.Tensor]) -> None:
        if is_trans is None:
            is_trans = torch.ones(frame_q.shape[0], dtype=torch.bool, device=frame_q.device)
        if ori is not None:
            assert ori.shape[0] == frame_q.shape[0] == is_trans.shape[0]
        else:
            ori = torch.zeros(frame_q.shape[0], 3, dtype=frame_q.dtype, device=frame_q.device)
            assert frame_q.shape[0] == is_trans.shape[0]
        self.ori = ori
        self.frame_q = frame_q
        self.is_trans = is_trans
        self.using_def_loc = DEF_LOC
        self.tensor_kwargs = dict(dtype=self.frame_q.dtype, device=self.frame_q.device)
    
    def get_reconstruct_ori_sequential(self, loc_ca_i: torch.Tensor, loc_ca_ia1: torch.Tensor):
        reconstruct_ori = [self.ori[0]]
        for i in range(1, len(self)):
            reconstruct_ori.append(quat_apply(self.frame_q[i-1], loc_ca_ia1[i-1]) + reconstruct_ori[-1] - quat_apply(self.frame_q[i], loc_ca_i[i]))
        return torch.stack(reconstruct_ori)
    
    def get_reconstruct_ori(self, loc_ca_i: torch.Tensor, loc_ca_ia1: torch.Tensor):
        reconstruct_ori = torch.zeros_like(self.ori)
        reconstruct_ori[0] = self.ori[0]
        return self.get_reconstruct_ori_base(reconstruct_ori, self.frame_q, loc_ca_i, loc_ca_ia1)

    @staticmethod
    def get_reconstruct_ori_base(reconstruct_ori: Optional[torch.Tensor], frame_q: torch.Tensor, loc_ca_i: torch.Tensor, loc_ca_ia1: torch.Tensor):
        '''
        Parallelized via binary merge.

        NOTE: input shape: L(xB)x...
        '''
        if reconstruct_ori is None:
            reconstruct_ori = torch.zeros(frame_q.shape[:-1]+(3,), dtype=frame_q.dtype, device=frame_q.device)
        first_part = np.floor(np.log2(reconstruct_ori.shape[0])).astype(int)
        for blevel in range(first_part):
            idx = torch.arange(2**blevel, reconstruct_ori.shape[0], 2**(blevel+1))
            to_assign = quat_apply(frame_q[idx-1], loc_ca_ia1[idx-1]) + reconstruct_ori[idx-1] - quat_apply(frame_q[idx], loc_ca_i[idx])
            #to_assign = (frame_q[idx-1] @ loc_ca_ia1[idx-1]) + reconstruct_ori[idx-1] - (frame_q[idx] @ loc_ca_i[idx])
            delta = to_assign - reconstruct_ori[idx]
            reconstruct_ori[idx] = to_assign
            if blevel > 0:
                idx_cat = torch.cat([idx+step for step in range(1, 2**blevel)])
                idx_cat_mask = idx_cat < reconstruct_ori.shape[0]
                idx_cat_masked = idx_cat[idx_cat_mask]
                if len(reconstruct_ori.shape) == 2:
                    delta_repeat = delta.repeat(2**blevel-1, 1)
                else:
                    delta_repeat = delta.repeat(2**blevel-1, 1, 1)
                reconstruct_ori[idx_cat_masked] = reconstruct_ori[idx_cat_masked] + delta_repeat[idx_cat_mask]
        complete_progress = 2**first_part
        while complete_progress < reconstruct_ori.shape[0]:
            remain_part = np.floor(np.log2(reconstruct_ori.shape[0] - complete_progress)).astype(int)
            idx = complete_progress
            to_assign = quat_apply(frame_q[[idx-1]], loc_ca_ia1[[idx-1]]) + reconstruct_ori[[idx-1]] - quat_apply(frame_q[[idx]], loc_ca_i[[idx]])
            #to_assign = (frame_q[[idx-1]] @ loc_ca_ia1[[idx-1]]) + reconstruct_ori[[idx-1]] - (frame_q[[idx]] @ loc_ca_i[[idx]])
            delta = to_assign - reconstruct_ori[[idx]]
            reconstruct_ori[[idx]] = to_assign
            idx_range = torch.arange(idx+1, idx+2**remain_part)
            reconstruct_ori[idx_range] = reconstruct_ori[idx_range] + delta
            complete_progress += 2**remain_part
        return reconstruct_ori
    
    def to_W_n_ca_c(self, loc_n_ter: torch.Tensor, loc_c_ter: torch.Tensor, loc_ca_i: torch.Tensor, loc_n_ia1: torch.Tensor, loc_ca_ia1: torch.Tensor):
        '''
        Returns:
        rebuilt_n_ca_c (torch.Tensor): shape(NumAtom x NumRes x 3 tensor) batch of points.
        '''
        rebuilt_n_ca_c = torch.zeros((3, self.frame_q.shape[0]+1, 3), dtype=self.frame_q.dtype, device=self.frame_q.device)
        rebuilt_n_ca_c[0, 0] = self.to_W_pos_i(loc_n_ter, 0)
        rebuilt_n_ca_c[0, 1:] = self.to_W_pos(loc_n_ia1)
        rebuilt_n_ca_c[1, :-1] = self.to_W_pos(loc_ca_i)
        rebuilt_n_ca_c[1, -1] = self.to_W_pos_i(loc_ca_ia1[-1], -1)
        rebuilt_n_ca_c[2, :-1] = self.ori
        rebuilt_n_ca_c[2, -1] = self.to_W_pos_i(loc_c_ter, -1)
        return rebuilt_n_ca_c

    def to_W_o(self, loc_o_ter: torch.Tensor, loc_o_i: torch.Tensor):
        return torch.cat((self.to_W_pos(loc_o_i), self.to_W_pos_i(loc_o_ter, -1).unsqueeze(0)))
    
    def to_W_backbone(self, loc_n_ter: torch.Tensor, loc_c_ter: torch.Tensor, loc_o_ter: torch.Tensor, loc_ca_i: torch.Tensor, loc_n_ia1: torch.Tensor, loc_ca_ia1: torch.Tensor, loc_o_i: torch.Tensor, numres_first: bool = False):
        w_n_ca_c = self.to_W_n_ca_c(loc_n_ter, loc_c_ter, loc_ca_i, loc_n_ia1, loc_ca_ia1)
        w_o = self.to_W_o(loc_o_ter, loc_o_i)
        ret = torch.cat((w_n_ca_c, w_o.unsqueeze(0)), dim=0)
        if not numres_first:
            return ret
        else:
            return ret.transpose(0, 1)
    
    def to_W_avg_backbone(self, loc_n_ter: Optional[torch.Tensor], loc_c_ter: Optional[torch.Tensor], loc_o_ter: Optional[torch.Tensor], loc_ca_i: Optional[torch.Tensor] = None, loc_ca_ia1_wrt_n_ia1: Optional[torch.Tensor] = None, **kwargs):
        # TODO: adapt to torch.nn.Parameter?
        avg_loc_n_ia1 = torch.tensor(self.using_def_loc['n_ia1'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        if loc_ca_ia1_wrt_n_ia1 is None:
            loc_ca_ia1 = torch.tensor(self.using_def_loc['ca_ia1_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_ia1[~self.is_trans] = torch.tensor(self.using_def_loc['ca_ia1_is_cis'], **self.tensor_kwargs)
        else:
            loc_ca_ia1 = loc_ca_ia1_wrt_n_ia1 + avg_loc_n_ia1
        if loc_ca_i is None:
            loc_ca_i = torch.tensor(self.using_def_loc['ca_i_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_i[~self.is_trans] = torch.tensor(self.using_def_loc['ca_i_is_cis'], **self.tensor_kwargs)
        avg_loc_o_i = torch.tensor(self.using_def_loc['o_i'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        reconstruct_mean_ori = self.get_reconstruct_ori(loc_ca_i, loc_ca_ia1)
        avg_loc_frame = PeptideUnitFrame(reconstruct_mean_ori, self.frame_q, self.is_trans)
        if loc_n_ter is None:
            loc_n_ter = torch.tensor(self.using_def_loc['n_ter_wrt_ca_i'], **self.tensor_kwargs) + loc_ca_i[0]
        if loc_c_ter is None:
            loc_c_ter = torch.tensor(self.using_def_loc['c_ter_wrt_ca_ia1'], **self.tensor_kwargs) + loc_ca_ia1[-1]
        if loc_o_ter is None:
            loc_o_ter = torch.tensor(self.using_def_loc['o_ter_wrt_ca_ia1'], **self.tensor_kwargs) + loc_ca_ia1[-1]
        return avg_loc_frame.to_W_backbone(loc_n_ter, loc_c_ter, loc_o_ter, loc_ca_i, avg_loc_n_ia1, loc_ca_ia1, avg_loc_o_i, **kwargs)
    
    def to_W_avg_backbone_addter(self, loc_ca_i: Optional[torch.Tensor] = None, loc_ca_ia1_wrt_n_ia1: Optional[torch.Tensor] = None, numres_first: bool = False):
        avg_loc_n_ia1 = torch.tensor(self.using_def_loc['n_ia1'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        if loc_ca_ia1_wrt_n_ia1 is None:
            loc_ca_ia1 = torch.tensor(self.using_def_loc['ca_ia1_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_ia1[~self.is_trans] = torch.tensor(self.using_def_loc['ca_ia1_is_cis'], **self.tensor_kwargs)
        else:
            loc_ca_ia1 = loc_ca_ia1_wrt_n_ia1 + avg_loc_n_ia1[:-1]
        if loc_ca_i is None:
            loc_ca_i = torch.tensor(self.using_def_loc['ca_i_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_i[~self.is_trans] = torch.tensor(self.using_def_loc['ca_i_is_cis'], **self.tensor_kwargs)
        avg_loc_o_i = torch.tensor(self.using_def_loc['o_i'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        reconstruct_mean_ori = self.get_reconstruct_ori(loc_ca_i, loc_ca_ia1)
        avg_loc_frame = PeptideUnitFrame(reconstruct_mean_ori, self.frame_q, self.is_trans)

        #rebuilt_backbone = torch.zeros((4, self.frame_q.shape[0]-1, 3), dtype=self.frame_q.dtype, device=self.frame_q.device)
        rebuilt_backbone = torch.stack([
            avg_loc_frame.to_W_pos(avg_loc_n_ia1)[:-1],  #rebuilt_backbone[0]
            avg_loc_frame.to_W_pos(loc_ca_i)[1:],  #rebuilt_backbone[1]
            avg_loc_frame.ori[1:],  #rebuilt_backbone[2]
            avg_loc_frame.to_W_pos(avg_loc_o_i)[1:],  #rebuilt_backbone[3]
        ], dim=0)
        if not numres_first:
            return rebuilt_backbone
        else:
            return rebuilt_backbone.transpose(0, 1)

    def to_W_avg_o_i_and_n_ia1_only(self, loc_ca_i: Optional[torch.Tensor] = None, loc_ca_ia1_wrt_n_ia1: Optional[torch.Tensor] = None):
        avg_loc_n_ia1 = torch.tensor(self.using_def_loc['n_ia1'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        if loc_ca_ia1_wrt_n_ia1 is None:
            loc_ca_ia1 = torch.tensor(self.using_def_loc['ca_ia1_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_ia1[~self.is_trans] = torch.tensor(self.using_def_loc['ca_ia1_is_cis'], **self.tensor_kwargs)
        else:
            loc_ca_ia1 = loc_ca_ia1_wrt_n_ia1 + avg_loc_n_ia1
        if loc_ca_i is None:
            loc_ca_i = torch.tensor(self.using_def_loc['ca_i_is_trans'], **self.tensor_kwargs).repeat(self.frame_q.shape[0], 1)
            loc_ca_i[~self.is_trans] = torch.tensor(self.using_def_loc['ca_i_is_cis'], **self.tensor_kwargs)
        avg_loc_o_i = torch.tensor(self.using_def_loc['o_i'], **self.tensor_kwargs).expand(self.frame_q.shape[0], -1)
        reconstruct_mean_ori = self.get_reconstruct_ori(loc_ca_i, loc_ca_ia1)
        avg_loc_frame = PeptideUnitFrame(reconstruct_mean_ori, self.frame_q, self.is_trans)
        return avg_loc_frame.to_W_pos(avg_loc_o_i), avg_loc_frame.to_W_pos(avg_loc_n_ia1)
    
    def update_is_trans(self, local_ca_ia1: torch.Tensor):
        # TODO: use the geodesics on the 2-sphere? (Related to the Fréchet Mean.)
        dist2trans = (local_ca_ia1 - torch.tensor(self.using_def_loc['ca_ia1_is_trans'], **self.tensor_kwargs)).norm(dim=1)
        dist2cis = (local_ca_ia1 - torch.tensor(self.using_def_loc['ca_ia1_is_cis'], **self.tensor_kwargs)).norm(dim=1)
        self.is_trans = dist2trans <= dist2cis
    
    @staticmethod
    def b1_mat():
        return [
            [-1/2, math.sqrt(3)/2, 0],
            [math.sqrt(3)/2, 1/2, 0],
            [0, 0, -1]
        ]
    
    @staticmethod
    def b2_mat(phi, psi):
        C = math.cos((phi+psi)/2)
        S = math.sin((phi+psi)/2)
        return [
            [1-3/2*(S**2), math.sqrt(3)/2*(S**2), math.sqrt(3)*C*S],
            [math.sqrt(3)/2*(S**2), 1-1/2*(S**2), -C*S],
            [-math.sqrt(3)*C*S, C*S, 1-2*(S**2)]
        ]

    @staticmethod
    def b3_mat(psi):
        C = math.cos(psi)
        S = math.sin(psi)
        return [
            [2/3-(C**2)/3+(S**2)/6, -2*(math.sqrt(2)*C/3+(S**2)/(4*math.sqrt(3))), 2*(C*S/(2*math.sqrt(3)) - S/(3*math.sqrt(2)))],
            [2*(math.sqrt(2)*C/3-(S**2)/(4*math.sqrt(3))), 2/3-C**2/3-(S**2)/6, -2*(C*S/6+S/math.sqrt(6))],
            [2*(C*S/(2*math.sqrt(3)) + S/(3*math.sqrt(2))), 2 * (S/math.sqrt(6)-C*S/6), 2/3+(C**2)/3-(S**2)/3],
        ]
    
    def relative_rotmat_from_phi_psi(self, phi, psi):
        return torch.tensor(self.b3_mat(phi), **self.tensor_kwargs) @ torch.tensor(self.b2_mat(phi, psi), **self.tensor_kwargs) @ torch.tensor(self.b1_mat(), **self.tensor_kwargs)
    
    def relative_quat_from_phi_psi(self, phi, psi):
        return roma.rotmat_to_unitquat(self.relative_rotmat_from_phi_psi(phi, psi))

    @staticmethod
    def get_ter_dihedral(n_coords, ca_coords, c_coords, o_coords):
        nter_n = n_coords[0]
        nter_ca = ca_coords[0]
        nter_c = c_coords[0]
        nter_n_a1 = n_coords[1]
        nter_psi = dihedral(nter_n, nter_ca, nter_c, nter_n_a1)
        cter_c_m1 = c_coords[-2]
        cter_n = n_coords[-1]
        cter_ca = ca_coords[-1]
        cter_c = c_coords[-1]
        cter_o = o_coords[-1]
        cter_psi = rad_unwrap(dihedral(cter_n, cter_ca, cter_c, cter_o)+torch.pi)
        cter_phi = dihedral(cter_c_m1, cter_n, cter_ca, cter_c)
        return nter_psi, cter_phi, cter_psi

    def to_W_avg_backbone_addter_from_dihedral(self, nter_psi, cter_phi, cter_psi, **kwargs):
        '''
        NOTE: The N/C terminal atoms are approximated by terminal dihedrals, likely to induce some tiny deviation.
        '''
        nter_frame_q = self.relative_quat_from_phi_psi(torch.pi, nter_psi)[None]
        cter_frame_q = self.relative_quat_from_phi_psi(cter_phi, cter_psi)[None]
        relative_frame_q = roma.quat_product(roma.quat_conjugation(self.frame_q[:-1]), self.frame_q[1:])
        return PeptideUnitFrame(None,
            quat_cumprod(torch.cat((nter_frame_q, relative_frame_q, cter_frame_q))),
            torch.cat((torch.tensor([True]), self.is_trans, torch.tensor([True])))
        ).to_W_avg_backbone_addter(**kwargs)
    
    @classmethod
    def to_W_batch_avg_ori(cls, frame_q: torch.Tensor, loc_ca_ia1_wrt_n_ia1: torch.Tensor):
        '''
        NOTE: input shape: L(xB)x...
        '''
        tensor_kwargs = dict(dtype=frame_q.dtype, device=frame_q.device)
        avg_loc_n_ia1 = torch.tensor(DEF_LOC['n_ia1'], **tensor_kwargs).expand(*frame_q.shape[:-1], -1)
        loc_ca_ia1 = loc_ca_ia1_wrt_n_ia1 + avg_loc_n_ia1[:-1]
        loc_ca_i = torch.tensor(DEF_LOC['ca_i_is_trans'], **tensor_kwargs).expand(*frame_q.shape[:-1], -1)
        reconstruct_ori = cls.get_reconstruct_ori_base(None, frame_q, loc_ca_i, loc_ca_ia1)
        return reconstruct_ori, avg_loc_n_ia1, loc_ca_i

    @classmethod
    def to_W_batch_avg_backbone_addter(cls, frame_q: torch.Tensor, loc_ca_ia1_wrt_n_ia1: torch.Tensor, stack_dim: int = 1):
        '''
        NOTE: input shape: L(xB)x...
        '''
        tensor_kwargs = dict(dtype=frame_q.dtype, device=frame_q.device)
        avg_loc_o_i = torch.tensor(DEF_LOC['o_i'], **tensor_kwargs).expand(*frame_q.shape[:-1], -1)
        reconstruct_ori, avg_loc_n_ia1, loc_ca_i = cls.to_W_batch_avg_ori(frame_q, loc_ca_ia1_wrt_n_ia1)
        to_W_pos = lambda some_loc_coords: quat_apply(frame_q, some_loc_coords) + reconstruct_ori
        #to_W_pos = lambda some_loc_coords: (frame_q @ some_loc_coords) + reconstruct_ori
        return torch.stack([
            to_W_pos(avg_loc_n_ia1)[:-1],  #rebuilt_backbone[0] N
            to_W_pos(loc_ca_i)[1:],        #rebuilt_backbone[1] Cα
            reconstruct_ori[1:],           #rebuilt_backbone[2] C
            to_W_pos(avg_loc_o_i)[1:],     #rebuilt_backbone[3] O
        ], dim=stack_dim)  #4xL(xB)x3, dim=0
                           #Lx4(xB)x3, dim=1
                           #L(xB)x4x3, dim=2
    
    @classmethod
    def to_W_batch_avg_backbone_addter_via_rotmat(cls, frame_rotmat: torch.Tensor, loc_ca_ia1_wrt_n_ia1: torch.Tensor, stack_dim: int = 1):
        '''
        NOTE: input shape: L(xB)x...
        '''
        tensor_kwargs = dict(dtype=frame_rotmat.dtype, device=frame_rotmat.device)
        avg_loc_o_i = torch.tensor(DEF_LOC['o_i'], **tensor_kwargs).expand(*frame_rotmat.shape[:-2], -1)
        reconstruct_ori, avg_loc_n_ia1, loc_ca_i = cls.to_W_batch_avg_ori_via_rotmat(frame_rotmat, loc_ca_ia1_wrt_n_ia1)
        to_W_pos = lambda some_loc_coords: torch.einsum('...ij,...j->...i', frame_rotmat, some_loc_coords) + reconstruct_ori
        return torch.stack([
            to_W_pos(avg_loc_n_ia1)[:-1],  #rebuilt_backbone[0] N
            to_W_pos(loc_ca_i)[1:],        #rebuilt_backbone[1] Cα
            reconstruct_ori[1:],           #rebuilt_backbone[2] C
            to_W_pos(avg_loc_o_i)[1:],     #rebuilt_backbone[3] O
        ], dim=stack_dim)  #4xL(xB)x3, dim=0
                           #Lx4(xB)x3, dim=1
                           #L(xB)x4x3, dim=2
    
    @classmethod
    def to_W_batch_avg_backbone_addter_via_rotmat_trans(cls, frame_rotmat: torch.Tensor, frame_trans: torch.Tensor):
        '''
        convert global pose to full atom backbone coordinates

        NOTE: input shape: BxLx...
        '''
        loc_ca_i = torch.tensor(DEF_LOC['ca_i_is_trans'], dtype=frame_rotmat.dtype, device=frame_rotmat.device)
        frame_rotmat = frame_rotmat.transpose(0, 1)
        reconstruct_ori = frame_trans.transpose(0, 1)
        to_W_pos = lambda some_loc_coords: torch.einsum('...ij,...j->...i', frame_rotmat, some_loc_coords) + reconstruct_ori
        tensor_kwargs = dict(dtype=frame_rotmat.dtype, device=frame_rotmat.device)
        avg_loc_n_ia1 = torch.tensor(DEF_LOC['n_ia1'], **tensor_kwargs).expand(*frame_rotmat.shape[:-2], -1)
        avg_loc_o_i = torch.tensor(DEF_LOC['o_i'], **tensor_kwargs).expand(*frame_rotmat.shape[:-2], -1)
        avg_loc_ca_i = loc_ca_i.expand(*frame_rotmat.shape[:-2], -1)
        return torch.stack([
                to_W_pos(avg_loc_n_ia1)[:-1],  #rebuilt_backbone[0] N
                to_W_pos(avg_loc_ca_i)[1:],    #rebuilt_backbone[1] Cα
                reconstruct_ori[1:],           #rebuilt_backbone[2] C
                to_W_pos(avg_loc_o_i)[1:],     #rebuilt_backbone[3] O
            ], dim=1).permute(2, 0, 1, 3)      # Lx4xBx3 -> BxLx4x3

    @classmethod
    def to_W_batch_avg_ori_via_rotmat(cls, frame_rotmat: torch.Tensor, loc_ca_ia1_wrt_n_ia1: torch.Tensor):
        '''
        NOTE: input shape: L(xB)x...
        '''
        tensor_kwargs = dict(dtype=frame_rotmat.dtype, device=frame_rotmat.device)
        avg_loc_n_ia1 = torch.tensor(DEF_LOC['n_ia1'], **tensor_kwargs).expand(*frame_rotmat.shape[:-2], -1)
        loc_ca_ia1 = loc_ca_ia1_wrt_n_ia1 + avg_loc_n_ia1[:-1]
        loc_ca_i = torch.tensor(DEF_LOC['ca_i_is_trans'], **tensor_kwargs).expand(*frame_rotmat.shape[:-2], -1)
        reconstruct_ori = cls.get_reconstruct_ori_base_via_rotmat(None, frame_rotmat, loc_ca_i, loc_ca_ia1)
        return reconstruct_ori, avg_loc_n_ia1, loc_ca_i
    
    @staticmethod
    def get_reconstruct_ori_base_via_rotmat(reconstruct_ori: Optional[torch.Tensor], frame_rotmat: torch.Tensor, loc_ca_i: torch.Tensor, loc_ca_ia1: torch.Tensor):
        '''
        Parallelized via binary merge.

        NOTE: input shape: L(xB)x...
        '''
        if reconstruct_ori is None:
            reconstruct_ori = torch.zeros(frame_rotmat.shape[:-2]+(3,), dtype=frame_rotmat.dtype, device=frame_rotmat.device)
        bmm = lambda a,b: torch.einsum('...ij,...j->...i', a, b)
        first_part = np.floor(np.log2(reconstruct_ori.shape[0])).astype(int)
        for blevel in range(first_part):
            idx = torch.arange(2**blevel, reconstruct_ori.shape[0], 2**(blevel+1))
            to_assign = bmm(frame_rotmat[idx-1], loc_ca_ia1[idx-1]) + reconstruct_ori[idx-1] - bmm(frame_rotmat[idx], loc_ca_i[idx])
            delta = to_assign - reconstruct_ori[idx]
            reconstruct_ori[idx] = to_assign
            if blevel > 0:
                idx_cat = torch.cat([idx+step for step in range(1, 2**blevel)])
                idx_cat_mask = idx_cat < reconstruct_ori.shape[0]
                idx_cat_masked = idx_cat[idx_cat_mask]
                if len(reconstruct_ori.shape) == 2:
                    delta_repeat = delta.repeat(2**blevel-1, 1)
                else:
                    delta_repeat = delta.repeat(2**blevel-1, 1, 1)
                reconstruct_ori[idx_cat_masked] = reconstruct_ori[idx_cat_masked] + delta_repeat[idx_cat_mask]
        complete_progress = 2**first_part
        while complete_progress < reconstruct_ori.shape[0]:
            remain_part = np.floor(np.log2(reconstruct_ori.shape[0] - complete_progress)).astype(int)
            idx = complete_progress
            to_assign = bmm(frame_rotmat[[idx-1]], loc_ca_ia1[[idx-1]]) + reconstruct_ori[[idx-1]] - bmm(frame_rotmat[[idx]], loc_ca_i[[idx]])
            delta = to_assign - reconstruct_ori[[idx]]
            reconstruct_ori[[idx]] = to_assign
            idx_range = torch.arange(idx+1, idx+2**remain_part)
            reconstruct_ori[idx_range] = reconstruct_ori[idx_range] + delta
            complete_progress += 2**remain_part
        return reconstruct_ori

    @classmethod
    def to_rottrans(cls, bb_coords: torch.Tensor, bb_masks: Optional[torch.Tensor] = None):
        pep_frame = cls.from_W_n_ca_c(*bb_coords[:3])
        nter_psi, cter_phi, cter_psi = pep_frame.get_ter_dihedral(*bb_coords[:4])
        nter_frame_q = pep_frame.relative_quat_from_phi_psi(torch.pi, nter_psi)
        cter_frame_q = pep_frame.relative_quat_from_phi_psi(cter_phi, cter_psi)
        nter_frame_q = roma.quat_product(pep_frame.frame_q[0], roma.quat_conjugation(nter_frame_q))[None]
        cter_frame_q = roma.quat_product(pep_frame.frame_q[-1], cter_frame_q)[None]
        ter_loc_ca_ia1_wrt_n_ia1= (torch.tensor(DEF_LOC['ca_ia1_is_trans']) - torch.tensor(DEF_LOC['n_ia1']))[None]
        global_rots_q = torch.cat((nter_frame_q, pep_frame.frame_q, cter_frame_q))
        virtual_Cm1 = bb_coords[1, 0] - roma.unitquat_to_rotmat(nter_frame_q) @ torch.tensor(DEF_LOC['ca_ia1_is_trans'])
        global_trans = torch.cat((virtual_Cm1, bb_coords[2]))
        loc_ca_ia1 = pep_frame.to_local_pos(bb_coords[1, 1:])
        loc_n_ia1 = pep_frame.to_local_pos(bb_coords[0, 1:])
        loc_ca_ia1_wrt_n_ia1 = loc_ca_ia1 - loc_n_ia1
        ret_loc_ca_ia1_wrt_n_ia1 = torch.cat((ter_loc_ca_ia1_wrt_n_ia1, loc_ca_ia1_wrt_n_ia1))
        if bb_masks is not None:
            n_ter_mask = bb_masks[0, [0, 1]].all() & bb_masks[1, 0] & bb_masks[2, 0]
            c_ter_mask = bb_masks[2, [-2, -1]].all() & bb_masks[0, -1] & bb_masks[1, -1] & bb_masks[3, -1]
            pep_mask = bb_masks[0, 1:] & bb_masks[1, :-1] & bb_masks[2, :-1]
            pep_mask = torch.cat([n_ter_mask[None], pep_mask, c_ter_mask[None]])
            assert (pep_mask[:-1] & pep_mask[1:]).any()
            loc_ca_ia1_wrt_n_ia1_mask = torch.cat([torch.tensor([True]), bb_masks[1, 1:] & bb_masks[0, 1:] & pep_mask[1:-1]])
        else:
            pep_mask = torch.ones((bb_coords.shape[0]+1, ), dtype=torch.bool, device=bb_coords.device)
            loc_ca_ia1_wrt_n_ia1_mask = torch.ones((bb_coords.shape[0], ), dtype=torch.bool, device=bb_coords.device)
        return roma.unitquat_to_rotmat(global_rots_q), global_trans, ret_loc_ca_ia1_wrt_n_ia1, pep_mask, loc_ca_ia1_wrt_n_ia1_mask


class ResidueFrame(FrameClass):

    '''
    Local Coordinate System defined as:
        
    * Cα as Origin
    * Cα-C as X-axis
    * Y-axis that perpendicular to X-axis in the N-Cα-C plane
    * Z-axis that perpendicular to the N-Cα-C plane and form a right-handed coordinate system
    '''

    @classmethod
    def from_W_n_ca_c(cls, n_coords: torch.Tensor, ca_coords: torch.Tensor, c_coords: torch.Tensor):
        return cls(ca_coords, roma.rotmat_to_unitquat(roma.special_gramschmidt(torch.stack([c_coords - ca_coords, n_coords - ca_coords], dim=2))))
    
    def __init__(self, ori: torch.Tensor, frame_q: torch.Tensor):
        self.ori = ori
        self.frame_q = frame_q


class SidechainFrame(FrameClass):

    """
    Pytorch implementation of `SidechainFrame` based on the SO(3) connection model on side chain.
    
    Many researchers are aware of similar kinds of local frames:

    Hanson, A., and Thakur, S. 2012.
    Quaternion maps of global protein structure.
    Journal of Molecular Graphics and Modelling, 38, p.256-278. doi:10.1016/j.jmgm.2012.06.004

    Lundgren, M., and Niemi, A. 2012.
    Correlation between protein secondary structure, backbone bond angles, and side-chain orientations.
    Phys. Rev. E, 86, p.021904. doi:10.1103/PhysRevE.86.021904

    Lee, J., Yadollahpour, P., Watkins, A., Frey, N., Leaver-Fay, A., Ra, S., Cho, K., Gligorĳevic, V., Regev, A., and Bonneau, R. 2022.
    EquiFold: Protein Structure Prediction with a Novel Coarse-Grained Structure Representation.
    bioRxiv. doi:10.1101/2022.10.07.511322
    """

    NCAC_FRAME_ONE_AA = ("SER", "CYS", "SEC", "THR", "VAL", "MET", "GLN", "GLU")
    try:
        quat_cumprod = torch.vmap(quat_cumprod_sequential)
    except AttributeError:
        from functorch import vmap
        quat_cumprod = vmap(quat_cumprod_sequential)

    @classmethod
    def from_W_3(cls, coords_1: torch.Tensor, coords_2: torch.Tensor, coords_3: torch.Tensor):
        return cls(coords_2, roma.rotmat_to_unitquat(roma.special_gramschmidt(torch.stack([coords_3 - coords_2, coords_1 - coords_2], dim=2))))
    
    def __init__(self, ori: Optional[torch.Tensor], frame_q: torch.Tensor):
        if ori is not None:
            assert ori.shape[0] == frame_q.shape[0]
        else:
            ori = torch.zeros(frame_q.shape[0], 3, dtype=frame_q.dtype, device=frame_q.device)
        self.ori = ori
        self.frame_q = frame_q

    def relative_quat(self, another):
        return roma.quat_product(roma.quat_conjugation(self.frame_q), another.frame_q)

    @classmethod
    def get_sc_relative_quat(cls, three_letter_seq_arr, bb_coords, sc_coords):
        sc_relative_quat = torch.zeros((bb_coords.shape[0], 2, 4), dtype=bb_coords.dtype, device=bb_coords.device)
        sc_relative_quat[:,:,-1] = 1
        for aa_kind in set(three_letter_seq_arr):
            cur_aa_kind_idx = np.where(three_letter_seq_arr==aa_kind)[0]
            cur_aa_kind_bb_coords = bb_coords[cur_aa_kind_idx]
            cur_aa_kind_sc_coords = sc_coords[cur_aa_kind_idx]
            
            n  = cur_aa_kind_bb_coords[:, 0]
            ca = cur_aa_kind_bb_coords[:, 1]
            c  = cur_aa_kind_bb_coords[:, 2]
            cb = cur_aa_kind_sc_coords[:, 0]

            if aa_kind in ("PRO", "ASP", "ASN", "LEU", "LYS", "ARG", "HIS", "PHE", "TYR", "TRP"):
                frame_list = [
                    cls.from_W_3(cb, ca, c),
                    cls.from_W_3(cur_aa_kind_sc_coords[:, 2], cur_aa_kind_sc_coords[:, 1], cb)
                ]
            elif aa_kind == "ILE":
                frame_list = [
                    cls.from_W_3(cb, ca, c),
                    cls.from_W_3(cur_aa_kind_sc_coords[:, 3], cur_aa_kind_sc_coords[:, 1], cb)
                ]
            elif aa_kind in cls.NCAC_FRAME_ONE_AA:
                frame_list = [
                    cls.from_W_3(n, ca, c),
                    cls.from_W_3(cur_aa_kind_sc_coords[:, 1], cb, ca)
                ]
            else:
                #print(f"skip '{aa_kind}'")
                continue
            if aa_kind in ("ARG", "LYS"):
                frame_list.append(cls.from_W_3(cur_aa_kind_sc_coords[:, 4], cur_aa_kind_sc_coords[:, 3], cur_aa_kind_sc_coords[:, 2]))
            elif aa_kind in ("MET", "GLN", "GLU"):
                frame_list.append(cls.from_W_3(cur_aa_kind_sc_coords[:, 3], cur_aa_kind_sc_coords[:, 2], cur_aa_kind_sc_coords[:, 1]))

            for rf_idx in range(1, len(frame_list)):
                sc_relative_quat[cur_aa_kind_idx, rf_idx-1] = frame_list[rf_idx-1].relative_quat(frame_list[rf_idx])
        return sc_relative_quat

    @classmethod
    def to_W_avg_sidechain(cls, three_letter_seq_arr, bb_coords, sc_relative_quat):
        # TODO: batch
        frame_mask = np.isin(three_letter_seq_arr, cls.NCAC_FRAME_ONE_AA)
        frame_rev_mask_exlude_gly = (~frame_mask) & (three_letter_seq_arr != 'GLY')
        n_ca_c_frameone = cls.from_W_3(*bb_coords.transpose(0, 1)[:3])
        tensor_kwargs = dict(dtype=bb_coords.dtype, device=bb_coords.device)
        n_ca_c_frameone_s_loc_cb = torch.tensor([CB_LOC[aa] for aa in three_letter_seq_arr[frame_rev_mask_exlude_gly]], **tensor_kwargs).reshape(-1, 3)
        n_ca_c_frameone_s_W_cb = n_ca_c_frameone.to_W_pos_i(n_ca_c_frameone_s_loc_cb, frame_rev_mask_exlude_gly)
        n_ca_c_frameone = cls(n_ca_c_frameone.ori[frame_mask], n_ca_c_frameone.frame_q[frame_mask])
        cb_ca_c_frameone = cls.from_W_3(n_ca_c_frameone_s_W_cb, bb_coords[frame_rev_mask_exlude_gly, 1], bb_coords[frame_rev_mask_exlude_gly, 2])
        frameone_quat = torch.zeros((sc_relative_quat.shape[0], 4), **tensor_kwargs)
        frameone_ori = torch.zeros((sc_relative_quat.shape[0], 3), **tensor_kwargs)
        anchor_atoms = torch.tensor([SC_F_ANCHOR_LOC[aa] for aa in three_letter_seq_arr], **tensor_kwargs)

        frameone_quat[:, -1] = 1.
        frameone_quat[frame_mask] = n_ca_c_frameone.frame_q
        frameone_quat[frame_rev_mask_exlude_gly] = cb_ca_c_frameone.frame_q
        frameone_ori[frame_mask] = n_ca_c_frameone.ori
        frameone_ori[frame_rev_mask_exlude_gly] = cb_ca_c_frameone.ori
        anchor_atoms[frame_rev_mask_exlude_gly, 0] = cb_ca_c_frameone.to_local_pos(n_ca_c_frameone_s_W_cb)
        sc_relative_quat = torch.cat((frameone_quat.unsqueeze(1), sc_relative_quat), dim=1)
        sc_glo_quat = cls.quat_cumprod(sc_relative_quat)
        
        frameone_ori_1 = quat_apply(sc_glo_quat[:, 0], anchor_atoms[:, 0]) + frameone_ori - quat_apply(sc_glo_quat[:, 1], anchor_atoms[:, 1])
        frame_joint = quat_apply(sc_glo_quat[:, 1], anchor_atoms[:, 2]) + frameone_ori_1
        frameone_ori_2 = frame_joint - quat_apply(sc_glo_quat[:, 2], anchor_atoms[:, 3])
        # frameone_ori_1: CB;CG;CG1(ILE)
        # frame_joint:  CD(ARG,LYS);CG(MET,GLN,GLU)
        # frameone_ori_2: NE(ARG);CE(LYS);SD(MET);CD(GLN,GLU)
        avg_sc_coords = torch.zeros((sc_relative_quat.shape[0], 10, 3), **tensor_kwargs)
        avg_sc_mask = torch.tensor([[True]*len(AA_SIDECHAIN_ATOMS[aa])+[False]*(10-len(AA_SIDECHAIN_ATOMS[aa])) for aa in three_letter_seq_arr], dtype=torch.bool)

        avg_sc_coords[frame_rev_mask_exlude_gly, 0] = n_ca_c_frameone_s_W_cb
        avg_sc_coords[frame_mask, 0] = frameone_ori_1[frame_mask]
        avg_sc_coords[frame_rev_mask_exlude_gly, 1] = frameone_ori_1[frame_rev_mask_exlude_gly]
        atom_1_aaidx = np.isin(three_letter_seq_arr, ('SER', 'CYS', 'SEC', 'VAL', 'THR', ))
        atom_1_loc = torch.tensor([SC_F_REMAIN_LOC[aa][0] for aa in three_letter_seq_arr[atom_1_aaidx]], **tensor_kwargs).reshape(-1, 3)
        avg_sc_coords[atom_1_aaidx, 1] = quat_apply(sc_glo_quat[atom_1_aaidx, 1], atom_1_loc) + frameone_ori_1[atom_1_aaidx]
        atom_1_joint_aaidx = np.isin(three_letter_seq_arr, ('MET', 'GLN', 'GLU'))
        avg_sc_coords[atom_1_joint_aaidx, 1] = frame_joint[atom_1_joint_aaidx]
        avg_sc_coords[atom_1_joint_aaidx, 2] = frameone_ori_2[atom_1_joint_aaidx]
        atom_2_joint_aaidx = np.isin(three_letter_seq_arr, ('LYS', 'ARG'))
        avg_sc_coords[atom_2_joint_aaidx, 2] = frame_joint[atom_2_joint_aaidx]
        avg_sc_coords[atom_2_joint_aaidx, 3] = frameone_ori_2[atom_2_joint_aaidx]
        atom_2_aaidx = np.isin(three_letter_seq_arr, ('VAL', 'THR', 'ILE', 'PRO', 'ASP', 'ASN', 'LEU', 'HIS', 'PHE', 'TYR', 'TRP')) # skip ILE
        atom_2_loc = torch.tensor([SC_F_REMAIN_LOC[aa][int(aa in ('VAL', 'THR'))] for aa in three_letter_seq_arr[atom_2_aaidx]], **tensor_kwargs).reshape(-1, 3)
        avg_sc_coords[atom_2_aaidx, 2] = quat_apply(sc_glo_quat[atom_2_aaidx, 1], atom_2_loc) + frameone_ori_1[atom_2_aaidx]

        ile_aaidx = three_letter_seq_arr == "ILE"
        avg_sc_coords[ile_aaidx, 3] = avg_sc_coords[ile_aaidx, 2]
        avg_sc_coords[ile_aaidx, 2] = cls.from_W_3(avg_sc_coords[ile_aaidx, 1], avg_sc_coords[ile_aaidx, 0], bb_coords[ile_aaidx, 1]).to_W_pos(
            torch.tensor([SC_F_LOC['ILE']['CG2']], **tensor_kwargs).expand(ile_aaidx.sum(), 3)
        )
        atom_3_aaidx_a = np.isin(three_letter_seq_arr, ('ASP', 'ASN', 'LEU', 'HIS', 'PHE', 'TYR', 'TRP'))
        atom_3_aaidx_b = np.isin(three_letter_seq_arr, ('MET', 'GLU', 'GLN'))
        avg_sc_coords[atom_3_aaidx_a, 3] = quat_apply(
                sc_glo_quat[atom_3_aaidx_a, 1],
                torch.tensor([SC_F_REMAIN_LOC[aa][1] for aa in three_letter_seq_arr[atom_3_aaidx_a]], **tensor_kwargs).reshape(-1, 3)
            ) + frameone_ori_1[atom_3_aaidx_a]
        avg_sc_coords[atom_3_aaidx_b, 3] = quat_apply(
                sc_glo_quat[atom_3_aaidx_b, 2],
                torch.tensor([SC_F_REMAIN_LOC[aa][0] for aa in three_letter_seq_arr[atom_3_aaidx_b]], **tensor_kwargs).reshape(-1, 3)
            ) + frameone_ori_2[atom_3_aaidx_b]
        atom_4_aaidx_a = np.isin(three_letter_seq_arr, ('HIS', 'PHE', 'TYR', 'TRP'))
        atom_4_aaidx_b = np.isin(three_letter_seq_arr, ('LYS', 'ARG', 'GLU', 'GLN'))

        avg_sc_coords[atom_4_aaidx_a, 4] = quat_apply(
                sc_glo_quat[atom_4_aaidx_a, 1],
                torch.tensor([SC_F_REMAIN_LOC[aa][2] for aa in three_letter_seq_arr[atom_4_aaidx_a]], **tensor_kwargs).reshape(-1, 3)
            ) + frameone_ori_1[atom_4_aaidx_a]
        avg_sc_coords[atom_4_aaidx_b, 4] = quat_apply(
                sc_glo_quat[atom_4_aaidx_b, 2],
                torch.tensor([SC_F_REMAIN_LOC[aa][int(aa in ('GLU', 'GLN'))] for aa in three_letter_seq_arr[atom_4_aaidx_b]], **tensor_kwargs).reshape(-1, 3)
            ) + frameone_ori_2[atom_4_aaidx_b]
        arg_aaidx = three_letter_seq_arr == "ARG"
        his_aaidx = three_letter_seq_arr == "HIS"
        phe_aaidx = three_letter_seq_arr == "PHE"
        tyr_aaidx = three_letter_seq_arr == "TYR"
        trp_aaidx = three_letter_seq_arr == "TRP"
        avg_sc_coords[arg_aaidx, 5:7] = roma.quat_action(
            sc_glo_quat[arg_aaidx, 2].unsqueeze(1).expand(-1, 2, -1),
            torch.tensor([SC_F_REMAIN_LOC["ARG"][1:]], **tensor_kwargs).expand(arg_aaidx.sum(),-1,3),
            True
        ) + frameone_ori_2[arg_aaidx].unsqueeze(1).expand(-1, 2, -1)
        avg_sc_coords[his_aaidx, 5] = quat_apply(
            sc_glo_quat[his_aaidx, 1],
            torch.tensor([SC_F_REMAIN_LOC["HIS"][-1]], **tensor_kwargs).expand(his_aaidx.sum(), 3)
        ) + frameone_ori_1[his_aaidx]
        avg_sc_coords[phe_aaidx, 5:7] = roma.quat_action(
            sc_glo_quat[phe_aaidx, 1].unsqueeze(1).expand(-1, 2, -1),
            torch.tensor([SC_F_REMAIN_LOC["PHE"][-2:]], **tensor_kwargs).expand(phe_aaidx.sum(),-1,3),
            True
        ) + frameone_ori_1[phe_aaidx].unsqueeze(1).expand(-1, 2, -1)
        avg_sc_coords[tyr_aaidx, 5:8] = roma.quat_action(
            sc_glo_quat[tyr_aaidx, 1].unsqueeze(1).expand(-1, 3, -1),
            torch.tensor([SC_F_REMAIN_LOC["TYR"][-3:]], **tensor_kwargs).expand(tyr_aaidx.sum(),-1,3),
            True
        ) + frameone_ori_1[tyr_aaidx].unsqueeze(1).expand(-1, 3, -1)
        avg_sc_coords[trp_aaidx, 5:10] = roma.quat_action(
            sc_glo_quat[trp_aaidx, 1].unsqueeze(1).expand(-1, 5, -1),
            torch.tensor([SC_F_REMAIN_LOC["TRP"][-5:]], **tensor_kwargs).expand(trp_aaidx.sum(),-1,3),
            True
        ) + frameone_ori_1[trp_aaidx].unsqueeze(1).expand(-1, 5, -1)

        return avg_sc_coords, avg_sc_mask
