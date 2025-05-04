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

# @Created Date: 2022-01-16 01:29:29 pm
# @Filename: __init__.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-05-04 07:36:02 pm
import torch
import numpy as np
import roma
from typing import Union, Optional, Literal
from enum import Enum
from .frame import PeptideUnitFrame
from .utils import mat_cumops, quat_cumprod


class to_rottrans_mode(Enum):
    PeptideUnitFrame = 'peptide_frame "Ca-C-N" global_rots:L+1 global_trans:L+1'
    ResidueFrame = 'residue_frame "N-Ca-C" global_rots:L global_trans:L'


def to_rottrans(bb_coords: torch.Tensor, bb_masks: Optional[torch.Tensor] = None, mode: Literal[to_rottrans_mode.PeptideUnitFrame, to_rottrans_mode.ResidueFrame] = to_rottrans_mode.PeptideUnitFrame, rot_repr_is_q: bool = False):
    '''
    shape: L x 4 x ...
    '''
    if mode == to_rottrans_mode.PeptideUnitFrame:
        return PeptideUnitFrame.to_rottrans(bb_coords, bb_masks, rot_repr_is_q=rot_repr_is_q)
    elif mode == to_rottrans_mode.ResidueFrame:
        raise NotImplementedError('TODO.')


class to_bb_mode(Enum):
    Pep_GlobalRots_GlobalTrans = 'peptide_frame global_rots:L+1 global_trans:L+1'
    Pep_GlobalRots_IsoRots = 'peptide_frame global_rots:L+1 loc_ca_ia1_wrt_n_ia1:L'
    Pep_RelativeRots_IsoRots = 'peptide_frame relative_rots:L loc_ca_ia1_wrt_n_ia1:L'
    Res_GlobalRots_GlobalTrans = 'residue_frame global_rots:L global_trans:L'


def to_backbone(rots: torch.Tensor,
                trans_or_loc_ca_ia1_wrt_n_ia1: torch.Tensor,
                mode: Literal[to_bb_mode.Pep_GlobalRots_GlobalTrans, to_bb_mode.Pep_GlobalRots_IsoRots, to_bb_mode.Pep_RelativeRots_IsoRots],
                aatype: Optional[Union[torch.Tensor, np.ndarray]] = None,
                with_cb: bool = False,
                init_global_rots: Optional[torch.Tensor] = None,
                init_global_trans: Optional[torch.Tensor] = None,
                rot_repr_is_q: bool = False
                ):
    '''
    input shape: B x L x ...
    output shape: B x L x 4 x 3
    TODO: flexible length dim
    '''
    
    # TODO: make use of aatype

    if init_global_trans is not None: init_global_trans = init_global_trans.transpose(0, 1)
    
    if mode == to_bb_mode.Pep_GlobalRots_GlobalTrans:
        global_rots, global_trans = rots, trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter_via_rot_trans(global_rots, global_trans, rot_repr_is_q=rot_repr_is_q)
    
    elif mode == to_bb_mode.Pep_GlobalRots_IsoRots:
        global_rots = rots
        loc_ca_ia1_wrt_n_ia1 = trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter(global_rots.transpose(0, 1), loc_ca_ia1_wrt_n_ia1.transpose(0, 1), init_global_trans=init_global_trans, rot_repr_is_q=rot_repr_is_q).permute(2, 0, 1, 3)
    
    elif mode == to_bb_mode.Pep_RelativeRots_IsoRots:
        if rot_repr_is_q:
            if init_global_rots is None:
                init_global_rots = torch.tensor([[0., 0., 0., 1.]], dtype=rots.dtype, device=rots.device).expand(rots.shape[0], 1, 4)
            rrots = torch.cat((init_global_rots, rots), dim=1)
            global_rots = quat_cumprod(rrots, 1)
        else:
            global_rots = mat_cumops(rots, 1)
            global_rots = torch.cat((
                torch.eye(3, device=global_rots.device, dtype=global_rots.dtype).unsqueeze(0).expand(global_rots.shape[0], -1, -1).unsqueeze(1),
                global_rots), dim=1)
            if init_global_rots is not None:
                global_rots = init_global_rots @ global_rots
        loc_ca_ia1_wrt_n_ia1 = trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter(global_rots.transpose(0, 1), loc_ca_ia1_wrt_n_ia1.transpose(0, 1), init_global_trans=init_global_trans, rot_repr_is_q=rot_repr_is_q).permute(2, 0, 1, 3)
    
    elif mode == to_bb_mode.Res_GlobalRots_GlobalTrans:
        raise NotImplementedError('TODO.')

    else:
        raise NotImplementedError(f'Only support {to_bb_mode.__members__}')
    
    """
    if with_cb:
        # TODO: make it allow batch processing
        # TODO: simplify the code when it comes to to_bb_mode.Res_GlobalRots_GlobalTrans
        i = 0
        n_ca_c_frameone = SidechainFrame.from_W_3(*bb_coords[i, :, :3].transpose(0,1))
        n_ca_c_frameone_s_loc_cb = torch.tensor([CB_LOC.get(aa, CB_LOC['ALA']) for aa in aatype[i]], device=bb_coords.device, dtype=bb_coords.dtype).reshape(-1, 3)
        n_ca_c_frameone_s_W_cb = n_ca_c_frameone.to_W_pos_i(n_ca_c_frameone_s_loc_cb, ...)
        n_ca_c_frameone_s_W_cb = n_ca_c_frameone_s_W_cb[None]
        bb_coords = torch.cat((bb_coords, n_ca_c_frameone_s_W_cb[:, :, None]), dim=2)
    """
    
    return bb_coords # shape: B x L x 4 x 3