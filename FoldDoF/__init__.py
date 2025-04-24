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
# @Last Modified: 2025-04-24 10:55:35 am
import torch
import numpy as np
from typing import Union, Optional, Literal
from enum import Enum
from .frame import PeptideUnitFrame
from .utils import mat_cumops


class to_bb_mode(Enum):
    Pep_GlobalRots_GlobalTrans = 'peptide_frame global_rots:L+1 trans:L+1'
    Pep_GlobalRots_IsoRots = 'peptide_frame global_rots:L+1 loc_ca_ia1_wrt_n_ia1:L'
    Pep_RelativeRots_IsoRots = 'peptide_frame relative_rots:L loc_ca_ia1_wrt_n_ia1:L'
    Res_GlobalRots_GlobalTrans = 'residue_frame global_rots:L trans:L'


def to_backbone(rots: torch.Tensor,
                trans_or_loc_ca_ia1_wrt_n_ia1: torch.Tensor,
                mode: Literal[to_bb_mode.Pep_GlobalRots_GlobalTrans, to_bb_mode.Pep_GlobalRots_IsoRots, to_bb_mode.Pep_RelativeRots_IsoRots],
                aatype: Optional[Union[torch.Tensor, np.ndarray]] = None):
    '''
    input shape: B x L x ...
    output shape: B x L x 4 x 3
    '''
    
    # TODO: make use of aatype
    
    if mode == to_bb_mode.Pep_GlobalRots_GlobalTrans:
        global_rots, global_trans = rots, trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter_via_rotmat_trans(global_rots, global_trans)
    
    elif mode == to_bb_mode.Pep_GlobalRots_IsoRots:
        global_rots = rots
        loc_ca_ia1_wrt_n_ia1 = trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter_via_rotmat(global_rots.transpose(0, 1), loc_ca_ia1_wrt_n_ia1.transpose(0, 1)).permute(2, 0, 1, 3)
    
    elif mode == to_bb_mode.Pep_RelativeRots_IsoRots:
        global_rots = mat_cumops(rots.clone(), 1)
        global_rots = torch.cat((
            torch.eye(3, device=global_rots.device, dtype=global_rots.dtype).unsqueeze(0).expand(global_rots.shape[0], -1, -1).unsqueeze(1),
            global_rots), dim=1)
        loc_ca_ia1_wrt_n_ia1 = trans_or_loc_ca_ia1_wrt_n_ia1
        bb_coords = PeptideUnitFrame.to_W_batch_avg_backbone_addter_via_rotmat(global_rots.transpose(0, 1), loc_ca_ia1_wrt_n_ia1.transpose(0, 1)).permute(2, 0, 1, 3)
    
    elif mode == to_bb_mode.Res_GlobalRots_GlobalTrans:
        raise NotImplementedError('TODO.')

    else:
        raise NotImplementedError(f'Only support {to_bb_mode.__members__}')
    
    return bb_coords # shape: B x L x 4 x 3