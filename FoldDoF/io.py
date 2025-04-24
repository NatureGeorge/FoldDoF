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

# @Created Date: 2022-04-08 09:03:28 pm
# @Filename: io.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-04-24 11:04:20 am
from typing import Sequence
import gemmi
import torch
from collections import defaultdict


def wrap_res_get_atom(res, atom:str):
    if atom in res:
        return (res[atom]#[0].pos.tolist()
               , True)
    else:
        return [0., 0., 0.], False


def atomgroup_get(self, key, alt_key):
    '''helper function for `gemmi`'''
    try:
        return self[key]
    except RuntimeError:
        return self[alt_key]


def wrap_res_atomgroup(cur_res) -> dict:
    '''helper function for `gemmi`'''
    atom_altloc_map = defaultdict(set)
    for atom in cur_res:
        if atom.name.startswith('H'): continue
        atom_altloc_map[atom.altloc].add(atom.name)
    for key in atom_altloc_map.keys()-{'\x00'}:
        atom_altloc_map[key] = atom_altloc_map[key] | atom_altloc_map['\x00']
    cur_atomgroup_list = []
    cur_atomgroup_list_size = []
    for key, val in atom_altloc_map.items():
        if len(val):
            #cur_atomgroup_list.append(dict([(atom, [atomgroup_get(cur_res[atom], key, '\x00')]) for atom in val]))
            cur_atomgroup_list.append(dict([(atom, atomgroup_get(cur_res[atom], key, '\x00').pos.tolist()) for atom in val]))
            cur_atomgroup_list_size.append(len(cur_atomgroup_list[-1]))
    return cur_atomgroup_list[cur_atomgroup_list_size.index(max(cur_atomgroup_list_size))]


def get_coords_with_mask(chain: Sequence[gemmi.Residue], atoms: Sequence[str] = ('CA',), dtype=torch.float, numres_first: bool = False, **kwargs):
    coords = []
    masks = []
    for res_ in chain:
        res = wrap_res_atomgroup(res_)
        cur_coord, cur_mask = zip(*(wrap_res_get_atom(res, atom) if not (atom == 'CB' and res_.name == 'GLY') else wrap_res_get_atom(res, 'CA') for atom in atoms))
        coords.append(cur_coord)
        masks.append(cur_mask)
    coords = torch.as_tensor(coords, dtype=dtype, **kwargs)
    masks = torch.as_tensor(masks, dtype=torch.bool, **kwargs)
    if numres_first:
        return coords, masks
    else:
        return coords.transpose(0, 1), masks.transpose(0, 1)


def savebb2pdb(seq: Sequence[str], backbone_coords: torch.Tensor, output_path: str, with_cb: bool = False):
    """
    Saving Backbone(N-Cα-CO) Cartesian coordinates into PDB file

    Args:
        seq (Sequence[str]): protein three-letter-code sequence
        backbone_coords (torch.Tensor): shape(NumRes x NumAtom(i.e. 4) x 3)
        output_path (str): the output file path to save
    """
    serial = 1
    atom_names = 'N', 'CA', 'C', 'O'
    with open(output_path, 'wt') as handle:
        for aa_idx in range(backbone_coords.shape[0]):
            aa_name = seq[aa_idx]
            handle.write(f'''ATOM  {serial:>5}  {atom_names[0]:<4}{aa_name} A{aa_idx+1:>4}    {backbone_coords[aa_idx, 0, 0]:>8.3f}{backbone_coords[aa_idx, 0, 1]:>8.3f}{backbone_coords[aa_idx, 0, 2]:>8.3f}  1.00 20.00           N
ATOM  {serial+1:>5}  {atom_names[1]:<4}{aa_name} A{aa_idx+1:>4}    {backbone_coords[aa_idx, 1, 0]:>8.3f}{backbone_coords[aa_idx, 1, 1]:>8.3f}{backbone_coords[aa_idx, 1, 2]:>8.3f}  1.00 20.00           C
ATOM  {serial+2:>5}  {atom_names[2]:<4}{aa_name} A{aa_idx+1:>4}    {backbone_coords[aa_idx, 2, 0]:>8.3f}{backbone_coords[aa_idx, 2, 1]:>8.3f}{backbone_coords[aa_idx, 2, 2]:>8.3f}  1.00 20.00           C
ATOM  {serial+3:>5}  {atom_names[3]:<4}{aa_name} A{aa_idx+1:>4}    {backbone_coords[aa_idx, 3, 0]:>8.3f}{backbone_coords[aa_idx, 3, 1]:>8.3f}{backbone_coords[aa_idx, 3, 2]:>8.3f}  1.00 20.00           O\n''')
            serial += 4
            if with_cb and aa_name != 'GLY':
                cb_name = 'CB'
                handle.write(f'ATOM  {serial:>5}  {cb_name:<4}{aa_name} A{aa_idx+1:>4}    {backbone_coords[aa_idx, 4, 0]:>8.3f}{backbone_coords[aa_idx, 4, 1]:>8.3f}{backbone_coords[aa_idx, 4, 2]:>8.3f}  1.00 20.00           C\n')
                serial += 1
        handle.write('END\n')