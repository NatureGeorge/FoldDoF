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
# @Last Modified: 2026-01-05 09:58:52 pm
from typing import Sequence, Optional, Literal
import gemmi
import torch
from collections import defaultdict
from pathlib import Path
from pdbecif.mmcif_io import CifFileWriter
from .data import AA_SIDECHAIN_ATOMS, AA_THREE2ONE_MOD


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


def get_sidechain_coords(chain: Sequence[gemmi.Residue], length: Optional[int] = None, dtype=torch.float, numres_first: bool = True, **kwargs):
    """
    Returns:
        sidechain_coords (torch.Tensor): shape(NumRes x NumAtom(i.e. 10) x 3)
        sidechain_coords_mask (torch.Tensor): shape(NumRes x NumAtom(i.e. 10) x 3)
    """
    if length is None: length = len(chain)
    sidechain_coords = torch.zeros((length, 10, 3), dtype=dtype, **kwargs)
    sidechain_coords_mask = torch.zeros_like(sidechain_coords[:,:,0], dtype=torch.bool)
    for res_idx, res_ in enumerate(chain):
        res = wrap_res_atomgroup(res_)
        for atom_idx, atom_name in enumerate(AA_SIDECHAIN_ATOMS.get(res_.name, 'GLY')):
            if atom_name in res:
                sidechain_coords[res_idx, atom_idx] = torch.tensor(res[atom_name], device=sidechain_coords.device)  # [0].pos.tolist()
                sidechain_coords_mask[res_idx, atom_idx] = True
    if numres_first:
        return sidechain_coords, sidechain_coords_mask
    else:
        return sidechain_coords.transpose(0, 1), sidechain_coords_mask.transpose(0, 1)


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


def savebb2pdb(
    seq: Sequence[str], 
    backbone_coords: torch.Tensor, 
    output_path: str, 
    with_cb: bool = False,
    mol_type: Literal['protein', 'rna'] = 'protein',
    chain_id: str = 'A'
):
    """
    Saving backbone Cartesian coordinates into PDB file.
    
    For proteins: N-Cα-CO atoms (and optionally CB)
    For RNA: P-O5'-C5'-C4'-C3'
            /\
         OP1  OP2
    
    Args:
        seq (Sequence[str]): 
            For proteins: three-letter-code sequence (e.g., "ALA", "GLY")
            For RNA: one-letter-code sequence (e.g., "A", "U", "G", "C")
        backbone_coords (torch.Tensor): 
            For proteins: shape(NumRes x 4 x 3) for [N, CA, C, O] 
                          or shape(NumRes x 5 x 3) if with_cb=True
            For RNA: shape(NumRes x 8 x 3) for [O3', P, O5', C5', C4', C3', OP1, OP2]
        output_path (str): the output file path to save
        with_cb (bool): for proteins only, whether to include CB atoms
        mol_type (str): 'protein' or 'rna'
        chain_id (str): chain identifier (default: 'A')
    """
    serial = 1
    residue_number = 1
    
    with open(output_path, 'wt') as handle:
        if mol_type == 'protein':
            atom_names = ['N', 'CA', 'C', 'O']
            atom_elements = ['N', 'C', 'C', 'O']
            
            for aa_idx in range(backbone_coords.shape[0]):
                aa_name = seq[aa_idx]
                for atom_idx in range(4):
                    x, y, z = backbone_coords[aa_idx, atom_idx].tolist()
                    handle.write(
                        f"ATOM  {serial:>5}  {atom_names[atom_idx]:<3} {aa_name:>3} {chain_id}{residue_number:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00           {atom_elements[atom_idx]:>2}\n"
                    )
                    serial += 1
                if with_cb and aa_name != 'GLY' and backbone_coords.shape[1] > 4:
                    x, y, z = backbone_coords[aa_idx, 4].tolist()
                    handle.write(
                        f"ATOM  {serial:>5}  CB  {aa_name:>3} {chain_id}{residue_number:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00           C\n"
                    )
                    serial += 1
                
                residue_number += 1
                
        elif mol_type == 'rna':
            atom_names = ["O3'", "P", "O5'", "C5'", "C4'", "C3'", "OP1", "OP2"]
            atom_elements = ['O', 'P', 'O', 'C', 'C', 'C', 'O', 'O']
            
            rna_code_map = {
                'A': '  A', 'U': '  U', 'G': '  G', 'C': '  C',
                'a': '  A', 'u': '  U', 'g': '  G', 'c': '  C'
            }
            
            for nt_idx in range(backbone_coords.shape[0]):
                nt_1letter = seq[nt_idx]
                nt_name = rna_code_map.get(nt_1letter, f' {nt_1letter:<2}')
                for atom_idx in range(8):
                    x, y, z = backbone_coords[nt_idx, atom_idx].tolist()
                    atom_name = atom_names[atom_idx]
                    padded_atom_name = atom_name.rjust(3) if "'" in atom_name else f" {atom_name:<2}"
                    
                    handle.write(
                        f"ATOM  {serial:>5} {padded_atom_name} {nt_name} {chain_id}{residue_number:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00           {atom_elements[atom_idx]:>2}\n"
                    )
                    serial += 1
                
                residue_number += 1
        else:
            raise ValueError(f"Unknown molecule type: {mol_type}. Must be 'protein' or 'rna'")
        
        handle.write('END\n')


def save2cif(seq: Sequence[str], backbone_coords: torch.Tensor, output_path: str, sidechain_coords: Optional[torch.Tensor] = None, sidechain_coords_mask: Optional[torch.Tensor] = None, decimals=3, B_iso_or_equiv: Optional[torch.Tensor] = None, **kwargs):
    """
    Saving Backbone(N-Cα-CO) and side-chain Cartesian coordinates into mmCIF file

    NOTE: current implementation does not allow any heavy atom of the backbone to be missed
    (although it is easy to fix), -> TODO
    but tolerant of missing side-chain heavy atoms

    TODO: deal with OXT

    Args:
        seq (Sequence[str]): protein three-letter-code sequence
        backbone_coords (torch.Tensor): shape(NumRes x NumAtom(i.e. 4) x 3)
        sidechain_coords (Optional[torch.Tensor]): shape(NumRes x NumAtom(i.e. 10) x 3)
        sidechain_coords_mask (Optional[torch.Tensor]): shape(NumRes x NumAtom(i.e. 10) x 3)
        output_path (str): the output file path to save
        decimals (int, optional): Defaults to 3.
    """
    protein_length = len(seq)
    oneletter_seq = ''.join(AA_THREE2ONE_MOD[aa] if aa in AA_SIDECHAIN_ATOMS else f"({aa})" for aa in seq)
    oneletter_seq_can = ''.join(AA_THREE2ONE_MOD.get(aa, 'X') for aa in seq)
    assert protein_length == backbone_coords.shape[0]
    backbone_coords = backbone_coords.detach().cpu().numpy()
    seq_num = list(range(1, protein_length+1))
    mon_id = list(seq)
    cif_dict = {        
        '_entity': {
            'id': 1,
            'type': 'polymer'
        },
        '_entity_poly_seq': {
            'entity_id': [1]*protein_length,
            'hetero': ['n']*protein_length,
            'mon_id': mon_id,
            'num': seq_num,
        },
        '_entity_poly': {
            'entity_id': '1',
            'type': 'polypeptide(L)',
            'nstd_linkage': 'no',
            'nstd_monomer': 'no',
            'pdbx_seq_one_letter_code': oneletter_seq,
            'pdbx_seq_one_letter_code_can': oneletter_seq_can,
            'pdbx_strand_id': 'A',
            'pdbx_target_identifier': '?'
        },
        '_pdbx_poly_seq_scheme': {  # TODO: deal with missing residues
            'asym_id': ['A']*protein_length,
            'auth_seq_num': seq_num,
            'entity_id': [1]*protein_length,
            'hetero': ['n']*protein_length,
            'mon_id': mon_id,
            'pdb_ins_code': ['.']*protein_length,
            'pdb_mon_id': mon_id,
            'pdb_seq_num': seq_num,
            'pdb_strand_id': ['A']*protein_length,
            'seq_id': seq_num
        },
        '_atom_site': {
            'type_symbol': [],
            'label_atom_id': [],
            'label_comp_id': [],
            'label_seq_id': [],
            'Cartn_x': [],
            'Cartn_y': [],
            'Cartn_z': [],
        }
    }
    if sidechain_coords is None:
        for idx in range(protein_length):
            cif_dict['_atom_site']['type_symbol'].extend(['N', 'C', 'C', 'O'])
            cif_dict['_atom_site']['label_atom_id'].extend(['N', 'CA', 'C', 'O'])
            cif_dict['_atom_site']['label_comp_id'].extend([seq[idx]]*4)
            cif_dict['_atom_site']['label_seq_id'].extend([idx+1]*4)
            cur_bb_coords_x, cur_bb_coords_y, cur_bb_coords_z = backbone_coords[idx].transpose(-1, -2).round(decimals).astype(str).tolist()
            cif_dict['_atom_site']['Cartn_x'].extend(cur_bb_coords_x)
            cif_dict['_atom_site']['Cartn_y'].extend(cur_bb_coords_y)
            cif_dict['_atom_site']['Cartn_z'].extend(cur_bb_coords_z)
    else:
        sidechain_coords = sidechain_coords.detach().numpy()
        sidechain_coords_mask = sidechain_coords_mask.detach().numpy()
        assert protein_length == sidechain_coords.shape[0] == sidechain_coords_mask.shape[0]
        for idx in range(protein_length):
            cur_bb_coords_x, cur_bb_coords_y, cur_bb_coords_z = backbone_coords[idx].transpose(-1, -2).round(decimals).astype(str).tolist()
            aa = seq[idx]
            cur_mask = sidechain_coords_mask[idx]
            cur_side_coords_x, cur_side_coords_y, cur_side_coords_z = sidechain_coords[idx][cur_mask].transpose(-1, -2).round(decimals).astype(str).tolist()
            cur_atoms = [atom_i for atom_i ,mask_i in zip(AA_SIDECHAIN_ATOMS[aa], cur_mask) if mask_i]  # NOTE: this would allow missing side-chain heavy atoms
            #cur_atoms = AA_SIDECHAIN_ATOMS[aa]
            cur_atomtypes = [atom_i[0] if atom_i != 'SE' else 'SE' for atom_i in cur_atoms]
            cur_side_atom_num = len(cur_side_coords_x)
            assert cur_side_atom_num == len(cur_atoms)
            cif_dict['_atom_site']['type_symbol'].extend(['N', 'C', 'C', 'O']+cur_atomtypes)
            cif_dict['_atom_site']['label_atom_id'].extend(['N', 'CA', 'C', 'O']+cur_atoms)
            cif_dict['_atom_site']['label_comp_id'].extend([aa]*(4+cur_side_atom_num))
            cif_dict['_atom_site']['label_seq_id'].extend([idx+1]*(4+cur_side_atom_num))
            cif_dict['_atom_site']['Cartn_x'].extend(cur_bb_coords_x+cur_side_coords_x)
            cif_dict['_atom_site']['Cartn_y'].extend(cur_bb_coords_y+cur_side_coords_y)
            cif_dict['_atom_site']['Cartn_z'].extend(cur_bb_coords_z+cur_side_coords_z)
    num_lines = len(cif_dict['_atom_site']['Cartn_x'])
    cif_dict['_atom_site']['group_PDB'] = ['ATOM']*num_lines
    cif_dict['_atom_site']['id'] = list(range(1, num_lines+1))
    cif_dict['_atom_site']['label_alt_id'] = ['.']*num_lines
    cif_dict['_atom_site']['label_asym_id'] = ['A']*num_lines
    cif_dict['_atom_site']['label_entity_id'] = [1]*num_lines
    cif_dict['_atom_site']['pdbx_PDB_ins_code'] = ['?']*num_lines
    cif_dict['_atom_site']['occupancy'] = ['1']*num_lines
    if B_iso_or_equiv is None:
        cif_dict['_atom_site']['B_iso_or_equiv'] = ['0']*num_lines
    else:
        assert B_iso_or_equiv.shape[0] == num_lines, (B_iso_or_equiv.shape[0], num_lines)
        cif_dict['_atom_site']['B_iso_or_equiv'] = B_iso_or_equiv.detach().numpy().round(decimals).astype(str).tolist()
    cif_dict['_atom_site']['pdbx_formal_charge'] = ['.']*num_lines
    cif_dict['_atom_site']['pdbx_PDB_model_num'] = [1]*num_lines
    cif_dict['_atom_site']['auth_atom_id'] = cif_dict['_atom_site']['label_atom_id']
    cif_dict['_atom_site']['auth_comp_id'] = cif_dict['_atom_site']['label_comp_id']
    cif_dict['_atom_site']['auth_seq_id'] = cif_dict['_atom_site']['label_seq_id']
    cif_dict['_atom_site']['auth_asym_id'] = cif_dict['_atom_site']['label_asym_id']
    CifFileWriter(output_path, **kwargs).write(cif_dict)


def merge_pdb(*files, output_path, assert_func = None, model_format='PDB', unlink: bool = False):
    sts = [gemmi.read_structure(file) for file in files]
    if assert_func is not None:
        try:
            assert_func(sts)
        except AssertionError as e:
            print(files, e)
            return
    global_counter = len(sts[0])
    for st in sts[1:]:
        for model_idx in range(len(st)):
            global_counter += 1
            if hasattr(st[model_idx], 'name'):
                st[model_idx].name = str(global_counter)
            else:
                st[model_idx].num = global_counter
            sts[0].add_model(st[model_idx])
    if model_format == 'MMCIF':
        doc = sts[0].make_mmcif_document()
        doc.write_file(output_path)
    else:
        sts[0].write_pdb(output_path)
    if unlink:
        for file in files: Path(file).unlink()
