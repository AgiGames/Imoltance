import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

from typing import List
from collections import Counter

from .topology import compute_weighted_topological_indices

def get_weighted_ve_df(
    smiles_dataset_path: str,
    smiles_index_col: str,
    weights_dataset_path: str,
    weights_index_col: str,
    weights_col: str,
    topo_idxs_cols: List[str],
):
    weights_df = pd.read_csv(weights_dataset_path, index_col=weights_index_col)
    weights = {}
    for index, row in weights_df.iterrows():
        weights[index] = row[weights_col]
    
    indices = []
    smiles_df = pd.read_csv(smiles_dataset_path, index_col=smiles_index_col)
    n = len(smiles_df)
    
    for i, (idx, row) in enumerate(smiles_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(idx)
        if mol is None:
            continue

        ids_and_elements = {atom.GetIdx(): atom for atom in mol.GetAtoms()}
        
        indices.append(
            compute_weighted_topological_indices(
                ids_and_elements,
                weights,
                None,
                None
            )
        )
        print(f"{i} / {n}")
    
    indices_df = pd.DataFrame(indices, columns=topo_idxs_cols, index=smiles_df.index)
    return indices_df

def get_weighted_ev_df(
    smiles_dataset_path: str,
    smiles_index_col: str,
    weights_dataset_path: str,
    weights_index_col: str,
    weights_col: str,
    topo_idxs_cols: List[str],
):
    weights_df = pd.read_csv(weights_dataset_path, index_col=weights_index_col)
    weights = {}
    for index, row in weights_df.iterrows():
        weights[index] = row[weights_col]
    
    indices = []
    smiles_df = pd.read_csv(smiles_dataset_path, index_col=smiles_index_col)
    n = len(smiles_df)
    
    for i, (idx, row) in enumerate(smiles_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(idx)
        if mol is None:
            continue

        ids_and_elements = {atom.GetIdx(): atom for atom in mol.GetAtoms()}
        
        indices.append(
            compute_weighted_topological_indices(
                ids_and_elements,
                None,
                weights,
                mol
            )
        )
        print(f"{i} / {n}")
    
    indices_df = pd.DataFrame(indices, columns=topo_idxs_cols, index=smiles_df.index)
    return indices_df