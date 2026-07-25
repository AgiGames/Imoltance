import pandas as pd
from rdkit import Chem
from collections import defaultdict

def count_bonds(ids_and_elements: dict[int, Chem.rdchem.Atom], mol: Chem.rdchem.Mol):
    visited_edges = set()
    counts = defaultdict(int)
    for atom in ids_and_elements.values():
        for neigh_atom in atom.GetNeighbors():
            edge = tuple(sorted([atom.GetIdx(), neigh_atom.GetIdx()]))
            if edge in visited_edges: continue
            visited_edges.add(edge)
            bond_type = mol.GetBondBetweenAtoms(edge[0], edge[1])
            edge_names = tuple(sorted([atom.GetSymbol(), neigh_atom.GetSymbol()]))
            counts[f'{edge_names[0]}-{bond_type.GetBondType()}-{bond_type.GetStereo()}-{edge_names[1]}'] += 1
    return counts

def count_elements(ids_and_elements: dict[int, Chem.rdchem.Atom]):
    counts = defaultdict(int)
    for atom in ids_and_elements.values():
        key = f'{atom.GetSymbol()}-{atom.GetChiralTag()}'
        counts[key] += 1
    return counts

def get_bond_counts_df(dataset_path: str, smiles_col_name: str) -> pd.DataFrame:
    smiles_df = pd.read_csv(dataset_path)
    rows = []
    n = len(smiles_df)
    
    for i, (idx, row) in enumerate(smiles_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(row[smiles_col_name])
        ids_and_elements = {atom.GetIdx(): atom for atom in mol.GetAtoms()}
        
        counts = dict(count_bonds(ids_and_elements, mol))
        counts['Tm'] = row["Tm"]
        
        rows.append(counts)

        print(f"{i} / {n}")
    
    print(rows)
    final_count_df = pd.DataFrame(rows).fillna(0)
    return final_count_df

def get_element_counts_df(dataset_path: str, smiles_col_name: str) -> pd.DataFrame:
    smiles_df = pd.read_csv(dataset_path)
    rows = []
    n = len(smiles_df)
    
    for i, (idx, row) in enumerate(smiles_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(row[smiles_col_name])
        ids_and_elements = {atom.GetIdx(): atom for atom in mol.GetAtoms()}
        
        counts = dict(count_elements(ids_and_elements))
        counts['Tm'] = row["Tm"]
        rows.append(counts)

        print(f"{i} / {n}")
    
    final_count_df = pd.DataFrame(rows).fillna(0)
    return final_count_df