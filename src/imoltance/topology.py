from collections import deque
from rdkit import Chem
import math
import pandas as pd

def dve(v: Chem.rdchem.Atom) -> int:
    neighborhood = set([v] + list(v.GetNeighbors()))

    edges = set()
    for u in neighborhood:
        for w in list(u.GetNeighbors()):
            # if w in neighborhood:
            edge = tuple(sorted((u.GetIdx(), w.GetIdx())))
            edges.add(edge)

    return len(edges)

def compute_dve_map(ids_and_elements: dict[int, Chem.rdchem.Atom]):
    return {
        i: dve(atom)
        for i, atom in ids_and_elements.items()
    }
    
def compute_dev_map(ids_and_elements):
    dev_map = {}

    for u in ids_and_elements.values():
        for v in u.GetNeighbors():
            edge = tuple(sorted((u.GetIdx(), v.GetIdx())))
            if edge in dev_map:
                continue

            # Closed neighborhoods
            Nu = set([u.GetIdx()] + [n.GetIdx() for n in u.GetNeighbors()])
            Nv = set([v.GetIdx()] + [n.GetIdx() for n in v.GetNeighbors()])

            # Union
            union_nodes = Nu.union(Nv)

            dev_map[edge] = len(union_nodes)

    return dev_map

def compute_weighted_ve_topological_indices(
        ids_and_elements: dict[int, Chem.rdchem.Atom],
        dve_map: dict[int, int],
        weights: dict[str, float]
    ):
    visited_edges = set()
    M1_alpha = M1_beta = M2 = R = ABC = GA = H = X = HM1 = HM2 = F = F1 = ReZG3 = AG = ISI = T = ID = ZD = mM1 = 0
    
    for u in ids_and_elements.values(): # Iterate through all elements in the graph
        for v in u.GetNeighbors(): # For each connection of the current element
            edge = tuple(sorted((u.GetIdx(), v.GetIdx())))
            if edge in visited_edges:
                continue
            
            key_u = f'{u.GetSymbol()}-{u.GetChiralTag()}'
            key_v = f'{v.GetSymbol()}-{v.GetChiralTag()}'
            
            if key_u not in weights or key_v not in weights:
                continue
            
            du = dve_map[u.GetIdx()] * weights[key_u]
            dv = dve_map[v.GetIdx()] * weights[key_v]
            
            M1_alpha += (dv * dv)
            M1_beta += (du + dv)
            M2 += (du * dv)
            HM1 += ((du + dv) ** 2)
            HM2 += ((du * dv) ** 2)
            F += (du ** 2 + dv ** 2)
            F1 += (dv ** 3)
            ReZG3 += ((du * dv) * (du + dv))
            T += (du)

            if dv != 0:
                ID += (1 / dv)
                ZD += (1 / (dv ** 0.5))
                mM1 += (1 / (dv * dv))

            if du != 0 and dv != 0:
                R += ((du * dv) ** -0.5)
                if (du + dv - 2) >= 0:
                    ABC += math.sqrt((du + dv - 2) / (du * dv))
                GA += ((2 * math.sqrt(du * dv)) / (du + dv)) if (du + dv) != 0 else 0
                AG += ((du + dv) / (2 * math.sqrt(du * dv)))
                ISI += ((du * dv) / (du + dv)) if (du + dv) != 0 else 0

            if (du + dv) != 0:
                H += (2 / (du + dv))
                X += ((du + dv) ** -0.5)
            
    return [M1_alpha, M1_beta, M2, R, ABC, GA, H, X, HM1, HM2, F, F1, ReZG3, AG, ISI, T, ID, ZD, mM1]

def compute_ve_topological_indices(
        ids_and_elements: dict[int, Chem.rdchem.Atom],
        dve_map: dict[int, int]
    ):
    visited_edges = set()
    M1_alpha = M1_beta = M2 = R = ABC = GA = H = X = HM1 = HM2 = F = F1 = ReZG3 = AG = ISI = T = ID = ZD = mM1 = 0
    
    for u in ids_and_elements.values(): # Iterate through all elements in the graph
        for v in u.GetNeighbors(): # For each connection of the current element
            edge = tuple(sorted((u.GetIdx(), v.GetIdx())))
            if edge in visited_edges:
                continue

            du = dve_map[u.GetIdx()]
            dv = dve_map[v.GetIdx()]
            
            M1_alpha += (dv * dv)
            M1_beta += (du + dv)
            M2 += (du * dv)
            R += ((du * dv) ** -0.5)
            ABC += math.sqrt((du + dv - 2) / (du * dv))
            GA += ((2 * math.sqrt(du * dv)) / (du + dv))
            H += (2 / (du + dv))
            X += ((du + dv) ** -0.5)
            HM1 += ((du + dv) ** 2)
            HM2 += ((du * dv) ** 2)
            F += (du ** 2 + dv ** 2)
            F1 += (dv ** 3)
            ReZG3 += ((du * dv) * (du + dv))
            AG += ((du + dv) / (2 * math.sqrt(du * dv)))
            ISI += ((du * dv) / (du + dv))
            T += (du)
            ID += (1 / dv)
            ZD += (1 / (dv ** 0.5))
            mM1 += (1 / (dv * dv))
            
    return [M1_alpha, M1_beta, M2, R, ABC, GA, H, X, HM1, HM2, F, F1, ReZG3, AG, ISI, T, ID, ZD, mM1]

def compute_weighted_ev_topological_indices(
        ids_and_elements: dict[int, Chem.rdchem.Atom],
        dev_map: dict[int, int],
        weights: dict[str, float],
        mol: Chem.rdchem.Mol
    ):
    visited_edges = set()
    
    T = M = F = mM = ID = R = RR = 0
    
    for u in ids_and_elements.values():
        for v in u.GetNeighbors():
            edge = tuple(sorted((u.GetIdx(), v.GetIdx())))
            if edge in visited_edges:
                continue
            edge_symbs = tuple(sorted((u.GetSymbol(), v.GetSymbol())))
            bond_type = mol.GetBondBetweenAtoms(edge[0], edge[1])
            key = f'{edge_symbs[0]}-{bond_type.GetBondType()}-{bond_type.GetStereo()}-{edge_symbs[1]}'
            if key not in weights:
                continue
            
            dev = dev_map[edge] * weights[key]
            
            T += (dev)
            M += (dev * dev)
            F += (dev * dev * dev)
            if dev > 0:
                mM += (1 / (dev * dev))
                ID += (1 / dev)
                R += (dev ** -0.5)
            RR += (dev ** 0.5)
    
    return [T, M, F, mM, ID, R, RR]

def compute_ev_topological_indices(ids_and_elements: dict[int, Chem.rdchem.Atom], dev_map: dict[int, int]):
    visited_edges = set()
    
    T = M = F = mM = ID = R = RR = 0
    
    for u in ids_and_elements.values():
        for v in u.GetNeighbors():
            edge = tuple(sorted((u.GetIdx(), v.GetIdx())))
            if edge in visited_edges:
                continue
            
            dev = dev_map[edge]
            
            T += (dev)
            M += (dev * dev)
            F += (dev * dev * dev)
            mM += (1 / (dev * dev))
            ID += (1 / dev)
            R += (dev ** -0.5)
            RR += (dev ** 0.5)
    
    return [T, M, F, mM, ID, R, RR]

def compute_topological_indices(
        ids_and_elements: dict[int, Chem.rdchem.Atom],
        include_ve: bool,
        include_ev: bool,
    ):
    result = []
    if include_ve:
        dve_map = compute_dve_map(ids_and_elements)
        result = result + compute_ve_topological_indices(ids_and_elements, dve_map)
    if include_ev:
        dev_map = compute_dev_map(ids_and_elements)
        result = result + compute_ev_topological_indices(ids_and_elements, dev_map)
    return result

def compute_weighted_topological_indices(
        ids_and_elements: dict[int, Chem.rdchem.Atom],
        ve_weights: dict[str, float],
        ev_weights: dict[str, float],
        mol: Chem.rdchem.Mol
    ):
    result = []
    if ve_weights is not None:
        dve_map = compute_dve_map(ids_and_elements)
        result = result + compute_weighted_ve_topological_indices(ids_and_elements, dve_map, ve_weights)
    if ev_weights is not None:
        dev_map = compute_dev_map(ids_and_elements)
        result = result + compute_weighted_ev_topological_indices(ids_and_elements, dev_map, ev_weights, mol)
    return result