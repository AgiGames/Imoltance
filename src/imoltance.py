import pandas as pd

import os
from typing import List

from importance import get_importance
from helper import train_test_split_csv
from features import get_weighted_ev_df, get_weighted_ve_df
from counter import get_bond_counts_df, get_element_counts_df

class Imoltance:
    def __init__(self,
                 dataset_path: str,
                 smiles_col_name: str, 
                 target_cols: List[str], 
                 random_state: int, 
                 test_size=0.3,
                 dump_path='imoltance_dump'):
        self.smiles_col_name = smiles_col_name
        self.target_cols = target_cols
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.test_size = test_size
        os.makedirs(dump_path, exist_ok='True')
        self.dump_path = dump_path
        
    def save_counts(self, data_path: str, bond_counts_out_path: str, element_counts_out_path: str):
        bond_counts_df = get_bond_counts_df(data_path, self.smiles_col_name)
        bond_counts_df.to_csv(bond_counts_out_path, index=False)
        
        element_counts_df = get_element_counts_df(data_path, self.smiles_col_name)
        element_counts_df.to_csv(element_counts_out_path, index=False)
    
    def save_importance(self,
                        bond_counts_path: str,
                        element_counts_path: str,
                        bond_importance_out_path: str,
                        element_importance_out_path: str):
        get_importance(bond_counts_path, self.target_cols).to_csv(bond_importance_out_path, index=False)
        get_importance(element_counts_path, self.target_cols).to_csv(element_importance_out_path, index=False)
    
    def save_features(
        self,
        data_path: str,
        bond_importance_path: str,
        element_importance_path: str,
        out_features_path: str
    ):  
        wve_df = get_weighted_ve_df(
            data_path,
            self.smiles_col_name,
            element_importance_path,
            'index',
            'normalized_weight',
            ["M1_alpha_ve", "M1_beta_ve", "M2_ve", "R_ve", "ABC_ve", "GA_ve",
            "H_ve", "X_ve", "HM1_ve", "HM2_ve", "F_ve", "F1_ve", "ReZG3_ve",
            "AG_ve", "ISI_ve", "T_ve", "ID_ve", "ZD_ve", "mM1_ve"]
        )
        
        wev_df = get_weighted_ev_df(
            data_path,
            self.smiles_col_name,
            bond_importance_path,
            'index',
            'normalized_weight',
            ["T_ev", "M_ev", "F_ev", "mM_ev", "ID_ev", "R_ev", "RR_ev"]
        )
        
        target = pd.read_csv('Bradley_Melting_Point_Dataset.csv', index_col='SMILES')[['Tm']]
        target = target.loc[wve_df.index]
        
        final_df = pd.concat([wve_df, wev_df, target], axis=1)
        invalid = {"Br", "C", "F", "O", "S", "N"}
        final_df = final_df[~final_df.index.isin(invalid)]
        final_df.to_csv(out_features_path)
    
    def run(self):
            train_dataset, test_dataset = train_test_split_csv(
                self.dataset_path,
                self.random_state,
                self.test_size,
                True
            )
            
            train_dataset_path = os.path.join(self.dump_path, 'train_dataset.csv')
            train_dataset.to_csv(train_dataset_path)
            test_dataset_path = os.path.join(self.dump_path, 'test_dataset.csv')
            test_dataset.to_csv(test_dataset_path)
            
            train_bond_counts_path = os.path.join(self.dump_path, 'train_bond_counts.csv')
            train_element_counts_path = os.path.join(self.dump_path, 'train_element_counts.csv')
            self.save_counts(
                train_dataset_path,
                train_bond_counts_path,
                train_element_counts_path
            )
            
            train_bond_importance_path = os.path.join(self.dump_path, 'train_bond_importance.csv')
            train_element_importance_path = os.path.join(self.dump_path, 'train_element_importance.csv')
            self.save_importance(
                train_bond_counts_path,
                train_element_counts_path,
                train_bond_importance_path,
                train_element_importance_path
            )
            
            train_features_path = os.path.join(self.dump_path, 'train_features.csv')
            test_features_path = os.path.join(self.dump_path, 'test_features.csv')
            
            self.save_features(
                train_dataset_path,
                train_bond_importance_path,
                train_element_importance_path,
                train_features_path
            )
            
            self.save_features(
                test_dataset_path,
                train_bond_importance_path,
                train_element_importance_path,
                test_features_path
            )