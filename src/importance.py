from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from typing import List

def get_importance(
        counts_dataset_path: str,
        target_cols: List[str]
    ) -> pd.DataFrame:
    bond_counts = pd.read_csv(counts_dataset_path)
    
    X_cols = bond_counts.columns.tolist()
    for target_col in target_cols:
        X_cols.remove(target_col)
        
    X = bond_counts[X_cols].to_numpy()
    Y = bond_counts[target_cols].to_numpy()
    
    model = LinearRegression()
    model.fit(X, Y)
    coeffs = model.coef_.flatten()
    min_w = coeffs.min()
    max_w = coeffs.max()
    if max_w - min_w == 0:
        normalized = np.zeros_like(coeffs)
    else:
        normalized = (coeffs - min_w) / (max_w - min_w)

    importance_df = pd.DataFrame({
        "index": X_cols,
        "weight": coeffs,
        "normalized_weight": normalized
    })
    importance_df = importance_df.sort_values(by="weight", ascending=False)
    
    print("\nTop 10 positive influences:")
    print(importance_df.head(10))

    print("\nTop 10 negative influences:")
    print(importance_df.tail(10))
    
    return importance_df