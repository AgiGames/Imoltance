import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple

def train_test_split_csv(data_path: str, random_state: int, test_size: float, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size >= 1:
        raise Exception("Test ratio must be less than 1.")
    
    data = pd.read_csv(data_path)
    train_data, test_data = train_test_split(
                                data,
                                test_size=test_size,
                                random_state=random_state,
                                shuffle=shuffle
                            )
    return train_data, test_data    