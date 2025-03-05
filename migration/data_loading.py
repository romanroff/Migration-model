import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy

def load_datasets(base_path: str = 'data') -> tuple:
    """Загрузка тренировочных и тестовых данных"""
    data_path = Path(base_path)
    
    # Загрузка данных
    with open(data_path / 'fold1_sample_dataframes_2022_11_02-02_54_03.pkl', "rb") as f:
        sample_train = pickle.load(f)
    
    with open(data_path / 'fold2_dataframes_2022_11_02-02_54_03.pkl', "rb") as f:
        test = pickle.load(f)
    
    with open(data_path / 'list_states_dataframes_2022_11_02-02_53_58.pkl', "rb") as f:
        list_states_dataframes = pickle.load(f)

    # Инициализация структур данных
    train_list_dataframes = sample_train
    test_list_dataframes = test
    
    return train_list_dataframes, test_list_dataframes, list_states_dataframes
