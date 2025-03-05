import numpy as np

def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred)**2))

def MRE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))  # +1e-6 для избежания деления на 0

def MLogRatio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))

def common_part_of_commuters(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.minimum(y_true, y_pred).sum() / y_true.sum()

def common_part_of_commuters_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.minimum(y_true, y_pred) / y_true)
