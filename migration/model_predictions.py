import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH, FEATURE_COLS

def make_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Генерация предсказаний и объединение с исходными данными"""
    predictions = np.exp(model.predict(features[FEATURE_COLS]))
    return pd.concat([features, pd.DataFrame(predictions, columns=['total_pop_flow'])], axis=1)

def load_model() -> object:
    """Загрузка обученной модели"""
    return joblib.load(MODEL_PATH)
