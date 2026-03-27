import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH, FEATURE_COLS

def make_predictions(model, features: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Генерация предсказаний и объединение с исходными данными"""
    feature_cols = feature_cols or FEATURE_COLS
    features = features.reset_index(drop=True)
    predictions = np.exp(model.predict(features[feature_cols]))
    return pd.concat([features, pd.DataFrame(predictions, columns=['total_pop_flow'])], axis=1)

def load_model(model_path: str | None = None) -> object:
    """Загрузка обученной модели"""
    return joblib.load(model_path or MODEL_PATH)
