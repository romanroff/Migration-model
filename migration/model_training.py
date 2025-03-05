import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from metrics import RMSE, MRE, MLogRatio, common_part_of_commuters, common_part_of_commuters_accuracy

def train_and_save_model(train_data: dict, test_data: dict, model_path: str = "models/RF/rf_model.joblib"):
    """Обучение модели и сохранение результатов"""
    # Объединение данных
    all_train_data = pd.concat(train_data.values())
    
    # Подготовка фичей и таргета
    train_labels = np.log(np.array(all_train_data['total_pop_flow']))
    train_features = np.array(all_train_data[['d','m_o','m_d']])
    
    # Обучение модели
    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, verbose=1)
    rf.fit(train_features, train_labels)

    from pathlib import Path
    Path("models/RF").mkdir(parents=True, exist_ok=True)
    
    # Сохранение модели
    joblib.dump(rf, model_path)
    
    # Оценка качества
    results = evaluate_model(rf, test_data)
    
    return rf, results

def evaluate_model(model, test_data: dict) -> pd.DataFrame:
    """Оценка модели на тестовых данных"""
    res = {}
    
    for key, df in test_data.items():
        test_labels = np.array(df['total_pop_flow'])
        test_features = np.array(df[['d','m_o','m_d']])
        predictions = np.exp(model.predict(test_features))

        res[key] = [
            common_part_of_commuters(test_labels, predictions),
            common_part_of_commuters_accuracy(test_labels, predictions),
            RMSE(test_labels, predictions),
            MRE(test_labels, predictions),
            MLogRatio(test_labels, predictions),
            '-'
        ]
    
    return pd.DataFrame(
        res, 
        index=[['Random Forest'] * 5, ['CPC', 'ACC', 'RMSE', 'RE', 'LR']]
    )
