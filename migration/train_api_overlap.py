from IPython.display import display

from config import API_OVERLAP_FEATURE_COLS, API_OVERLAP_MODEL_PATH, DATA_PATH
from data_loading import load_datasets
from model_training import train_and_save_model


def training_pipeline():
    print("Шаг 1: Загрузка данных")
    train_data, test_data, _ = load_datasets(DATA_PATH, train_variant="full")
    print(f"Train rows total: {sum(len(df) for df in train_data.values())}")
    print(f"Test rows total: {sum(len(df) for df in test_data.values())}")

    print("Шаг 2: Обучение overlap-модели")
    model, results = train_and_save_model(
        train_data=train_data,
        test_data=test_data,
        model_path=API_OVERLAP_MODEL_PATH,
        feature_cols=API_OVERLAP_FEATURE_COLS,
    )

    print("Результаты оценки overlap-модели:")
    display(results)
    return model, results


if __name__ == "__main__":
    training_pipeline()
