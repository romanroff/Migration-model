from IPython.display import display
from data_loading import load_datasets
from model_training import train_and_save_model, evaluate_model
from config import DATA_PATH

def training_pipeline():
    # Загрузка данных
    print('Шаг 1: Загрузка данных')
    train_data, test_data, _ = load_datasets(DATA_PATH)
    
    # Обучение и сохранение модели
    print('Шаг 2: Обучение и сохранение модели')
    model, results = train_and_save_model(train_data, test_data)
    
    # Вывод результатов
    print("Результаты оценки модели:")
    display(results)
    
    # Дополнительная логика сохранения отчетов


if __name__ == "__main__":
    training_pipeline()