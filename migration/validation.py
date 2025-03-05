from data_fetching import fetch_geodata, process_geo_data
from data_processing import calculate_distances
from model_predictions import load_model, make_predictions
from visualization import create_migration_graph
import pandas as pd
from config import TOP_PERCENT

def pipeline():
    # Шаг 1: Загрузка данных
    print('Шаг 1: Загрузка данных')
    raw_geo = fetch_geodata()
    city_df = process_geo_data(raw_geo)
    
    # Шаг 2: Подготовка данных
    print('Шаг 2: Подготовка данных')
    final_df = calculate_distances(city_df)
    
    # Шаг 3: Прогнозирование
    print('Шаг 3: Прогнозирование')
    model = load_model()
    predictions_df = make_predictions(model, final_df)
    
    # Шаг 4: Визуализация
    print('Шаг 4: Визуализация')
    migration_map = create_migration_graph(
        predictions_df, 
        city_df,
        top_percent=TOP_PERCENT
    )
    migration_map.save("migration_map_py.html")
    print('Запуск завершен. Проверьте файл migration_map_py.html')

if __name__ == "__main__":
    pipeline()


