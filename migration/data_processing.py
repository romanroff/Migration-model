from geopy.distance import geodesic
import pandas as pd
from config import FEATURE_COLS

from geopy.distance import geodesic
import pandas as pd

def calculate_distances(city_df: pd.DataFrame) -> pd.DataFrame:
    """Вычисление матрицы расстояний между населенными пунктами"""
    distances = []
    
    for i, city1 in city_df.iterrows():
        for j, city2 in city_df.iterrows():
            if i < j and city1["name"] != city2["name"]:
                coord1 = (city1["lat"], city1["lon"])
                coord2 = (city2["lat"], city2["lon"])
                distances.append({
                    "name_o": city1["name"],  # Название начального пункта
                    "name_d": city2["name"],  # Название конечного пункта
                    "lat_o": city1["lat"],    # Широта начального пункта
                    "lon_o": city1["lon"],    # Долгота начального пункта
                    "lat_d": city2["lat"],    # Широта конечного пункта
                    "lon_d": city2["lon"],    # Долгота конечного пункта
                    "d": geodesic(coord1, coord2).miles,  # Расстояние в милях
                    "m_o": city1["population"],  # Население начального пункта
                    "m_d": city2["population"]   # Население конечного пункта
                })
    
    # Возвращаем DataFrame с указанными колонками
    return pd.DataFrame(distances)[FEATURE_COLS + ['name_o', 'name_d', 'lat_o', 'lon_o', 'lat_d', 'lon_d']]