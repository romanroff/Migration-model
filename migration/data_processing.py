from geopy.distance import geodesic
import pandas as pd
from config import FEATURE_COLS

def calculate_distances(city_df: pd.DataFrame) -> pd.DataFrame:
    """Вычисление матрицы расстояний между населенными пунктами"""
    distances = []
    
    for i, city1 in city_df.iterrows():
        for j, city2 in city_df.iterrows():
            if i < j and city1["name"] != city2["name"]:
                coord1 = (city1["lat"], city1["lon"])
                coord2 = (city2["lat"], city2["lon"])
                distances.append({
                    "name_o": city1["name"],
                    "name_d": city2["name"],
                    "d": geodesic(coord1, coord2).km,
                    "m_o": city1["population"],
                    "m_d": city2["population"]
                })
    
    return pd.DataFrame(distances)[FEATURE_COLS + ['name_o', 'name_d']]