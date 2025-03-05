import osmnx as ox
import geopandas as gpd
import pandas as pd
from config import REGION_NAME, PLACE_FILTERS

def fetch_geodata() -> gpd.GeoDataFrame:
    """Загрузка географических данных из OSM"""
    return ox.features_from_place(REGION_NAME, PLACE_FILTERS)

def process_geo_data(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Преобразование сырых геоданных в DataFrame"""
    filtered = gdf[["name", "geometry", "population"]].dropna(subset=["population"])
    
    city_data = [
        {
            "name": row["name"],
            "population": row["population"],
            "lat": row["geometry"].y,
            "lon": row["geometry"].x
        }
        for _, row in filtered.iterrows()
        if row["geometry"].geom_type == "Point"
    ]
    
    return pd.DataFrame(city_data).drop_duplicates()
