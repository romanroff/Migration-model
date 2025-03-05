import branca.colormap as cmap
import folium
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def create_migration_graph(df, df_coords, top_percent=100, 
                                        node_o='name_o', node_d='name_d', 
                                        coord_name='name', lat_col='lat', lon_col='lon', 
                                        label_col='total_pop_flow', figsize=(15, 10), 
                                        map_type='folium'):
    """
    Создает граф миграции с топ-% рёбер, подложкой карты, легендой и всплывающими подсказками.
    """
    # Создаем граф
    G = nx.from_pandas_edgelist(df, node_o, node_d, edge_attr=[label_col])

    # Фильтрация топ-% рёбер
    edges = sorted(((u, v, G[u][v]['total_pop_flow']) for u, v in G.edges()), key=lambda x: x[2], reverse=True)
    top_edges = edges[:int(len(edges) * (top_percent / 100))]

    # Новый граф с отфильтрованными рёбрами
    G_filtered = nx.DiGraph()
    G_filtered.add_nodes_from(G.nodes(data=True))
    G_filtered.add_edges_from((u, v, {'total_pop_flow': w}) for u, v, w in top_edges)

    # Определяем позиции узлов
    pos = {
        node: (df_coords[df_coords[coord_name] == node][lon_col].values[0], 
               df_coords[df_coords[coord_name] == node][lat_col].values[0])
        for node in G_filtered.nodes() if node in df_coords[coord_name].values
    }

    # Определяем границы карты
    lon_min, lon_max = min(x[0] for x in pos.values()), max(x[0] for x in pos.values())
    lat_min, lat_max = min(x[1] for x in pos.values()), max(x[1] for x in pos.values())

    # Нормализуем значения предсказаний (логарифмируем и нормируем)
    predictions = np.array([G[u][v][label_col] for u, v in G_filtered.edges()])
    predictions_log = np.log1p([G[u][v][label_col] for u, v in G_filtered.edges()])
    normalizer = QuantileTransformer(n_quantiles=10)
    predictions_norm = normalizer.fit_transform(predictions.reshape(-1, 1)).flatten()

    # Создаем пользовательскую палитру
    custom_colormap = cmap.LinearColormap(
        colors=['#5902d0', '#eded2e'],  # Задаем цвета
        vmin=min(predictions_log),  # Минимальное значение
        vmax=max(predictions_log)   # Максимальное значение
    )

    # Выбираем цвета на основе нормализованных значений
    edge_colors = [custom_colormap(val) for val in predictions_log]

    # Создание карты через folium
    if map_type == 'folium':
        m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=7, tiles="cartodbpositron")
        
        # Создаем цветовую шкалу (легенду)
        colormap_legend = custom_colormap
        colormap_legend.caption = 'Значения предсказаний'
        colormap_legend.add_to(m)

        # Добавление узлов и рёбер
        for node, (lon, lat) in pos.items():
            folium.CircleMarker([lat, lon], radius=5, color="blue", fill=True, fill_opacity=0.6).add_to(m)
        
        # Добавляем рёбра с нормализованными цветами и всплывающими подсказками
        for (u, v), color, pred in zip(G_filtered.edges(), edge_colors, predictions):
            lon1, lat1 = pos[u]
            lon2, lat2 = pos[v]
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=2,
                tooltip=f"Значение: {pred:.2f}"  # Всплывающая подсказка с реальным значением
            ).add_to(m)

        return m
