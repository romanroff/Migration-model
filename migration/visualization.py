import branca.colormap as cmap
import folium
import networkx as nx
import math
import numpy as np
import pandas as pd
from folium.features import GeoJsonTooltip

FLOW_TOOLTIP_COLUMNS = {
    "total_pop_flow",
    "predicted_outflow",
    "predicted_inflow",
    "predicted_balance",
}


def build_directed_graph(df, node_o='name_o', node_d='name_d', label_col='total_pop_flow'):
    """Строит направленный граф без схлопывания рёбер A->B и B->A."""
    graph = nx.DiGraph()
    for row in df[[node_o, node_d, label_col]].itertuples(index=False):
        graph.add_edge(row[0], row[1], **{label_col: row[2]})
    return graph


def select_top_flows(df, label_col='total_pop_flow', top_percent=100, top_n=None):
    """Возвращает топ-потоки, сохраняя направление."""
    sorted_df = df.sort_values(label_col, ascending=False).reset_index(drop=True)
    if sorted_df.empty:
        return sorted_df

    if top_n is not None:
        count = min(len(sorted_df), max(1, int(top_n)))
    else:
        count = max(1, math.ceil(len(sorted_df) * (top_percent / 100)))
    return sorted_df.head(count).copy()


def _build_tooltip(row, label_col, tooltip_cols):
    if tooltip_cols:
        values = [
            f"{col}: {_format_tooltip_value(col, row[col])}"
            for col in tooltip_cols
            if col in row.index and pd.notna(row[col])
        ]
        return "<br>".join(values)
    return f"Значение: {_format_tooltip_value(label_col, row[label_col])}"


def _format_tooltip_value(column, value):
    if column in FLOW_TOOLTIP_COLUMNS:
        return int(round(float(value)))
    return value


def build_polygon_flow_stats(
    predictions_df,
    polygons_gdf,
    polygon_id_col,
    node_o='name_o',
    node_d='name_d',
    label_col='total_pop_flow',
):
    """Считает прогнозный outflow/inflow/balance по полигонам."""
    polygons = polygons_gdf.copy()

    outflow = (
        predictions_df.groupby(node_o)[label_col]
        .sum()
        .rename("predicted_outflow")
        .reset_index()
        .rename(columns={node_o: polygon_id_col})
    )
    inflow = (
        predictions_df.groupby(node_d)[label_col]
        .sum()
        .rename("predicted_inflow")
        .reset_index()
        .rename(columns={node_d: polygon_id_col})
    )

    polygons = polygons.merge(outflow, on=polygon_id_col, how="left")
    polygons = polygons.merge(inflow, on=polygon_id_col, how="left")
    polygons[["predicted_outflow", "predicted_inflow"]] = polygons[["predicted_outflow", "predicted_inflow"]].fillna(0.0)
    polygons["predicted_balance"] = polygons["predicted_inflow"] - polygons["predicted_outflow"]
    return polygons


def _add_polygon_layer(
    folium_map,
    polygon_stats_gdf,
    polygon_value_col,
    polygon_name_col=None,
    polygon_tooltip_cols=None,
):
    polygon_values = polygon_stats_gdf[polygon_value_col].fillna(0.0)
    max_abs = float(np.abs(polygon_values).max()) if len(polygon_values) else 0.0
    max_abs = max(max_abs, 1.0)

    polygon_colormap = cmap.LinearColormap(
        colors=["#b2182b", "#f7f7f7", "#1a9850"],
        index=[-max_abs, 0.0, max_abs],
        vmin=-max_abs,
        vmax=max_abs,
    )
    polygon_colormap.caption = "Баланс миграции: красный -> уехало, зелёный -> приехало"
    polygon_colormap.add_to(folium_map)

    tooltip_cols = polygon_tooltip_cols or []
    tooltip_df = polygon_stats_gdf.copy()
    tooltip_fields = []
    tooltip_aliases = []

    for column in tooltip_cols:
        tooltip_aliases.append(f"{column}:")
        if column in FLOW_TOOLTIP_COLUMNS and column in tooltip_df.columns:
            display_column = f"{column}_display"
            tooltip_df[display_column] = tooltip_df[column].round().astype("Int64")
            tooltip_fields.append(display_column)
        else:
            tooltip_fields.append(column)

    geojson = folium.GeoJson(
        data=tooltip_df.to_json(),
        name="Территории",
        style_function=lambda feature: {
            "fillColor": polygon_colormap(feature["properties"].get(polygon_value_col, 0.0)),
            "color": "#404040",
            "weight": 1,
            "fillOpacity": 0.55,
        },
        highlight_function=lambda _: {
            "weight": 2,
            "fillOpacity": 0.75,
        },
        tooltip=GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            labels=True,
            sticky=False,
        ) if tooltip_fields else None,
    )
    geojson.add_to(folium_map)

    if polygon_name_col and polygon_name_col in polygon_stats_gdf.columns:
        for row in polygon_stats_gdf[[polygon_name_col, "geometry"]].dropna().itertuples(index=False):
            centroid = row[1].representative_point()
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=(
                        '<div style="font-size: 10px; color: #222; '
                        'background: rgba(255,255,255,0.75); padding: 1px 3px; '
                        'border-radius: 3px;">'
                        f'{row[0]}</div>'
                    )
                ),
            ).add_to(folium_map)


def create_migration_graph(
    df,
    df_coords,
    top_percent=100,
    top_n=None,
    node_o='name_o',
    node_d='name_d',
    coord_name='name',
    lat_col='lat',
    lon_col='lon',
    label_col='total_pop_flow',
    figsize=(15, 10),
    map_type='folium',
    tooltip_cols=None,
    polygon_gdf=None,
    polygon_id_col=None,
    polygon_name_col=None,
    polygon_value_col='predicted_balance',
    polygon_tooltip_cols=None,
):
    """
    Создает граф миграции с топ-% рёбер, подложкой карты, легендой и всплывающими подсказками.
    """
    top_df = select_top_flows(df, label_col=label_col, top_percent=top_percent, top_n=top_n)
    graph = build_directed_graph(top_df, node_o=node_o, node_d=node_d, label_col=label_col)
    if graph.number_of_edges() == 0:
        raise ValueError("Для визуализации нет рёбер.")

    coords = df_coords.drop_duplicates(subset=[coord_name]).set_index(coord_name)

    # Определяем позиции узлов
    pos = {
        node: (coords.at[node, lon_col], coords.at[node, lat_col])
        for node in graph.nodes()
        if node in coords.index
    }
    if not pos:
        raise ValueError("Не удалось сопоставить координаты узлам графа.")

    # Определяем границы карты
    lon_min, lon_max = min(x[0] for x in pos.values()), max(x[0] for x in pos.values())
    lat_min, lat_max = min(x[1] for x in pos.values()), max(x[1] for x in pos.values())

    predictions = np.array(top_df[label_col])
    predictions_log = np.log1p(predictions)

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

        if polygon_gdf is not None:
            if polygon_id_col is None:
                raise ValueError("Для полигонального слоя нужно передать polygon_id_col.")
            polygon_stats_gdf = build_polygon_flow_stats(
                predictions_df=df,
                polygons_gdf=polygon_gdf,
                polygon_id_col=polygon_id_col,
                node_o=node_o,
                node_d=node_d,
                label_col=label_col,
            )
            _add_polygon_layer(
                folium_map=m,
                polygon_stats_gdf=polygon_stats_gdf,
                polygon_value_col=polygon_value_col,
                polygon_name_col=polygon_name_col,
                polygon_tooltip_cols=polygon_tooltip_cols,
            )
        
        # Создаем цветовую шкалу (легенду)
        colormap_legend = custom_colormap
        colormap_legend.caption = 'Значения предсказаний'
        colormap_legend.add_to(m)

        nodes_layer = folium.FeatureGroup(name="Точки", show=True)
        flows_layer = folium.FeatureGroup(name="OD-линии", show=True)

        # Добавление узлов и рёбер
        for node, (lon, lat) in pos.items():
            folium.CircleMarker([lat, lon], radius=5, color="blue", fill=True, fill_opacity=0.6).add_to(nodes_layer)
        
        # Добавляем рёбра с нормализованными цветами и всплывающими подсказками
        for (_, row), color in zip(top_df.iterrows(), edge_colors):
            if row[node_o] not in pos or row[node_d] not in pos:
                continue
            lon1, lat1 = pos[row[node_o]]
            lon2, lat2 = pos[row[node_d]]
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=2,
                tooltip=_build_tooltip(row, label_col=label_col, tooltip_cols=tooltip_cols),
            ).add_to(flows_layer)

        nodes_layer.add_to(m)
        flows_layer.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        return m
