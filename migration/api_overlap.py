import json
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from shapely import wkt
from shapely.geometry import shape

from config import (
    API_OVERLAP_FEATURE_COLS,
    API_OVERLAP_MODEL_PATH,
    LOOP_REQUEST_SLEEP_SECONDS,
    MAIN_INFO_API_URL,
    MAIN_INFO_TIMEOUT,
    TOP_PERCENT,
)
from model_predictions import load_model, make_predictions
from visualization import create_migration_graph

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


MAIN_INFO_FIELD_MAPPING = {
    "m": "Численность населения (чел.)",
    "health_point": "Лечебно-профилактические организации (шт.)",
    "main_road_line": "Длина дорог (км)",
    "school_point": "Число общеобразовательных организаций (шт.)",
}


def fetch_main_info(
    territory_id: int,
    down_by: int = 1,
    base_url: str = MAIN_INFO_API_URL,
    timeout: int = MAIN_INFO_TIMEOUT,
    allow_empty: bool = False,
) -> dict:
    """Запрашивает список дочерних территорий и их признаки из main_info API."""
    query = urlencode(
        {
            "territory_id": territory_id,
            "down_by": down_by,
            "mig_destinations": "false",
        }
    )
    url = f"{base_url}?{query}"
    with urlopen(url, timeout=timeout) as response:
        data = json.load(response)

    if data.get("type") != "FeatureCollection":
        raise ValueError("main_info API не вернул GeoJSON FeatureCollection.")
    if not data.get("features") and not allow_empty:
        raise ValueError("main_info API вернул пустой список территорий.")
    return data


def _deduplicate_features(features: list[dict]) -> list[dict]:
    deduplicated = []
    seen_keys = set()

    for feature in features:
        props = feature.get("properties") or {}
        territory_id = props.get("territory_id")
        dedupe_key = territory_id if territory_id is not None else json.dumps(feature, sort_keys=True, ensure_ascii=False)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduplicated.append(feature)

    return deduplicated


def fetch_main_info_feature_collection(
    territory_id: int,
    down_by: int = 1,
    loop: bool = False,
    base_url: str = MAIN_INFO_API_URL,
    timeout: int = MAIN_INFO_TIMEOUT,
    loop_request_sleep_seconds: float = LOOP_REQUEST_SLEEP_SECONDS,
    show_progress: bool = True,
) -> tuple[dict, list[int], int]:
    """Возвращает итоговый FeatureCollection для прогноза.

    Если ``loop=True``, то:
    1. выполняется запрос ``territory_id -> down_by``,
    2. из ответа берутся все ``territory_id`` первого уровня,
    3. по каждому из них выполняется ещё один запрос с ``down_by=1``,
    4. объединённый второй уровень используется как итоговый набор территорий.
    """
    root_collection = fetch_main_info(
        territory_id=territory_id,
        down_by=down_by,
        base_url=base_url,
        timeout=timeout,
    )
    request_count = 1

    if not loop:
        return root_collection, [], request_count

    first_level_ids = []
    nested_features = []

    for feature in root_collection.get("features", []):
        props = feature.get("properties") or {}
        child_territory_id = props.get("territory_id")
        if child_territory_id is None:
            continue

        child_territory_id = int(child_territory_id)
        first_level_ids.append(child_territory_id)

    child_ids_iterator = tqdm(
        first_level_ids,
        total=len(first_level_ids),
        desc="Second-level API requests",
        disable=not show_progress,
    )

    for index, child_territory_id in enumerate(child_ids_iterator):
        child_collection = fetch_main_info(
            territory_id=child_territory_id,
            down_by=1,
            base_url=base_url,
            timeout=timeout,
            allow_empty=True,
        )
        request_count += 1
        nested_features.extend(child_collection.get("features", []))
        if loop_request_sleep_seconds > 0 and index < len(first_level_ids) - 1:
            sleep(loop_request_sleep_seconds)

    nested_features = _deduplicate_features(nested_features)
    if not nested_features:
        raise ValueError("LOOP=True, но второй уровень не вернул ни одной территории.")

    return {
        "type": "FeatureCollection",
        "features": nested_features,
    }, first_level_ids, request_count


def _to_float(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def build_territory_dataframe(feature_collection: dict) -> pd.DataFrame:
    """Преобразует GeoJSON main_info в таблицу территорий с overlap-признаками."""
    records = []
    geometries = []

    for feature in feature_collection.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry")
        territory_id = props.get("territory_id")
        name = props.get("name")
        centre_point = props.get("centre_point")
        population = _to_float(props.get(MAIN_INFO_FIELD_MAPPING["m"]))

        if territory_id is None or not name or not centre_point or geometry is None or population <= 0:
            continue

        try:
            centre_geom = wkt.loads(centre_point)
            polygon_geom = shape(geometry)
        except Exception:
            continue

        if centre_geom.geom_type != "Point":
            continue

        geometries.append(polygon_geom)
        records.append(
            {
                "territory_id": int(territory_id),
                "name": name,
                "lat": centre_geom.y,
                "lon": centre_geom.x,
                "m": population,
                "health_point": _to_float(props.get(MAIN_INFO_FIELD_MAPPING["health_point"])),
                "main_road_line": _to_float(props.get(MAIN_INFO_FIELD_MAPPING["main_road_line"])),
                "school_point": _to_float(props.get(MAIN_INFO_FIELD_MAPPING["school_point"])),
            }
        )

    if not records:
        raise ValueError("После фильтрации не осталось валидных территорий для прогноза.")

    territories = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    areas = gpd.GeoSeries(geometries, crs="EPSG:4326").to_crs(3857).area / 1_000_000
    territories["area"] = areas.to_numpy()
    territories = territories[
        [
            "territory_id",
            "name",
            "lat",
            "lon",
            "area",
            "m",
            "health_point",
            "main_road_line",
            "school_point",
            "geometry",
        ]
    ].copy()

    return territories


def build_api_overlap_dataset(territories: pd.DataFrame) -> pd.DataFrame:
    """Строит полную направленную OD-матрицу признаков внутри выбранного региона."""
    if len(territories) < 2:
        raise ValueError("Для прогноза нужно минимум две территории.")

    territory_features = territories[
        [
            "territory_id",
            "name",
            "lat",
            "lon",
            "area",
            "m",
            "health_point",
            "main_road_line",
            "school_point",
        ]
    ].copy()

    origin = territory_features.rename(
        columns={
            "territory_id": "territory_id_o",
            "name": "name_o",
            "lat": "lat_o",
            "lon": "lon_o",
            "area": "area_o",
            "m": "m_o",
            "health_point": "health_point_o",
            "main_road_line": "main_road_line_o",
            "school_point": "school_point_o",
        }
    )
    destination = territory_features.rename(
        columns={
            "territory_id": "territory_id_d",
            "name": "name_d",
            "lat": "lat_d",
            "lon": "lon_d",
            "area": "area_d",
            "m": "m_d",
            "health_point": "health_point_d",
            "main_road_line": "main_road_line_d",
            "school_point": "school_point_d",
        }
    )

    od_pairs = origin.merge(destination, how="cross")
    od_pairs = od_pairs[od_pairs["territory_id_o"] != od_pairs["territory_id_d"]].copy()
    od_pairs["d"] = od_pairs.apply(
        lambda row: geodesic((row["lat_o"], row["lon_o"]), (row["lat_d"], row["lon_d"])).miles,
        axis=1,
    )

    ordered_cols = API_OVERLAP_FEATURE_COLS + [
        "territory_id_o",
        "name_o",
        "lat_o",
        "lon_o",
        "territory_id_d",
        "name_d",
        "lat_d",
        "lon_d",
    ]
    return od_pairs[ordered_cols].copy()


def run_api_overlap_pipeline(
    territory_id: int,
    down_by: int = 1,
    loop: bool = False,
    top_percent: int = TOP_PERCENT,
    top_n: int | None = None,
    model_path: str = API_OVERLAP_MODEL_PATH,
    base_url: str = MAIN_INFO_API_URL,
    timeout: int = MAIN_INFO_TIMEOUT,
    loop_request_sleep_seconds: float = LOOP_REQUEST_SLEEP_SECONDS,
    show_progress: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """Полный inference-поток по API main_info для directed OD-прогноза."""
    feature_collection, first_level_ids, request_count = fetch_main_info_feature_collection(
        territory_id=territory_id,
        down_by=down_by,
        loop=loop,
        base_url=base_url,
        timeout=timeout,
        loop_request_sleep_seconds=loop_request_sleep_seconds,
        show_progress=show_progress,
    )
    territories = build_territory_dataframe(feature_collection)
    features = build_api_overlap_dataset(territories)

    model = load_model(model_path=model_path)
    predictions = make_predictions(
        model,
        features,
        feature_cols=API_OVERLAP_FEATURE_COLS,
    )

    loop_suffix = "_loop" if loop else ""
    if output_dir is None:
        output_dir = Path("artifacts") / "api_overlap" / f"territory_{territory_id}_down_{down_by}{loop_suffix}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"predictions_territory_{territory_id}_down_{down_by}{loop_suffix}.csv"
    map_path = output_dir / f"migration_map_territory_{territory_id}_down_{down_by}{loop_suffix}.html"
    predictions.to_csv(predictions_path, index=False)

    migration_map = create_migration_graph(
        predictions,
        territories,
        top_percent=top_percent,
        top_n=top_n,
        node_o="territory_id_o",
        node_d="territory_id_d",
        coord_name="territory_id",
        label_col="total_pop_flow",
        tooltip_cols=[
            "territory_id_o",
            "name_o",
            "territory_id_d",
            "name_d",
            "d",
            "total_pop_flow",
        ],
        polygon_gdf=territories,
        polygon_id_col="territory_id",
        polygon_name_col="name",
        polygon_tooltip_cols=[
            "name",
            "predicted_outflow",
            "predicted_inflow",
            "predicted_balance",
        ],
    )
    migration_map.save(str(map_path))

    return {
        "territories": territories,
        "features": features,
        "predictions": predictions,
        "predictions_path": predictions_path,
        "map_path": map_path,
        "loop": loop,
        "first_level_ids": first_level_ids,
        "request_count": request_count,
        "feature_collection": feature_collection,
    }
