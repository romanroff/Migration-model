import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from api_overlap import build_api_overlap_dataset, build_territory_dataframe
from config import API_OVERLAP_FEATURE_COLS
from data_loading import load_datasets
from model_predictions import make_predictions
from model_training import train_and_save_model
from visualization import build_directed_graph, create_migration_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_response_feature_collection(index: int = 0) -> dict:
    outer = json.loads((PROJECT_ROOT / "response1.json").read_text(encoding="utf-8"))
    return json.loads(outer[index])


def test_feature_mapping_from_response_json():
    territories = build_territory_dataframe(load_response_feature_collection(0))

    expected_columns = {
        "territory_id",
        "name",
        "lat",
        "lon",
        "area",
        "m",
        "health_point",
        "main_road_line",
        "school_point",
    }
    assert expected_columns.issubset(territories.columns)
    assert len(territories) > 0
    assert territories["m"].gt(0).all()
    assert territories["area"].gt(0).all()


def test_pair_builder_creates_full_directed_matrix():
    territories = build_territory_dataframe(load_response_feature_collection(0)).head(3).copy()
    dataset = build_api_overlap_dataset(territories)

    assert len(dataset) == 6
    assert set(API_OVERLAP_FEATURE_COLS).issubset(dataset.columns)
    assert dataset[API_OVERLAP_FEATURE_COLS].isna().sum().sum() == 0


def test_visualization_preserves_both_edge_directions():
    predictions = pd.DataFrame(
        [
            {
                "territory_id_o": 1,
                "name_o": "A",
                "territory_id_d": 2,
                "name_d": "B",
                "d": 10.0,
                "total_pop_flow": 5.0,
            },
            {
                "territory_id_o": 2,
                "name_o": "B",
                "territory_id_d": 1,
                "name_d": "A",
                "d": 10.0,
                "total_pop_flow": 7.0,
            },
        ]
    )
    territories = pd.DataFrame(
        [
            {"territory_id": 1, "name": "A", "lat": 59.0, "lon": 30.0},
            {"territory_id": 2, "name": "B", "lat": 60.0, "lon": 31.0},
        ]
    )

    graph = build_directed_graph(
        predictions,
        node_o="territory_id_o",
        node_d="territory_id_d",
        label_col="total_pop_flow",
    )

    assert graph.number_of_edges() == 2
    assert graph.has_edge(1, 2)
    assert graph.has_edge(2, 1)

    migration_map = create_migration_graph(
        predictions,
        territories,
        top_n=2,
        node_o="territory_id_o",
        node_d="territory_id_d",
        coord_name="territory_id",
        tooltip_cols=["territory_id_o", "name_o", "territory_id_d", "name_d", "d", "total_pop_flow"],
    )
    assert migration_map is not None


def test_overlap_training_smoke(tmp_path):
    train_data, test_data, _ = load_datasets("data", train_variant="sample")
    model_path = tmp_path / "rf_model_api_overlap.joblib"

    model, results = train_and_save_model(
        train_data=train_data,
        test_data=test_data,
        model_path=str(model_path),
        feature_cols=API_OVERLAP_FEATURE_COLS,
        n_estimators=10,
        random_state=0,
        verbose=0,
    )

    assert model is not None
    assert model_path.exists()
    assert not results.empty
    assert "California" in results.columns


def test_overlap_inference_smoke():
    train_data, _, _ = load_datasets("data", train_variant="sample")
    all_train = pd.concat(train_data.values()).head(300)
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(all_train[API_OVERLAP_FEATURE_COLS], np.log(all_train["total_pop_flow"]))

    territories = build_territory_dataframe(load_response_feature_collection(0)).head(4).copy()
    features = build_api_overlap_dataset(territories)
    predictions = make_predictions(model, features, feature_cols=API_OVERLAP_FEATURE_COLS)

    assert set(API_OVERLAP_FEATURE_COLS).issubset(predictions.columns)
    assert predictions[API_OVERLAP_FEATURE_COLS].isna().sum().sum() == 0
    assert predictions["total_pop_flow"].gt(0).all()
